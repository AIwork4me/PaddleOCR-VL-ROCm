from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from .constants import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    MARKDOWN_IGNORE_LABEL_SET,
    MARKDOWN_IGNORE_LABELS,
    NON_MERGE_LABELS,
    SKIP_ORDER_LABELS,
)
from .content import _normalize_vlm_result, _truncate_repetitive_content
from .encoding import _jpeg_bytes, _png_bytes, _sha256_hex
from .geometry import _filter_overlap_boxes
from .imageio import _crop_margin, _open_crop_source, _open_crop_source_bgr
from .layout import PADDLEOCR_VL_LAYOUT_MERGE_MODE, LayoutBox, PPDocLayoutV3Onnx
from .markdown import _markdown_from_blocks
from .models import LightBlock
from .preprocess import (
    _construct_img_path,
    _gather_imgs_for_table_tokens,
    _make_blocks,
    _merge_blocks,
    _tokenize_figure_of_table,
    _untokenize_figure_of_table,
)
from .server import check_openai_compatible_server
from .table import _convert_otsl_to_html
from .utils import write_json
from .vlm.client import LlamaCppClient, _load_vlm_compat_cache, _prompt_for_label


def _result_payload(
    input_path: Path,
    display_input_path: str,
    width: int,
    height: int,
    boxes: list[LayoutBox],
    blocks: list[LightBlock],
    use_layout_detection: bool,
    use_chart_recognition: bool,
    use_seal_recognition: bool,
) -> dict[str, Any]:
    return {
        "input_path": display_input_path,
        "page_index": None,
        "page_count": None,
        "width": width,
        "height": height,
        "model_settings": {
            "use_doc_preprocessor": False,
            "use_layout_detection": use_layout_detection,
            "use_chart_recognition": use_chart_recognition,
            "use_seal_recognition": use_seal_recognition,
            "use_ocr_for_image_block": False,
            "format_block_content": False,
            "merge_layout_blocks": True,
            "markdown_ignore_labels": MARKDOWN_IGNORE_LABELS,
            "return_layout_polygon_points": True,
        },
        "parsing_res_list": [
            _block_to_json(block, idx, order)
            for idx, (block, order) in enumerate(_blocks_with_orders(blocks))
        ],
        "layout_det_res": {
            "input_path": None,
            "page_index": None,
            "boxes": _layout_boxes_to_json(boxes),
        },
    }


def _layout_boxes_to_json(boxes: list[LayoutBox]) -> list[dict[str, Any]]:
    result = []
    order_index = 1
    for box in boxes:
        item = box.to_dict()
        if box.label in SKIP_ORDER_LABELS:
            item["order"] = None
        else:
            item["order"] = order_index
            order_index += 1
        result.append(item)
    return result


def _blocks_with_orders(blocks: list[LightBlock]) -> list[tuple[LightBlock, int | None]]:
    result: list[tuple[LightBlock, int | None]] = []
    order_index = 1
    skip_labels = SKIP_ORDER_LABELS | MARKDOWN_IGNORE_LABEL_SET
    for block in blocks:
        if block.label in skip_labels:
            order = None
        else:
            order = order_index
            order_index += 1
        result.append((block, order))
    return result


def _block_to_json(block: LightBlock, idx: int, order: int | None) -> dict[str, Any]:
    polygon_points = block.polygon_points or [
        [float(block.bbox[0]), float(block.bbox[1])],
        [float(block.bbox[2]), float(block.bbox[1])],
        [float(block.bbox[2]), float(block.bbox[3])],
        [float(block.bbox[0]), float(block.bbox[3])],
    ]
    return {
        "block_label": block.label,
        "block_content": block.content,
        "block_bbox": block.bbox,
        "block_id": idx,
        "block_order": order,
        "group_id": block.group_id if block.group_id is not None else idx,
        "block_polygon_points": polygon_points,
    }


def run_light_parser(
    input_path: Path,
    output_dir: Path,
    model_dir: Path,
    server_url: str,
    vlm_backend: str,
    api_model_name: str,
    max_new_tokens: int | None,
    timeout: float,
    prompt_label: str | None,
    use_layout_detection: bool,
    use_chart_recognition: bool,
    use_seal_recognition: bool,
    seed: int,
    threshold: float,
    compat_cache_path: Path | None = None,
    display_input_path: str | None = None,
    vlm_repeats: int = 1,
    vlm_max_workers: int = 200,
    layout_model: PPDocLayoutV3Onnx | None = None,
    skip_server_check: bool = False,
    vlm_trace_events: list[dict[str, Any]] | None = None,
) -> Path:
    compat_cache = _load_vlm_compat_cache(compat_cache_path)
    if not compat_cache and not skip_server_check:
        check_openai_compatible_server(server_url)
    client = LlamaCppClient(
        server_url,
        api_model_name,
        timeout=timeout,
        backend=vlm_backend,
        seed=seed,
        compat_cache=compat_cache,
    )
    full_image = _open_crop_source(input_path)
    bgr_image = _open_crop_source_bgr(input_path)
    width, height = full_image.size

    if use_layout_detection:
        layout = layout_model or PPDocLayoutV3Onnx(model_dir)
        boxes, _ = layout.predict(
            input_path,
            threshold=threshold,
            layout_nms=True,
            layout_merge_bboxes_mode=PADDLEOCR_VL_LAYOUT_MERGE_MODE,
        )
        figures_in_doc = _gather_imgs_for_table_tokens(boxes)
    else:
        label = (prompt_label or "ocr").lower()
        boxes = [
            LayoutBox(
                cls_id=0,
                label=label,
                score=1.0,
                coordinate=[0, 0, width, height],
                order=0,
            )
        ]
        figures_in_doc = []

    if use_layout_detection:
        blocks = _merge_blocks(
            _make_blocks(full_image, _filter_overlap_boxes(boxes), bgr_image=bgr_image),
            non_merge_labels=NON_MERGE_LABELS,
        )
    else:
        blocks = _make_blocks(full_image, boxes, bgr_image=bgr_image)

    vlm_tasks = []
    drop_figures_set: set[str] = set()
    for block in blocks:
        prompt = _prompt_for_label(
            block.label,
            use_chart_recognition=use_chart_recognition,
            use_seal_recognition=use_seal_recognition,
        )
        if prompt is None or block.image is None:
            block.content = ""
        else:
            image_for_vlm = block.image
            if block.label == "table":
                image_for_vlm, block.figure_token_map, drop_figures = _tokenize_figure_of_table(
                    image_for_vlm,
                    block.bbox,
                    figures_in_doc,
                )
                drop_figures_set.update(drop_figures)
            if "formula" in block.label and block.label != "formula_number":
                image_for_vlm = _crop_margin(image_for_vlm)
            trace_event: dict[str, Any] | None = None
            if vlm_trace_events is not None:
                image_bytes = (
                    _jpeg_bytes(image_for_vlm)
                    if vlm_backend == "vllm-server"
                    else _png_bytes(image_for_vlm)
                )
                trace_event = {
                    "backend": vlm_backend,
                    "model": api_model_name,
                    "request_order": len(vlm_trace_events),
                    "block_label": block.label,
                    "block_bbox": block.bbox,
                    "prompt": prompt,
                    "image_format": "JPEG" if vlm_backend == "vllm-server" else "PNG",
                    "image_sha256": _sha256_hex(image_bytes),
                    "image_size": list(image_for_vlm.size),
                    "max_new_tokens": max_new_tokens,
                    "min_pixels": DEFAULT_MIN_PIXELS,
                    "max_pixels": DEFAULT_MAX_PIXELS,
                    "skip_special_tokens": True,
                }
                vlm_trace_events.append(trace_event)
            vlm_tasks.append((block, prompt, image_for_vlm, trace_event))

    def _run_vlm_task(
        task: tuple[LightBlock, str, Image.Image, dict[str, Any] | None],
    ) -> tuple[LightBlock, str, dict[str, Any] | None]:
        block, prompt, image_for_vlm, trace_event = task
        if vlm_repeats <= 1:
            content = client.complete_image(
                prompt,
                image=image_for_vlm,
                max_new_tokens=max_new_tokens,
                min_pixels=DEFAULT_MIN_PIXELS,
                max_pixels=DEFAULT_MAX_PIXELS,
            )
        else:
            candidates = [
                client.complete_image(
                    prompt,
                    image=image_for_vlm,
                    max_new_tokens=max_new_tokens,
                    min_pixels=DEFAULT_MIN_PIXELS,
                    max_pixels=DEFAULT_MAX_PIXELS,
                    use_client_cache=False,
                )
                for _ in range(vlm_repeats)
            ]
            counts = Counter(candidates)
            content = max(candidates, key=lambda item: (counts[item], -candidates.index(item)))
        return block, content, trace_event

    if vlm_tasks:
        max_workers = min(max(1, vlm_max_workers), len(vlm_tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for block, content, trace_event in executor.map(_run_vlm_task, vlm_tasks):
                if trace_event is not None:
                    trace_event["raw_result_sha256"] = _sha256_hex(content.encode("utf-8"))
                    trace_event["raw_result_chars"] = len(content)
                min_count = 5000 if block.label == "table" else 50
                content = _truncate_repetitive_content(content, min_count=min_count)
                content = _normalize_vlm_result(block.label, content)
                if block.label == "table":
                    table_html = _convert_otsl_to_html(content)
                    if table_html:
                        content = table_html
                    if block.figure_token_map:
                        content = _untokenize_figure_of_table(content, block.figure_token_map)
                block.content = content
                if trace_event is not None:
                    trace_event["final_result_sha256"] = _sha256_hex(content.encode("utf-8"))
                    trace_event["final_result_chars"] = len(content)

    if drop_figures_set:
        blocks = [
            block
            for block in blocks
            if _construct_img_path(block.label, block.bbox) not in drop_figures_set
        ]

    result = _result_payload(
        input_path,
        display_input_path=display_input_path or str(input_path),
        width=width,
        height=height,
        boxes=boxes,
        blocks=blocks,
        use_layout_detection=use_layout_detection,
        use_chart_recognition=use_chart_recognition,
        use_seal_recognition=use_seal_recognition,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(output_dir / "result.json", result)
    markdown = _markdown_from_blocks(blocks, width)
    if markdown:
        (output_dir / "result.md").write_text(markdown, encoding="utf-8")
    return json_path
