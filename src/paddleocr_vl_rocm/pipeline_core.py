from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local
from time import perf_counter
from typing import Any

from PIL import Image

from .constants import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    DEFAULT_VLM_MAX_WORKERS,
    NON_MERGE_LABELS,
)
from .content import _normalize_vlm_result, _truncate_repetitive_content
from .contracts import VLMRequestContract
from .encoding import _sha256_hex
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
from .serialize import _result_payload
from .server import check_openai_compatible_server
from .table import _convert_otsl_to_html
from .timing import _covered_seconds
from .utils import write_json, write_text_lf
from .vlm.client import LlamaCppClient, _load_vlm_compat_cache, _prompt_for_label


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
    vlm_max_workers: int = DEFAULT_VLM_MAX_WORKERS,
    layout_model: PPDocLayoutV3Onnx | None = None,
    skip_server_check: bool = False,
    vlm_trace_events: list[dict[str, Any]] | None = None,
    layout_provider_requested: str | None = None,
    layout_providers_active: list[str] | None = None,
    timing_events: list[dict[str, float]] | None = None,
) -> Path:
    timing_enabled = timing_events is not None
    total_started = perf_counter() if timing_enabled else 0.0
    compat_cache = _load_vlm_compat_cache(compat_cache_path)
    if not compat_cache and not skip_server_check:
        check_openai_compatible_server(server_url)
    trace_context = local()

    def _observe_vlm_request(request: VLMRequestContract) -> None:
        trace_event = getattr(trace_context, "event", None)
        if trace_event is None:
            return
        mm_processor_kwargs = request.payload.get("mm_processor_kwargs", {})
        trace_event.update(
            {
                "backend": request.backend,
                "model": request.model,
                "prompt": request.prompt,
                "image_format": request.image_format,
                "image_sha256": request.image_sha256,
                "image_size": list(request.image_size),
                "max_new_tokens": request.payload.get(
                    "max_tokens", request.payload.get("max_completion_tokens")
                ),
                "min_pixels": mm_processor_kwargs.get("min_pixels"),
                "max_pixels": mm_processor_kwargs.get("max_pixels"),
                "skip_special_tokens": request.payload.get("skip_special_tokens"),
                "payload": request.payload,
            }
        )

    def _observe_vlm_timing(event: dict[str, float]) -> None:
        task_timings = getattr(trace_context, "timings", None)
        if task_timings is not None:
            task_timings.append(event)

    client = LlamaCppClient(
        server_url,
        api_model_name,
        timeout=timeout,
        backend=vlm_backend,
        seed=seed,
        compat_cache=compat_cache,
        request_observer=_observe_vlm_request if vlm_trace_events is not None else None,
        timing_observer=_observe_vlm_timing if timing_enabled else None,
    )
    decode_started = perf_counter() if timing_enabled else 0.0
    full_image = _open_crop_source(input_path)
    bgr_image = _open_crop_source_bgr(input_path)
    width, height = full_image.size
    decode_seconds = perf_counter() - decode_started if timing_enabled else 0.0

    layout_started = perf_counter() if timing_enabled else 0.0
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
    layout_seconds = perf_counter() - layout_started if timing_enabled else 0.0

    crop_encode_started = perf_counter() if timing_enabled else 0.0
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
                trace_event = {
                    "request_order": len(vlm_trace_events),
                    "block_label": block.label,
                    "block_bbox": block.bbox,
                    "layout_provider_requested": layout_provider_requested,
                    "layout_providers_active": list(layout_providers_active or []),
                }
                vlm_trace_events.append(trace_event)
            task_timings: list[dict[str, float]] | None = [] if timing_enabled else None
            vlm_tasks.append((block, prompt, image_for_vlm, trace_event, task_timings))
    crop_prepare_seconds = perf_counter() - crop_encode_started if timing_enabled else 0.0

    def _run_vlm_task(
        task: tuple[
            LightBlock,
            str,
            Image.Image,
            dict[str, Any] | None,
            list[dict[str, float]] | None,
        ],
    ) -> tuple[LightBlock, str, dict[str, Any] | None, list[dict[str, float]] | None]:
        block, prompt, image_for_vlm, trace_event, task_timings = task
        trace_context.event = trace_event
        trace_context.timings = task_timings
        try:
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
        finally:
            trace_context.event = None
            trace_context.timings = None
        return block, content, trace_event, task_timings

    vlm_timing_events: list[dict[str, float]] = []
    if vlm_tasks:
        max_workers = min(max(1, vlm_max_workers), len(vlm_tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for block, content, trace_event, task_timings in executor.map(_run_vlm_task, vlm_tasks):
                if task_timings is not None:
                    vlm_timing_events.extend(task_timings)
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
    encode_seconds = _covered_seconds(
        [(event["encode_started"], event["encode_finished"]) for event in vlm_timing_events]
    )
    request_seconds = _covered_seconds(
        [(event["request_started"], event["request_finished"]) for event in vlm_timing_events]
    )
    crop_encode_seconds = crop_prepare_seconds + encode_seconds
    vlm_seconds = request_seconds

    finalize_started = perf_counter() if timing_enabled else 0.0
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
        write_text_lf(output_dir / "result.md", markdown)
    finalize_seconds = perf_counter() - finalize_started if timing_enabled else 0.0
    if timing_events is not None:
        timing_events.append(
            {
                "decode_seconds": decode_seconds,
                "layout_seconds": layout_seconds,
                "crop_encode_seconds": crop_encode_seconds,
                "vlm_seconds": vlm_seconds,
                "finalize_seconds": finalize_seconds,
                "total_seconds": perf_counter() - total_started,
            }
        )
    return json_path
