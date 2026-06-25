from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .constants import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    IMAGE_LABELS,
    MARKDOWN_IGNORE_LABEL_SET,
    MARKDOWN_IGNORE_LABELS,
    NON_MERGE_LABELS,
    PROMPTS,
    SKIP_ORDER_LABELS,
)
from .content import _normalize_vlm_result, _truncate_repetitive_content
from .encoding import (
    _data_url_from_bytes,
    _image_data_url,
    _jpeg_bytes,
    _png_bytes,
    _sha256_hex,
)
from .geometry import _filter_overlap_boxes, _overlap_ratio, _projection_overlap_ratio
from .imageio import (
    _crop,
    _crop_from_bgr,
    _crop_margin,
    _merge_images,
    _open_crop_source,
    _open_crop_source_bgr,
)
from .layout import PADDLEOCR_VL_LAYOUT_MERGE_MODE, LayoutBox, PPDocLayoutV3Onnx
from .models import LightBlock
from .server import check_openai_compatible_server, normalize_server_url
from .table import _convert_otsl_to_html
from .utils import write_json


def _vlm_cache_key(
    model: str,
    prompt: str,
    image_sha256: str,
    max_new_tokens: int | None,
    seed: int,
) -> str:
    return json.dumps(
        {
            "schema": "paddleocr-vl-local-vlm-cache-v1",
            "model": model,
            "prompt": prompt,
            "image_sha256": image_sha256,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _load_vlm_compat_cache(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("entries"), dict):
        data = data["entries"]
    if not isinstance(data, dict):
        raise ValueError(f"Invalid VLM compatibility cache: {path}")
    cache: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"Invalid VLM compatibility cache entry in {path}")
        cache[key] = value
    return cache


def _content_from_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _completion_payload(
    backend: str,
    model: str,
    prompt: str,
    image_url: str,
    max_new_tokens: int | None,
    seed: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    if backend == "llama-cpp-server":
        payload.update(
            {
                "seed": seed,
                "top_p": 1.0,
                "skip_special_tokens": True,
                "top_k": 1,
                "min_p": 0.0,
                "repeat_penalty": 1.0,
                "cache_prompt": False,
            }
        )
        if max_new_tokens is not None:
            payload["max_tokens"] = max_new_tokens
    elif backend == "vllm-server":
        payload["skip_special_tokens"] = True
        if min_pixels is not None or max_pixels is not None:
            payload["mm_processor_kwargs"] = {}
            if min_pixels is not None:
                payload["mm_processor_kwargs"]["min_pixels"] = min_pixels
            if max_pixels is not None:
                payload["mm_processor_kwargs"]["max_pixels"] = max_pixels
        if max_new_tokens is not None:
            payload["max_completion_tokens"] = max_new_tokens
    elif max_new_tokens is not None:
        payload["max_tokens"] = max_new_tokens
    return payload


class OpenAICompatibleVLMClient:
    def __init__(
        self,
        server_url: str,
        model: str,
        timeout: float,
        backend: str = "llama-cpp-server",
        seed: int = 1,
        compat_cache: dict[str, str] | None = None,
    ) -> None:
        if backend not in {"llama-cpp-server", "vllm-server"}:
            raise ValueError(
                "Unsupported VLM backend for the lightweight parser: "
                f"{backend}. Expected 'llama-cpp-server' or 'vllm-server'."
            )
        self.backend = backend
        self.base_url = normalize_server_url(server_url)
        self.model = model
        self.timeout = timeout
        self.seed = seed
        self._cache: dict[str, str] = {}
        self._compat_cache = compat_cache or {}

    def complete_image(
        self,
        prompt: str,
        image: Image.Image | None = None,
        image_path: Path | None = None,
        max_new_tokens: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        use_client_cache: bool = True,
    ) -> str:
        if image is None and image_path is None:
            raise ValueError("Either image or image_path is required.")
        if image is not None:
            if self.backend == "vllm-server":
                image_bytes = _jpeg_bytes(image)
                image_url = _data_url_from_bytes(image_bytes, "image/jpeg")
            else:
                image_bytes = _png_bytes(image)
                image_url = _data_url_from_bytes(image_bytes, "image/png")
        else:
            image_path = image_path  # type: ignore[assignment]
            image_bytes = image_path.read_bytes()  # type: ignore[union-attr]
            image_url = _image_data_url(image_path)  # type: ignore[arg-type]
        image_sha256 = _sha256_hex(image_bytes)
        cache_key = _vlm_cache_key(
            self.model,
            prompt=prompt,
            image_sha256=image_sha256,
            max_new_tokens=max_new_tokens,
            seed=self.seed,
        )
        if use_client_cache and cache_key in self._cache:
            return self._cache[cache_key]
        if use_client_cache and cache_key in self._compat_cache:
            text = self._compat_cache[cache_key]
            self._cache[cache_key] = text
            return text
        payload = _completion_payload(
            self.backend,
            self.model,
            prompt=prompt,
            image_url=image_url,
            max_new_tokens=max_new_tokens,
            seed=self.seed,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError("VLM request failed without a response.")
        text = _content_from_response(response.json())
        if use_client_cache:
            self._cache[cache_key] = text
        return text


LlamaCppClient = OpenAICompatibleVLMClient


def _prompt_for_label(
    label: str, use_chart_recognition: bool, use_seal_recognition: bool
) -> str | None:
    if label in IMAGE_LABELS:
        return None
    if label == "chart" and not use_chart_recognition:
        return None
    if label == "seal" and not use_seal_recognition:
        return None
    if "formula" in label and label != "formula_number":
        return "Formula Recognition:"
    return PROMPTS.get(label, "OCR:")


def _construct_img_path(label: str, box: list[int] | tuple[int, int, int, int]) -> str:
    x_min, y_min, x_max, y_max = [int(v) for v in box]
    return f"imgs/img_in_{label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg"


def _gather_imgs_for_table_tokens(boxes: list[LayoutBox]) -> list[dict[str, Any]]:
    figures = []
    for box in boxes:
        if box.label not in {"image", "figure", "seal"}:
            continue
        figures.append(
            {
                "path": _construct_img_path(box.label, box.coordinate),
                "label": box.label,
                "coordinate": tuple(int(v) for v in box.coordinate),
                "score": box.score,
            }
        )
    return figures


def _paint_token(image: Any, box: list[int], token_str: str) -> Any:
    import cv2

    def get_optimal_font_scale(
        text: str, font_face: int, square_size: int, fill_ratio: float = 0.9
    ) -> tuple[float, int, int]:
        left, right = 0.2, 10.0
        optimal_scale = left
        while right - left > 1e-2:
            mid = (left + right) / 2
            (width, height), _ = cv2.getTextSize(text, font_face, mid, thickness=1)
            if width < square_size * fill_ratio and height < square_size * fill_ratio:
                optimal_scale = mid
                left = mid
            else:
                right = mid
        return optimal_scale, width, height

    x1, y1, x2, y2 = [int(v) for v in box]
    box_w = x2 - x1
    box_h = y2 - y1
    img = image.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, text_w, text_h = get_optimal_font_scale(
        token_str, font, min(box_w, box_h), fill_ratio=0.9
    )
    font_thickness = max(1, math.floor(font_scale * 4))
    text_x = x1 + (box_w - text_w) // 2
    text_y = y1 + (box_h + text_h) // 2
    cv2.putText(
        img,
        token_str,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        font_thickness,
        lineType=cv2.LINE_AA,
    )
    return img


def _tokenize_figure_of_table(
    table_block_img: Image.Image,
    table_box: list[int],
    figures: list[dict[str, Any]],
) -> tuple[Image.Image, dict[str, str], list[str]]:
    def gen_random_map(num: int) -> list[int]:
        exclude_digits = {"0", "1", "9"}
        seq: list[int] = []
        idx = 0
        while len(seq) < num:
            if not (set(str(idx)) & exclude_digits):
                seq.append(idx)
            idx += 1
        return seq

    import numpy as np

    random.seed(1024)
    token_map: dict[str, str] = {}
    table_x_min, table_y_min, table_x_max, table_y_max = table_box
    drop_idxes = []
    random_map = gen_random_map(len(figures))
    random.shuffle(random_map)
    table_array = np.asarray(table_block_img.convert("RGB")).copy()
    for figure_id, figure in enumerate(figures):
        figure_x_min, figure_y_min, figure_x_max, figure_y_max = figure["coordinate"]
        if (
            figure_x_min >= table_x_min
            and figure_y_min >= table_y_min
            and figure_x_max <= table_x_max
            and figure_y_max <= table_y_max
        ):
            drop_idxes.append(figure_id)
            if min(figure_x_max - figure_x_min, figure_y_max - figure_y_min) < 25:
                continue
            draw_box = [
                figure_x_min - table_x_min,
                figure_y_min - table_y_min,
                figure_x_max - table_x_min,
                figure_y_max - table_y_min,
            ]
            token_str = "[F" + str(random_map[figure_id]) + "]"
            table_array = _paint_token(table_array, draw_box, token_str)
            token_map[token_str] = figure["path"]
    drop_figures = [figure["path"] for idx, figure in enumerate(figures) if idx in drop_idxes]
    return Image.fromarray(table_array), token_map, drop_figures


def _untokenize_figure_of_table(table_res_str: str, figure_token_map: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        token_id = match.group(1)
        token = f"[F{token_id}]"
        img_path = figure_token_map.get(token)
        if img_path is None:
            return match.group(0)
        return '<img src="{}" alt="Image"" />'.format(
            img_path.replace("-\n", "").replace("\n", " ")
        )

    return re.sub(r"\[F(\d+)\]", repl, table_res_str)


def _make_blocks(
    full_image: Image.Image, boxes: list[LayoutBox], bgr_image: Any | None = None
) -> list[LightBlock]:
    def crop_box(box: LayoutBox) -> Image.Image:
        if bgr_image is not None:
            return _crop_from_bgr(bgr_image, box)
        return _crop(full_image, box)

    return [
        LightBlock(
            label=box.label,
            content="",
            bbox=box.coordinate,
            score=box.score,
            cls_id=box.cls_id,
            image=crop_box(box),
            polygon_points=box.to_dict()["polygon_points"],
        )
        for box in boxes
    ]


def _merge_blocks(blocks: list[LightBlock], non_merge_labels: set[str]) -> list[LightBlock]:
    blocks_to_merge: list[tuple[int, LightBlock]] = []
    non_merge_blocks: dict[int, LightBlock] = {}
    for idx, block in enumerate(blocks):
        if block.label in non_merge_labels:
            non_merge_blocks[idx] = block
        else:
            blocks_to_merge.append((idx, block))

    def is_aligned(a: int, b: int) -> bool:
        return abs(a - b) <= 5

    def alignment(block_bbox: list[int], prev_bbox: list[int]) -> str:
        if is_aligned(block_bbox[0], prev_bbox[0]):
            return "left"
        if is_aligned(block_bbox[2], prev_bbox[2]):
            return "right"
        return "center"

    def overlaps_non_merge(block_idx: int, prev_idx: int) -> bool:
        prev_bbox = blocks[prev_idx].bbox
        block_bbox = blocks[block_idx].bbox
        merged = [
            min(prev_bbox[0], block_bbox[0]),
            min(prev_bbox[1], block_bbox[1]),
            max(prev_bbox[2], block_bbox[2]),
            max(prev_bbox[3], block_bbox[3]),
        ]
        for idx, other in enumerate(blocks):
            if idx in {block_idx, prev_idx} or other.label not in non_merge_labels:
                continue
            if _overlap_ratio(merged, other.bbox) > 0:
                return True
        return False

    merged_groups: list[tuple[list[int], list[str]]] = []
    current_indices: list[int] = []
    current_aligns: list[str] = []
    for pos, (idx, block) in enumerate(blocks_to_merge):
        if not current_indices:
            current_indices = [idx]
            current_aligns = []
            continue
        prev_idx, prev_block = blocks_to_merge[pos - 1]
        prev_bbox = prev_block.bbox
        block_bbox = block.bbox
        iou_h = _projection_overlap_ratio(block_bbox, prev_bbox, "horizontal")
        is_cross = (
            iou_h == 0
            and block.label == "text"
            and block.label == prev_block.label
            and block_bbox[0] > prev_bbox[2]
            and block_bbox[1] < prev_bbox[3]
            and block_bbox[0] - prev_bbox[2]
            < max(prev_bbox[2] - prev_bbox[0], block_bbox[2] - block_bbox[0]) * 0.3
        )
        is_updown_align = (
            iou_h > 0
            and block.label == "text"
            and block.label == prev_block.label
            and block_bbox[3] >= prev_bbox[1]
            and abs(block_bbox[1] - prev_bbox[3])
            < max(prev_bbox[3] - prev_bbox[1], block_bbox[3] - block_bbox[1]) * 0.5
            and (is_aligned(block_bbox[0], prev_bbox[0]) ^ is_aligned(block_bbox[2], prev_bbox[2]))
            and overlaps_non_merge(idx, prev_idx)
        )
        if is_cross or is_updown_align:
            current_indices.append(idx)
            current_aligns.append("center" if is_cross else alignment(block_bbox, prev_bbox))
        else:
            merged_groups.append((current_indices, current_aligns))
            current_indices = [idx]
            current_aligns = []
    if current_indices:
        merged_groups.append((current_indices, current_aligns))

    group_ranges = [
        (min(indices), max(indices), indices, aligns) for indices, aligns in merged_groups
    ]
    result: list[LightBlock] = []
    used: set[int] = set()
    idx = 0
    while idx < len(blocks):
        group_found = False
        for start, end, group_indices, aligns in group_ranges:
            if idx != start or any(group_idx in used for group_idx in group_indices):
                continue
            group_found = True
            images = [
                img for group_idx in group_indices if (img := blocks[group_idx].image) is not None
            ]
            width = max((image.width for image in images), default=0)
            height = sum(image.height for image in images)
            aspect_ratio = height / width if width else float("inf")
            if aspect_ratio >= 3:
                for block_idx in group_indices:
                    result.append(blocks[block_idx])
                    used.add(block_idx)
            else:
                merged_image = _merge_images(images, aligns) if images else None
                is_merged_group = len(group_indices) > 1
                for group_pos, block_idx in enumerate(group_indices):
                    block = blocks[block_idx]
                    if group_pos == 0:
                        block.image = merged_image
                        block.group_id = group_indices[0]
                        block.merged = is_merged_group
                    else:
                        block.image = None
                        block.group_id = group_indices[0]
                        block.merged = is_merged_group
                    result.append(block)
                    used.add(block_idx)
            for inner_idx in range(start + 1, end):
                if inner_idx in non_merge_blocks:
                    result.append(non_merge_blocks[inner_idx])
                    used.add(inner_idx)
            idx = end + 1
            break
        if group_found:
            continue
        if idx in non_merge_blocks and idx not in used:
            result.append(non_merge_blocks[idx])
            used.add(idx)
        idx += 1
    return result


def _markdown_from_blocks(blocks: list[LightBlock], page_width: int) -> str:
    markdown = ""
    for _idx, block in enumerate(blocks):
        if block.label in MARKDOWN_IGNORE_LABEL_SET:
            continue
        content = _markdown_content_for_block(block, page_width)
        if content is None:
            continue
        if markdown:
            markdown += "\n\n" + content
        else:
            markdown = content
    return markdown


def _collapse_soft_newlines(text: str) -> str:
    return text.replace("-\n", "").replace("\n", " ")


def _normalize_markdown_newlines(text: str) -> str:
    return text.replace("\n\n", "\n").replace("\n", "\n\n")


def _format_title_text(content: str) -> str:
    title = content.rstrip(".")
    match = re.match(
        r"^\s*((?:\d+(?:\.\d+)*\.?|[一二三四五六七八九十百千万零]+[、.．]?))\s*(.*)$", title
    )
    if match and match.group(2):
        title = f"{match.group(1).strip()} {match.group(2).lstrip()}"
    level = title.count(".") + 1 if "." in title else 1
    return _collapse_soft_newlines(f"{'#' * (level + 1)} {title}")


def _markdown_content_for_block(block: LightBlock, page_width: int) -> str | None:
    label = block.label
    content = block.content
    if label == "doc_title":
        return _collapse_soft_newlines(f"# {content}")
    if label == "paragraph_title":
        return _format_title_text(content)
    if label in {"text", "ocr", "vertical_text", "reference_content", "vision_footnote"}:
        return _normalize_markdown_newlines(content)
    if label == "content":
        return content.replace("-\n", "  \n").replace("\n", "  \n")
    if label in {"formula", "display_formula", "inline_formula"}:
        return content
    if label == "table":
        return "\n\n" + content.replace("<html>", "").replace("</html>", "").replace(
            "<body>", ""
        ).replace("</body>", "")
    if label in {"image", "chart", "seal"} and block.image is not None:
        x1, y1, x2, y2 = block.bbox
        return f"![](imgs/img_in_{label}_box_{x1}_{y1}_{x2}_{y2}.jpg)"
    if label == "chart":
        return content
    if label == "algorithm":
        return content.strip("\n")
    return content


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
