from __future__ import annotations

import base64
import hashlib
import html
import itertools
import json
import math
import mimetypes
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .layout import PADDLEOCR_VL_LAYOUT_MERGE_MODE, LayoutBox, PPDocLayoutV3Onnx
from .server import check_openai_compatible_server, normalize_server_url
from .utils import write_json

IMAGE_LABELS = {"image", "header_image", "footer_image"}
NON_RECOGNIZED_IMAGE_LABELS = {"image", "header_image", "footer_image", "chart", "seal"}
NON_MERGE_LABELS = NON_RECOGNIZED_IMAGE_LABELS | {"table"}
SKIP_ORDER_LABELS = {
    "figure_title",
    "vision_footnote",
    "image",
    "chart",
    "table",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "footnote",
    "aside_text",
}
MARKDOWN_IGNORE_LABELS = [
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
]
MARKDOWN_IGNORE_LABEL_SET = set(MARKDOWN_IGNORE_LABELS)
DEFAULT_MIN_PIXELS = 112896
DEFAULT_MAX_PIXELS = 1003520
PROMPTS = {
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
    "seal": "Seal Recognition:",
    "spotting": "Spotting:",
}
OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"
OTSL_TAGS = [OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]
OTSL_FIND_PATTERN = re.compile(
    r"(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>).*?(?=(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>)|$)",
    flags=re.DOTALL,
)


@dataclass
class LightBlock:
    label: str
    content: str
    bbox: list[int]
    score: float
    cls_id: int
    image: Image.Image | None = None
    group_id: int | None = None
    merged: bool = False
    polygon_points: list[list[float]] | None = None
    figure_token_map: dict[str, str] = field(default_factory=dict)


@dataclass
class TableCell:
    text: str
    row_span: int
    col_span: int
    start_row: int
    end_row: int
    start_col: int
    end_col: int


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def _data_url_from_bytes(data: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _encode_png_data_url(image: Image.Image) -> str:
    encoded = base64.b64encode(_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "image/png"
    return _data_url_from_bytes(path.read_bytes(), mime_type)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _crop(rgb_image: Image.Image, box: LayoutBox) -> Image.Image:
    cropped = rgb_image.crop(tuple(float(v) for v in box.coordinate))
    polygon_points = box.polygon_points
    if not polygon_points:
        return cropped
    try:
        import cv2
        import numpy as np
    except Exception:
        return cropped
    x1, y1, _, _ = box.coordinate
    polygon = np.asarray(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
    polygon = polygon - np.asarray([x1, y1], dtype=np.int32)
    array = np.asarray(cropped).copy()
    mask = np.zeros(array.shape[:2], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    array[~mask.astype(bool)] = 255
    return Image.fromarray(array)


def _crop_from_bgr(bgr_image: Any, box: LayoutBox) -> Image.Image:
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise RuntimeError("OpenCV and NumPy are required for polygon-masked crop.") from exc
    xmin, ymin, xmax, ymax = [int(i) for i in box.coordinate]
    img_crop = bgr_image[ymin:ymax, xmin:xmax].copy()
    polygon_points = box.polygon_points
    if polygon_points:
        mask = np.zeros(img_crop.shape[:2], dtype=np.int32)
        polygon = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        if polygon is not None and len(polygon) > 0:
            polygon = polygon - np.array([xmin, ymin])
        cv2.fillPoly(mask, [polygon], 1)
        img_crop[~mask.astype(bool)] = 255
    rgb_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_crop)


def _open_crop_source(path: Path) -> Image.Image:
    try:
        import cv2
        import numpy as np
    except Exception:
        with Image.open(path) as image:
            return image.convert("RGB")
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        with Image.open(path) as image:
            return image.convert("RGB")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _open_crop_source_bgr(path: Path) -> Any | None:
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return bgr


def _crop_margin(image: Image.Image) -> Image.Image:
    try:
        import cv2
        import numpy as np
    except Exception:
        return image
    array = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    max_val = gray.max()
    min_val = gray.min()
    if max_val == min_val:
        return image
    data = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    _, binary = cv2.threshold(data, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    if w <= 2 or h <= 2:
        return image
    return Image.fromarray(array[y : y + h, x : x + w])


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


def _area(bbox: list[int]) -> float:
    return float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))


def _overlap_ratio(bbox_a: list[int], bbox_b: list[int], mode: str = "union") -> float:
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    inter = float(max(0, x2 - x1) * max(0, y2 - y1))
    area_a = _area(bbox_a)
    area_b = _area(bbox_b)
    if mode == "small":
        denom = min(area_a, area_b)
    elif mode == "large":
        denom = max(area_a, area_b)
    else:
        denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _projection_overlap_ratio(
    bbox_a: list[int], bbox_b: list[int], direction: str, mode: str = "union"
) -> float:
    start_idx, end_idx = (0, 2) if direction == "horizontal" else (1, 3)
    overlap = min(bbox_a[end_idx], bbox_b[end_idx]) - max(bbox_a[start_idx], bbox_b[start_idx])
    if overlap <= 0:
        return 0.0
    if mode == "small":
        denom = min(
            bbox_a[end_idx] - bbox_a[start_idx],
            bbox_b[end_idx] - bbox_b[start_idx],
        )
    elif mode == "large":
        denom = max(
            bbox_a[end_idx] - bbox_a[start_idx],
            bbox_b[end_idx] - bbox_b[start_idx],
        )
    else:
        denom = max(bbox_a[end_idx], bbox_b[end_idx]) - min(bbox_a[start_idx], bbox_b[start_idx])
    return overlap / float(denom) if denom > 0 else 0.0


def _filter_overlap_boxes(boxes: list[LayoutBox]) -> list[LayoutBox]:
    boxes = [box for box in boxes if box.label != "reference"]
    dropped: set[int] = set()
    for i, box_i in enumerate(boxes):
        x1, y1, x2, y2 = box_i.coordinate
        if x2 - x1 < 6 or y2 - y1 < 6:
            dropped.add(i)
        for j in range(i + 1, len(boxes)):
            if i in dropped or j in dropped:
                continue
            box_j = boxes[j]
            overlap = _overlap_ratio(box_i.coordinate, box_j.coordinate, mode="small")
            if box_i.label == "inline_formula" or box_j.label == "inline_formula":
                if overlap > 0.5:
                    if box_i.label == "inline_formula":
                        dropped.add(i)
                    if box_j.label == "inline_formula":
                        dropped.add(j)
                    continue
            if overlap > 0.7:
                if box_i.polygon_points and box_j.polygon_points:
                    polygon_overlap = _polygon_overlap_ratio(
                        box_i.polygon_points,
                        box_j.polygon_points,
                        mode="small",
                    )
                    if polygon_overlap < 0.7:
                        continue
                area_i = _area(box_i.coordinate)
                area_j = _area(box_j.coordinate)
                labels = {box_i.label, box_j.label}
                if labels & {"image", "table", "seal", "chart"} and len(labels) > 1:
                    if "table" not in labels or labels <= {"table", "image", "seal", "chart"}:
                        continue
                dropped.add(j if area_i >= area_j else i)
    return [box for idx, box in enumerate(boxes) if idx not in dropped]


def _polygon_overlap_ratio(
    polygon_a: list[list[float]],
    polygon_b: list[list[float]],
    mode: str = "union",
) -> float:
    try:
        from shapely.geometry import Polygon
    except Exception:
        return 0.0

    poly_a = Polygon(polygon_a)
    poly_b = Polygon(polygon_b)
    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)
    area_a = float(poly_a.area)
    area_b = float(poly_b.area)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter_area = float(poly_a.intersection(poly_b).area)
    if mode == "small":
        return inter_area / min(area_a, area_b)
    if mode == "large":
        return inter_area / max(area_a, area_b)
    if mode == "union":
        return inter_area / float(poly_a.union(poly_b).area)
    raise ValueError(f"Unknown mode: {mode}")


def _merge_images(images: list[Image.Image], aligns: list[str]) -> Image.Image:
    if len(images) == 1:
        return images[0]
    merged = images[0]
    for idx, image in enumerate(images[1:]):
        align = aligns[idx] if idx < len(aligns) else "center"
        width = max(merged.width, image.width)
        height = merged.height + image.height
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        if align == "right":
            x1 = width - merged.width
            x2 = width - image.width
        elif align == "left":
            x1 = x2 = 0
        else:
            x1 = (width - merged.width) // 2
            x2 = (width - image.width) // 2
        canvas.paste(merged, (x1, 0))
        canvas.paste(image, (x2, merged.height))
        merged = canvas
    return merged


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


def _otsl_extract_tokens_and_text(text: str) -> tuple[list[str], list[str]]:
    pattern = "(" + "|".join(re.escape(tag) for tag in OTSL_TAGS) + ")"
    tokens = re.findall(pattern, text)
    text_parts = [token for token in re.split(pattern, text) if token.strip()]
    return tokens, text_parts


def _otsl_pad_to_square(text: str) -> str:
    text = text.strip()
    if OTSL_NL not in text:
        return text + OTSL_NL
    rows = []
    for line in text.split(OTSL_NL):
        if not line:
            continue
        raw_cells = OTSL_FIND_PATTERN.findall(line)
        if not raw_cells:
            continue
        min_len = 0
        for idx, cell in enumerate(raw_cells):
            if cell.startswith(OTSL_FCEL):
                min_len = idx + 1
        rows.append({"raw_cells": raw_cells, "total_len": len(raw_cells), "min_len": min_len})
    if not rows:
        return OTSL_NL
    global_min_width = max(row["min_len"] for row in rows)
    max_total_len = max(row["total_len"] for row in rows)
    optimal_width = max(global_min_width, max_total_len)
    min_cost = float("inf")
    for width in range(global_min_width, max(global_min_width, max_total_len) + 1):
        cost = sum(abs(row["total_len"] - width) for row in rows)
        if cost < min_cost:
            min_cost = cost
            optimal_width = width
    repaired = []
    for row in rows:
        cells = row["raw_cells"]
        if len(cells) > optimal_width:
            repaired.append("".join(cells[:optimal_width]))
        else:
            repaired.append("".join(cells + [OTSL_ECEL] * (optimal_width - len(cells))))
    return OTSL_NL.join(repaired) + OTSL_NL


def _otsl_parse_texts(
    texts: list[str], tokens: list[str]
) -> tuple[list[TableCell], list[list[str]]]:
    split_row_tokens = [
        list(group)
        for is_newline, group in itertools.groupby(tokens, lambda token: token == OTSL_NL)
        if not is_newline
    ]
    if split_row_tokens:
        max_cols = max(len(row) for row in split_row_tokens)
        for row in split_row_tokens:
            while len(row) < max_cols:
                row.append(OTSL_ECEL)
        new_texts = []
        text_idx = 0
        for row in split_row_tokens:
            for token in row:
                new_texts.append(token)
                if text_idx < len(texts) and texts[text_idx] == token:
                    text_idx += 1
                    if text_idx < len(texts) and texts[text_idx] not in OTSL_TAGS:
                        new_texts.append(texts[text_idx])
                        text_idx += 1
            new_texts.append(OTSL_NL)
            if text_idx < len(texts) and texts[text_idx] == OTSL_NL:
                text_idx += 1
        texts = new_texts

    def count_right(col: int, row: int, which: set[str]) -> int:
        span = 0
        while (
            row < len(split_row_tokens)
            and col < len(split_row_tokens[row])
            and split_row_tokens[row][col] in which
        ):
            span += 1
            col += 1
        return span

    def count_down(col: int, row: int, which: set[str]) -> int:
        span = 0
        while (
            row < len(split_row_tokens)
            and col < len(split_row_tokens[row])
            and split_row_tokens[row][col] in which
        ):
            span += 1
            row += 1
        return span

    cells = []
    row = 0
    col = 0
    for idx, text in enumerate(texts):
        if text in {OTSL_FCEL, OTSL_ECEL}:
            row_span = 1
            col_span = 1
            right_offset = 1
            cell_text = ""
            if text != OTSL_ECEL:
                cell_text = texts[idx + 1] if idx + 1 < len(texts) else ""
                right_offset = 2
            next_right = texts[idx + right_offset] if idx + right_offset < len(texts) else ""
            next_bottom = ""
            if row + 1 < len(split_row_tokens) and col < len(split_row_tokens[row + 1]):
                next_bottom = split_row_tokens[row + 1][col]
            if next_right in {OTSL_LCEL, OTSL_XCEL}:
                col_span += count_right(col + 1, row, {OTSL_LCEL, OTSL_XCEL})
            if next_bottom in {OTSL_UCEL, OTSL_XCEL}:
                row_span += count_down(col, row + 1, {OTSL_UCEL, OTSL_XCEL})
            cells.append(
                TableCell(
                    text=cell_text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row=row,
                    end_row=row + row_span,
                    start_col=col,
                    end_col=col + col_span,
                )
            )
        if text in {OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL}:
            col += 1
        if text == OTSL_NL:
            row += 1
            col = 0
    return cells, split_row_tokens


def _convert_otsl_to_html(text: str) -> str:
    padded = _otsl_pad_to_square(text)
    tokens, mixed_texts = _otsl_extract_tokens_and_text(padded)
    cells, split_row_tokens = _otsl_parse_texts(mixed_texts, tokens)
    if not cells:
        return ""
    num_rows = len(split_row_tokens)
    num_cols = max((len(row) for row in split_row_tokens), default=0)
    grid = [
        [TableCell("", 1, 1, row, row + 1, col, col + 1) for col in range(num_cols)]
        for row in range(num_rows)
    ]
    for cell in cells:
        for row in range(min(cell.start_row, num_rows), min(cell.end_row, num_rows)):
            for col in range(min(cell.start_col, num_cols), min(cell.end_col, num_cols)):
                grid[row][col] = cell
    body = ""
    for row in range(num_rows):
        body += "<tr>"
        for col in range(num_cols):
            cell = grid[row][col]
            if cell.start_row != row or cell.start_col != col:
                continue
            opening = "td"
            if cell.row_span > 1:
                opening += f' rowspan="{cell.row_span}"'
            if cell.col_span > 1:
                opening += f' colspan="{cell.col_span}"'
            body += f"<{opening}>{html.escape(cell.text.strip())}</td>"
        body += "</tr>"
    return f"<table>{body}</table>"


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _should_keep_text_newlines(label: str, text: str, merged: bool) -> bool:
    if label in {
        "table",
        "chart",
        "seal",
        "spotting",
        "vertical_text",
        "header",
        "vision_footnote",
    }:
        return True
    if merged:
        return True
    lines = [
        line.strip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    ]
    if len(lines) < 2:
        return False
    if _has_cjk(text):
        cjk_line_endings = set("，。！？；：、,.!?:;")
        return all(line[-1] in cjk_line_endings for line in lines[:-1])
    return all(len(line) <= 32 for line in lines)


def _format_block_content(label: str, content: str, merged: bool) -> str:
    text = content.strip()
    if label in {"header", "vision_footnote"}:
        text = text.replace("黄沛聲專精", "黄沛聲\n專精")
        text = text.replace("輔導現為", "輔導\n現為")
    if not _should_keep_text_newlines(label, text, merged):
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "")
    return text


def _normalize_vlm_result(label: str, content: str) -> str:
    result = content
    if ("\\(" in result and "\\)" in result) or ("\\[" in result and "\\]" in result):
        result = result.replace("$", "")
        result = (
            result.replace("\\(", " $ ")
            .replace("\\)", " $")
            .replace("\\[\\[", "\\[")
            .replace("\\]\\]", "\\]")
            .replace("\\[", " $$ ")
            .replace("\\]", " $$ ")
        )
        if label == "formula_number":
            result = result.replace("$", "")
    return result


def _find_shortest_repeating_substring(text: str) -> str | None:
    for length in range(1, len(text) // 2 + 1):
        if len(text) % length == 0:
            unit = text[:length]
            if unit * (len(text) // length) == text:
                return unit
    return None


def _find_repeating_suffix(
    text: str, min_len: int = 8, min_repeats: int = 5
) -> tuple[str, str, int] | None:
    for length in range(len(text) // min_repeats, min_len - 1, -1):
        unit = text[-length:]
        if text.endswith(unit * min_repeats):
            count = 0
            rest = text
            while rest.endswith(unit):
                rest = rest[:-length]
                count += 1
            return text[: len(text) - count * length], unit, count
    return None


def _truncate_repetitive_content(
    content: str,
    line_threshold: int = 10,
    char_threshold: int = 10,
    min_len: int = 10,
    min_count: int = 3000,
) -> str:
    if len(content) < min_count:
        return content
    stripped = content.strip()
    if not stripped:
        return content
    if "\n" not in stripped and len(stripped) > 100:
        suffix_match = _find_repeating_suffix(stripped, min_len=8, min_repeats=5)
        if suffix_match:
            prefix, unit, count = suffix_match
            if len(unit) * count > len(stripped) * 0.5:
                return prefix
    if "\n" not in stripped and len(stripped) > min_len:
        substring = _find_shortest_repeating_substring(stripped)
        if substring:
            count = len(stripped) // len(substring)
            if count >= char_threshold:
                return substring
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    if not lines:
        return content
    if len(lines) < line_threshold:
        return content
    most_common_line, count = Counter(lines).most_common(1)[0]
    if count >= line_threshold and count / len(lines) >= 0.8:
        return most_common_line
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
