from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LABEL_LIST = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
]

HEADING_LABELS = {"doc_title", "paragraph_title", "figure_title"}
BODY_TEXT_LABELS = {
    "abstract",
    "aside_text",
    "content",
    "reference_content",
    "text",
    "vertical_text",
}

PADDLEOCR_VL_LAYOUT_MERGE_MODE = {
    0: "union",
    1: "union",
    2: "union",
    3: "large",
    4: "union",
    5: "large",
    6: "large",
    7: "union",
    8: "union",
    9: "union",
    10: "union",
    11: "union",
    12: "union",
    13: "union",
    14: "union",
    15: "large",
    16: "union",
    17: "large",
    18: "union",
    19: "union",
    20: "union",
    21: "union",
    22: "union",
    23: "union",
    24: "union",
}


@dataclass(frozen=True)
class LayoutBox:
    cls_id: int
    label: str
    score: float
    coordinate: list[int]
    order: int | None = None
    polygon_points: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.coordinate
        polygon_points = self.polygon_points or [
            [float(x1), float(y1)],
            [float(x2), float(y1)],
            [float(x2), float(y2)],
            [float(x1), float(y2)],
        ]
        item: dict[str, Any] = {
            "cls_id": self.cls_id,
            "label": self.label,
            "score": self.score,
            "coordinate": self.coordinate,
            "order": self.order,
            "polygon_points": polygon_points,
        }
        return item


def _resize_rgb(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(array, size, interpolation=cv2.INTER_CUBIC)


def _read_layout_rgb(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    import cv2

    image_bytes = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    bgr = cv2.imdecode(image_bytes, flags=cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Image read Error: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return rgb, (width, height)


def _normalize_rgb(array: np.ndarray) -> np.ndarray:
    import cv2

    channels = list(cv2.split(array))
    for idx, channel in enumerate(channels):
        channels[idx] = channel.astype(np.float32)
        channels[idx] = channels[idx] * (1.0 / 255.0)
    return cv2.merge(channels)


def preprocess_layout_image(
    path: Path, target_size: int = 800
) -> tuple[dict[str, np.ndarray], tuple[int, int], np.ndarray]:
    rgb, (width, height) = _read_layout_rgb(path)
    resized = _normalize_rgb(_resize_rgb(rgb, (target_size, target_size)))
    chw = np.transpose(resized, (2, 0, 1))[None, :, :, :]
    feeds = {
        "im_shape": np.asarray([[target_size, target_size]], dtype=np.float32),
        "image": np.ascontiguousarray(chw),
        "scale_factor": np.asarray(
            [[target_size / float(height), target_size / float(width)]],
            dtype=np.float32,
        ),
    }
    return feeds, (width, height), rgb


def _box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[4] - box[2])) * max(0.0, float(box[5] - box[3]))


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[2]), float(box_b[2]))
    y1 = max(float(box_a[3]), float(box_b[3]))
    x2 = min(float(box_a[4]), float(box_b[4]))
    y2 = min(float(box_a[5]), float(box_b[5]))
    inter = max(0.0, x2 - x1 + 1.0) * max(0.0, y2 - y1 + 1.0)
    area_a = max(0.0, float(box_a[4] - box_a[2] + 1.0)) * max(0.0, float(box_a[5] - box_a[3] + 1.0))
    area_b = max(0.0, float(box_b[4] - box_b[2] + 1.0)) * max(0.0, float(box_b[5] - box_b[3] + 1.0))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _layout_nms(boxes: np.ndarray, iou_same: float = 0.6, iou_diff: float = 0.98) -> np.ndarray:
    return boxes[_layout_nms_indices(boxes, iou_same=iou_same, iou_diff=iou_diff)]


def _layout_nms_indices(
    boxes: np.ndarray, iou_same: float = 0.6, iou_diff: float = 0.98
) -> list[int]:
    if boxes.size == 0:
        return []
    indices = np.argsort(boxes[:, 1])[::-1].tolist()
    selected: list[int] = []
    while indices:
        current = indices.pop(0)
        selected.append(current)
        survivors: list[int] = []
        for idx in indices:
            threshold = iou_same if boxes[idx, 0] == boxes[current, 0] else iou_diff
            if _iou(boxes[current], boxes[idx]) < threshold:
                survivors.append(idx)
        indices = survivors
    return selected


def _is_contained(box_a: np.ndarray, box_b: np.ndarray) -> bool:
    x1 = max(float(box_a[2]), float(box_b[2]))
    y1 = max(float(box_a[3]), float(box_b[3]))
    x2 = min(float(box_a[4]), float(box_b[4]))
    y2 = min(float(box_a[5]), float(box_b[5]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, float(box_a[4] - box_a[2])) * max(0.0, float(box_a[5] - box_a[3]))
    return area > 0 and inter / area >= 0.9


def _check_containment(
    boxes: np.ndarray,
    formula_index: int | None = None,
    category_index: int | None = None,
    mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    contains_other = np.zeros(len(boxes), dtype=int)
    contained_by_other = np.zeros(len(boxes), dtype=int)
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j:
                continue
            if (
                formula_index is not None
                and boxes[i][0] == formula_index
                and boxes[j][0] != formula_index
            ):
                continue
            if category_index is not None and mode is not None:
                if (
                    mode == "large"
                    and boxes[j][0] == category_index
                    and _is_contained(boxes[i], boxes[j])
                ):
                    contained_by_other[i] = 1
                    contains_other[j] = 1
                elif (
                    mode == "small"
                    and boxes[i][0] == category_index
                    and _is_contained(boxes[i], boxes[j])
                ):
                    contained_by_other[i] = 1
                    contains_other[j] = 1
            elif _is_contained(boxes[i], boxes[j]):
                contained_by_other[i] = 1
                contains_other[j] = 1
    return contains_other, contained_by_other


def _layout_merge_keep_mask(boxes: np.ndarray, merge_mode: dict[int, str] | None) -> np.ndarray:
    keep_mask = np.ones(len(boxes), dtype=bool)
    if not merge_mode or boxes.size == 0:
        return keep_mask
    formula_index = LABEL_LIST.index("formula") if "formula" in LABEL_LIST else None
    for category_index, layout_mode in merge_mode.items():
        if layout_mode == "union":
            continue
        contains_other, contained_by_other = _check_containment(
            boxes[:, :6],
            formula_index=formula_index,
            category_index=category_index,
            mode=layout_mode,
        )
        if layout_mode == "large":
            keep_mask &= contained_by_other == 0
        elif layout_mode == "small":
            keep_mask &= (contains_other == 0) | (contained_by_other == 1)
    return keep_mask


def _apply_layout_merge_mode(boxes: np.ndarray, merge_mode: dict[int, str] | None) -> np.ndarray:
    return boxes[_layout_merge_keep_mask(boxes, merge_mode)]


def _unclip_boxes(
    boxes: np.ndarray,
    unclip_ratio: float | tuple[float, float] | list[float] | dict[int, tuple[float, float]] | None,
) -> np.ndarray:
    if unclip_ratio is None or boxes.size == 0:
        return boxes
    if isinstance(unclip_ratio, (int, float)):
        unclip_ratio = (float(unclip_ratio), float(unclip_ratio))
    if isinstance(unclip_ratio, dict):
        expanded = []
        for box in boxes:
            cls_id, score, x1, y1, x2, y2 = box[:6]
            if int(cls_id) not in unclip_ratio:
                expanded.append(box[:6])
                continue
            width_ratio, height_ratio = unclip_ratio[int(cls_id)]
            width = x2 - x1
            height = y2 - y1
            center_x = x1 + width / 2
            center_y = y1 + height / 2
            new_w = width * width_ratio
            new_h = height * height_ratio
            expanded.append(
                [
                    cls_id,
                    score,
                    center_x - new_w / 2,
                    center_y - new_h / 2,
                    center_x + new_w / 2,
                    center_y + new_h / 2,
                ]
            )
        return np.asarray(expanded)

    width_ratio, height_ratio = unclip_ratio
    widths = boxes[:, 4] - boxes[:, 2]
    heights = boxes[:, 5] - boxes[:, 3]
    center_x = boxes[:, 2] + widths / 2
    center_y = boxes[:, 3] + heights / 2
    new_w = widths * width_ratio
    new_h = heights * height_ratio
    return np.column_stack(
        (
            boxes[:, 0],
            boxes[:, 1],
            center_x - new_w / 2,
            center_y - new_h / 2,
            center_x + new_w / 2,
            center_y + new_h / 2,
        )
    )


def _rect_from_box(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box).astype(np.int32)
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _polygon_overlap_ratio(
    poly_a: np.ndarray | list[list[float]], poly_b: np.ndarray | list[list[float]], mode: str
) -> float:
    try:
        from shapely.geometry import Polygon

        polygon_a = Polygon(poly_a)
        polygon_b = Polygon(poly_b)
        if not polygon_a.is_valid:
            polygon_a = polygon_a.buffer(0)
        if not polygon_b.is_valid:
            polygon_b = polygon_b.buffer(0)
        intersection = float(polygon_a.intersection(polygon_b).area)
        if mode == "small":
            denom = min(float(polygon_a.area), float(polygon_b.area))
        elif mode == "large":
            denom = max(float(polygon_a.area), float(polygon_b.area))
        else:
            denom = float(polygon_a.union(polygon_b).area)
        return intersection / denom if denom > 0 else 0.0
    except Exception:
        pass

    try:
        import cv2
    except Exception:
        return 0.0
    a = np.asarray(poly_a, dtype=np.float32)
    b = np.asarray(poly_b, dtype=np.float32)
    area_a = abs(float(cv2.contourArea(a)))
    area_b = abs(float(cv2.contourArea(b)))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(a, b)
    inter_area = max(0.0, float(inter_area))
    if mode == "small":
        return inter_area / min(area_a, area_b)
    if mode == "large":
        return inter_area / max(area_a, area_b)
    return inter_area / max(area_a + area_b - inter_area, 1e-6)


def _polygon_to_quad(polygon: np.ndarray) -> np.ndarray | None:
    try:
        import cv2
    except Exception:
        return None
    points = np.asarray(polygon, dtype=np.float32)
    if len(points) < 3:
        return None
    quad = cv2.boxPoints(cv2.minAreaRect(points))
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    quad = quad[np.argsort(angles)]
    top_left_idx = np.argmin(quad[:, 0] + quad[:, 1])
    return np.roll(quad, -top_left_idx, axis=0)


def _is_convex(prev_point: np.ndarray, point: np.ndarray, next_point: np.ndarray) -> bool:
    v1 = point - prev_point
    v2 = next_point - point
    return v1[0] * v2[1] - v1[1] * v2[0] < 0


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    return float(np.degrees(np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0))))


def _extract_custom_vertices(
    polygon: np.ndarray,
    max_allowed_dist: float,
    sharp_angle_thresh: float = 45,
    max_dist_ratio: float = 0.3,
) -> list[tuple[float, float]]:
    poly = np.asarray(polygon)
    point_count = len(poly)
    max_allowed_dist *= max_dist_ratio
    point_info = []
    for idx in range(point_count):
        prev_point = poly[(idx - 1) % point_count]
        point = poly[idx]
        next_point = poly[(idx + 1) % point_count]
        v1 = prev_point - point
        v2 = next_point - point
        point_info.append(
            {
                "index": idx,
                "is_convex": _is_convex(prev_point, point, next_point),
                "angle": _angle_between(v1, v2),
                "v1": v1,
                "v2": v2,
            }
        )

    concave_indices = [idx for idx, info in enumerate(point_info) if not info["is_convex"]]
    preserve_concave: set[int] = set()
    if concave_indices:
        groups = []
        current_group = [concave_indices[0]]
        for idx in range(1, len(concave_indices)):
            if concave_indices[idx] - concave_indices[idx - 1] == 1 or (
                concave_indices[idx - 1] == point_count - 1 and concave_indices[idx] == 0
            ):
                current_group.append(concave_indices[idx])
            else:
                if len(current_group) >= 2:
                    groups.extend(current_group)
                current_group = [concave_indices[idx]]
        if len(current_group) >= 2:
            groups.extend(current_group)
        if (
            len(concave_indices) >= 2
            and concave_indices[0] == 0
            and concave_indices[-1] == point_count - 1
        ):
            if 0 in groups and point_count - 1 in groups:
                preserve_concave.update(groups)
        else:
            preserve_concave.update(groups)

    kept_points = [
        idx
        for idx, info in enumerate(point_info)
        if info["is_convex"] or (idx in preserve_concave and info["angle"] >= 120)
    ]
    final_points = []
    for idx in range(len(kept_points)):
        current_idx = kept_points[idx]
        next_idx = kept_points[(idx + 1) % len(kept_points)]
        final_points.append(current_idx)
        distance = np.linalg.norm(poly[current_idx] - poly[next_idx])
        if distance > max_allowed_dist:
            intermediate = (
                list(range(current_idx + 1, next_idx))
                if next_idx > current_idx
                else list(range(current_idx + 1, point_count)) + list(range(0, next_idx))
            )
            if intermediate:
                needed = int(np.ceil(distance / max_allowed_dist)) - 1
                if len(intermediate) <= needed:
                    final_points.extend(intermediate)
                else:
                    step = len(intermediate) / needed
                    final_points.extend([intermediate[int(i * step)] for i in range(needed)])

    result = []
    for idx in sorted(set(final_points)):
        info = point_info[idx]
        point = poly[idx]
        if info["is_convex"] and abs(info["angle"] - sharp_angle_thresh) < 1:
            v1_norm = info["v1"] / np.linalg.norm(info["v1"])
            v2_norm = info["v2"] / np.linalg.norm(info["v2"])
            direction = v1_norm + v2_norm
            direction /= np.linalg.norm(direction)
            distance = (np.linalg.norm(info["v1"]) + np.linalg.norm(info["v2"])) / 2
            result.append(tuple(point + direction * distance))
        else:
            result.append(tuple(point))
    return result


def _normalize_layout_polygon(
    box: np.ndarray,
    polygon: np.ndarray | None,
    previous_polygon: np.ndarray | None,
) -> np.ndarray:
    rect = _rect_from_box(box)
    if polygon is None:
        return rect
    polygon = np.asarray(polygon, dtype=np.float32)
    if polygon.ndim == 1:
        polygon = polygon.reshape(-1, 2)
    if len(polygon) < 4:
        return rect
    quad = _polygon_to_quad(polygon)
    if quad is not None:
        rect_list = rect.tolist()
        quad_list = quad.tolist()
        if _polygon_overlap_ratio(rect_list, quad_list, "union") >= 0.95:
            return rect
        polygon_list = polygon.tolist()
        iou_poly_quad = _polygon_overlap_ratio(polygon_list, quad_list, "union")
        iou_pre = 0.0
        if previous_polygon is not None:
            iou_pre = _polygon_overlap_ratio(previous_polygon.tolist(), rect_list, "small")
        if iou_poly_quad >= 0.8 and iou_pre < 0.01:
            return quad
    return polygon


def _mask_to_polygon(mask: np.ndarray, max_allowed_dist: float) -> np.ndarray | None:
    try:
        import cv2
    except Exception:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.004 * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True).squeeze()
    polygon = np.atleast_2d(polygon)
    if polygon.shape[0] < 4:
        return None
    return np.asarray(_extract_custom_vertices(polygon, max_allowed_dist))


def _extract_polygon_points_by_masks(
    boxes: np.ndarray,
    masks: np.ndarray | None,
    image_size: tuple[int, int],
) -> list[list[list[float]] | None]:
    if masks is None or len(boxes) == 0:
        return [None for _ in boxes]
    try:
        import cv2
    except Exception:
        return [None for _ in boxes]
    width, height = image_size
    scale_w = masks.shape[2] / float(width)
    scale_h = masks.shape[1] / float(height)
    max_box_w = max(boxes[:, 4] - boxes[:, 3])
    polygons: list[np.ndarray] = []
    output: list[list[list[float]] | None] = []
    for idx, box in enumerate(boxes):
        rounded_box = np.asarray(box[2:6], dtype=np.float32)
        x1, y1, x2, y2 = rounded_box.astype(np.int32)
        box_w = int(x2 - x1)
        box_h = int(y2 - y1)
        if box_w <= 0 or box_h <= 0:
            polygon = _rect_from_box(rounded_box)
        else:
            x_s = np.clip([int(round(x1 * scale_w)), int(round(x2 * scale_w))], 0, masks.shape[2])
            y_s = np.clip([int(round(y1 * scale_h)), int(round(y2 * scale_h))], 0, masks.shape[1])
            cropped = masks[idx, y_s[0] : y_s[1], x_s[0] : x_s[1]]
            if cropped.size == 0 or np.sum(cropped) == 0:
                polygon = _rect_from_box(rounded_box)
            else:
                resized = cv2.resize(
                    cropped.astype(np.uint8),
                    (box_w, box_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                max_allowed_dist = box_w if box_w > max_box_w * 0.6 else max_box_w
                mask_polygon = _mask_to_polygon(resized, max_allowed_dist)
                if mask_polygon is not None and len(mask_polygon) > 0:
                    mask_polygon = mask_polygon + np.asarray([x1, y1])
                polygon = _normalize_layout_polygon(
                    rounded_box,
                    mask_polygon,
                    previous_polygon=polygons[-1] if polygons else None,
                )
        polygons.append(polygon)
        output.append([[float(x), float(y)] for x, y in polygon.tolist()])
    return output


def _horizontal_overlap_ratio(a: list[int], b: list[int]) -> float:
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    min_width = max(1, min(a[2] - a[0], b[2] - b[0]))
    return overlap / float(min_width)


def _heading_should_precede_body(heading: LayoutBox, body: LayoutBox) -> bool:
    heading_box = heading.coordinate
    body_box = body.coordinate
    heading_center_y = (heading_box[1] + heading_box[3]) / 2.0
    body_center_y = (body_box[1] + body_box[3]) / 2.0
    if heading_center_y >= body_center_y:
        return False
    vertical_gap = body_box[1] - heading_box[3]
    if vertical_gap < -4 or vertical_gap > 96:
        return False
    return _horizontal_overlap_ratio(heading_box, body_box) >= 0.45


def _same_column_should_precede(upper: LayoutBox, lower: LayoutBox) -> bool:
    if upper.label != lower.label:
        return False
    if upper.label not in {"text", "content", "reference_content", "vertical_text"}:
        return False
    if upper.coordinate[1] >= lower.coordinate[1]:
        return False
    vertical_gap = lower.coordinate[1] - upper.coordinate[3]
    if vertical_gap < 0 or vertical_gap > 8:
        return False
    return _horizontal_overlap_ratio(upper.coordinate, lower.coordinate) >= 0.45


def normalize_layout_order(boxes: list[LayoutBox]) -> list[LayoutBox]:
    ordered = list(boxes)
    index = 0
    while index + 1 < len(ordered):
        current = ordered[index]
        following = ordered[index + 1]
        if _same_column_should_precede(following, current) or (
            current.label in BODY_TEXT_LABELS
            and following.label in HEADING_LABELS
            and _heading_should_precede_body(following, current)
        ):
            ordered[index], ordered[index + 1] = following, current
            index = max(0, index - 1)
            continue
        index += 1
    return ordered


def postprocess_layout(
    raw_boxes: np.ndarray,
    image_size: tuple[int, int],
    raw_masks: np.ndarray | None = None,
    threshold: float = 0.5,
    layout_nms: bool = False,
    layout_unclip_ratio: float
    | tuple[float, float]
    | list[float]
    | dict[int, tuple[float, float]]
    | None = None,
    layout_merge_bboxes_mode: dict[int, str] | None = None,
) -> list[LayoutBox]:
    raw_boxes = raw_boxes.copy()
    raw_boxes[:, 2:6] = np.round(raw_boxes[:, 2:6]).astype(int)
    keep = (raw_boxes[:, 1] > threshold) & (raw_boxes[:, 0] > -1)
    boxes = raw_boxes[keep]
    masks = raw_masks[keep] if raw_masks is not None else None
    if boxes.size == 0:
        return []

    if layout_nms:
        selected = _layout_nms_indices(boxes)
        boxes = boxes[selected]
        if masks is not None:
            masks = masks[selected]

    width, height = image_size
    if len(boxes) > 1 and boxes.shape[1] in (6, 7, 8):
        image_index = LABEL_LIST.index("image")
        area_threshold = 0.82 if width > height else 0.93
        image_area = width * height
        filtered = []
        filtered_masks = []
        for idx, box in enumerate(boxes):
            if int(box[0]) != image_index:
                filtered.append(box)
                if masks is not None:
                    filtered_masks.append(masks[idx])
                continue
            clipped = box.copy()
            clipped[2] = max(0, clipped[2])
            clipped[3] = max(0, clipped[3])
            clipped[4] = min(width, clipped[4])
            clipped[5] = min(height, clipped[5])
            if _box_area(clipped) <= area_threshold * image_area:
                filtered.append(box)
                if masks is not None:
                    filtered_masks.append(masks[idx])
        if filtered:
            boxes = np.asarray(filtered)
            if masks is not None:
                masks = np.asarray(filtered_masks)

    keep = _layout_merge_keep_mask(boxes, layout_merge_bboxes_mode)
    boxes = boxes[keep]
    if masks is not None:
        masks = masks[keep]

    if boxes.shape[1] == 7:
        sorted_idx = np.argsort(boxes[:, 6])
        boxes = boxes[sorted_idx]
        if masks is not None:
            masks = masks[sorted_idx]
    elif boxes.shape[1] == 8:
        sorted_idx = np.lexsort((-boxes[:, 7], boxes[:, 6]))
        boxes = boxes[sorted_idx]
        if masks is not None:
            masks = masks[sorted_idx]

    polygon_points = _extract_polygon_points_by_masks(boxes, masks, image_size)
    boxes = _unclip_boxes(boxes[:, :6], layout_unclip_ratio)

    results: list[LayoutBox] = []
    for idx, raw in enumerate(boxes):
        cls_id = int(raw[0])
        x1 = max(0, int(raw[2]))
        y1 = max(0, int(raw[3]))
        x2 = min(width, int(raw[4]))
        y2 = min(height, int(raw[5]))
        if x2 <= x1 or y2 <= y1:
            continue
        results.append(
            LayoutBox(
                cls_id=cls_id,
                label=LABEL_LIST[cls_id],
                score=float(raw[1]),
                coordinate=[x1, y1, x2, y2],
                order=idx + 1,
                polygon_points=polygon_points[idx],
            )
        )
    return results


def resolve_layout_providers(
    available: Sequence[str], requested: str, platform_name: str
) -> list[str]:
    choice = requested.strip().lower()
    if choice == "auto":
        choice = "directml" if platform_name == "Windows" else "cpu"
    mapping = {
        "directml": "DmlExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }
    if choice not in mapping:
        raise ValueError(f"Unsupported layout provider: {requested}")
    provider = mapping[choice]
    if provider not in available:
        if choice == "directml":
            raise RuntimeError(
                "DmlExecutionProvider is unavailable. Install onnxruntime-directml with "
                "pip install -e '.[gpu]' and verify the AMD graphics driver."
            )
        raise RuntimeError(f"{provider} is unavailable")
    return [provider]


class PPDocLayoutV3Onnx:
    def __init__(
        self,
        model_dir: Path,
        providers: list[str] | None = None,
        intra_op_threads: int | None = None,
        requested_provider: str | None = None,
    ) -> None:
        import onnxruntime as ort

        model_path = model_dir / "inference.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"PP-DocLayoutV3 ONNX not found: {model_path}")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if intra_op_threads:
            options.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers or ["CPUExecutionProvider"],
        )
        disable_fallback = getattr(self.session, "disable_fallback", None)
        if disable_fallback is not None:
            disable_fallback()
        requested = providers or ["CPUExecutionProvider"]
        self.layout_provider_requested = requested_provider or (
            "directml" if requested[0] == "DmlExecutionProvider" else "cpu"
        )
        self.layout_providers_active = list(self.session.get_providers())
        self.active_providers = self.layout_providers_active
        if (
            providers
            and providers[0] == "DmlExecutionProvider"
            and (
                not self.layout_providers_active
                or self.layout_providers_active[0] != "DmlExecutionProvider"
            )
        ):
            raise RuntimeError(
                "DmlExecutionProvider failed to activate; refusing CPU fallback for "
                "layout inference"
            )

    def predict(
        self,
        image_path: Path,
        threshold: float = 0.5,
        layout_nms: bool = False,
        layout_unclip_ratio: float
        | tuple[float, float]
        | list[float]
        | dict[int, tuple[float, float]]
        | None = (1.0, 1.0),
        layout_merge_bboxes_mode: dict[int, str] | None = None,
    ) -> tuple[list[LayoutBox], np.ndarray]:
        feeds, image_size, rgb = preprocess_layout_image(image_path)
        outputs = self.session.run(None, feeds)
        return (
            postprocess_layout(
                outputs[0],
                image_size=image_size,
                raw_masks=outputs[2] if len(outputs) > 2 else None,
                threshold=threshold,
                layout_nms=layout_nms,
                layout_unclip_ratio=layout_unclip_ratio,
                layout_merge_bboxes_mode=layout_merge_bboxes_mode,
            ),
            rgb,
        )
