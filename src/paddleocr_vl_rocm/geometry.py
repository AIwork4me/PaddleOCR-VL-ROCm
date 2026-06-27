from __future__ import annotations

from .layout import LayoutBox


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
