from __future__ import annotations

import math
import random
import re
from typing import Any

from PIL import Image

from .geometry import _overlap_ratio, _projection_overlap_ratio
from .imageio import _crop, _crop_from_bgr, _merge_images
from .layout import LayoutBox
from .models import LightBlock


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
