from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import MARKDOWN_IGNORE_LABEL_SET, MARKDOWN_IGNORE_LABELS, SKIP_ORDER_LABELS
from .layout import LayoutBox
from .models import LightBlock


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
