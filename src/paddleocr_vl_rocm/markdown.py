from __future__ import annotations

import re

from .constants import MARKDOWN_IGNORE_LABEL_SET
from .models import LightBlock


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
