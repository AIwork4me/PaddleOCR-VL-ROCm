from __future__ import annotations

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
