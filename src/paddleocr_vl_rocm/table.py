from __future__ import annotations

import html
import itertools
import re

from .models import TableCell

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
