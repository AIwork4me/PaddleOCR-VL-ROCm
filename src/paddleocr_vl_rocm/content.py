from __future__ import annotations

from collections import Counter


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
