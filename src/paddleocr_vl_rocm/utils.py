from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console


def get_console() -> Console:
    return Console(highlight=False)


def get_logger(name: str = "paddleocr_vl_rocm") -> logging.Logger:
    return logging.getLogger(name)


def ensure_input_file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.exists():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Input path is not a file: {resolved}")
    return resolved


def ensure_output_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if hasattr(value, "numpy") and callable(value.numpy):
        return value.numpy().tolist()
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def to_jsonable(value: Any, depth: int = 0, max_depth: int = 8) -> Any:
    if depth > max_depth:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, depth + 1, max_depth) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v, depth + 1, max_depth) for v in value]
    try:
        converted = json_default(value)
    except Exception:
        return str(value)
    if converted is value:
        return str(value)
    return to_jsonable(converted, depth + 1, max_depth)


def write_json(path: str | Path, value: Any) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(to_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return resolved


def _call_noarg(value: Any, name: str) -> Any:
    attr = getattr(value, name, None)
    if callable(attr):
        return attr()
    return attr


def extract_markdown(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        for key in ("markdown", "md", "content", "text"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
        parts: list[str] = []
        for item in value.values():
            extracted = extract_markdown(item)
            if extracted:
                parts.append(extracted)
        return "\n\n".join(parts) if parts else None
    if isinstance(value, (list, tuple)):
        parts = [part for item in value if (part := extract_markdown(item))]
        return "\n\n".join(parts) if parts else None

    for name in ("to_markdown", "markdown", "md"):
        try:
            extracted = _call_noarg(value, name)
        except Exception:
            continue
        if isinstance(extracted, str) and extracted.strip():
            return extracted.strip()
        if extracted is not None and extracted is not value:
            nested = extract_markdown(extracted)
            if nested:
                return nested

    return None


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = json.loads(json.dumps(payload, default=json_default))
    messages = summary.get("messages", [])
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_url = part.get("image_url", {})
                    if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                        url = image_url["url"]
                        image_url["url"] = f"{url[:64]}...<base64 omitted, {len(url)} chars>"
    return summary
