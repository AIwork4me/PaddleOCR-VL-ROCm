from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PaddleOCRVLROCmResult(dict):
    """PaddleOCR-VL-style result with lightweight save helpers."""

    def __init__(self, payload: dict[str, Any], markdown_text: str = "") -> None:
        super().__init__(payload)
        self.markdown_text = markdown_text

    @property
    def input_stem(self) -> str:
        input_path = str(self.get("input_path") or "result")
        stem = Path(input_path).stem
        return stem or "result"

    def print(self) -> None:
        blocks = self.get("parsing_res_list") or []
        print(json.dumps({"input_path": self.get("input_path"), "blocks": len(blocks)}, ensure_ascii=False, indent=2))

    def save_to_json(self, save_path: str | Path) -> Path:
        base = Path(save_path)
        path = base if base.suffix.lower() == ".json" else base / f"{self.input_stem}_res.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def save_to_markdown(self, save_path: str | Path, pretty: bool = False) -> Path:
        base = Path(save_path)
        path = base if base.suffix.lower() in {".md", ".markdown"} else base / f"{self.input_stem}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.markdown_text, encoding="utf-8")
        return path

