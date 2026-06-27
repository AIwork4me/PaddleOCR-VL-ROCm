from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_adapter():
    script = Path("eval/PaddleOCRVLROCm_img2md.py")
    spec = importlib.util.spec_from_file_location("paddleocrvl_rocm_img2md", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_md_name_strips_extension():
    mod = _load_adapter()
    # OmniDocBench matcher looks up <img_name[:-4]>.md first
    assert mod.expected_md_name("page_001.png") == "page_001.md"
    assert mod.expected_md_name("doc.jpeg") == "doc.md"


def test_image_extensions_lowercase():
    mod = _load_adapter()
    exts = {e.lower() for e in mod.IMAGE_EXTENSIONS}
    assert {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"} <= exts
