from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_checker():
    script = Path("scripts/check_official_paddleocr.py")
    spec = importlib.util.spec_from_file_location("check_official_paddleocr", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_check_reports_import_success(monkeypatch):
    mod = _load_checker()
    fake = types.SimpleNamespace(PaddleOCRVL=object)
    monkeypatch.setitem(sys.modules, "paddleocr", fake)

    result = mod.check_official_dependency(
        construct=False,
        server_url="http://127.0.0.1:8111/v1",
        api_model_name="PaddleOCR-VL-1.6-GGUF.gguf",
    )

    assert result["ok"] is True
    assert result["paddleocr_found"] is True
    assert result["PaddleOCRVL_found"] is True
    assert result["constructed"] is False


def test_dependency_check_constructs_with_local_llamacpp_arguments(monkeypatch):
    mod = _load_checker()
    captured = {}

    class FakePaddleOCRVL:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCRVL=FakePaddleOCRVL))

    result = mod.check_official_dependency(
        construct=True,
        server_url="http://127.0.0.1:8111/v1",
        api_model_name="PaddleOCR-VL-1.6-GGUF.gguf",
    )

    assert result["ok"] is True
    assert result["constructed"] is True
    assert captured == {
        "pipeline_version": "v1.6",
        "vl_rec_backend": "llama-cpp-server",
        "vl_rec_server_url": "http://127.0.0.1:8111/v1",
        "vl_rec_api_model_name": "PaddleOCR-VL-1.6-GGUF.gguf",
    }
