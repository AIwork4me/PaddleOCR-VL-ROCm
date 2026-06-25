from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_download_module():
    script = Path("scripts/download_ppdoclayoutv3_onnx.py")
    spec = importlib.util.spec_from_file_location("download_ppdoclayoutv3_onnx", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_script_defaults_to_public_hf_repo():
    module = _load_download_module()

    assert module.DEFAULT_REPO_ID == "AlexTransformer/PP-DocLayoutV3-onnx"
    assert module.REQUIRED_FILES == ["inference.onnx", "inference.yml"]
