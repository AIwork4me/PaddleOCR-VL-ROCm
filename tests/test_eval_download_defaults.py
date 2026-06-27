from __future__ import annotations

import importlib.util
from pathlib import Path


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(file))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_defaults():
    mod = _load("dl", "eval/download_omnidocbench.py")
    assert mod.DEFAULT_REPO_ID == "opendatalab/OmniDocBench"
    assert {"v15", "v16"} <= set(mod.VERSIONS)
