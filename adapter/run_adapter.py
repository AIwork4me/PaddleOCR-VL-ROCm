#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026 AIwork4me
"""omnidocbench-rocm platform adapter — delegates to PaddleOCRVLROCm_img2md.

The engine invokes this as a subprocess. Maps platform CLI flags to the
existing PaddleOCR-VL-ROCm run_adapter API.
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from PaddleOCRVLROCm_img2md import run_adapter as _run_adapter  # noqa: E402


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="PaddleOCR-VL-ROCm OmniDocBench adapter",
    )
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--platform", required=True, choices=["linux-rocm", "windows-hip"])
    p.add_argument("--backend", default="lightweight")
    p.add_argument("--server-url", default="http://127.0.0.1:8111/v1")
    p.add_argument("--api-model-name", default="PaddleOCR-VL-1.6-GGUF.gguf")
    p.add_argument("--skip-existing", action="store_true")
    a = p.parse_args(argv)

    try:
        _run_adapter(
            Path(a.img_dir),
            Path(a.out_dir),
            a.server_url,
            engine=a.backend,
            api_model_name=a.api_model_name,
            vlm_backend="llama-cpp-server",
            page_retries=1,
        )
    except SystemExit as e:
        if e.code == 2:
            return 1  # adapter exit 2 = too many failures
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
