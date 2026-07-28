# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""OmniDocBench-ROCm standard CLI contract surface (ADR-0011), locked to central
commit ccd466ef317fd6a710131db3a19ec9d55a65ce2e. version/capabilities/parse share
ONE inference core (the OpenAI-compatible VLM client) with `run` and the adapter.
`--json` prints exactly one JSON document to stdout; logs go to stderr.
Exit codes 0/1/2/3/4/5.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_USAGE = 2
EXIT_BACKEND_MISMATCH = 3
EXIT_CONTRACT = 4
EXIT_FATAL = 5

SCHEMA_VERSION = 1
NAME = "paddleocr-vl-rocm"

# MUST agree with rocmdoc.yaml; `manifest --card` enforces result<->capability.
DECLARED_PLATFORMS = [
    {
        "platform": "windows-hip",
        "backend": "llama-cpp",
        "precision": "bf16",
        "interface": "standard-cli",
    },
    {
        "platform": "windows-hip",
        "backend": "onnx-directml",
        "precision": "fp32",
        "interface": "adapter-script",
    },
    {
        "platform": "linux-rocm",
        "backend": "llama-cpp",
        "precision": "bf16",
        "interface": "adapter-script",
    },
]
DECLARED_INTERFACES = ["standard-cli", "adapter-script", "api-server"]

DEFAULT_MODEL = os.environ.get("PADDLEOCRVL_API_MODEL_NAME", "PaddleOCR-VL-1.6-GGUF.gguf")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
# VLM backends served via an OpenAI-compatible server; the ONNX layout component
# is not served by `parse` (it is part of the mixed pipeline run by `run`).
_VLM_BACKENDS = {"llama-cpp", "vllm", "openai"}


def emit_json(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def cmd_version() -> int:
    try:
        from importlib.metadata import version

        v = version("paddleocr-vl-rocm")
    except Exception:
        v = "0.0.0+unknown"
    emit_json(
        {
            "name": NAME,
            "version": v,
            "engine_version": "PaddleOCR-VL via llama.cpp HIP (VLM) + ONNX DirectML (layout)",
            "schema_version": SCHEMA_VERSION,
            "central_spec_commit": "ccd466ef317fd6a710131db3a19ec9d55a65ce2e",
            "cli_contract": "omnidocbench-rocm cli-contract.md (ADR-0011)",
        }
    )
    return EXIT_OK


def cmd_capabilities() -> int:
    emit_json({"platforms": DECLARED_PLATFORMS, "interfaces": DECLARED_INTERFACES})
    return EXIT_OK


def _list_images(img_dir: Path) -> list[Path]:
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS and p.is_file())


class _UsageError(Exception):
    pass


def _resolve_server(server_url: str | None) -> str:
    url = (
        server_url or os.environ.get("PADDLEOCRVL_SERVER_URL") or os.environ.get("VLLM_SERVER_URL")
    )
    if not url:
        raise _UsageError("no server URL: pass --server-url or set PADDLEOCRVL_SERVER_URL")
    return url


def _build_client(server_url: str):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise _UsageError("openai client not installed") from exc
    return OpenAI(api_key="EMPTY", base_url=server_url, timeout=3600.0)


def _real_infer(client, image_path: str, *, model: str) -> str:
    """Delegate to the shared VLM inference core (same as `run`/adapter)."""
    from paddleocr_vl_rocm.server import infer_one  # shared inference core

    return infer_one(client, image_path, model=model)


INFER = _real_infer


def cmd_parse(
    *,
    img_dir: Path,
    out_dir: Path,
    platform: str,
    backend: str | None,
    server_url: str | None,
    model: str | None,
    limit: int | None,
) -> int:
    """Parse images -> <out_dir>/<stem>.md; emit cli_result. R2 robust; page_count
    always == #images. The ONNX layout backend is not served here (use `run`)."""
    backend = (backend or "llama-cpp").lower()
    if backend not in _VLM_BACKENDS:
        _stderr(
            f"parse serves OpenAI-compatible VLM backends {_VLM_BACKENDS}; "
            f"backend {backend!r} (ONNX layout) is part of the mixed pipeline "
            "— use `paddleocr-vl-rocm run`."
        )
        return EXIT_USAGE
    if not img_dir.is_dir():
        _stderr(f"img-dir not found: {img_dir}")
        return EXIT_USAGE
    images = _list_images(img_dir)
    if limit is not None:
        images = images[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    actual_backend = backend
    try:
        client = _build_client(_resolve_server(server_url))
    except _UsageError as exc:
        _stderr(str(exc))
        return EXIT_USAGE
    mdl = model or DEFAULT_MODEL
    pages: list[dict] = []
    ok = failed = 0
    for img in images:
        md_path = out_dir / (img.stem + ".md")
        t0 = time.monotonic()
        try:
            md = INFER(client, str(img), model=mdl)
            if not isinstance(md, str) or not md.strip():
                raise RuntimeError("empty prediction")
            md_path.write_text(md, encoding="utf-8")
            pages.append(
                {"image": img.name, "status": "ok", "seconds": round(time.monotonic() - t0, 3)}
            )
            ok += 1
        except Exception as exc:  # R2: per-page failure caught, run continues
            if md_path.exists():
                try:
                    md_path.unlink()
                except OSError:
                    pass
            pages.append({"image": img.name, "status": "failed", "error": str(exc)[:300]})
            failed += 1
    status = "ok" if failed == 0 else ("partial" if ok > 0 else "failed")
    assert len(pages) == len(images), "page conservation violated"
    emit_json(
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "backend": actual_backend,
            "engine": actual_backend,
            "page_count": len(images),
            "ok": ok,
            "failed": failed,
            "skipped": 0,
            "output_dir": str(out_dir),
            "full_set": limit is None,
            "pages": pages,
            "requested_backend": backend,
        }
    )
    if status == "ok":
        return EXIT_OK
    if status == "partial":
        return EXIT_PARTIAL
    return EXIT_FATAL
