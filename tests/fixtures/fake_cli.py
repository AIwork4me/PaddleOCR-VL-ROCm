#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""FAKE standard-CLI fixture for benchmark-omnidocbench-v16 conformance — exercises
the central profile machinery WITHOUT a GPU/model (as the central repo does in CI).
TEST FIXTURE; the real-model benchmark is a GPU workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _emit(obj):
    sys.stdout.write(json.dumps(obj, indent=2) + "\n")


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "version":
        _emit({"name": "paddleocr-vl-rocm-fake", "version": "0.0.0-fixture", "schema_version": 1})
        return 0
    if cmd == "capabilities":
        _emit(
            {
                "platforms": [{"platform": "windows-hip", "backend": "llama-cpp"}],
                "interfaces": ["standard-cli"],
            }
        )
        return 0
    if cmd == "doctor":
        _emit({"status": "ready"})
        return 0
    if cmd == "parse":
        img_dir = out_dir = backend = ""
        i = 1
        while i < len(argv):
            a = argv[i]
            if a == "--img-dir":
                img_dir = argv[i + 1]
                i += 2
            elif a == "--out-dir":
                out_dir = argv[i + 1]
                i += 2
            elif a == "--backend":
                backend = argv[i + 1]
                i += 2
            elif a in ("--platform", "--benchmark"):
                i += 2
            elif a == "--json":
                i += 1
            else:
                i += 1
        imgs = (
            sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in _IMAGE_EXTS)
            if img_dir
            else []
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pages = []
        for img in imgs:
            (Path(out_dir) / (img.stem + ".md")).write_text("# fake\n", encoding="utf-8")
            pages.append({"image": img.name, "status": "ok", "seconds": 0.001})
        actual = backend or "llama-cpp"
        _emit(
            {
                "schema_version": 1,
                "status": "ok",
                "backend": actual,
                "engine": actual,
                "page_count": len(imgs),
                "ok": len(imgs),
                "failed": 0,
                "skipped": 0,
                "output_dir": out_dir,
                "full_set": True,
                "pages": pages,
            }
        )
        return 0
    print(f"fake_cli: unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
