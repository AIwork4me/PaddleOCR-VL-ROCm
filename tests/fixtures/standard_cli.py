#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Entry shim so the central conformance runner can invoke the REAL
``paddleocr-vl-rocm`` standard CLI as a subprocess. Adds NO behavior of its own.
Run with PYTHONPATH=src (or a wheel install)."""

import sys

from paddleocr_vl_rocm.cli import main

if __name__ == "__main__":
    sys.exit(main())
