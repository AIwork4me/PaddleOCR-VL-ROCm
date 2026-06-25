#!/usr/bin/env bash
set -euo pipefail
python -m compileall -q src
ruff check src tests scripts
ruff format --check src tests scripts
mypy src
python -m pytest -q
