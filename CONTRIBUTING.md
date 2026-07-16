# Contributing

Use Python 3.10-3.13 and a focused branch. Do not commit models, datasets, predictions, raw traces, credentials, private documents, or `eval/.omnidocbench/`.

```powershell
pip install -e .[dev]
python -m compileall -q src/paddleocr_vl_rocm eval
ruff check src tests scripts eval
ruff format --check src tests scripts eval
mypy src
python -m pytest -q
python -m build
```

Tests and CI must remain offline. Add a regression test before changing behavior. Benchmark claims must identify the exact scorer commit, coverage, hardware, runtime, hashes, and provenance. Prompt, crop, normalization, model-output, or serializer changes require authenticated same-boundary oracle evidence.

