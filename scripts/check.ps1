$ErrorActionPreference = "Stop"
python -m compileall -q src
ruff check src tests scripts eval
ruff format --check src tests scripts eval
mypy src
python -m pytest -q
