$ErrorActionPreference = "Stop"
python -m compileall -q src
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
ruff check src tests scripts eval
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
ruff format --check src tests scripts eval
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
mypy src
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
