# OmniDocBench v1.6 Evidence And Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every local score use the exact official OmniDocBench v1.6 page-level fields, notebook rounding order, full-coverage gates, and auditable provenance.

**Architecture:** Keep inference, scoring, and evidence publication separate. Pin a dedicated OmniDocBench v1.6 checkout, validate the official scoring file blobs before every release run, then extract leaderboard values through one tested function in `eval/artifact_utils.py`. Refuse to publish evidence unless prediction coverage and CDM/TEDS quality gates are clean.

**Tech Stack:** Python 3.10+, pytest, PowerShell, Git, OmniDocBench v1.6, JSON, SHA-256.

> **Superseded G0 requirement (2026-07-13):** The project owner approved the
> immutable issue #18248 exception defined in
> `docs/superpowers/specs/2026-07-13-release-gate-recovery-design.md`.
> Release evidence now requires 1,650 successful predictions plus the sole
> approved `peg-native` failure while scoring all 1,651 GT pages. The original
> 1,651-success text below is retained only as historical plan context.

## Global Constraints

- OmniDocBench v1.6 commit is `147cd5ac9472002f5751221d390bf00abdbc0d2f`.
- The full dataset contains 1651 pages.
- Formula leaderboard value is `display_formula.page.CDM.ALL * 100`.
- Table leaderboard value is `table.page.TEDS.ALL * 100`.
- Compute Overall only after rounding Text Edit, Formula CDM, and Table TEDS to three decimals.
- Official-local evidence requires `ok=1651`, `fail=0`, and `fallback=0`.
- CDM and TEDS require zero timeouts and zero exceptions/errors.
- Do not commit `data/`, `predictions/`, `logs/`, `eval/.omnidocbench/`, model files, or secrets.

---

## File Structure

- Create `eval/benchmark_contract.py`: v1.6 commit, scoring blob IDs, checkout validation, and file hashing.
- Create `tests/test_benchmark_contract.py`: checkout and provenance contract tests.
- Modify `eval/artifact_utils.py`: exact notebook metric extraction and metric-quality gates.
- Modify `tests/test_eval_artifact_utils.py`: field-selection, rounding, and invalid-run tests.
- Modify `eval/run_eval.py`: strict full-run evidence gate and contract validation.
- Modify `tests/test_eval_report_path.py`: full-run and checkout-gate tests.
- Create `eval/patches/omnidocbench-v16-windows-cdm.patch`: isolated Windows execution patch for CDM.
- Create `scripts/prepare_omnidocbench_v16.ps1`: reproducible pinned checkout preparation.
- Modify `scripts/run_official_local_v16.ps1`: repair-page and strict evidence commands.
- Modify `results/omnidocbench/v16/README.md`, `README.md`, and `README.zh-CN.md`: corrected official values and provenance.

### Task 1: Implement the exact official notebook metric extractor

**Files:**
- Modify: `eval/artifact_utils.py`
- Test: `tests/test_eval_artifact_utils.py`

**Interfaces:**
- Consumes: OmniDocBench metric-result dictionaries.
- Produces: `extract_notebook_metrics(metric: dict[str, Any]) -> dict[str, float | None]`.
- Produces: `analyze_metric_quality(metric: dict[str, Any]) -> dict[str, Any]` with Formula CDM and Table TEDS status.

- [ ] **Step 1: Write failing field-selection and rounding tests**

Add tests that distinguish page-level values from sample-level values:

```python
def test_extract_notebook_metrics_uses_official_page_fields_and_rounding():
    metric = {
        "text_block": {"all": {"Edit_dist": {"ALL_page_avg": 0.0344448}}},
        "display_formula": {
            "all": {"CDM": {"all": 0.9617079}},
            "page": {"CDM": {"ALL": 0.96502201}},
        },
        "table": {
            "all": {"TEDS": {"all": 0.9304263}},
            "page": {
                "TEDS": {"ALL": 0.94239317},
                "TEDS_structure_only": {"ALL": 0.955},
            },
        },
        "reading_order": {"all": {"Edit_dist": {"ALL_page_avg": 0.1294874}}},
    }

    values = artifact_utils.extract_notebook_metrics(metric)

    assert values == {
        "text_edit_dist": 0.034,
        "formula_cdm_percent": 96.502,
        "table_teds_percent": 94.239,
        "table_teds_structure_only_percent": 95.5,
        "reading_order_edit_dist": 0.129,
        "overall": 95.78033333333333,
    }
```

Add a regression test using the tracked official-local artifact and assert
`overall == 95.78033333333333`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_eval_artifact_utils.py -q
```

Expected: FAIL because `extract_notebook_metrics` does not exist.

- [ ] **Step 3: Implement the extractor**

Add this implementation and make `extract_readme_metrics` delegate to it:

```python
def _rounded(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def extract_notebook_metrics(metric: dict[str, Any]) -> dict[str, float | None]:
    text = _nested_number(metric, "text_block", "all", "Edit_dist", "ALL_page_avg")
    formula_raw = _nested_number(metric, "display_formula", "page", "CDM", "ALL")
    table_raw = _nested_number(metric, "table", "page", "TEDS", "ALL")
    table_s_raw = _nested_number(
        metric, "table", "page", "TEDS_structure_only", "ALL"
    )
    reading = _nested_number(
        metric, "reading_order", "all", "Edit_dist", "ALL_page_avg"
    )

    text_value = _rounded(text)
    formula_value = _rounded(None if formula_raw is None else formula_raw * 100.0)
    table_value = _rounded(None if table_raw is None else table_raw * 100.0)
    overall = None
    if text_value is not None and formula_value is not None and table_value is not None:
        overall = ((1.0 - text_value) * 100.0 + formula_value + table_value) / 3.0

    return {
        "text_edit_dist": text_value,
        "formula_cdm_percent": formula_value,
        "table_teds_percent": table_value,
        "table_teds_structure_only_percent": _rounded(
            None if table_s_raw is None else table_s_raw * 100.0
        ),
        "reading_order_edit_dist": _rounded(reading),
        "overall": overall,
    }
```

- [ ] **Step 4: Add Formula CDM and Table TEDS quality checks**

Refactor quality extraction through this helper:

```python
def _debug_quality(
    metric: dict[str, Any], element: str, name: str, error_key: str
) -> dict[str, Any]:
    debug = _nested(metric, element, "metric_debug", name) or {}
    sample_count = debug.get("sample_count")
    timeout_count = int(debug.get("timeout_case_count") or 0)
    error_count = int(debug.get(error_key) or 0)
    valid = (
        isinstance(sample_count, int)
        and sample_count > 0
        and timeout_count == 0
        and error_count == 0
    )
    return {
        "valid": valid,
        "sample_count": sample_count,
        "timeout_case_count": timeout_count,
        error_key: error_count,
        "reason": "" if valid else (
            f"{name} requires samples>0, timeouts=0, errors=0; "
            f"samples={sample_count}, timeouts={timeout_count}, errors={error_count}"
        ),
    }
```

Return `formula_cdm` using `exception_case_count` and `table_teds` using
`error_case_count`. Make `extract_readme_metrics` set affected values and
Overall to `None` when either quality gate fails.

- [ ] **Step 5: Run focused and full tests**

Run:

```powershell
python -m pytest tests/test_eval_artifact_utils.py -q
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add eval/artifact_utils.py tests/test_eval_artifact_utils.py
git commit -m "fix(eval): match official v16 notebook scoring"
```

### Task 2: Pin and validate the official v1.6 scorer

**Files:**
- Create: `eval/benchmark_contract.py`
- Create: `tests/test_benchmark_contract.py`
- Create: `eval/patches/omnidocbench-v16-windows-cdm.patch`
- Create: `scripts/prepare_omnidocbench_v16.ps1`

**Interfaces:**
- Produces: `validate_checkout(checkout: Path) -> dict[str, object]`.
- Produces: `sha256_file(path: Path) -> str`.
- The preparation script creates `eval/.omnidocbench` at the pinned commit and applies only the tracked Windows CDM patch.

- [ ] **Step 1: Write checkout contract tests**

```python
def test_validate_checkout_accepts_expected_commit_and_blobs(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "_git", lambda *args: {
        ("rev-parse", "HEAD"): contract.OMNIDOCBENCH_V16_COMMIT,
        ("rev-parse", "HEAD:tools/generate_result_tables.ipynb"):
            contract.SCORING_BLOBS["tools/generate_result_tables.ipynb"],
    }[args])
    monkeypatch.setattr(
        contract,
        "SCORING_BLOBS",
        {"tools/generate_result_tables.ipynb": "72fb7a5c7d40bb6f1b2b839fc33d31856c756ee8"},
    )

    result = contract.validate_checkout(tmp_path)

    assert result["commit"] == contract.OMNIDOCBENCH_V16_COMMIT


def test_validate_checkout_rejects_v17_head(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "_git", lambda *args: "0c7db667")
    with pytest.raises(RuntimeError, match="OmniDocBench v1.6"):
        contract.validate_checkout(tmp_path)
```

- [ ] **Step 2: Run the tests and verify RED**

Run `python -m pytest tests/test_benchmark_contract.py -q`.

Expected: import failure because `eval/benchmark_contract.py` is absent.

- [ ] **Step 3: Implement the contract module**

Define these exact constants:

```python
OMNIDOCBENCH_V16_COMMIT = "147cd5ac9472002f5751221d390bf00abdbc0d2f"
SCORING_BLOBS = {
    "tools/generate_result_tables.ipynb": "72fb7a5c7d40bb6f1b2b839fc33d31856c756ee8",
    "src/core/metrics.py": "6039ff87c463be88c988e7ec017860b8f0687b2a",
    "src/metrics/cal_metric.py": "8993efdc2f55769e96d04f634645a00de7d5b900",
    "src/metrics/table_metric.py": "705e294919bb1ff96cf1a69655b1267958a66407",
    "src/metrics/cdm_metric.py": "c82d5a405f92cf7493e6cf9201b4ba5531759ba8",
    "src/dataset/end2end_dataset.py": "633a28a2629d7cd30d9d49c10cecc619b57519ac",
}
```

Implement `_git()` with `subprocess.run(..., check=True, text=True,
capture_output=True)` and reject a checkout unless `HEAD` and every blob match.
Return a dictionary containing commit and blob IDs for provenance.

- [ ] **Step 4: Capture the isolated Windows CDM patch**

Run:

```powershell
New-Item -ItemType Directory -Force eval/patches | Out-Null
git -C eval/.omnidocbench diff -- `
  src/metrics/cdm/modules/latex2bbox_color.py `
  src/metrics/cdm/modules/texlive_env.py `
  | Set-Content -Encoding utf8 eval/patches/omnidocbench-v16-windows-cdm.patch
```

Verify the patch touches only those two files:

```powershell
rg -n '^diff --git' eval/patches/omnidocbench-v16-windows-cdm.patch
```

Expected: exactly two matches.

- [ ] **Step 5: Write the pinned preparation script**

The script must run these operations with native-command exit checks:

```powershell
$Commit = "147cd5ac9472002f5751221d390bf00abdbc0d2f"
$Checkout = "eval/.omnidocbench"
$Patch = (Resolve-Path "eval/patches/omnidocbench-v16-windows-cdm.patch")

if (-not (Test-Path "$Checkout/.git")) {
  git clone https://github.com/opendatalab/OmniDocBench.git $Checkout
  if ($LASTEXITCODE -ne 0) { throw "OmniDocBench clone failed" }
}
git -C $Checkout fetch origin $Commit
if ($LASTEXITCODE -ne 0) { throw "OmniDocBench fetch failed" }
git -C $Checkout checkout --detach $Commit
if ($LASTEXITCODE -ne 0) { throw "OmniDocBench checkout failed" }
git -C $Checkout apply --check $Patch
if ($LASTEXITCODE -eq 0) { git -C $Checkout apply $Patch }
python eval/benchmark_contract.py --checkout $Checkout
if ($LASTEXITCODE -ne 0) { throw "v1.6 contract validation failed" }
```

Add a `main()` to `benchmark_contract.py` so the last command prints JSON and
returns exit code 2 on mismatch.

- [ ] **Step 6: Verify and commit**

Run:

```powershell
python -m pytest tests/test_benchmark_contract.py -q
python eval/benchmark_contract.py --checkout eval/.omnidocbench
git diff --check
```

Expected: tests pass and JSON names commit `147cd5ac...`.

Commit:

```powershell
git add eval/benchmark_contract.py eval/patches/omnidocbench-v16-windows-cdm.patch scripts/prepare_omnidocbench_v16.ps1 tests/test_benchmark_contract.py
git commit -m "feat(eval): pin official omnidocbench v16 scorer"
```

### Task 3: Enforce release-grade prediction and metric gates

**Files:**
- Modify: `eval/run_eval.py`
- Test: `tests/test_eval_report_path.py`

**Interfaces:**
- Consumes: `_run_stats.json`, dataset image count, and the pinned checkout.
- Produces: `_validate_release_prediction_stats(args, predictions_dir) -> None`.

- [ ] **Step 1: Write failing strict-gate tests**

Add parameterized tests for `fail=1`, `fallback=1`, `ok=1650`, and
`limit_pages=16`. Each must raise `SystemExit`. Add a passing case with:

```python
stats = {
    "count": 1651,
    "ok": 1651,
    "fail": 0,
    "fallback": 0,
    "limit_pages": None,
    "engine": "official",
    "stats": [],
}
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_eval_report_path.py -q
```

Expected: at least the failure/fallback cases are incorrectly accepted.

- [ ] **Step 3: Tighten the gate and validate the scorer before evaluation**

After the existing dataset-count checks, add:

```python
if run_stats.get("ok") != actual_count:
    raise SystemExit(f"Release evidence requires ok={actual_count}: {stats_path}")
if run_stats.get("fail") != 0:
    raise SystemExit(f"Release evidence requires fail=0: {stats_path}")
if run_stats.get("fallback") != 0:
    raise SystemExit(f"Release evidence requires fallback=0: {stats_path}")
```

At the start of `stage_eval`, load `eval/benchmark_contract.py` and call
`validate_checkout(checkout)` before rendering the config.

- [ ] **Step 4: Run focused and full tests**

Run:

```powershell
python -m pytest tests/test_eval_report_path.py -q
python -m pytest -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add eval/run_eval.py tests/test_eval_report_path.py
git commit -m "fix(eval): require complete clean benchmark runs"
```

### Task 4: Expand provenance and run summaries

**Files:**
- Modify: `eval/artifact_utils.py`
- Modify: `eval/run_eval.py`
- Test: `tests/test_eval_artifact_utils.py`

**Interfaces:**
- Provenance adds `omnidocbench`, `dataset_sha256`, `config_sha256`, and `prediction_manifest_sha256`.
- Run summary adds exact notebook metrics and CDM/TEDS quality records.

- [ ] **Step 1: Write failing provenance tests**

Assert the written provenance contains:

```python
assert provenance["omnidocbench"]["commit"] == benchmark_contract.OMNIDOCBENCH_V16_COMMIT
assert len(provenance["dataset_sha256"]) == 64
assert len(provenance["config_sha256"]) == 64
assert len(provenance["prediction_manifest_sha256"]) == 64
assert summary["notebook_metrics"]["overall"] == 95.78033333333333
```

- [ ] **Step 2: Run tests and verify RED**

Run `python -m pytest tests/test_eval_artifact_utils.py -q`.

Expected: missing-key failures.

- [ ] **Step 3: Implement deterministic hashing**

Add:

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_manifest_sha256(predictions_dir: Path) -> str:
    rows = [
        f"{path.name}\t{sha256_file(path)}"
        for path in sorted(predictions_dir.glob("*.md"), key=lambda item: item.name)
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
```

Pass the checkout contract and hashes from `stage_eval` into the artifact writer.
Store `notebook_metrics=extract_notebook_metrics(metric_result)` in summaries.

- [ ] **Step 4: Verify and commit**

Run:

```powershell
python -m pytest tests/test_eval_artifact_utils.py tests/test_eval_report_path.py -q
git diff --check
```

Commit:

```powershell
git add eval/artifact_utils.py eval/run_eval.py tests/test_eval_artifact_utils.py tests/test_eval_report_path.py
git commit -m "feat(eval): record reproducible benchmark provenance"
```

### Task 5: Regenerate corrected official and lightweight evidence

**Files:**
- Modify: `scripts/run_official_local_v16.ps1`
- Modify: `results/omnidocbench/v16/README.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Update: small JSON artifacts under `results/omnidocbench/v16/`

**Interfaces:**
- Produces: corrected official-local and lightweight v1.6 artifacts generated by the same scorer.

- [ ] **Step 1: Repair the one official failed page with the official engine**

Remove only the stale failed-page record and rerun that exact image through
`run_official_folder`; do not copy a lightweight fallback. Then merge the new
successful record into `_run_stats.json` and assert:

```powershell
$s = Get-Content -Raw predictions/paddleocr_official_local_llamacpp_gguf_v16/_run_stats.json | ConvertFrom-Json
if ($s.count -ne 1651 -or $s.ok -ne 1651 -or $s.fail -ne 0 -or $s.fallback -ne 0) {
  throw "official prediction repair did not produce a clean 1651-page run"
}
```

- [ ] **Step 2: Prepare and validate the scorer**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_omnidocbench_v16.ps1
python eval/benchmark_contract.py --checkout eval/.omnidocbench
```

Expected: pinned commit and all scoring blob IDs match.

- [ ] **Step 3: Score official-local and lightweight predictions**

Run the two full CDM evaluation commands with distinct artifact profiles and
copy destinations. Expected: 1651 pages, 2352 CDM samples, zero CDM/TEDS
timeouts, and zero CDM/TEDS errors.

- [ ] **Step 4: Verify the corrected values**

Run a short script importing `extract_notebook_metrics` for both artifacts.
Before any inference-quality change, expected tracked-artifact regressions are:

```text
official-local overall = 95.78033333333333
lightweight overall = 95.948
```

Fresh re-scoring may differ only if the repaired official page or pinned scorer
changes a real metric. Record and explain any difference before editing docs.

- [ ] **Step 5: Update English and Chinese documentation**

Publish the exact notebook values, identify the external Linux CUDA 96.33 value
as context, link each local row to its provenance, and remove the incorrect
95.7657 notebook claim.

- [ ] **Step 6: Run the evidence release gate**

Run:

```powershell
python -m pytest -q
python -m compileall -q src/paddleocr_vl_rocm eval
ruff check src tests scripts eval
ruff format --check src tests scripts eval
mypy src
git diff --check
```

Expected: all pass.

- [ ] **Step 7: Commit**

```powershell
git add README.md README.zh-CN.md eval scripts/run_official_local_v16.ps1 results/omnidocbench/v16 tests
git commit -m "data: publish corrected omnidocbench v16 evidence"
```
