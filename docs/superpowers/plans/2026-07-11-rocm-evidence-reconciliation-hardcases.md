# ROCm Evidence Reconciliation And Hard-Case Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring PaddleOCR-VL-ROCm's published local OmniDocBench evidence, hard-case diagnosis, and any accepted accuracy fixes into one reproducible Windows + AMD + llama.cpp/GGUF loop.

**Architecture:** Reuse the already implemented local adapter/orchestrator and the now-validated `omnidocbench-amd-windows` scoring infrastructure. First reconcile score artifacts and result provenance, then add small analysis tooling that compares local ROCm/lightweight and official-local formula cases, and only then apply narrow output fixes proven by tests and local evidence.

**Tech Stack:** Python 3.11, pytest, Markdown/JSON evidence artifacts, local OmniDocBench v1.6 result JSON, Windows PowerShell, llama.cpp/GGUF OpenAI-compatible VLM endpoint.

## Global Constraints

- Validation is local-only: Windows + AMD + llama.cpp/GGUF + this machine's OmniDocBench/CDM environment.
- Do not set up Linux vLLM, BF16, SGLang, FastDeploy, Docker inference, or any cross-machine reference path.
- Do not rewrite ground truth or tune benchmark scores.
- Keep official PaddleOCR imports lazy so `--help` and unit tests work when PaddleOCR is not installed.
- Do not commit `data/`, `eval/.omnidocbench/`, `logs/`, generated predictions, or large local eval outputs.
- Commit only lightweight evidence summaries and small copied score artifacts that are already suitable for source control.
- Any production normalization must have a reproducible failing case, a focused unit test, and evidence that it preserves formula meaning.

---

## File Structure

- Modify `results/omnidocbench/v16/README.md`: provenance index for tracked local score artifacts.
- Copy or update small tracked artifacts under `results/omnidocbench/v16/`: latest Windows-native `paddleocrvl_rocm_cdm` metric/run summaries if they are small enough and useful for review.
- Create `scripts/analyze_formula_cdm_cases.py`: local-only diagnostic script that extracts and ranks Formula CDM cases from metric/result JSON files.
- Create `docs/formula-cdm-rocm-hardcase-analysis-2026-07-11.md`: evidence report with root-cause categories and recommended fixes.
- Modify `README.md`, `README.zh-CN.md`, and `eval/README.md`: align published scores and commands with reconciled local evidence.
- Modify `tests/test_vlm_payload.py`, `tests/test_eval_adapter.py`, or a new targeted test only when a code behavior change is accepted.
- Modify production code only after a test proves the specific hard-case failure.

---

### Task 1: Reconcile Latest Windows-Native ROCm CDM Evidence

**Files:**
- Create or modify: `results/omnidocbench/v16/README.md`
- Copy if accepted: `results/omnidocbench/v16/paddleocrvl_rocm_cdm_quick_match_metric_result_windows_native_2026-07-11.json`
- Copy if accepted: `results/omnidocbench/v16/paddleocrvl_rocm_cdm_quick_match_run_summary_windows_native_2026-07-11.json`
- Modify: `README.md`
- Modify: `README.zh-CN.md`

**Interfaces:**
- Consumes: `C:\Users\rocm\Desktop\omnidocbench-amd-windows\eval-infra\01-omnidocbench\OmniDocBench\result\paddleocrvl_rocm_cdm_quick_match_metric_result.json`
- Consumes: `C:\Users\rocm\Desktop\omnidocbench-amd-windows\eval-infra\01-omnidocbench\OmniDocBench\result\paddleocrvl_rocm_cdm_quick_match_run_summary.json`
- Produces: a tracked provenance record that explains which local score row is current and which artifacts are historical.

- [ ] **Step 1: Inspect artifact sizes and copied values**

Run:

```powershell
Get-Item C:\Users\rocm\Desktop\omnidocbench-amd-windows\eval-infra\01-omnidocbench\OmniDocBench\result\paddleocrvl_rocm_cdm_quick_match_metric_result.json
Get-Item C:\Users\rocm\Desktop\omnidocbench-amd-windows\eval-infra\01-omnidocbench\OmniDocBench\result\paddleocrvl_rocm_cdm_quick_match_run_summary.json
```

Expected: both files are small enough for review (`metric_result` about 16 KB, `run_summary` about 10 KB).

- [ ] **Step 2: Copy small evidence artifacts with provenance-preserving names**

Run:

```powershell
Copy-Item `
  C:\Users\rocm\Desktop\omnidocbench-amd-windows\eval-infra\01-omnidocbench\OmniDocBench\result\paddleocrvl_rocm_cdm_quick_match_metric_result.json `
  results\omnidocbench\v16\paddleocrvl_rocm_cdm_quick_match_metric_result_windows_native_2026-07-11.json

Copy-Item `
  C:\Users\rocm\Desktop\omnidocbench-amd-windows\eval-infra\01-omnidocbench\OmniDocBench\result\paddleocrvl_rocm_cdm_quick_match_run_summary.json `
  results\omnidocbench\v16\paddleocrvl_rocm_cdm_quick_match_run_summary_windows_native_2026-07-11.json
```

- [ ] **Step 3: Write provenance README**

Create or update `results/omnidocbench/v16/README.md` with:

```markdown
# OmniDocBench v1.6 Local Evidence

All artifacts in this directory are local Windows + AMD + llama.cpp/GGUF
measurements. They are not Linux vLLM/BF16 reference-path measurements.

## Current ROCm Lightweight/Local Evidence

| Artifact | Source | Notes |
|---|---|---|
| `paddleocrvl_rocm_cdm_quick_match_metric_result_windows_native_2026-07-11.json` | `omnidocbench-amd-windows` Windows-native CDM run | Current local ROCm CDM evidence for `predictions/paddleocrvl_rocm_cdm` |
| `paddleocrvl_rocm_cdm_quick_match_run_summary_windows_native_2026-07-11.json` | same run | Records 1651 pages, 2352 CDM samples, 0 CDM errors/exceptions |

## Historical Artifacts

Existing `paddleocrvl_rocm_*` and `paddleocr_official_local_llamacpp_gguf_*`
artifacts are retained for comparison. Do not mix score rows unless prediction
directory, adapter version, config, and CDM environment are explicitly named.
```

- [ ] **Step 4: Update README score row to use the reconciled ROCm CDM evidence**

In `README.md`, update the local evaluation table so the lightweight/local row names the exact artifact and uses the latest local ROCm CDM value:

```markdown
| Lightweight local engine | 0.03402 | 0.12824 | 93.1345 | 96.7129 | Latest Windows-native CDM artifact for `predictions/paddleocrvl_rocm_cdm` |
```

Add one sentence under the table:

```markdown
Older tracked lightweight artifacts are retained under `results/omnidocbench/v16/`
for comparison; the dated Windows-native artifact is the current local ROCm CDM
evidence.
```

- [ ] **Step 5: Mirror the same wording in `README.zh-CN.md`**

Use the same numbers and explicitly label the row as local Windows-native CDM evidence. Keep the public PaddleOCR-VL target row as context only.

- [ ] **Step 6: Verify docs and commit**

Run:

```powershell
python -m pytest tests/test_eval_artifact_utils.py tests/test_eval_adapter.py -q
rg -n "96\\.7129|windows_native_2026-07-11|Linux vLLM/BF16" README.md README.zh-CN.md results\omnidocbench\v16\README.md
git diff --check
```

Expected: tests pass, grep finds the new evidence references, and diff check is clean.

Commit:

```powershell
git add README.md README.zh-CN.md results\omnidocbench\v16
git commit -m "docs: reconcile local rocm cdm evidence"
```

---

### Task 2: Add Formula CDM Hard-Case Analyzer

**Files:**
- Create: `scripts/analyze_formula_cdm_cases.py`
- Create: `tests/test_formula_cdm_case_analysis.py`

**Interfaces:**
- Produces: `load_formula_scores(path: Path) -> list[dict[str, object]]`
- Produces: `summarize_cases(cases: list[dict[str, object]], threshold: float) -> dict[str, object]`
- CLI output: JSON with `count`, `below_threshold_count`, `zero_count`, and `lowest_cases`.

- [ ] **Step 1: Write failing tests for per-sample CDM extraction**

Create `tests/test_formula_cdm_case_analysis.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_formula_cdm_cases import load_formula_scores, summarize_cases


def test_load_formula_scores_accepts_per_sample_mapping(tmp_path: Path):
    sample_path = tmp_path / "per_sample.json"
    sample_path.write_text(
        json.dumps(
            {
                "page_a.png": [
                    {"sample_id": "a-1", "CDM": 1.0, "gt": "x", "pred": "x"},
                    {"sample_id": "a-2", "CDM": 0.25, "gt": "\\\\frac{1}{2}", "pred": ""},
                ],
                "page_b.png": [
                    {"sample_id": "b-1", "CDM": 0.0, "gt": "y", "pred": "bad"}
                ],
            }
        ),
        encoding="utf-8",
    )

    cases = load_formula_scores(sample_path)

    assert [case["page"] for case in cases] == ["page_a.png", "page_a.png", "page_b.png"]
    assert [case["cdm"] for case in cases] == [1.0, 0.25, 0.0]
    assert cases[1]["pred"] == ""


def test_summarize_cases_ranks_lowest_cases():
    cases = [
        {"page": "a", "sample_id": "1", "cdm": 0.5},
        {"page": "b", "sample_id": "2", "cdm": 0.0},
        {"page": "c", "sample_id": "3", "cdm": 0.9},
    ]

    summary = summarize_cases(cases, threshold=0.8)

    assert summary["count"] == 3
    assert summary["below_threshold_count"] == 2
    assert summary["zero_count"] == 1
    assert [case["page"] for case in summary["lowest_cases"]] == ["b", "a", "c"]
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/test_formula_cdm_case_analysis.py -q
```

Expected: FAIL because `scripts/analyze_formula_cdm_cases.py` does not exist.

- [ ] **Step 3: Implement analyzer**

Create `scripts/analyze_formula_cdm_cases.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _coerce_case(page: str, item: dict[str, Any]) -> dict[str, object] | None:
    raw_cdm = item.get("CDM", item.get("cdm"))
    if raw_cdm is None:
        return None
    cdm = float(raw_cdm)
    return {
        "page": page,
        "sample_id": str(item.get("sample_id", item.get("id", ""))),
        "cdm": cdm,
        "gt": item.get("gt", item.get("ground_truth", "")),
        "pred": item.get("pred", item.get("prediction", "")),
    }


def load_formula_scores(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: list[dict[str, object]] = []
    if isinstance(data, dict):
        for page, value in data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        case = _coerce_case(str(page), item)
                        if case is not None:
                            cases.append(case)
            elif isinstance(value, dict):
                case = _coerce_case(str(page), value)
                if case is not None:
                    cases.append(case)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                page = str(item.get("page", item.get("image", "")))
                case = _coerce_case(page, item)
                if case is not None:
                    cases.append(case)
    return cases


def summarize_cases(cases: list[dict[str, object]], threshold: float) -> dict[str, object]:
    ranked = sorted(cases, key=lambda case: float(case["cdm"]))
    return {
        "count": len(cases),
        "below_threshold_count": sum(1 for case in cases if float(case["cdm"]) < threshold),
        "zero_count": sum(1 for case in cases if float(case["cdm"]) == 0.0),
        "lowest_cases": ranked[:50],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Formula CDM per-sample cases.")
    parser.add_argument("--per-sample-cdm", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cases = load_formula_scores(args.per_sample_cdm)
    summary = summarize_cases(cases, threshold=args.threshold)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_formula_cdm_case_analysis.py -q
python scripts/analyze_formula_cdm_cases.py --help
```

Expected: tests pass and help exits 0.

- [ ] **Step 5: Run analyzer on available local per-sample files**

If the following file exists, run:

```powershell
python scripts/analyze_formula_cdm_cases.py `
  --per-sample-cdm C:\Users\rocm\Desktop\omnidocbench-amd-windows\eval-infra\01-omnidocbench\OmniDocBench\result\paddleocrvl_rocm_cdm_quick_match_display_formula_per_sample_CDM.json `
  --threshold 0.8 `
  --out docs\formula-cdm-rocm-hardcase-summary-2026-07-11.json
```

Expected: JSON summary is written under `docs/`.

- [ ] **Step 6: Commit analyzer**

```powershell
git add scripts/analyze_formula_cdm_cases.py tests/test_formula_cdm_case_analysis.py docs\formula-cdm-rocm-hardcase-summary-2026-07-11.json
git commit -m "feat: analyze formula cdm hard cases"
```

If the local per-sample file is unavailable, omit the generated summary from `git add` and state that live analysis was skipped.

---

### Task 3: Decide And Implement Only Proven Output Fixes

**Files:**
- Modify only after evidence: `src/paddleocr_vl_rocm/markdown.py`, `src/paddleocr_vl_rocm/content.py`, `src/paddleocr_vl_rocm/serialize.py`, or `eval/PaddleOCRVLROCm_img2md.py`
- Test: focused test file matching the touched module.
- Create: `docs/formula-cdm-rocm-hardcase-analysis-2026-07-11.md`

**Interfaces:**
- Consumes: `docs/formula-cdm-rocm-hardcase-summary-2026-07-11.json`
- Produces: one evidence report and zero or more narrow code fixes.

- [ ] **Step 1: Classify hard cases before editing production code**

Inspect the lowest cases from the analyzer and write `docs/formula-cdm-rocm-hardcase-analysis-2026-07-11.md` with this structure:

```markdown
# Formula CDM ROCm Hard-Case Analysis - 2026-07-11

## Scope

Local Windows + AMD + llama.cpp/GGUF only. No Linux vLLM/BF16 reference path.

## Evidence

- Source per-sample file: `<absolute or repo-relative path>`
- Total samples: `<number>`
- Samples with CDM < 0.8: `<number>`
- Samples with CDM == 0: `<number>`

## Categories

| Category | Count | Example pages | Action |
|---|---:|---|---|
| Empty prediction | 0 |  | No code fix |
| Malformed LaTeX | 0 |  | Add test only if pattern is deterministic |
| Markdown wrapper mismatch | 0 |  | Normalize only if semantics are preserved |
| True model-output difference | 0 |  | Document, do not mask |

## Accepted Fixes

No production fix is accepted until a focused test reproduces the exact case.
```

- [ ] **Step 2: If no safe deterministic pattern is found, commit report only**

Run:

```powershell
git add docs\formula-cdm-rocm-hardcase-analysis-2026-07-11.md
git commit -m "docs: classify rocm formula cdm hard cases"
```

- [ ] **Step 3: If a safe deterministic normalization is found, write the failing test first**

Example only for an HTML wrapper issue in `src/paddleocr_vl_rocm/markdown.py`:

```python
def test_formula_markdown_normalization_preserves_latex():
    assert normalize_formula_markdown("<span>$x^2$</span>") == "$x^2$"
```

Run:

```powershell
python -m pytest tests/test_markdown.py::test_formula_markdown_normalization_preserves_latex -q
```

Expected: FAIL for the exact reason found in the hard-case evidence.

- [ ] **Step 4: Implement the minimal production fix**

Only change the smallest module needed for the proven pattern. Do not change prompts, crop geometry, or decoding defaults in this task.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
python -m pytest tests/test_markdown.py tests/test_formula_cdm_case_analysis.py -q
python -m pytest -q
git diff --check
```

Commit:

```powershell
git add src tests docs\formula-cdm-rocm-hardcase-analysis-2026-07-11.md
git commit -m "fix: normalize proven formula cdm hard case"
```

If no production fix was made, do not create this commit.

---

### Task 4: Final Verification, PR-Ready Push, And Main Push Decision

**Files:**
- Inspect: all modified files.
- Do not commit: `data/`, `eval/.omnidocbench/`, `logs/`, generated predictions.

**Interfaces:**
- Consumes: Tasks 1-3 commits.
- Produces: verified branch pushed to `https://github.com/AIwork4me/PaddleOCR-VL-ROCm`.

- [ ] **Step 1: Run fast verification**

Run:

```powershell
python -m compileall -q src/paddleocr_vl_rocm eval scripts
ruff check src tests scripts eval
ruff format --check src tests scripts eval
mypy src
python -m pytest -q
python eval/PaddleOCRVLROCm_img2md.py --help
python eval/run_eval.py --help
```

Expected: all commands exit 0.

- [ ] **Step 2: Run local server smoke only if endpoint is reachable**

Run:

```powershell
paddleocr-vl-rocm-check-server --server-url http://127.0.0.1:8111/v1
```

If it exits 0, run a bounded inference:

```powershell
python eval/run_eval.py --stage infer --version v16 `
  --engine lightweight `
  --vlm-backend llama-cpp-server `
  --server-url http://127.0.0.1:8111/v1 `
  --api-model-name PaddleOCR-VL-1.6-GGUF.gguf `
  --limit-pages 3 `
  --predictions-dir predictions/paddleocrvl_rocm_smoke
```

If the server is unreachable, record that live inference was skipped and do not claim smoke inference passed.

- [ ] **Step 3: Check generated/untracked files**

Run:

```powershell
git status --short --branch
```

Expected: only intended tracked changes are committed. Untracked `data/`, `eval/.omnidocbench/`, and `logs/` may remain untracked.

- [ ] **Step 4: Push branch**

Run:

```powershell
git push origin codex/local-reference-quality
```

Expected: push succeeds.

- [ ] **Step 5: Decide merge route**

If user asks to merge, prefer a GitHub PR from `codex/local-reference-quality` to `main` unless they explicitly request direct push to `main`. Direct push command, only after explicit approval:

```powershell
git push https://github.com/AIwork4me/PaddleOCR-VL-ROCm HEAD:main
```

