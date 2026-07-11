# Task 3 Report

Status: DONE_WITH_CONCERNS

## Commits

- `a3e9677` (`docs: classify rocm formula cdm hard cases`)
- `0c6181a` (`docs: record rocm hard-case verification`)
- `760b0a2` (`docs: fix rocm hard-case report provenance`)
- `27fef75` (`docs: clarify rocm hard-case report revisions`)

## Files Changed

- `docs/formula-cdm-rocm-hardcase-analysis-2026-07-11.md`
- `docs/formula-cdm-rocm-hardcase-summary-2026-07-11.json`

## Summary

Task 3 produced an evidence report from the checked-in scalar Formula CDM hard-case summary. No production code was changed because the available artifact contains only page/sample_id/CDM scalar values; it does not contain GT or prediction formula text.

## Verification Commands and Results

- `python scripts/analyze_formula_cdm_cases.py --per-sample-cdm C:\Users\rocm\Desktop\omnidocbench-amd-windows\eval-infra\01-omnidocbench\OmniDocBench\result\paddleocrvl_rocm_cdm_quick_match_display_formula_per_sample_CDM.json --threshold 0.8 --out docs\formula-cdm-rocm-hardcase-summary-2026-07-11.json`: PASS; regenerated the checked-in scalar summary from the real source.
- `python -m pytest tests/test_formula_cdm_case_analysis.py -q`: PASS; 3 tests passed.
- `python -m pytest -q`: PASS; 61 tests passed.
- PowerShell summary schema/count check: PASS; verified `count=2352`, `below_threshold_count=93`, `zero_count=17`, and scalar `page`/`sample_id`/`cdm` fields.
- `git diff --check`: PASS.
- `git status --short`: only pre-existing untracked `data/`, `eval/.omnidocbench/`, and `logs/` remain.

## Production Code Changed

No.

## Concerns

The scalar summary supports ranking and counts only. It cannot distinguish empty predictions, malformed LaTeX, Markdown wrapper mismatch, or true model-output differences. The 17 zero-CDM samples must not be labeled as empty predictions from this artifact alone.

## Provenance Correction

- The analysis document now uses a non-self-referential revision-history section instead of
  trying to list every final commit hash inside the file being amended.
- The invalid summary-of-summary analyzer invocation was removed from the report and replaced
  with the real source per-sample scalar map command.

## Reviewer Fixes

- Restored the required `Source per-sample file` evidence field.
- Replaced `N/A` category counts with explicit `Unclassifiable from this artifact` status because
  the available per-sample source has scalar CDM values but no paired GT/pred text.
