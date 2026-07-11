# Task 3 Report

Status: DONE_WITH_CONCERNS

## Commits

- `a3e9677` (`docs: classify rocm formula cdm hard cases`)
- `0c6181a` (`docs: record rocm hard-case verification`)

## Files Changed

- `docs/formula-cdm-rocm-hardcase-analysis-2026-07-11.md`

## Summary

Task 3 produced an evidence report from the checked-in scalar Formula CDM hard-case summary. No production code was changed because the available artifact contains only page/sample_id/CDM scalar values; it does not contain GT or prediction formula text.

## Verification Commands and Results

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

- The analysis document now records both commits: `a3e9677` and `0c6181a`.
- The analyzer command was executed against the generated summary and is recorded with an explicit warning because it produced invalid summary-of-summary counts (`count=53`, `below_threshold_count=50`, `zero_count=17`).
