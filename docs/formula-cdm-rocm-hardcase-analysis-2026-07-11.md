# Formula CDM ROCm Hard-Case Analysis - 2026-07-11

## Status

DONE_WITH_CONCERNS

## Scope

Local Windows + AMD + llama.cpp/GGUF only. No Linux vLLM/BF16 reference path.

## Evidence

- Source summary: `docs/formula-cdm-rocm-hardcase-summary-2026-07-11.json`
- Underlying per-sample source: not present in this repository; the checked-in summary is scalar-only.
- Total samples: 2352
- Samples with CDM < 0.8: 93
- Samples with CDM == 0: 17
- Lowest observed samples include pages `page-21967f5d-667d-488e-a5b3-76b9d6f53656.png`, `page-cdb92c2f-f43f-45ef-ace7-91d4664a7834.png`, `page-ad5a110f-a4b4-430b-b5db-ecd0ee394451.png`, and `book_en_国外数学教材-数论-Melvyn B. Nathanson—Elementary Methods in Number Theory_0451.png`.

The summary contains `page`, `sample_id`, and scalar `cdm` values. Its empty `gt` and `pred` fields are produced by the scalar-map analyzer because source text is unavailable; they are not evidence that either prediction or ground truth was empty.

## Categories

| Category | Count | Example pages | Action |
|---|---:|---|---|
| Empty prediction | N/A | Not determinable from scalar CDM data | No code fix; obtain prediction text before classifying |
| Malformed LaTeX | N/A | Not determinable from scalar CDM data | No code fix; obtain formula text before adding a test |
| Markdown wrapper mismatch | N/A | Not determinable from scalar CDM data | No normalization change without text-level evidence |
| True model-output difference | N/A | Low/zero CDM pages are candidates only | Document as unresolved until GT/pred text is available |

## Decision

No safe deterministic output pattern is proven by the available evidence. No production code was changed, and no focused normalization test was added because there is no exact failing text case to reproduce.

## Accepted Fixes

No production fix is accepted until a focused test reproduces the exact case and demonstrates that the proposed normalization preserves formula semantics.

## Concerns

- The scalar summary supports ranking and counts only; it cannot distinguish empty predictions, malformed LaTeX, wrapper mismatches, or genuine model-output differences.
- The 17 zero-CDM samples must not be labeled as empty predictions from this artifact alone.
- A follow-up requires the original per-sample records or paired GT/pred formula text for representative low and zero cases.

## Files Changed

- `docs/formula-cdm-rocm-hardcase-analysis-2026-07-11.md`

## Commit

Recorded after committing this report.

## Verification Commands

Commands and results are recorded here after execution:

- `python scripts/analyze_formula_cdm_cases.py --per-sample-cdm docs/formula-cdm-rocm-hardcase-summary-2026-07-11.json --threshold 0.8` - NOT USED; this input is the generated summary rather than the raw scalar map, so the analyzer interpreted summary fields as cases and produced invalid counts (`count=53`, `below_threshold_count=50`, `zero_count=17`).
- `python -m pytest tests/test_formula_cdm_case_analysis.py -q` - PASS; 3 tests passed.
- `python -m pytest -q` - PASS; 61 tests passed.
- PowerShell summary schema/count check - PASS; verified `count=2352`, `below_threshold_count=93`, `zero_count=17` and scalar `page`/`sample_id`/`cdm` fields.
- `git diff --check` - PASS.
- `git status --short` - only the report was an intended change; pre-existing untracked `data/`, `eval/.omnidocbench/`, and `logs/` remained untouched.

Production code changed: no.
