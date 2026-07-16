# Evidence Task 4 Report

## Outcome

- Added generic ordered single-variable oracle attribution for `crop`,
  `payload`, `raw_vlm`, and `final_output`.
- Each observable swap is scored from the lightweight baseline and the
  baseline is restored before the next boundary.
- The replay callback is the only scorer. The implementation contains no edit
  distance or other proxy, so callers retain Formula CDM F1 and content-aware
  Table TEDS semantics.
- Missing, unobservable, and fingerprint-only official observations are
  reported as `status="unproven"` without a synthesized contribution.
- No production inference code changed.

## TDD evidence

The first focused run failed during collection with the expected
`ModuleNotFoundError: No module named 'scripts.attribute_accuracy_deltas'`.
After the minimal implementation, the focused file passed 10 tests. Coverage
includes fixed ordering, restoration after every tested oracle, the specified
formula and table score deltas, negative and authenticated zero effects, each
unobservable boundary, and metadata-only observations.

## Current 20-case evidence

The committed `v16-trace-capture-summary.json` contains no authenticated
same-boundary official replay input. A regression test evaluates all 20 cases:
all 20 return `status="unproven"`, no replay callback is invoked, and no result
contains a synthetic top-level contribution. No swaps were fabricated from
fingerprints.

## Verification

- Focused attribution tests: 10 passed.
- Complete accuracy analyzer/manifest suite: 45 passed.
- Full pytest suite: 161 passed, 7 skipped.
- Ruff check and format check for changed Python files: passed.
- Mypy for the new script: passed.
- Mypy for `src` plus the new script under the active Python 3.13 runtime:
  passed, 22 source files checked. The configured Python 3.10 target cannot
  parse the installed NumPy stub's Python 3.12 `type` statement under mypy
  2.2.0, so the verification override matches the active supported runtime.
- `git diff --check`: passed.

## Scope protection

The pre-existing untracked `eval/.omnidocbench/` checkout was not modified or
staged. Raw evidence under ignored `.superpowers/sdd` paths was not committed.
