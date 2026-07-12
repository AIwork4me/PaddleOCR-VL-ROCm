# Evidence Task 2 Report

## Outcome

- Commit: `976635976c0a1f215ed615d43daa163d041da73a`
- Commit subject: `test(eval): lock v16 root-cause case manifest`
- The schema-1 manifest contains 20 scalar-only cases: five distinct pages for
  each of Formula CDM, Table TEDS, Text Edit, and reading order.
- Formula identity is the canonical `gt_position` plus SHA-256 of raw GT;
  scorer-local indices are retained only under `metadata.gt_idx`.
- Every case requires `layout`, `crop`, `payload`, `raw_vlm`, and
  `final_output` boundaries.
- The trace contract requires `DmlExecutionProvider` first and sets
  `allow_cpu_fallback` to `false`.
- No GT text, predictions, responses, crops, images, or traces were added.

## Review follow-up

The review findings on commit `9766359` were addressed without changing the
authentic manifest:

- the regression lock now asserts the complete 20-case identity map, including
  exact component, page, canonical `gt_position`, and GT SHA-256 for Formula,
  Table, Text, and reading order;
- the former forbidden-key search was replaced by exact allowlists at the
  manifest, `trace_contract`, case, `source_identity`, `metadata`, and `gt_idx`
  object levels; and
- the schema test also constrains booleans, non-empty strings, nonnegative
  integer index lists, nullable reading-order scorer indices, finite scalar
  scores in `[0, 1]`, exact boundary values, and lowercase SHA-256 values.

The added tests were observed RED with two expected `NameError` failures for
the missing complete identity map and missing strict schema validator, then
GREEN after the test contract was implemented (`4 passed`).

## TDD evidence

RED command:

```powershell
python -m pytest tests/test_accuracy_case_manifest.py -q
```

Result before adding the manifest: `4 failed`. Each failure was the expected
`FileNotFoundError` for
`tests/fixtures/accuracy/v16-root-cause-cases.json`.

GREEN command:

```powershell
python -m pytest tests/test_accuracy_case_manifest.py -q
```

Result: `4 passed`.

## Authentic scalar validation

An in-memory Python check read the historical official and lightweight
Formula, Table, Text, and reading-order scorer artifacts cited by
`docs/accuracy-root-cause-v16.md`. For every manifest row it verified:

- page and canonical source position;
- SHA-256 of raw GT without printing or storing GT;
- official and lightweight scorer-index metadata;
- official and lightweight representative scalar scores; and
- the page delta reconstructed with the v1.6 component-specific formula.

Result: `validated 20 scalar cases against both authentic artifact sets`.

The review follow-up repeated this read-only validation for the complete
identity map. Reading-order GT is a page-level JSON list and was hashed using
the scorer artifact's default JSON representation; the other components hash
the raw GT string. Result:
`validated 20 identities against both authentic artifact sets`.

## Verification commands and results

```powershell
python -m pytest tests/test_analyze_omnidocbench_deltas.py tests/test_accuracy_case_manifest.py -q
```

Result: `12 passed`.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result: the complete repository suite passed (`7 skipped`, zero failures).

```powershell
python -m ruff check tests/test_accuracy_case_manifest.py
python -m ruff format --check tests/test_accuracy_case_manifest.py
git diff --check
```

Results: Ruff check passed, the test file was already formatted, and diff
whitespace validation passed.

Repository-wide Ruff was also run. It remains non-zero because it traverses
the pre-existing untracked `eval/.omnidocbench/` checkout (1,242 findings in
that external tree and other unrelated legacy files); repository-wide format
check lists 89 pre-existing files. Focused Ruff check and format check for the
only changed test file both pass. No Ruff fixes were applied outside Task 2.

The repository-wide `python -m pytest -q` check remains blocked by five
pre-existing collection errors outside Task 2: missing
`paddleocr_vl_rocm.contracts` imports and missing
`resolve_layout_providers` exports. Repository-wide Ruff also reports an
unrelated import-order issue in `tests/test_benchmark_contract.py` and six
pre-existing files needing formatting. These out-of-scope files were not
changed.

## File hashes

- `tests/fixtures/accuracy/v16-root-cause-cases.json`:
  `69d136ebab952330d986e929c98e8e16bc504f417df302f77d2c66a26078baf0`
- `tests/test_accuracy_case_manifest.py`:
  `8af7eb4170b41eae7e1948b8ed6561e2731decdc2a6d1872d455fd24f4ae5375`

## Touched files

- `tests/fixtures/accuracy/v16-root-cause-cases.json` (committed)
- `tests/test_accuracy_case_manifest.py` (committed)
- `.superpowers/sdd/evidence-task-2-report.md` (review evidence update)

The existing untracked `eval/.omnidocbench/` directory was neither touched nor
staged.
