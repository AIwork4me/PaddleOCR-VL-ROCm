# Accuracy Root-Cause Evidence Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make historical/fresh v1.6 artifact pairing fail closed and produce DirectML-qualified canonical evidence that identifies the earliest divergence for the exact Formula, Table, Text, and reading-order loss fixtures before any production inference change.

**Architecture:** Extend the diagnostic analyzer with an artifact-consistency gate, keep a small manifest of exact case identifiers and expected score deltas, then use the existing recorder/comparator for paired capture and boundary-oracle attribution. This plan deliberately makes no prompt, crop, normalization, model, or serializer change because the available scorer records observe only final output.

**Tech Stack:** Python 3.10+, JSON, SHA-256, pytest, OmniDocBench v1.6 at `147cd5ac9472002f5751221d390bf00abdbc0d2f`, ONNX Runtime DirectML.

## Global Constraints

- Do not claim fresh release evidence until official inference is `count=1651`, `ok=1651`, `fail=0`, `fallback=0`.
- Formula uses `display_formula.page.CDM.ALL * 100`; Table uses `table.page.TEDS.ALL * 100`.
- Text page score is `sum(Edit_num) / sum(upper_len)`; Formula CDM and Table TEDS use equal sample means within each GT page and equal page means.
- Round Text, Formula, and Table to three decimals before computing Overall.
- Every Windows lightweight trace must record `DmlExecutionProvider` first in `layout_providers_active`; reject absent provider metadata and CPU fallback.
- Keep raw predictions, images, crops, responses, and traces untracked.
- Do not implement a prompt, crop, model, normalization, or serialization change in this plan.

---

## File Structure

- Modify `scripts/analyze_omnidocbench_deltas.py`: validate per-sample reconstruction against companion metric fields and emit full-page loss summaries.
- Modify `tests/test_analyze_omnidocbench_deltas.py`: stale/mispaired CDM rejection and official v1.6 aggregation tests.
- Create `tests/fixtures/accuracy/v16-root-cause-cases.json`: exact page/GT-index manifest, historical expected deltas, and required trace boundaries.
- Create `tests/test_accuracy_case_manifest.py`: manifest schema, uniqueness, thresholds, and DirectML trace contract tests.
- Create `scripts/attribute_accuracy_deltas.py`: run deterministic crop/payload/raw/final oracle comparisons over trace pairs.
- Create `tests/test_attribute_accuracy_deltas.py`: first-divergence and oracle-contribution tests.
- Modify `docs/accuracy-root-cause-v16.md`: replace unproven boundaries only when authenticated traces support them.
- Create `docs/superpowers/plans/2026-07-12-accuracy-inference-fixes.md` only after attribution: separate TDD plan for proven production fixes.

### Task 1: Reject inconsistent Formula CDM evidence

**Files:**
- Modify: `scripts/analyze_omnidocbench_deltas.py`
- Modify: `tests/test_analyze_omnidocbench_deltas.py`

**Interfaces:**
- Produces: `validate_component_metrics(result_dir: Path, samples: list[dict[str, object]]) -> dict[str, object]`.
- The CLI exits non-zero when reconstructed Formula CDM or Table TEDS differs from the companion official metric by more than `1e-12`.

- [ ] **Step 1: Write the failing stale-CDM fixture test**

Add a test directory containing two formula samples with CDM `0.0` and a
companion metric whose `display_formula.page.CDM.ALL` is `0.75`:

```python
def test_rejects_per_sample_cdm_that_does_not_reconstruct_metric(tmp_path: Path):
    result_dir = tmp_path / "stale"
    result_dir.mkdir()
    _write_formula_rows(result_dir, [("a.png", 0, 0.0), ("b.png", 0, 0.0)])
    _write_metric(result_dir, formula_page_cdm=0.75, table_page_teds=None)

    with pytest.raises(ValueError, match="Formula CDM reconstruction mismatch"):
        validate_component_metrics(result_dir, load_component_samples(result_dir))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_analyze_omnidocbench_deltas.py::test_rejects_per_sample_cdm_that_does_not_reconstruct_metric -q`

Expected: FAIL because `validate_component_metrics` is undefined.

- [ ] **Step 3: Implement strict v1.6 reconstruction**

Group Formula CDM and Table TEDS samples by page, take the equal sample mean
within each page, then the equal mean across pages. Read only
`display_formula.page.CDM.ALL` and `table.page.TEDS.ALL` from the companion
metric. Raise with the component, reconstructed value, metric value, and
artifact paths when `abs(reconstructed - recorded) > 1e-12`.

- [ ] **Step 4: Test the authentic inconsistency and selected pair**

Run the validator on the excluded all-zero lightweight intermediate plus the
later 96.922 metric. Expected: non-zero exit with reconstruction `0.0` versus
recorded `0.9692192577813941`.

Run it on the selected later lightweight pair and historical official pair.
Expected Formula reconstructions: `0.9692192577813941` and
`0.9650220103430606`; expected Table reconstructions:
`0.9432215352965814` and `0.9423931666667129`.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
python -m pytest tests/test_analyze_omnidocbench_deltas.py -q
python -m ruff check scripts/analyze_omnidocbench_deltas.py tests/test_analyze_omnidocbench_deltas.py
python -m ruff format --check scripts/analyze_omnidocbench_deltas.py tests/test_analyze_omnidocbench_deltas.py
git diff --check
```

Expected: all pass. Commit only the analyzer and test with
`fix(eval): reject inconsistent v16 delta artifacts`.

### Task 2: Lock exact historical loss cases without raw predictions

**Files:**
- Create: `tests/fixtures/accuracy/v16-root-cause-cases.json`
- Create: `tests/test_accuracy_case_manifest.py`

**Interfaces:**
- Produces a schema-1 manifest with `component`, `page`, `gt_idx`,
  `official_score`, `lightweight_score`, `page_delta`, and
  `required_boundaries`.

- [ ] **Step 1: Create the failing manifest test**

Assert the manifest contains five distinct pages for each of `formula_cdm`,
`table_teds`, `text_edit`, and `reading_order`, and that every row requires:

```json
["layout", "crop", "payload", "raw_vlm", "final_output"]
```

Also assert `trace_contract.layout_providers_active_first` equals
`DmlExecutionProvider` and `allow_cpu_fallback` is `false`.

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_accuracy_case_manifest.py -q`

Expected: FAIL because the manifest does not exist.

- [ ] **Step 3: Add the exact case manifest**

Use the twenty cases listed in `docs/accuracy-root-cause-v16.md`. Store only
identifiers and scalar scores/deltas, never GT text or predictions. Exact
Formula page deltas are `0.9040`, `0.25309090909090914`, `0.2`, `0.1635`, and
`0.07966666666666655`. Exact Table page deltas are
`0.3409420289855073`, `0.15384615384615397`, `0.14681114064407297`,
`0.03286526420581071`, and `0.030612244897959218`. Exact Text page losses are
`0.2251210600925081`, `0.12666666666666668`, `0.07355387174274626`,
`0.06564501150780636`, and `0.06352941176470588`. Exact reading-order losses
are `0.2857142857142857`, `0.18518518518518517`, `0.16666666666666666`,
`0.14285714285714285`, and `0.13333333333333336`.

- [ ] **Step 4: Verify and commit**

Run the focused test, Ruff, and `git diff --check`. Expected: all pass. Commit
with `test(eval): lock v16 root-cause case manifest`.

### Task 3: Capture DirectML-qualified canonical traces

**Files:**
- Modify: `tests/test_accuracy_case_manifest.py`
- Modify after capture: `docs/accuracy-root-cause-v16.md`

**Interfaces:**
- Consumes the case manifest and existing `scripts/record_trace.py`.
- Produces untracked JSONL traces plus a small tracked scalar capture summary.

- [ ] **Step 1: Add trace-summary contract tests**

Test a synthetic summary and reject it unless all twenty case IDs are present,
all required boundaries have non-empty fingerprints, credentials are absent,
and every lightweight case has:

```json
{
  "layout_provider_requested": "auto",
  "layout_providers_active": ["DmlExecutionProvider", "CPUExecutionProvider"]
}
```

The test must reject `CPUExecutionProvider` in the first position.

- [ ] **Step 2: Run the contract test and verify RED**

Expected: FAIL until the summary validator is implemented in the test helper or
a focused diagnostic module.

- [ ] **Step 3: Capture each named lightweight page**

Run `scripts/record_trace.py` once per manifest page with
`--layout-provider auto --trace-jsonl <untracked-path>`. Immediately validate
the recorded provider before accepting the trace. Expected: DirectML is first;
any absent metadata or CPU-first trace aborts the batch.

- [ ] **Step 4: Capture the closest observable official boundaries**

Use only the official engine and its historical scorer input/output. If the
official adapter cannot expose a boundary, record `unobservable` rather than
copying a lightweight value. Expected: all twenty cases have official final
scorer output; deeper official boundaries may remain explicitly unobservable.

- [ ] **Step 5: Compare and publish only scalar trace metadata**

Run `scripts/compare_inference_traces.py` for each authentic pair. Keep JSONL,
crops, raw responses, and predictions untracked. Add to the report only case
ID, hashes, provider list, first observable divergence, and whether each
boundary was observable.

- [ ] **Step 6: Verify and commit**

Run the manifest/trace contract tests and `git status --short`. Expected: raw
trace paths are ignored and only the small report/summary/test changes are
staged. Commit with `docs(eval): record directml v16 trace coverage`.

### Task 4: Attribute loss with boundary oracle swaps

**Files:**
- Create: `scripts/attribute_accuracy_deltas.py`
- Create: `tests/test_attribute_accuracy_deltas.py`

**Interfaces:**
- Produces: `attribute_case(official: dict, lightweight: dict, replay: Callable[[str, object], float]) -> dict[str, object]`.
- Returns ordered contributions for `crop`, `payload`, `raw_vlm`, and
  `final_output`, preserving Formula CDM and Table TEDS scorer semantics.

- [ ] **Step 1: Write failing ordered-oracle tests**

Use a synthetic formula case where swapping crop changes score from `0.061` to
`0.965` and later swaps do nothing; expect first causal boundary `crop` and
contribution `0.904`. Use a table case where only raw output changes score from
`0.6340579710144927` to `0.975`; expect `raw_vlm` and contribution
`0.3409420289855073`. Add a test that unobservable official input yields
`status="unproven"`, never a zero contribution.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_attribute_accuracy_deltas.py -q`

Expected: import failure because the script does not exist.

- [ ] **Step 3: Implement one-boundary-at-a-time attribution**

Apply swaps in fixed order `crop`, `payload`, `raw_vlm`, `final_output`. Record
the score before and after each swap, restore the baseline before the next
single-variable test, and label absent inputs `unproven`. Never substitute an
Edit-distance proxy for CDM or TEDS.

- [ ] **Step 4: Replay all observable manifest cases**

Expected contract improvement: every one of the twenty cases is either
assigned an authenticated earliest causal boundary or explicitly `unproven`;
the report contains no inferred zero. Formula scores come from CDM F1 and Table
scores from content-aware TEDS.

- [ ] **Step 5: Verify and commit**

Run the focused tests, the complete analyzer/manifest suite, Ruff, mypy, and
`git diff --check`. Commit with `feat(eval): attribute v16 losses by trace boundary`.

### Task 5: Write the first production-fix plan only from proven attribution

**Files:**
- Modify: `docs/accuracy-root-cause-v16.md`
- Create: `docs/superpowers/plans/2026-07-12-accuracy-inference-fixes.md`

**Interfaces:**
- Consumes authenticated oracle-attribution output.
- Produces a separate TDD plan; it does not change production code.

- [ ] **Step 1: Update the evidence table**

For every proven cause, record exact page/GT index, before score, oracle score,
affected-case count, estimated v1.6 Overall contribution, first causal
boundary, and trace hashes. Keep all unproven cases labeled unproven.

- [ ] **Step 2: Apply the production-plan admission gate**

A proposed production task is admitted only if at least one exact manifest
fixture reproduces the divergence, the oracle swap improves the official
metric, neighboring gain fixtures are named, and the generic boundary is known.
Expected: zero speculative prompt/crop/normalization tasks pass by default.

- [ ] **Step 3: Write complete TDD tasks for admitted causes**

Each task must name the exact fixture, its current score/contract failure, its
expected post-fix score or boundary contract, and the neighboring fixtures
that must not regress. If no cause passes the gate, the plan must state that no
production inference fix is authorized and stop without placeholders.

- [ ] **Step 4: Verify document claims**

Re-run artifact consistency, trace-summary, and attribution tests. Search the
two documents for placeholder markers, unsupported “zero” classifications,
CPU-first provider evidence, and any Overall calculation that bypasses
notebook rounding. Expected: no findings.

- [ ] **Step 5: Commit**

Commit only the updated report and new plan with
`docs(eval): plan proven v16 accuracy fixes`. Raw evidence remains untracked.
