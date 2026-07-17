# Release Gate Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce fresh, authenticated OmniDocBench v1.6 accuracy and performance evidence under the approved 1,650-success/one-known-failure contract, diagnose any remaining accuracy gap before changing inference behavior, and release only after G0-G5 pass.

**Architecture:** Converge every active contract before starting expensive inference, then run official and lightweight evidence into isolated directories tied to immutable input hashes. Failed G3 enters a diagnostic-only same-boundary attribution loop; passed G3 admits separately reviewed performance work. Every stage is fail-closed and preserves historical artifacts.

**Tech Stack:** Python 3.10+, PowerShell 7+, pytest, Ruff, mypy, ONNX Runtime DirectML 1.24.4, llama.cpp b9884, OmniDocBench v1.6 commit `147cd5ac9472002f5751221d390bf00abdbc0d2f`.

## Global Constraints

- Official-local G0 accepts exactly `count=1651`, `ok=1650`, `fail=1`, `fallback=0`, `limit_pages=null`.
- The only accepted failure is `newspaper_The Times UK_0801@magazinesclubnew_page_031.png` with `peg-native` evidence and issue URL `https://github.com/PaddlePaddle/PaddleOCR/issues/18248`.
- Scoring retains all 1,651 GT pages; the failed page has no prediction file and scores as empty.
- Formula uses `display_formula.page.CDM.ALL`; Table uses `table.page.TEDS.ALL`; CDM/TEDS errors, timeouts, missing pages, or denominator mismatches fail the release.
- Text Edit distance, Formula percent, and Table percent are rounded to three decimals before Overall, calculated as `((1 - TextEdit) * 100 + FormulaPercent + TablePercent) / 3`. Reading order does not enter Overall.
- Every Windows lightweight run uses `layout_provider_requested=auto`, `DmlExecutionProvider` first, and `layout_fallback_disabled=true`; CPU-first or fallback fails closed.
- New evidence uses isolated paths and never overwrites historical predictions or tracked results.
- Raw images, crops, payloads, responses, predictions, model files, and trace JSONL remain ignored and untracked.
- Do not start performance behavior changes until fresh lightweight Overall is at least 96.13.
- Do not bump a version, push, open a PR, tag, or release until G0-G5 all pass.

---

## File Structure

- Modify `docs/superpowers/plans/2026-07-12-accuracy-inference-fixes.md`: replace its active obsolete success contract with the approved exception.
- Modify `docs/superpowers/plans/2026-07-12-accuracy-root-cause-fixes.md`: mark its original G0 constraint superseded.
- Modify `docs/superpowers/plans/2026-07-12-v16-evidence-and-scoring.md`: preserve historical steps but add a prominent supersession note.
- Modify `docs/superpowers/specs/2026-07-12-top-tier-quality-design.md`: annotate the two obsolete 1,651-success statements.
- Modify `tests/test_documentation_contract.py`: enforce the approved wording in active documents and reject active obsolete gates.
- Create `eval/release_evidence.py`: immutable evidence-input manifest and final G0/G3 decision validation.
- Create `tests/test_release_evidence.py`: input hashing, isolated-path, notebook-rounding, and gate tests.
- Modify `scripts/run_official_local_v16.ps1`: accept explicit evidence output paths and refuse historical-path reuse.
- Modify `tests/test_run_official_local_v16_script.py`: PowerShell path and isolation contract tests.
- Create `scripts/run_release_evidence_v16.ps1`: one resumable orchestrator for preflight, official inference/scoring, lightweight inference/scoring, and final decision.
- Create `tests/test_run_release_evidence_v16_script.py`: stage ordering and fail-closed contract tests.
- Modify `docs/releases/0.1.0-readiness.md`: record only authenticated stage results.
- Modify `results/omnidocbench/v16/README.md`: list new tracked summaries only after validation.
- Conditional on failed G3, create `eval/official_trace_observer.py` and `tests/test_official_trace_observer.py` for diagnostic-only official observation.
- Conditional on failed G3, modify `eval/PaddleOCRVLROCm_img2md.py` and `tests/test_eval_adapter.py` to accept an optional observer without changing normal output.
- Conditional on passed attribution, create `docs/superpowers/plans/2026-07-13-proven-accuracy-fixes.md`.

### Task 1: Converge the active G0 contract

**Files:**
- Modify: `docs/superpowers/plans/2026-07-12-accuracy-inference-fixes.md`
- Modify: `docs/superpowers/plans/2026-07-12-accuracy-root-cause-fixes.md`
- Modify: `docs/superpowers/plans/2026-07-12-v16-evidence-and-scoring.md`
- Modify: `docs/superpowers/specs/2026-07-12-top-tier-quality-design.md`
- Modify: `tests/test_documentation_contract.py`

**Interfaces:**
- Consumes: `eval.release_contract.KNOWN_V16_OFFICIAL_FAILURE`.
- Produces: one active textual G0 contract and CI rejection of obsolete active wording.

- [ ] **Step 1: Add a failing documentation-contract test**

Append a test that separates active execution documents from annotated historical records:

```python
def test_active_release_documents_use_approved_v16_exception() -> None:
    active = [
        ROOT / "docs/superpowers/plans/2026-07-12-accuracy-inference-fixes.md",
        ROOT / "docs/releases/0.1.0-readiness.md",
        ROOT / "eval/README.md",
    ]
    for path in active:
        text = path.read_text(encoding="utf-8")
        assert "ok=1651`, `fail=0" not in text
        assert "1,650" in text
        assert "1,651" in text
        assert "peg-native" in text
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_documentation_contract.py::test_active_release_documents_use_approved_v16_exception -q
```

Expected: FAIL because the active accuracy admission plan still requires `ok=1651, fail=0`.

- [ ] **Step 3: Update active documents and annotate historical requirements**

Use this exact supersession block in historical plans/specifications:

```markdown
> **Superseded G0 requirement (2026-07-13):** The project owner approved the
> immutable issue #18248 exception defined in
> `docs/superpowers/specs/2026-07-13-release-gate-recovery-design.md`.
> Release evidence now requires 1,650 successful predictions plus the sole
> approved `peg-native` failure while scoring all 1,651 GT pages. The original
> 1,651-success text below is retained only as historical plan context.
```

In the active accuracy admission plan, replace the first global constraint with:

```markdown
- Fresh official-local release evidence requires `count=1651`, `ok=1650`,
  `fail=1`, `fallback=0`, and `limit_pages=null`, with only the approved issue
  #18248 `peg-native` failure and no failed-page prediction; scoring retains all
  1,651 GT pages.
```

- [ ] **Step 4: Run contract and full verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_documentation_contract.py tests\test_release_contract.py tests\test_run_official_local_v16_script.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check tests\test_documentation_contract.py
.\.venv\Scripts\python.exe -m ruff format --check tests\test_documentation_contract.py
git diff --check
```

Expected: all pass; `git status --short` shows only the intended documents/test and `?? eval/.omnidocbench/`.

- [ ] **Step 5: Commit and independently review**

```powershell
git add docs/superpowers/plans/2026-07-12-accuracy-inference-fixes.md docs/superpowers/plans/2026-07-12-accuracy-root-cause-fixes.md docs/superpowers/plans/2026-07-12-v16-evidence-and-scoring.md docs/superpowers/specs/2026-07-12-top-tier-quality-design.md tests/test_documentation_contract.py
git commit -m "docs(eval): converge approved v16 release contract"
```

### Task 2: Build an immutable release-evidence preflight

**Files:**
- Create: `eval/release_evidence.py`
- Create: `tests/test_release_evidence.py`

**Interfaces:**
- Produces: `build_input_manifest(paths: Mapping[str, Path], *, git_commit: str) -> dict[str, object]`.
- Produces: `validate_isolated_output_paths(paths: Iterable[Path], protected: Iterable[Path]) -> None`.
- Produces: `decide_release_gates(official_stats: dict, lightweight_metric: dict) -> dict[str, object]`.
- Produces CLI: `python eval/release_evidence.py manifest|decide ...`.

- [ ] **Step 1: Write failing manifest, isolation, and decision tests**

```python
def test_manifest_hashes_every_immutable_input(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    manifest = build_input_manifest({"main_model": model}, git_commit="ac77c5b")
    assert manifest["git_commit"] == "ac77c5b"
    assert manifest["inputs"]["main_model"]["sha256"] == hashlib.sha256(b"model").hexdigest()


def test_isolation_rejects_historical_result_path() -> None:
    with pytest.raises(ValueError, match="protected historical path"):
        validate_isolated_output_paths(
            [Path("results/omnidocbench/v16/paddleocrvl_rocm_quick_match_metric_result.json")],
            [Path("results/omnidocbench/v16")],
        )


def test_gate_decision_requires_notebook_rounded_overall() -> None:
    decision = decide_release_gates(
        official_stats=accepted_known_failure_stats(),
        lightweight_metric=metric(text=0.0344, formula=0.969224, table=0.943224),
    )
    assert decision["components"] == {
        "text_edit_dist": 0.034,
        "formula_cdm_percent": 96.922,
        "table_teds_percent": 94.322,
    }
    assert decision["overall"] == 95.948
    assert decision["g3"] is False
```

- [ ] **Step 2: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_release_evidence.py -q`

Expected: collection fails because `eval.release_evidence` does not exist.

- [ ] **Step 3: Implement the minimal preflight module**

The implementation must:

```python
def notebook_overall(text: float, formula: float, table: float) -> tuple[dict[str, float], float]:
    components = {
        "text_edit_dist": round(text, 3),
        "formula_cdm_percent": round(formula * 100, 3),
        "table_teds_percent": round(table * 100, 3),
    }
    overall = (
        (1.0 - components["text_edit_dist"]) * 100.0
        + components["formula_cdm_percent"]
        + components["table_teds_percent"]
    ) / 3.0
    return components, overall
```

Use `eval.release_contract.validate_release_run_stats` and
`validate_approved_failure_predictions` for G0; do not duplicate the exception.
Reject missing input files, non-file inputs, output paths inside protected
historical roots, and non-64-character hashes. The orchestrator separately
requires a clean tracked worktree before it records the producing commit.

- [ ] **Step 4: Verify module and CLI**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_release_evidence.py -q
.\.venv\Scripts\python.exe -m ruff check eval\release_evidence.py tests\test_release_evidence.py
.\.venv\Scripts\python.exe -m ruff format --check eval\release_evidence.py tests\test_release_evidence.py
.\.venv\Scripts\python.exe -m mypy eval\release_evidence.py
git diff --check
```

- [ ] **Step 5: Commit and review**

```powershell
git add eval/release_evidence.py tests/test_release_evidence.py
git commit -m "feat(eval): add immutable release evidence preflight"
```

### Task 3: Add an isolated, resumable v1.6 evidence runner

**Files:**
- Modify: `scripts/run_official_local_v16.ps1`
- Modify: `tests/test_run_official_local_v16_script.py`
- Create: `scripts/run_release_evidence_v16.ps1`
- Create: `tests/test_run_release_evidence_v16_script.py`

**Interfaces:**
- Consumes: `eval/release_evidence.py` and `eval/release_contract.py` CLIs.
- Produces: `-EvidenceRoot <path>` containing separate `official`, `lightweight`, `results`, `logs`, and `manifest.json` paths.
- Produces stages: `Preflight`, `Official`, `Lightweight`, `Decide`, and `All`.

- [ ] **Step 1: Write failing PowerShell contract tests**

Assert the runner:

```python
def test_release_runner_isolates_and_orders_stages() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'ValidateSet("Preflight", "Official", "Lightweight", "Decide", "All")' in text
    assert "eval/release_evidence.py manifest" in text
    assert text.index('"Preflight"') < text.index('"Official"')
    assert "eval/release_contract.py" in text
    assert "layout_provider_requested" in text
    assert "DmlExecutionProvider" in text
    assert "--copy-report" in text
    assert "--run-summary" in text
    assert "--provenance" in text
```

Add an execution test with stub `python` that verifies paths containing spaces
remain single arguments and a failed preflight prevents the official command.

- [ ] **Step 2: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_run_release_evidence_v16_script.py -q`

Expected: FAIL because the release runner does not exist.

- [ ] **Step 3: Implement explicit path parameters**

Use these PowerShell parameters:

```powershell
param(
  [ValidateSet("Preflight", "Official", "Lightweight", "Decide", "All")]
  [string]$Stage = "Preflight",
  [Parameter(Mandatory = $true)] [string]$EvidenceRoot,
  [string]$ServerUrl = "http://127.0.0.1:8111/v1",
  [string]$ApiModelName = "PaddleOCR-VL-1.6-GGUF.gguf",
  [string]$DatasetDir = "data/omnidocbench/v16",
  [string]$LayoutModel = "models/PP-DocLayoutV3-onnx"
)
```

Resolve `$EvidenceRoot`, reject it when it is inside tracked
`results/omnidocbench/v16` or either historical prediction directory, and
create only stage-specific children. Write each native command and exit code to
`$EvidenceRoot/logs/commands.jsonl`. Resume a stage only when its input hashes
and producing commit match `manifest.json`; otherwise abort.

- [ ] **Step 4: Verify scripts without running the benchmark**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_run_official_local_v16_script.py tests\test_run_release_evidence_v16_script.py -q
.\.venv\Scripts\python.exe -m ruff check tests\test_run_release_evidence_v16_script.py
git diff --check
```

- [ ] **Step 5: Commit and review**

```powershell
git add scripts/run_official_local_v16.ps1 scripts/run_release_evidence_v16.ps1 tests/test_run_official_local_v16_script.py tests/test_run_release_evidence_v16_script.py
git commit -m "feat(eval): add isolated v16 release evidence runner"
```

### Task 4: Generate and validate fresh official G0 evidence

**Files:**
- Generate untracked: `<EvidenceRoot>/official/**`
- Generate untracked: `<EvidenceRoot>/results/official/**`
- Modify only after validation: `docs/releases/0.1.0-readiness.md`
- Modify only after validation: `results/omnidocbench/v16/README.md`

**Interfaces:**
- Consumes the Task 3 runner and the local b9884 server on port 8111.
- Produces authenticated official run stats, non-CDM/CDM metric files, summary, provenance, and alignment diagnostics.

- [ ] **Step 1: Freeze the evaluated commit and verify clean tracked state**

```powershell
$Worktree = '<repo>\.worktrees\top-tier-quality'
$EvidenceRoot = '<evidence-root>\v16-2026-07-13'
Set-Location $Worktree
git status --porcelain=v1
git rev-parse HEAD
.\.venv\Scripts\python.exe scripts\check_server.py --server-url http://127.0.0.1:8111/v1
```

Expected: the only status entry is `?? eval/.omnidocbench/`; server check passes.
Record the exact post-Task-3 commit in `manifest.json`, not the example commit
from this plan.

- [ ] **Step 2: Run preflight**

```powershell
.\scripts\run_release_evidence_v16.ps1 -Stage Preflight -EvidenceRoot $EvidenceRoot
```

Expected: hashes exist for code, GGUF, mmproj, layout ONNX/config, dataset,
scorer config, and pinned scorer blobs; DirectML is first and fallback disabled.

- [ ] **Step 3: Run full official inference and scoring**

```powershell
.\scripts\run_release_evidence_v16.ps1 -Stage Official -EvidenceRoot $EvidenceRoot
```

Expected final inference contract:

```text
count=1651 ok=1650 fail=1 fallback=0 limit_pages=null
```

The sole failure is the approved filename with `peg-native`, its `.md` file is
absent, and non-CDM plus CDM scoring complete with no Formula/Table timeout or
exception.

- [ ] **Step 4: Validate artifacts before updating tracked documentation**

```powershell
.\.venv\Scripts\python.exe eval\release_contract.py --stats "$EvidenceRoot\official\_run_stats.json" --version v16 --engine official
.\.venv\Scripts\python.exe eval\release_evidence.py decide --evidence-root $EvidenceRoot --official-only
```

Expected: G0 PASS. If either command fails, stop this plan at Task 4, preserve
the evidence directory, and record the exact failure without updating scores.

- [ ] **Step 5: Update readiness with hashes, not score claims**

Record the producing commit, manifest SHA-256, stats SHA-256, metric paths,
`approved_known_failures`, and G0 PASS. Do not mark G3 or G4 passed.

- [ ] **Step 6: Verify, commit small evidence metadata, and review**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_release_contract.py tests\test_release_evidence.py tests\test_eval_artifact_utils.py -q
git diff --check
git add docs/releases/0.1.0-readiness.md results/omnidocbench/v16/README.md
git commit -m "docs(eval): record fresh v16 official evidence"
```

Do not stage predictions, model files, raw scorer working files, or
`eval/.omnidocbench/`.

### Task 5: Generate fresh lightweight evidence and decide G3

**Files:**
- Generate untracked: `<EvidenceRoot>/lightweight/**`
- Generate untracked: `<EvidenceRoot>/results/lightweight/**`
- Modify: `docs/releases/0.1.0-readiness.md`
- Modify: `results/omnidocbench/v16/README.md`

**Interfaces:**
- Consumes the exact Task 4 manifest and server configuration.
- Produces a G3 decision with official notebook component fields, Overall, provider metadata, and artifact hashes.

- [ ] **Step 1: Run the lightweight stage**

```powershell
.\scripts\run_release_evidence_v16.ps1 -Stage Lightweight -EvidenceRoot $EvidenceRoot
```

Expected: 1,651 lightweight pages, `fallback=0`, DirectML first, fallback
disabled, full Formula 2352/313 and Table 665/458 coverage, and zero CDM/TEDS
error metadata.

- [ ] **Step 2: Run the gate decision**

```powershell
.\scripts\run_release_evidence_v16.ps1 -Stage Decide -EvidenceRoot $EvidenceRoot
```

Expected output includes the three rounded components, exact Overall, G0, G3,
and artifact hashes. Recompute independently with
`extract_notebook_metrics`; the two results must be identical.

- [ ] **Step 3: Apply the conditional exit**

- If Overall is at least 96.13 and no component/contract regression exists,
  mark G3 PASS and proceed to Task 8.
- If Overall is below 96.13, mark G3 BLOCKED with the exact measured value and
  proceed to Task 6.
- If evidence integrity fails, return to the failing Task 4/5 stage; do not
  diagnose accuracy from invalid artifacts.

- [ ] **Step 4: Verify and commit only authenticated metadata**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_release_evidence.py tests\test_eval_artifact_utils.py tests\test_analyze_omnidocbench_deltas.py -q
git diff --check
git add docs/releases/0.1.0-readiness.md results/omnidocbench/v16/README.md
git commit -m "docs(eval): record fresh v16 lightweight evidence"
```

### Task 6: Capture official same-boundary diagnostics when G3 fails

**Files:**
- Create: `eval/official_trace_observer.py`
- Create: `tests/test_official_trace_observer.py`
- Modify: `eval/PaddleOCRVLROCm_img2md.py`
- Modify: `tests/test_eval_adapter.py`
- Modify: `tests/fixtures/accuracy/v16-trace-capture-summary.json`
- Modify: `docs/accuracy-root-cause-v16.md`

**Interfaces:**
- Produces: `OfficialTraceObserver.observe(boundary: str, value: object) -> None`.
- Produces: `OfficialTraceObserver.summary() -> dict[str, object]` containing only strict scalar metadata and SHA-256 values.
- Adds optional `trace_observer: OfficialTraceObserver | None = None` to `run_official_folder`; `None` preserves byte-identical behavior.

- [ ] **Step 1: Write failing no-op and redaction tests**

```python
def test_none_observer_preserves_official_markdown(tmp_path: Path) -> None:
    baseline = run_fake_official(tmp_path / "a", trace_observer=None)
    observed = run_fake_official(tmp_path / "b", trace_observer=OfficialTraceObserver())
    assert baseline.read_bytes() == observed.read_bytes()


def test_summary_contains_hashes_not_raw_values() -> None:
    observer = OfficialTraceObserver()
    observer.observe("final_output", "secret markdown")
    summary = observer.summary()
    assert summary["boundaries"]["final_output"]["fingerprint"] == hashlib.sha256(
        b"secret markdown"
    ).hexdigest()
    assert "secret markdown" not in json.dumps(summary)
```

Add strict allowed boundaries `crop`, `payload`, `raw_vlm`, `final_output` and
reject credentials, arbitrary keys, missing observability status, or hash
prefixes.

- [ ] **Step 2: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_official_trace_observer.py -q`

Expected: import failure because the observer does not exist.

- [ ] **Step 3: Implement diagnostic-only observation**

The adapter may call `observe` only on values authentically exposed by the
official pipeline. It must record unavailable internals as
`{"status": "unobservable"}` and must never copy a lightweight value. Wrap the
final Markdown immediately before normalization and immediately after
normalization as distinct labeled observations; do not call either a crop,
payload, or raw-VLM boundary.

- [ ] **Step 4: Capture the fixed 20 cases**

Run the observer only for case IDs in
`tests/fixtures/accuracy/v16-root-cause-cases.json`. Keep raw results under the
isolated evidence root. Regenerate the scalar summary with complete hashes and
validate exact case IDs, official source hashes, and lightweight DirectML
provider metadata.

- [ ] **Step 5: Run ordered attribution**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_accuracy_case_manifest.py tests\test_attribute_accuracy_deltas.py tests\test_official_trace_observer.py -q
```

Use `scripts/attribute_accuracy_deltas.py`. A fingerprint without replayable
same-boundary content remains `unproven`. Do not publish a positive cause unless
the official v1.6 metric replay is actually invoked.

- [ ] **Step 6: Apply the admission gate and commit**

- If 20/20 remain unproven, update the report with the new observability result
  and stop production accuracy work.
- If at least one exact fixture proves a positive earliest causal boundary,
  record its before/oracle scores, affected count, Overall estimate, full
  hashes, and neighboring gain fixtures, then proceed to Task 7.

```powershell
git add eval/official_trace_observer.py eval/PaddleOCRVLROCm_img2md.py tests/test_official_trace_observer.py tests/test_eval_adapter.py tests/fixtures/accuracy/v16-trace-capture-summary.json docs/accuracy-root-cause-v16.md
git commit -m "feat(eval): capture official v16 trace boundaries"
```

### Task 7: Plan and execute only proven accuracy corrections

**Files:**
- Create: `docs/superpowers/plans/2026-07-13-proven-accuracy-fixes.md`
- Modify: only the production files named by the proven causal boundary.
- Test: exact failing fixtures and named gain fixtures from Task 6.

**Interfaces:**
- Consumes authenticated Task 6 attribution.
- Produces zero tasks when no cause passes; otherwise produces one TDD task per independently reviewable generic cause.

- [ ] **Step 1: Generate the admission table**

For every candidate require non-null values for:

```text
case_id, page, gt_position, before_score, oracle_score, metric,
first_causal_boundary, affected_case_count, overall_contribution,
official_trace_sha256, lightweight_trace_sha256, gain_fixture_ids
```

Reject any row with `status=unproven`, missing same-boundary replay, a
non-positive oracle effect, or no gain fixture.

- [ ] **Step 2: Stop when admission is empty**

If no row passes, the plan must contain the sentence:

```markdown
No production inference fix is authorized by the current authenticated evidence.
```

It must contain no production task, placeholder, prompt guess, crop heuristic,
normalization rule, or fixture-specific output.

- [ ] **Step 3: For each passing cause, use TDD and full regression**

Each admitted task follows RED, minimal GREEN, focused regression, full suite,
DirectML contract, public output contract, independent review, and a separate
commit. After all admitted fixes, rerun Tasks 4 and 5 into a new evidence root.

- [ ] **Step 4: Require fresh G3 before continuing**

Only a fresh Overall of at least 96.13 on the corrected exact configuration
admits Task 8. Otherwise return to Task 6 with new evidence; do not stack
unattributed changes.

### Task 8: Execute G4 performance work on the accepted G3 configuration

**Files:**
- Follow: `docs/superpowers/plans/2026-07-12-performance-and-benchmark.md`
- Modify as specified there: HTTP client reuse, disk cache, concurrency, and paired benchmark files/tests.
- Modify: `docs/releases/0.1.0-readiness.md`

**Interfaces:**
- Consumes the immutable G3 manifest and output hashes.
- Produces paired mean/P95 results and output-equivalence proof.

- [ ] **Step 1: Record the exact accepted G3 manifest hash**

The benchmark refuses any model, runtime, provider, prompt, decoding, cache,
concurrency, or code mismatch not explicitly under test.

- [ ] **Step 2: Execute Performance Tasks 2-5 separately**

For each task in the existing performance plan: RED test, minimal
implementation, output hash comparison, focused/full tests, independent review,
and separate commit. Do not combine connection reuse, cache, and concurrency in
one change.

- [ ] **Step 3: Run paired acceptance**

Require:

```text
mean_seconds_per_page <= 13.00
p95_seconds_per_page <= 34.82
output_hashes_equal = true
quality_gate = true
```

If latency fails, keep G4 BLOCKED. If hashes or accuracy fail, revert the
candidate behavior and keep the last accepted G3 configuration.

- [ ] **Step 4: Commit authenticated G4 metadata**

Update readiness with the exact benchmark manifest/result hashes and commit
only small reports.

### Task 9: Complete G5 and publish only after the final audit

**Files:**
- Modify: `docs/releases/0.1.0-windows-validation.md`
- Modify: `docs/releases/0.1.0-readiness.md`
- Modify: release notes/version files only after G0-G5 pass.

**Interfaces:**
- Consumes accepted G0-G4 manifests.
- Produces final package, onboarding, security, repository, and publication evidence.

- [ ] **Step 1: Repeat empty-cache public-network setup**

Use a new empty root and capture DNS, connect, first-byte, Range, checksum,
extraction, activation, and doctor stages separately. A network failure remains
G5 BLOCKED but must not be mislabeled an installer or checksum failure.

- [ ] **Step 2: Repeat both inference journeys**

Run verified-cache installation, full doctor, managed-server inference, and
existing-server inference. Require DirectML first, fallback disabled, redacted
diagnostics, stable exit codes, and expected JSON/Markdown artifacts.

- [ ] **Step 3: Run the final verification matrix**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check . --exclude eval/.omnidocbench
.\.venv\Scripts\python.exe -m ruff format --check . --exclude eval/.omnidocbench
.\.venv\Scripts\python.exe -m mypy src eval scripts
.\.venv\Scripts\python.exe -m build
git diff --check
git status --short
gh auth status
```

Install the built wheel into a new virtual environment and repeat setup,
doctor, and one inference before approving the package.

- [ ] **Step 4: Run final branch and release reviews**

Use `superpowers:requesting-code-review`,
`superpowers:verification-before-completion`, and
`superpowers:finishing-a-development-branch`. Confirm no model/raw evidence,
secret, absolute local path, ignored scorer checkout, or unsupported score/speed
claim is staged.

- [ ] **Step 5: Publish only when every gate is PASS**

When G0-G5 are all PASS, intentionally bump the version, commit release notes,
push `codex/top-tier-quality`, open the reviewed PR, merge according to project
policy, tag the merged commit, and publish the GitHub release. If any gate is
BLOCKED, stop before all publication mutations.
