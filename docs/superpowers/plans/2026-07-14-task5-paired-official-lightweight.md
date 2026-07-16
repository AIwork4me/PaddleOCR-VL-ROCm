# Task 5 Paired Official/Lightweight Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce append-only, independently auditable OmniDocBench v1.6 evidence that compares fresh Official and Lightweight inference on the same Windows AMD machine, renders separate strict-equivalence and AMD-adaptation verdicts, and decides G3 without mutating sealed r7 G0 evidence.

**Architecture:** Keep the r7 G0 namespace immutable and add a dedicated `task5/` evidence chain. Small Python modules own manifest binding, trace/output comparison, DirectML node-execution attestation, decisions, and receipts; a new PowerShell runner composes the existing inference/scoring entry points into immutable attempts. Optional observation is threaded through the existing adapters without changing default outputs.

**Tech Stack:** Python 3.10+, pytest, ONNX Runtime DirectML profiling, PowerShell 5.1+, OmniDocBench v1.6 commit `147cd5ac9472002f5751221d390bf00abdbc0d2f`, SHA-256, JSON/JSONL, Ruff, mypy.

## Global Constraints

- Work only in `C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm\.worktrees\top-tier-quality` on `codex/top-tier-quality`.
- Preserve every existing byte under `C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-evidence\v16-2026-07-14-official-r7-score-recovery-py310` except new files below its `task5/` child.
- Bind tracked G0 receipt SHA-256 `d0b7fcbe389e03439b5ba65126008fa5ee828a59e358ae0347c5bb6a51648a04` and the six r7 Official output hashes from `docs/releases/0.1.0-g0-evidence.md`.
- Use OmniDocBench v1.6 only; v1.7 must not enter configs, thresholds, tables, or claims.
- Score all 1,651 GT pages. Pair exactly the 1,650 successful Official pages; the sole approved issue #18248 `peg-native` page is excluded only from equivalence pairing.
- “100% output equivalent” means all 1,650 normalized Markdown strings and all required canonical trace boundaries are equal with zero unobservable boundaries. It does not mean raw-file byte identity.
- Canonical boundary order is `request_order -> label -> bbox -> crop_pixels -> prompt -> payload -> raw_result -> postprocess`.
- Any proven difference yields strict `FAIL`; otherwise any required unobservable boundary yields `UNKNOWN`; only complete equality yields `PASS`.
- `amd_adaptation` is independent from `strict_equivalence` and requires DirectML-first layout, disabled session fallback, DirectML ownership of strictly more than 50% of profiled layout `Node` events, zero missing/other-provider node events, transparent CPU graph-partition counts, public contracts, clean scorer quality, Lightweight Overall >= 96.13, and no accepted-component regression against paired Official.
- Formula uses `display_formula.page.CDM.ALL`; Table uses `table.page.TEDS.ALL`; components are rounded to three decimals before Overall; reading order is excluded.
- Default inference behavior and outputs must remain unchanged when observers/profiling are absent.
- No production accuracy fix or performance behavior optimization is authorized by this plan.
- Keep predictions, raw traces, models, datasets, scorer work products, and `eval/.omnidocbench/` untracked. Never stage `eval/.omnidocbench/`.
- Every code task uses TDD, a focused commit, full relevant verification, and independent review before the next task.

## File Structure

- Create `eval/task5_manifest.py`: append-only r7 binding, immutable input/environment schema, revalidation, atomic JSON writer.
- Create `eval/task5_comparison.py`: scorer-facing Markdown normalization, page pairing, canonical observation schema, exhaustive trace comparison, strict verdict.
- Create `eval/directml_attestation.py`: parse ORT profiling JSON and fail closed unless all layout node events are DML.
- Create `eval/task5_decision.py`: paired metric extraction, AMD/G3 decision, allowlisted receipt generation and CLI.
- Create `scripts/run_task5_paired_v16.ps1`: immutable attempt orchestration and sealed-r7 before/after verification.
- Modify `src/paddleocr_vl_rocm/layout.py`: optional ORT profiling and explicit fallback-disable state.
- Modify `src/paddleocr_vl_rocm/pipeline.py`: optional trace sink and profiling prefix, with unchanged defaults.
- Modify `eval/PaddleOCRVLROCm_img2md.py`: per-page Lightweight traces, conservative Official observability records, profile finalization.
- Modify `eval/run_eval.py`: forward optional trace/profile arguments to the adapter.
- Modify `scripts/compare_inference_traces.py`: consume explicit observable/unobservable boundary records and report `PASS`/`FAIL`/`UNKNOWN`.
- Create `tests/test_task5_manifest.py`, `tests/test_task5_comparison.py`, `tests/test_directml_attestation.py`, `tests/test_task5_decision.py`, and `tests/test_run_task5_paired_v16_script.py`.
- Modify `tests/test_layout_provider.py`, `tests/test_eval_adapter.py`, `tests/test_compare_inference_traces.py`, and `tests/test_eval_report_path.py` for optional-observer contracts.
- Generate external `r7/task5/**`; track only compact copies under `results/omnidocbench/v16/task5/` after the live run.
- Modify `docs/releases/0.1.0-readiness.md`, `results/omnidocbench/v16/README.md`, `README.md`, and `README.zh-CN.md` only from the authenticated final decision.

---

### Task 1: Bind Task 5 to sealed r7 without changing G0

**Files:**
- Create: `eval/task5_manifest.py`
- Create: `tests/test_task5_manifest.py`

**Interfaces:**
- Produces: `file_identity(path: Path) -> dict[str, object]`.
- Produces: `snapshot_sealed_g0(r7_root: Path, receipt_path: Path) -> dict[str, object]`.
- Produces: `build_task5_manifest(*, r7_root: Path, receipt_path: Path, git_commit: str, inputs: Mapping[str, Path], environment: Mapping[str, object], contracts: Mapping[str, object]) -> dict[str, object]`.
- Produces: `validate_task5_manifest(manifest: Mapping[str, object], *, task5_root: Path) -> None`.
- Produces CLI: `python -m eval.task5_manifest create|validate|snapshot`.

- [ ] **Step 1: Write failing sealed-root and schema tests**

```python
def test_manifest_binds_receipt_r7_manifest_and_six_outputs(tmp_path: Path) -> None:
    r7, receipt = make_sealed_r7(tmp_path)
    manifest = build_task5_manifest(
        r7_root=r7,
        receipt_path=receipt,
        git_commit="2" * 40,
        inputs={"dataset": tmp_path / "dataset.json"},
        environment={"os": "Windows", "gpu": "AMD"},
        contracts={"benchmark": "OmniDocBench-v1.6", "pair_pages": 1650},
    )
    assert manifest["g0"]["receipt"]["sha256"] == sha256_file(receipt)
    assert manifest["g0"]["manifest"]["sha256"] == sha256_file(r7 / "manifest.json")
    assert set(manifest["g0"]["official_outputs"]) == EXPECTED_OFFICIAL_OUTPUTS


def test_manifest_rejects_task5_outside_exact_r7_child(tmp_path: Path) -> None:
    manifest = valid_manifest(tmp_path)
    with pytest.raises(ValueError, match="exactly r7/task5"):
        validate_task5_manifest(manifest, task5_root=tmp_path / "task5-copy")


def test_revalidation_detects_any_sealed_g0_mutation(tmp_path: Path) -> None:
    r7, receipt = make_sealed_r7(tmp_path)
    before = snapshot_sealed_g0(r7, receipt)
    (r7 / "results/official/metric.json").write_text("changed", encoding="utf-8")
    assert snapshot_sealed_g0(r7, receipt) != before
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_task5_manifest.py -q
```

Expected: collection fails because `eval.task5_manifest` does not exist.

- [ ] **Step 3: Implement the strict manifest and snapshot**

Use this schema and reject extra/missing top-level keys:

```python
TASK5_SCHEMA = 1
OFFICIAL_OUTPUTS = (
    "results/official/metric.json",
    "results/official/metric-cdm.json",
    "results/official/provenance.json",
    "results/official/provenance-cdm.json",
    "results/official/run-summary.json",
    "results/official/run-summary-cdm.json",
)


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Evidence input must be a regular file: {path}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def snapshot_sealed_g0(r7_root: Path, receipt_path: Path) -> dict[str, object]:
    root = r7_root.resolve(strict=True)
    return {
        "receipt": file_identity(receipt_path),
        "manifest": file_identity(root / "manifest.json"),
        "official_outputs": {
            relative: file_identity(root / relative) for relative in OFFICIAL_OUTPUTS
        },
    }


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
```

`build_task5_manifest` must set `task5_root` to `str((r7_root / "task5").resolve(strict=False))`, sort all logical input names, hash every input file, copy only JSON scalar/list/object environment fields, and include the exact approved contracts. `validate_task5_manifest` must rehash every recorded file, require full lowercase SHA-256, reject symlink/path changes and non-finite JSON numbers, and verify the receipt digest equals the global constraint.

- [ ] **Step 4: Add CLI round-trip and mutation tests, then run GREEN**

```python
def test_cli_create_then_validate_rehashes_inputs(tmp_path: Path) -> None:
    completed = run_manifest_cli("create", tmp_path)
    assert completed.returncode == 0
    manifest_path = tmp_path / "r7/task5/manifest.json"
    assert run_manifest_cli("validate", tmp_path, manifest_path).returncode == 0
    dataset = tmp_path / "dataset.json"
    dataset.write_text("mutated", encoding="utf-8")
    failed = run_manifest_cli("validate", tmp_path, manifest_path)
    assert failed.returncode != 0
    assert "SHA-256" in failed.stderr
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_task5_manifest.py tests\test_release_evidence.py -q
.\.venv\Scripts\python.exe -m ruff check eval\task5_manifest.py tests\test_task5_manifest.py
```

Expected: all pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add eval/task5_manifest.py tests/test_task5_manifest.py
git commit -m "feat(eval): bind Task 5 to sealed G0 evidence"
```

### Task 2: Capture and compare all scorer-facing outputs and canonical boundaries

**Files:**
- Create: `eval/task5_comparison.py`
- Create: `tests/test_task5_comparison.py`
- Modify: `src/paddleocr_vl_rocm/pipeline.py`
- Modify: `eval/PaddleOCRVLROCm_img2md.py`
- Modify: `eval/run_eval.py`
- Modify: `scripts/compare_inference_traces.py`
- Modify: `tests/test_compare_inference_traces.py`
- Modify: `tests/test_eval_adapter.py`
- Modify: `tests/test_eval_report_path.py`

**Interfaces:**
- Produces: `normalize_scorer_markdown(text: str) -> str`.
- Produces: `compare_prediction_dirs(official_dir: Path, lightweight_dir: Path, approved_excluded_stem: str) -> dict[str, object]`.
- Produces: `observation(value: object) -> dict[str, str]` and `unobservable() -> dict[str, str]`.
- Produces: `compare_canonical_traces(official_dir: Path, lightweight_dir: Path) -> dict[str, object]`.
- Extends: `PaddleOCRVLROCm.predict(image_path, *, vlm_trace_events: list[dict[str, Any]] | None = None)`.
- Extends adapters with optional `trace_dir: Path | None`; `None` preserves existing files byte-for-byte.

- [ ] **Step 1: Write failing normalization, denominator, and verdict tests**

```python
def test_normalization_ignores_only_transport_newlines() -> None:
    assert normalize_scorer_markdown("a  \r\nformula\r\n\r\n") == "a  \nformula"
    assert normalize_scorer_markdown("a \n") != normalize_scorer_markdown("a\n")


def test_output_comparison_requires_exactly_1650_pairs(tmp_path: Path) -> None:
    official, lightweight = write_prediction_pairs(tmp_path, count=1650)
    report = compare_prediction_dirs(official, lightweight, APPROVED_STEM)
    assert report["paired_pages"] == 1650
    assert report["equal_pages"] == 1650
    assert report["different_pages"] == 0


def test_proven_difference_beats_unobservable() -> None:
    report = compare_boundary_documents(
        official=[event(raw_result=unobservable(), postprocess=observation("a"))],
        lightweight=[event(raw_result=observation("x"), postprocess=observation("b"))],
    )
    assert report["verdict"] == "FAIL"
    assert report["first_divergence_counts"]["postprocess"] == 1


def test_only_unobservable_boundaries_yield_unknown() -> None:
    report = compare_boundary_documents(
        official=[event(raw_result=unobservable())],
        lightweight=[event(raw_result=observation("x"))],
    )
    assert report["verdict"] == "UNKNOWN"
```

Normalization is intentionally narrow: decode as UTF-8, convert CRLF/CR to LF,
and remove terminal LF characters. Preserve every other character and all
in-line/trailing spaces because Markdown uses two trailing spaces semantically.

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_task5_comparison.py tests\test_compare_inference_traces.py -q
```

Expected: failures for missing `eval.task5_comparison` and unobservable verdict support.

- [ ] **Step 3: Implement explicit observation records and exhaustive comparison**

```python
BOUNDARIES = (
    "request_order", "label", "bbox", "crop_pixels",
    "prompt", "payload", "raw_result", "postprocess",
)


def observation(value: object) -> dict[str, str]:
    return {"status": "observable", "fingerprint": fingerprint(redact(value))}


def unobservable() -> dict[str, str]:
    return {"status": "unobservable"}


def normalize_scorer_markdown(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def boundary_relation(reference: Mapping[str, str], candidate: Mapping[str, str]) -> str:
    if reference["status"] == "unobservable" or candidate["status"] == "unobservable":
        return "unobservable"
    return "equal" if reference["fingerprint"] == candidate["fingerprint"] else "different"
```

Require each page event to include `page`, `block_index`, and exactly the eight
boundary observation objects. Pair events by `(page, block_index)`, not zip
position. A missing event is a proven structural difference. Retain at most 100
bounded detail rows, but count and hash all differences/unobservable records.
Never include prompt, payload, raw output, or Markdown bodies in the report.

- [ ] **Step 4: Thread optional Lightweight trace capture through the public pipeline**

Change only the optional path:

```python
def predict(
    self,
    image_path: str | Path,
    *,
    vlm_trace_events: list[dict[str, object]] | None = None,
) -> PaddleOCRVLROCmResult:
    # existing setup remains unchanged
    json_path = run_light_parser(
        # existing arguments unchanged
        vlm_trace_events=vlm_trace_events,
    )
```

In `run_lightweight_folder`, allocate a fresh list per page, call
`pipeline.predict(img, vlm_trace_events=events)`, convert the existing event
fields to explicit observations, and write deterministic JSONL to
`trace_dir/<image-stem>.jsonl`. Confirm `trace_dir=None` creates no extra files
and does not alter Markdown or `_run_stats.json`.

- [ ] **Step 5: Add conservative Official observation without inventing boundaries**

For each successful Official page, inspect only authenticated fields already
returned by PaddleOCR. If an ordered block collection exposes label, bbox,
prompt/request, raw result, or postprocessed block content, fingerprint it. For
every field not directly exposed, write `{"status": "unobservable"}`. Always
fingerprint the exact scorer-facing normalized page Markdown in a page summary;
never infer an earlier boundary from it.

```python
def official_page_trace(page: str, result: object, markdown: str) -> dict[str, object]:
    blocks = _extract_authenticated_official_blocks(result)
    if blocks is None:
        return {
            "page": page,
            "block_index": None,
            "block_structure": unobservable(),
            "boundaries": {name: unobservable() for name in BOUNDARIES},
            "page_postprocess": observation(normalize_scorer_markdown(markdown)),
        }
    return _fingerprint_official_blocks(page, blocks, markdown)
```

The trace schema validator must allow the page-level `block_index=null` record
only when `block_structure` is unobservable; this produces `UNKNOWN`, never
`PASS`. Add fake Official results for fully observable, partially observable,
and page-only cases.

- [ ] **Step 6: Forward `--trace-dir` and verify unchanged defaults**

Add `--trace-dir` to `eval.run_eval` and forward it through `stage_infer` to
`run_adapter`. Tests must compare baseline and observer-disabled Markdown and
stats bytes, then assert observer-enabled output contains only fingerprints and
status metadata.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_task5_comparison.py tests\test_compare_inference_traces.py tests\test_eval_adapter.py tests\test_eval_report_path.py tests\test_native_compat_contract.py -q
.\.venv\Scripts\python.exe -m ruff check eval\task5_comparison.py eval\PaddleOCRVLROCm_img2md.py eval\run_eval.py scripts\compare_inference_traces.py src\paddleocr_vl_rocm\pipeline.py tests\test_task5_comparison.py
```

Expected: all pass; native compatibility goldens stay unchanged.

- [ ] **Step 7: Commit Task 2**

```powershell
git add eval/task5_comparison.py scripts/compare_inference_traces.py eval/PaddleOCRVLROCm_img2md.py eval/run_eval.py src/paddleocr_vl_rocm/pipeline.py tests/test_task5_comparison.py tests/test_compare_inference_traces.py tests/test_eval_adapter.py tests/test_eval_report_path.py
git commit -m "feat(eval): compare paired outputs and observable traces"
```

### Task 3: Prove PP-DocLayoutV3 node execution on DirectML

**Files:**
- Create: `eval/directml_attestation.py`
- Create: `tests/test_directml_attestation.py`
- Modify: `src/paddleocr_vl_rocm/layout.py`
- Modify: `src/paddleocr_vl_rocm/pipeline.py`
- Modify: `eval/PaddleOCRVLROCm_img2md.py`
- Modify: `eval/run_eval.py`
- Modify: `tests/test_layout_provider.py`
- Modify: `tests/test_eval_adapter.py`
- Modify: `tests/test_eval_report_path.py`

**Interfaces:**
- Extends: `PPDocLayoutV3Onnx(..., profiling_prefix: Path | None = None)`.
- Produces: `PPDocLayoutV3Onnx.finish_profiling() -> Path | None`, idempotent.
- Produces: `attest_directml_profile(profile_path: Path, run_stats: Mapping[str, object]) -> dict[str, object]`.
- Extends: `PaddleOCRVLROCm(..., layout_profile_prefix: Path | None = None)` and
  `finish_layout_profiling() -> Path | None`.
- Extends: `run_lightweight_folder(..., layout_profile_prefix: Path | None = None)`.
- Extends CLI: `eval.run_eval --layout-profile-prefix PATH` for Lightweight inference only.

- [ ] **Step 1: Write failing profile parser and fallback-state tests**

```python
def test_attestation_accepts_directml_majority_with_cpu_graph_partitions(tmp_path: Path) -> None:
    profile = write_profile(
        tmp_path,
        providers=["DmlExecutionProvider"] * 7 + ["CPUExecutionProvider"] * 3,
    )
    report = attest_directml_profile(profile, valid_directml_stats())
    assert report["verdict"] == "PASS"
    assert report["dml_node_events"] == 7
    assert report["cpu_node_events"] == 3
    assert report["dml_node_share"] == 0.7


@pytest.mark.parametrize(
    "providers",
    [
        [],
        ["CPUExecutionProvider"],
        ["DmlExecutionProvider", "CPUExecutionProvider"],
        ["DmlExecutionProvider", "CPUExecutionProvider", "CPUExecutionProvider"],
    ],
)
def test_attestation_fails_without_strict_dml_majority(
    tmp_path: Path, providers: list[str]
) -> None:
    report = attest_directml_profile(write_profile(tmp_path, providers), valid_directml_stats())
    assert report["verdict"] == "FAIL"


def test_session_records_fallback_disabled_only_after_api_call(tmp_path: Path, monkeypatch) -> None:
    model = build_fake_layout(tmp_path, monkeypatch, has_disable_fallback=True)
    assert model.layout_fallback_disabled is True
```

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_directml_attestation.py tests\test_layout_provider.py -q
```

Expected: missing module and missing profiling/fallback properties.

- [ ] **Step 3: Add optional ORT profiling and explicit fallback state**

```python
options = ort.SessionOptions()
if profiling_prefix is not None:
    profiling_prefix.parent.mkdir(parents=True, exist_ok=True)
    options.enable_profiling = True
    options.profile_file_prefix = str(profiling_prefix.resolve())

self.layout_fallback_disabled = False
disable_fallback = getattr(self.session, "disable_fallback", None)
if disable_fallback is not None:
    disable_fallback()
    self.layout_fallback_disabled = True
self._profile_path: Path | None = None


def finish_profiling(self) -> Path | None:
    if self._profile_path is not None:
        return self._profile_path
    if not self._profiling_enabled:
        return None
    self._profile_path = Path(self.session.end_profiling()).resolve(strict=True)
    return self._profile_path
```

DirectML evidence mode must fail during setup when `disable_fallback()` is
unavailable. Normal non-evidence construction preserves compatibility and
records `layout_fallback_disabled=False` instead of claiming it was disabled.

- [ ] **Step 4: Implement the fail-closed profile attestor**

Parse ORT JSON trace events with `cat == "Node"`; read the provider from
`event["args"]["provider"]`; count DML, CPU, missing-provider, and other-provider
node events. Require stats to contain requested `auto`, active providers exactly
`["DmlExecutionProvider", "CPUExecutionProvider"]`, and fallback disabled.

```python
passed = (
    requested == "auto"
    and active == ["DmlExecutionProvider", "CPUExecutionProvider"]
    and fallback_disabled is True
    and dml_nodes > 0
    and dml_nodes / (dml_nodes + cpu_nodes) > 0.5
    and missing_provider_nodes == 0
    and not other_providers
)
```

Emit only counts, ordered provider names, profile SHA-256, profile bytes, and
`PASS`/`FAIL`; do not copy the raw profile into tracked evidence.

- [ ] **Step 5: Finalize profiling after the complete Lightweight corpus**

Pass `layout_profile_prefix` into the pipeline constructor, close the profile in
a `finally` block after all pages, and add these stats fields:

```python
summary.update(
    {
        "layout_fallback_disabled": pipeline.layout_fallback_disabled,
        "layout_profile_path": str(profile_path) if profile_path else None,
    }
)
```

Add a regression test proving an inference exception still calls
`finish_profiling()` and leaves a complete stats record.

Add `--layout-profile-prefix` to `eval.run_eval` and forward it through
`stage_infer` and `run_adapter`. Reject it for the Official engine so the paired
runner cannot accidentally label Official inference with a DirectML profile.

- [ ] **Step 6: Run GREEN and a real one-page DirectML smoke**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_directml_attestation.py tests\test_layout_provider.py tests\test_eval_adapter.py tests\test_eval_report_path.py tests\test_native_compat_contract.py -q
.\.venv\Scripts\python.exe -m ruff check eval\directml_attestation.py src\paddleocr_vl_rocm\layout.py src\paddleocr_vl_rocm\pipeline.py eval\PaddleOCRVLROCm_img2md.py eval\run_eval.py
```

Then run the first v1.6 dataset page through evidence-mode profiling using the
same r7-bound dataset/layout artifacts and validate the generated profile:

```powershell
$R7 = 'C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-evidence\v16-2026-07-14-official-r7-score-recovery-py310'
$Manifest = Get-Content -Raw "$R7\manifest.json" | ConvertFrom-Json
$DatasetDir = Split-Path -Parent $Manifest.inputs.dataset.path
$LayoutDir = Split-Path -Parent $Manifest.inputs.layout_model.path
$SmokeRoot = Join-Path $env:TEMP 'paddleocr-task5-dml-smoke'
New-Item -ItemType Directory -Force -Path $SmokeRoot | Out-Null
.\.venv\Scripts\python.exe -m eval.run_eval --stage infer --version v16 --engine lightweight --server-url http://127.0.0.1:8111/v1 --dataset-dir $DatasetDir --predictions-dir "$SmokeRoot\predictions" --layout-model $LayoutDir --limit-pages 1 --layout-profile-prefix "$SmokeRoot\layout-profile"
$Profile = Get-ChildItem -LiteralPath $SmokeRoot -Filter 'layout-profile*.json' | Sort-Object LastWriteTimeUtc | Select-Object -Last 1 -ExpandProperty FullName
.\.venv\Scripts\python.exe -m eval.directml_attestation --profile $Profile --stats "$SmokeRoot\predictions\_run_stats.json"
```

Expected: `verdict=PASS`, `dml_node_events>cpu_node_events`,
`dml_node_share>0.5`, and missing/other counts zero. CPU-assigned graph
partitions are reported and do not by themselves fail adaptation. If ORT emits
a different documented provider field, preserve the raw profile and adjust the
parser only after adding that exact real event as a sanitized fixture.

- [ ] **Step 7: Commit Task 3**

```powershell
git add eval/directml_attestation.py src/paddleocr_vl_rocm/layout.py src/paddleocr_vl_rocm/pipeline.py eval/PaddleOCRVLROCm_img2md.py eval/run_eval.py tests/test_directml_attestation.py tests/test_layout_provider.py tests/test_eval_adapter.py tests/test_eval_report_path.py
git commit -m "feat(directml): attest layout node execution"
```

### Task 4: Render independent equivalence, AMD-adaptation, and G3 decisions

**Files:**
- Create: `eval/task5_decision.py`
- Create: `tests/test_task5_decision.py`

**Interfaces:**
- Produces: `extract_paired_scores(non_cdm: Mapping[str, object], cdm: Mapping[str, object]) -> dict[str, object]`.
- Produces: `strict_equivalence_decision(output_report: Mapping[str, object], trace_report: Mapping[str, object]) -> dict[str, object]`.
- Produces: `amd_adaptation_decision(*, official_scores, lightweight_scores, provider_attestation, lightweight_stats, public_contracts_pass: bool) -> dict[str, object]`.
- Produces: `build_task5_receipt(task5_root: Path, relative_paths: Sequence[str]) -> dict[str, object]`.
- Produces CLI: `python -m eval.task5_decision decide|receipt|validate-receipt`.

- [ ] **Step 1: Write failing verdict-priority and G3 tests**

```python
def test_strict_fail_beats_unknown() -> None:
    decision = strict_equivalence_decision(
        {"paired_pages": 1650, "different_pages": 1},
        {"verdict": "UNKNOWN", "unobservable_count": 8},
    )
    assert decision["verdict"] == "FAIL"


def test_strict_unknown_does_not_block_independent_amd_pass() -> None:
    amd = amd_adaptation_decision(
        official_scores=scores(overall=96.20),
        lightweight_scores=scores(overall=96.20),
        provider_attestation={"verdict": "PASS"},
        lightweight_stats=valid_lightweight_stats(),
        public_contracts_pass=True,
    )
    assert amd["verdict"] == "PASS"
    assert amd["g3"] is True


@pytest.mark.parametrize("overall", [96.129, 95.743])
def test_g3_fails_below_9613(overall: float) -> None:
    assert decide_with_lightweight_overall(overall)["g3"] is False


def test_component_regression_fails_even_when_overall_is_high() -> None:
    official = scores(text_edit=0.030, formula=96.0, table=94.0, overall=96.2)
    lightweight = scores(text_edit=0.031, formula=97.0, table=95.0, overall=96.5)
    assert decide_amd(official, lightweight)["verdict"] == "FAIL"
```

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_task5_decision.py -q
```

Expected: missing `eval.task5_decision`.

- [ ] **Step 3: Implement score extraction and exact notebook arithmetic**

Reuse `eval.artifact_utils.extract_notebook_metrics` and
`analyze_metric_quality`. Require non-CDM and CDM reports to agree on Text Edit,
Table TEDS, and reading order after approved rounding. Select Formula CDM only
from the CDM report. Reject missing/non-finite/out-of-range components.

```python
def component_not_worse(official: Mapping[str, float], lightweight: Mapping[str, float]) -> bool:
    return (
        lightweight["text_edit_dist"] <= official["text_edit_dist"]
        and lightweight["formula_cdm_percent"] >= official["formula_cdm_percent"]
        and lightweight["table_teds_percent"] >= official["table_teds_percent"]
    )
```

The final `decision.json` has exactly:

```json
{
  "schema": 1,
  "benchmark": "OmniDocBench-v1.6",
  "coverage": {},
  "scores": {"official": {}, "lightweight": {}},
  "strict_equivalence": {},
  "amd_adaptation": {},
  "g3": false,
  "evidence": {}
}
```

- [ ] **Step 4: Implement an allowlisted, self-excluding receipt**

Only allow relative files under the selected Task 5 root, reject symlinks and
path escapes, sort paths, and never include `receipt.sha256.json` in its own
input list.

```python
def build_task5_receipt(task5_root: Path, relative_paths: Sequence[str]) -> dict[str, object]:
    if "receipt.sha256.json" in relative_paths:
        raise ValueError("Receipt cannot hash itself")
    files = {name: file_identity(_contained_file(task5_root, name)) for name in sorted(relative_paths)}
    return {"schema": 1, "algorithm": "sha256", "files": files}
```

The allowlist must include the manifest, selected attempt/stage state, both
score summaries and provenances, input/output/trace comparison, provider
attestation, and decision. Raw prediction and trace directories are bound by
their sorted manifest hashes, not copied into the receipt.

- [ ] **Step 5: Run GREEN and mutation tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_task5_decision.py tests\test_eval_artifact_utils.py tests\test_release_evidence.py -q
.\.venv\Scripts\python.exe -m ruff check eval\task5_decision.py tests\test_task5_decision.py
```

Expected: all pass; mutating any receipt input makes `validate-receipt` exit nonzero.

- [ ] **Step 6: Commit Task 4**

```powershell
git add eval/task5_decision.py tests/test_task5_decision.py
git commit -m "feat(eval): decide paired equivalence and AMD adaptation"
```

### Task 5: Orchestrate immutable paired inference, dual scoring, comparison, and receipt

**Files:**
- Create: `scripts/run_task5_paired_v16.ps1`
- Create: `tests/test_run_task5_paired_v16_script.py`

**Interfaces:**
- Produces PowerShell stages: `Preflight`, `Official`, `Lightweight`, `Score`, `Compare`, `Decide`, `All`.
- Consumes: sealed r7 root, current worktree interpreter, authenticated scorer interpreter, existing server, and paths already bound by the r7 manifest/runtime config.
- Produces: `r7/task5/manifest.json`, immutable attempt state, paired predictions/traces, two score pairs, comparisons, decisions, and receipt.

- [ ] **Step 1: Write failing script-structure and append-only tests**

```python
def test_runner_uses_separate_task5_manifest_and_never_calls_g0_runner() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'Join-Path $R7Root "task5"' in text
    assert "run_release_evidence_v16.ps1" not in text
    assert "Remove-Item $R7Root" not in text
    assert "snapshot-before.json" in text and "snapshot-after.json" in text


def test_lightweight_and_official_both_run_non_cdm_and_cdm_scoring() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count('"--cdm"') >= 2
    assert "results/official/metric-cdm.json" in text
    assert "results/lightweight/metric-cdm.json" in text


def test_stage_resume_rejects_commit_manifest_or_output_drift(tmp_path: Path) -> None:
    first = run_stubbed_runner(tmp_path, stage="Preflight")
    assert first.returncode == 0
    mutate_bound_input(tmp_path)
    resumed = run_stubbed_runner(tmp_path, stage="Official")
    assert resumed.returncode != 0
    assert "integrity mismatch" in resumed_output(resumed)
```

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_run_task5_paired_v16_script.py -q
```

Expected: missing script.

- [ ] **Step 3: Implement preflight and immutable attempts**

Use this parameter contract:

```powershell
param(
  [ValidateSet("Preflight", "Official", "Lightweight", "Score", "Compare", "Decide", "All")]
  [string]$Stage = "Preflight",
  [Parameter(Mandatory=$true)][string]$R7Root,
  [Parameter(Mandatory=$true)][ValidatePattern('^[a-z0-9][a-z0-9-]{0,63}$')][string]$AttemptId,
  [string]$PythonExe = ".\.venv\Scripts\python.exe",
  [string]$ScorerPythonExe = "C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-scorer-v16-py310\Scripts\python.exe",
  [string]$ServerUrl = "http://127.0.0.1:8111/v1"
)
$Task5Root = Join-Path (Resolve-Path -LiteralPath $R7Root) "task5"
$AttemptRoot = Join-Path (Join-Path $Task5Root "attempts") $AttemptId
```

Preflight must:

1. require a clean tracked worktree while explicitly tolerating only
   `eval/.omnidocbench/` as the known untracked checkout;
2. snapshot sealed G0 to `attempts/<id>/snapshot-before.json`;
3. authenticate both Python interpreters and the v1.6 scorer;
4. rehash every r7-bound model/dataset/config input;
5. check server and Official constructor;
6. create the Task 5 manifest atomically if absent, or revalidate it if present;
7. write stage state with producing commit, manifest SHA-256, command-log hash,
   output-map hash, start/end UTC, exit code, and orphan audit.

- [ ] **Step 4: Implement fresh paired inference without fallback**

Official command:

```powershell
& $PythonExe -m eval.run_eval --stage infer --version v16 --engine official `
  --server-url $ServerUrl --api-model-name $ApiModelName `
  --dataset-dir $DatasetDir --predictions-dir (Join-Path $Task5Root "paired-official") `
  --trace-dir (Join-Path $Task5Root "traces/official") --page-retries 1
```

Lightweight command:

```powershell
& $PythonExe -m eval.run_eval --stage infer --version v16 --engine lightweight `
  --server-url $ServerUrl --api-model-name $ApiModelName `
  --dataset-dir $DatasetDir --predictions-dir (Join-Path $Task5Root "lightweight") `
  --layout-model $LayoutModel --trace-dir (Join-Path $Task5Root "traces/lightweight") `
  --layout-profile-prefix (Join-Path $Task5Root "attempts/$AttemptId/layout-profile")
```

Do not pass `--fallback-pred-dir`. Validate Official with the approved
1,650/1/0 contract and Lightweight with 1,651/1,651/0/0/null. Create a new
AttemptId after any invalid attempt; never clear an earlier attempt directory.

- [ ] **Step 5: Implement both score modes for both engines**

For each engine run `eval.run_eval --stage eval` once without `--cdm` and once
with `--cdm`, always passing the authenticated scorer interpreter and distinct
`metric`, `run-summary`, and `provenance` destinations under
`task5/results/<engine>/`. Check metric quality immediately after each CDM run.

- [ ] **Step 6: Implement comparison, decisions, receipt, and sealed-r7 closure**

The Compare stage invokes `eval.task5_comparison` for predictions and traces.
The Decide stage invokes `eval.directml_attestation`, then
`eval.task5_decision decide`, then receipt generation. Finally snapshot sealed
G0 to `snapshot-after.json` and require canonical JSON equality with the before
snapshot before writing `selected-attempt.json`.

If strict equivalence is `UNKNOWN`, the runner exits 0 when the evidence is
complete and prints `strict_equivalence=UNKNOWN`; this is a valid measured
outcome. Evidence-integrity failures and AMD/G3 failures remain explicit in
`decision.json`; they must not be converted into process crashes after the
decision is successfully written.

- [ ] **Step 7: Add fault-injection coverage**

Stub tests must simulate and reject: CPU-first provider order, zero or at-most-
50% DML node share, missing/other provider node events, missing profile,
Official fallback, Lightweight partial coverage, stale score,
CDM timeout, TEDS error, orphan process, changed command log, changed output,
changed G0 file, symlinked task5 root, old-attempt file reuse, and receipt
mutation. Assert no case emits `strict_equivalence=PASS` or
`amd_adaptation=PASS` incorrectly.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_run_task5_paired_v16_script.py tests\test_run_release_evidence_v16_script.py -q
git diff --check
```

Expected: all pass and the existing G0 runner contract remains unchanged.

- [ ] **Step 8: Commit Task 5**

```powershell
git add scripts/run_task5_paired_v16.ps1 tests/test_run_task5_paired_v16_script.py
git commit -m "feat(eval): orchestrate paired Task 5 evidence"
```

### Task 6: Run the complete offline verification gate

**Files:**
- Modify only if a failing test exposes a Task 1-5 defect; return to that task's TDD cycle and commit the focused correction.

**Interfaces:**
- Consumes all Task 1-5 code.
- Produces a clean implementation commit suitable for live evidence generation.

- [ ] **Step 1: Run focused Task 5 tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_task5_manifest.py tests\test_task5_comparison.py tests\test_directml_attestation.py tests\test_task5_decision.py tests\test_run_task5_paired_v16_script.py -q
```

Expected: all pass.

- [ ] **Step 2: Run compatibility and release regressions**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_native_compat_contract.py tests\test_layout_provider.py tests\test_eval_adapter.py tests\test_eval_report_path.py tests\test_compare_inference_traces.py tests\test_release_contract.py tests\test_release_evidence.py tests\test_run_release_evidence_v16_script.py tests\test_scorer_preflight.py -q
```

Expected: all pass with only previously documented environment skips.

- [ ] **Step 3: Run repository quality gates**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy src
git diff --check
git status --short
```

Expected: tests/lint/format/type/diff checks pass. Status contains no unintended
tracked changes and may contain only the known untracked `eval/.omnidocbench/`.

- [ ] **Step 4: Independently review implementation commits**

Review each Task 1-5 commit against
`docs/superpowers/specs/2026-07-14-task5-paired-official-lightweight-design.md`.
Critical or Important findings return to the owning task with a failing test and
focused fix commit. Repeat full verification after the last fix.

### Task 7: Execute the real paired v1.6 run and seal Task 5 evidence

**Files:**
- Generate untracked external evidence: `C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-evidence\v16-2026-07-14-official-r7-score-recovery-py310\task5\**`
- No repository files change during inference/scoring.

**Interfaces:**
- Consumes the exact verified implementation commit and existing local server on `http://127.0.0.1:8111/v1`.
- Produces the selected attempt, both metric pairs, exhaustive comparisons, decisions, and receipt.

- [ ] **Step 1: Record exact pre-run state**

```powershell
$R7 = 'C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-evidence\v16-2026-07-14-official-r7-score-recovery-py310'
$Attempt = 'task5-20260714-paired-a1'
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m eval.task5_manifest snapshot --r7-root $R7 --receipt 'docs\releases\0.1.0-g0-evidence.md'
```

Expected: a clean tracked worktree, known untracked scorer checkout only, and
the G0 snapshot matches the approved receipt and six output hashes.

- [ ] **Step 2: Run Preflight and inspect its manifest before expensive inference**

```powershell
.\scripts\run_task5_paired_v16.ps1 -Stage Preflight -R7Root $R7 -AttemptId $Attempt
.\.venv\Scripts\python.exe -m eval.task5_manifest validate --manifest "$R7\task5\manifest.json" --task5-root "$R7\task5"
```

Expected: scorer commit `147cd5ac...`, current producing commit, model/mmproj,
layout, dataset, config, server, hardware, driver, and G0 identities all present
and revalidated.

- [ ] **Step 3: Run fresh Official inference**

```powershell
.\scripts\run_task5_paired_v16.ps1 -Stage Official -R7Root $R7 -AttemptId $Attempt
```

Expected: 1,651 accounted pages, 1,650 successes, the sole approved peg-native
failure, no fallback, no failed-page prediction, and a trace observability
record for every successful page.

- [ ] **Step 4: Run fresh Lightweight DirectML inference**

```powershell
.\scripts\run_task5_paired_v16.ps1 -Stage Lightweight -R7Root $R7 -AttemptId $Attempt
```

Expected: 1,651/1,651 success, fallback 0, requested `auto`, active providers
`DmlExecutionProvider` then `CPUExecutionProvider`, fallback disabled, DML node
share strictly above 50%, missing/other node events 0, transparent DML/CPU
counts, and per-page canonical traces.

- [ ] **Step 5: Score both engines with normal and CDM flows**

```powershell
.\scripts\run_task5_paired_v16.ps1 -Stage Score -R7Root $R7 -AttemptId $Attempt
```

Expected: four metric files plus summaries/provenance; Formula sample count
2,352 with zero timeout/exception; Table sample count 665 with zero
timeout/error; all metrics use page fields and notebook rounding.

- [ ] **Step 6: Compare and decide**

```powershell
.\scripts\run_task5_paired_v16.ps1 -Stage Compare -R7Root $R7 -AttemptId $Attempt
.\scripts\run_task5_paired_v16.ps1 -Stage Decide -R7Root $R7 -AttemptId $Attempt
.\.venv\Scripts\python.exe -m eval.task5_decision validate-receipt --task5-root "$R7\task5" --receipt "$R7\task5\receipt.sha256.json"
```

Expected: 1,650 paired normalized outputs, exhaustive trace counts, separate
strict/AMD verdicts, exact Official/Lightweight scores, G3 result, unchanged G0
snapshot, valid receipt, wrapper exit 0, and no orphan process. Record the
actual verdicts; never replace `UNKNOWN`/`FAIL` with a similarity claim.

- [ ] **Step 7: Apply the evidence-driven branch**

- If `strict_equivalence=PASS`, publication may say normalized outputs and
  canonical traces are 100% equivalent.
- If `strict_equivalence=UNKNOWN`, publish the exact unobservable Official
  boundaries and do not claim 100% equivalence.
- If `strict_equivalence=FAIL`, publish the first differing page/block/boundary
  counts and admit Task 6 diagnosis.
- If `g3=true`, admit the existing G4 plan on this exact manifest.
- If `g3=false`, keep G4 blocked and admit Task 6 evidence-led diagnosis.
- If evidence integrity is invalid, do not use the run for either conclusion;
  preserve it and create `task5-20260714-paired-a2` only after fixing the
  infrastructure cause.

### Task 8: Publish compact evidence, update readiness, and hand off Task 6 or G4

**Files:**
- Create: `results/omnidocbench/v16/task5/decision.json`
- Create: `results/omnidocbench/v16/task5/normalized-output.json`
- Create: `results/omnidocbench/v16/task5/trace-diff.json`
- Create: `results/omnidocbench/v16/task5/directml-attestation.json`
- Create: `results/omnidocbench/v16/task5/receipt.sha256.json`
- Create: `results/omnidocbench/v16/task5/README.md`
- Modify: `docs/releases/0.1.0-readiness.md`
- Modify: `results/omnidocbench/v16/README.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm\.superpowers\sdd\progress.md`

**Interfaces:**
- Consumes only the validated Task 5 receipt and its allowlisted small reports.
- Produces user-facing claims mechanically consistent with `decision.json`.

- [ ] **Step 1: Copy only allowlisted compact artifacts**

```powershell
$Source = 'C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-evidence\v16-2026-07-14-official-r7-score-recovery-py310\task5'
$Dest = 'results\omnidocbench\v16\task5'
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Copy-Item -LiteralPath "$Source\comparison\decision.json" -Destination "$Dest\decision.json"
Copy-Item -LiteralPath "$Source\comparison\normalized-output.json" -Destination "$Dest\normalized-output.json"
Copy-Item -LiteralPath "$Source\comparison\trace-diff.json" -Destination "$Dest\trace-diff.json"
Copy-Item -LiteralPath "$Source\comparison\directml-attestation.json" -Destination "$Dest\directml-attestation.json"
Copy-Item -LiteralPath "$Source\receipt.sha256.json" -Destination "$Dest\receipt.sha256.json"
```

Before staging, scan these files for absolute model paths, server secrets,
prompt/payload/raw-result content, and prediction text. Any finding blocks
publication and must be fixed at the report generator, not manually redacted.

- [ ] **Step 2: Update evidence and readiness wording from actual verdicts**

The Task 5 README must show the environment, exact 1,651/1,650 denominators,
Official and Lightweight component tables, strict verdict, AMD verdict, G3,
error counts, receipt SHA-256, reproduction commands, and external Linux CUDA
reference label. README/Chinese README may say “100% output equivalent” only
when the tracked strict verdict is PASS. Keep English and Chinese claims
semantically identical.

- [ ] **Step 3: Add documentation-contract tests before accepting wording**

Extend `tests/test_documentation_contract.py` so it loads tracked
`decision.json` and enforces:

```python
if decision["strict_equivalence"]["verdict"] != "PASS":
    assert "100% output equivalent" not in README.read_text(encoding="utf-8")
if decision["g3"] is not True:
    assert readiness_gate("G4") == "BLOCKED"
assert "OmniDocBench v1.6" in README.read_text(encoding="utf-8")
```

- [ ] **Step 4: Verify tracked evidence and all claims**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_documentation_contract.py tests\test_task5_decision.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
git diff --check
git status --short
```

Expected: all pass; no prediction, raw trace, model, scorer checkout, or secret
is staged.

- [ ] **Step 5: Commit compact Task 5 evidence and independently review it**

```powershell
git add results/omnidocbench/v16/task5 docs/releases/0.1.0-readiness.md results/omnidocbench/v16/README.md README.md README.zh-CN.md tests/test_documentation_contract.py
git commit -m "docs(eval): publish paired v16 Task 5 evidence"
```

Independently recompute every tracked file hash and receipt, reconstruct both
Overall values, verify denominators and DirectML counts, and compare every
public claim to `decision.json`. Record the final review hash in readiness.

- [ ] **Step 6: Update durable progress and select the next approved plan**

Mark Task 5 complete with actual verdicts and hashes. Route to Task 6 when G3
or strict equivalence fails/needs explanation; route to existing Task 8/G4 only
when G3 passes on the exact selected Task 5 manifest. Do not start either path
inside this Task 5 plan.

## Plan Self-Review

- Spec coverage: append-only r7 binding, fresh paired inference, dual official
  scoring, 1,650-page normalized parity, eight canonical boundaries,
  `FAIL > UNKNOWN > PASS`, independent AMD verdict, DirectML node proof,
  attempt preservation, receipt, tests, user claims, and Task 6/G4 routing are
  each assigned to a concrete task.
- Deferred-marker scan: no unresolved implementation markers; live outcomes are handled by
  explicit conditional rules rather than assumed values.
- Type consistency: manifest identities use `{path, bytes, sha256}` throughout;
  observation records use `{status}` or `{status, fingerprint}`; comparison
  verdicts are `PASS|FAIL|UNKNOWN`; AMD and G3 consume the same paired score
  schema; receipt paths are task5-root-relative.
- Scope: Task 5 creates evidence and decisions only. Accuracy corrections and
  performance mutations remain outside this plan.
