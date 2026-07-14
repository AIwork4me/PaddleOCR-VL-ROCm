# Task 5 Runner Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Task 5 runner review findings with attempt-local receipts, an atomic selection pointer, complete chain-of-custody revalidation, safe process handling, redacted logs, and executable end-to-end fault injection.

**Architecture:** Every attempt owns its raw work, compact evidence, immutable selection candidate, and receipt. The shared Task 5 root owns only the manifest and one create-if-absent pointer whose bytes match the receipted candidate. Python owns exact receipt/selection validation; PowerShell owns stage orchestration, environment snapshots, process-tree lifecycle, and selection finalization.

**Tech Stack:** Python 3.10+, pytest, PowerShell 5.1+, SHA-256, strict JSON/JSONL, Windows CIM process and GPU metadata, ONNX Runtime DirectML.

## Global Constraints

- Work only in `C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm\.worktrees\top-tier-quality` on `codex/top-tier-quality`.
- Never read, modify, or stage `eval/.omnidocbench/` during implementation or tests.
- Never modify the real external r7 evidence root; runtime tests use temporary stub roots only.
- OmniDocBench remains exactly v1.6; Formula is CDM, Table is TEDS, and Lightweight coverage is exactly 1,651/1,651/0/0/null.
- DirectML acceptance remains majority based: DML share strictly greater than 0.5, CPU partitions allowed, missing/other provider counts zero.
- Failed attempts are immutable and never deleted, moved, quarantined, repaired, or copied into a new attempt.
- Root-level `results/`, `comparison/`, and `receipt.sha256.json` are not evidence authorities and must not be created by the runner.
- Each task uses RED, GREEN, focused verification, a focused commit, and independent read-only review.

---

### Task 1: Validate attempt-local receipts and the atomic selection pointer

**Files:**
- Modify: `eval/task5_decision.py`
- Modify: `tests/test_task5_decision.py`

**Interfaces:**
- Produces: `required_attempt_receipt_paths(attempt_id: str) -> tuple[str, ...]`.
- Produces: `validate_task5_selection(task5_root: Path, pointer_path: Path | None = None) -> dict[str, object]`.
- Extends CLI: `python -m eval.task5_decision validate-selection --task5-root ROOT [--pointer PATH]`.
- Receipt authority: `attempts/<id>/receipt.sha256.json`; selection candidate: `attempts/<id>/selected-attempt.json`; root pointer: `selected-attempt.json`.

- [x] **Step 1: Write failing path-authority tests**

Add tests which assert that the receipt allowlist accepts only these attempt-local compact patterns:

```python
assert_receiptable("manifest.json")
assert_receiptable("attempts/a1/stage-state.json")
assert_receiptable("attempts/a1/snapshot-before.json")
assert_receiptable("attempts/a1/snapshot-after.json")
assert_receiptable("attempts/a1/selected-attempt.json")
assert_receiptable("attempts/a1/compact/results/official/metric.json")
assert_receiptable("attempts/a1/compact/results/lightweight/metric-cdm.json")
assert_receiptable("attempts/a1/compact/comparison/decision.json")
```

Reject root-level `results/**`, `comparison/**`, root `receipt.sha256.json`, attempt raw `work/**`, another attempt's path, unknown compact names, symlinks, and path escape.

- [x] **Step 2: Run RED for the new authority**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_task5_decision.py -q
```

Expected: the attempt-local compact paths are rejected and the root-level legacy paths remain incorrectly accepted.

- [x] **Step 3: Write failing selection-validation tests**

Build a complete temporary attempt and assert all of the following:

```python
validated = validate_task5_selection(task5_root)
assert validated["attempt_id"] == "a1"
assert validated["strict_equivalence"] in {"PASS", "UNKNOWN", "FAIL"}
assert validated["amd_adaptation"] in {"PASS", "FAIL"}
```

Mutation cases must fail: pointer/candidate byte mismatch; malformed AttemptId; missing/invalid receipt; receipt omits one required path; receipt includes an extra path; manifest digest mismatch; before/after G0 mismatch; candidate verdict differs from compact decision; candidate attempt differs from stage state; stage state is not sealed; symlinked attempt or pointer.

- [x] **Step 4: Implement exact attempt-local validation**

Use an exact AttemptId expression and an exact path set:

```python
ATTEMPT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

def required_attempt_receipt_paths(attempt_id: str) -> tuple[str, ...]:
    base = f"attempts/{attempt_id}"
    return tuple(sorted((
        "manifest.json",
        f"{base}/stage-state.json",
        f"{base}/snapshot-before.json",
        f"{base}/snapshot-after.json",
        f"{base}/selected-attempt.json",
        *attempt_result_paths(base, "official"),
        *attempt_result_paths(base, "lightweight"),
        *attempt_comparison_paths(base),
    )))
```

`validate_task5_selection` must stable-read strict JSON, require root pointer bytes to equal the attempt candidate bytes, validate the attempt-local receipt, require its file keys to equal `required_attempt_receipt_paths`, compare canonical G0 snapshots, compare the candidate with stage state, manifest SHA, and compact decision, and reject any absolute path disclosure in the candidate.

- [x] **Step 5: Run Task 1 GREEN and mutation checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_task5_decision.py tests\test_task5_comparison.py tests\test_directml_attestation.py -q
.\.venv\Scripts\python.exe -m ruff check eval\task5_decision.py tests\test_task5_decision.py
git diff --check
```

Expected: all pass, with only the existing Windows symlink-permission skip.

- [x] **Step 6: Commit Task 1**

```powershell
git add eval/task5_decision.py tests/test_task5_decision.py
git commit -m "fix(eval): validate attempt-local Task 5 selection"
```

### Task 2: Revalidate inputs, environment, commands, logs, and process trees

**Files:**
- Modify: `scripts/run_task5_paired_v16.ps1`
- Modify: `tests/test_run_task5_paired_v16_script.py`

**Interfaces:**
- Adds runner parameters: `[int]$CommandTimeoutSeconds = 86400`, `[int]$TerminationGraceSeconds = 10`.
- Produces an exact normalized environment object bound into the manifest and recomputed before every stage.
- `Invoke-LoggedNative` records strict command JSONL, redacted logs, timeout status, and recursive orphan audit.

- [x] **Step 1: Write executable RED tests for input and commit drift**

Create a temporary stub harness that completes Preflight, changes one bound file, and invokes Official. Parameterize over dataset manifest, layout ONNX, runtime config, worktree Python, and scorer Python. Each resumed stage must exit nonzero with `manifest integrity mismatch`. Add an existing-manifest case whose `git_commit` differs from HEAD and require rejection before inference.

- [x] **Step 2: Write executable RED tests for command-log integrity**

After a completed stub stage, mutate, delete, and exchange the raw `.log` files. Also inject duplicate JSON keys, NaN, a duplicate command name, nonzero exit, missing orphan audit, and a JSONL/log digest mismatch. The next stage must reject every case before launching its command.

- [x] **Step 3: Write executable RED tests for timeout and descendants**

Use a stub Python command that sleeps, one that leaves a direct child alive, and one that leaves a grandchild alive. Invoke with `-CommandTimeoutSeconds 1 -TerminationGraceSeconds 2`. Assert bounded completion, invalid attempt state, durable timeout/orphan evidence, and no surviving recorded PID.

- [x] **Step 4: Implement stage-start chain-of-custody validation**

At every stage entry, before any stage command, run manifest validation and compare commits:

```powershell
Invoke-LoggedNative $StageName "manifest-revalidate" $PythonExe @(
  "-m", "eval.task5_manifest", "validate",
  "--manifest", $ManifestPath,
  "--task5-root", $Task5Root
)
$manifest = Read-StrictJson $ManifestPath
if ($manifest.git_commit -cne (Get-GitCommit) -or
    $State.producing_commit -cne $manifest.git_commit) {
  throw "producing commit integrity mismatch"
}
```

Repeat the check immediately before decision sealing and receipt generation. Recompute the normalized environment object and require canonical strict-JSON equality with `manifest.environment`.

- [x] **Step 5: Bind the complete normalized environment**

The exact environment keys are:

```text
benchmark, os, machine, gpu_devices, python, scorer_python,
onnxruntime, available_providers, paddleocr, official_adapter,
lightweight_adapter, server_model_runtime
```

`gpu_devices` is a stable Name/PNPDeviceID/DriverVersion list sorted by PNPDeviceID from `Win32_VideoController`. Python package versions come from the bound interpreter. Adapter identities are full SHA-256 identities of `eval/PaddleOCRVLROCm_img2md.py`, `eval/run_eval.py`, `src/paddleocr_vl_rocm/layout.py`, and `src/paddleocr_vl_rocm/pipeline.py`. `server_model_runtime` is a redacted, canonically hashed `/v1/models` identity plus the requested API model name. Missing fields or a changed recomputation fail closed.

- [x] **Step 6: Implement strict command/log verification and redaction**

Parse every JSONL line with a strict Python helper or an equivalent duplicate-key/non-finite rejecting parser. Require exact record keys, unique command names, exit code zero, `orphan_audit == "PASS"`, a real log path relative to the attempt, and current log SHA equality.

Before disk write, replace credentials and sensitive values with deterministic markers:

```text
Authorization: <redacted>
Bearer <redacted>
api_key=<redacted>
token=<redacted>
signature=<redacted>
<absolute-model-path>
<prompt-redacted>
<payload-redacted>
<raw-result-redacted>
```

Captured stdout returned to the caller may remain raw in memory; only the redacted form is persisted. Tests must verify none of the original sentinel secrets appears anywhere below `attempts/<id>/commands`.

- [x] **Step 7: Implement bounded process-tree lifecycle**

Use timed `WaitForExit(milliseconds)`. On timeout, recursively snapshot descendants by ParentProcessId, terminate deepest descendants before the root, wait the configured grace period, rescan, and fail if any PID remains. After normal exit, perform the same recursive descendant scan and reject any survivor. Record `timed_out`, all observed descendant PIDs, termination result, exit code, and orphan verdict in the command record.

- [x] **Step 8: Run Task 2 GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_run_task5_paired_v16_script.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_run_release_evidence_v16_script.py -q
git diff --check
```

Expected: all executable drift, redaction, timeout, direct-child, and grandchild probes pass.

- [x] **Step 9: Commit Task 2**

```powershell
git add scripts/run_task5_paired_v16.ps1 tests/test_run_task5_paired_v16_script.py
git commit -m "fix(eval): harden Task 5 runner integrity"
```

### Task 3: Seal attempts locally and finalize selection atomically

**Files:**
- Modify: `scripts/run_task5_paired_v16.ps1`
- Modify: `tests/test_run_task5_paired_v16_script.py`

**Interfaces:**
- Compact authority: `attempts/<id>/compact/**`.
- Receipt authority: `attempts/<id>/receipt.sha256.json`.
- Candidate: `attempts/<id>/selected-attempt.json`.
- Global commit point: root `selected-attempt.json`, written create-if-absent and byte-equal to the candidate.

- [x] **Step 1: Write executable RED transaction tests**

Inject interruption during compact copy, receipt creation, receipt validation, and root-pointer creation. Assert:

```python
assert failed_attempt_bytes_unchanged()
assert not valid_root_selection(task5_root)
assert new_attempt_can_complete(task5_root)
assert no_file_from_failed_attempt_was_copied_into_new_attempt()
```

Also assert that an already valid root pointer blocks every later AttemptId and that pointer retry for the same already-sealed attempt changes no attempt-local byte.

- [x] **Step 2: Move compact production into the current attempt**

Write score and comparison outputs directly under `attempts/<id>/compact`; do not copy them to shared root directories. Before sealing, require exact file-name sets for both engines and comparison, and bind the output maps in final stage state.

- [x] **Step 3: Freeze candidate and stage state before the receipt**

The candidate has exactly:

```json
{
  "schema": 1,
  "attempt_id": "a1",
  "manifest_sha256": "<64 lowercase hex>",
  "strict_equivalence": "PASS|UNKNOWN|FAIL",
  "amd_adaptation": "PASS|FAIL",
  "g0_closure": "PASS",
  "effective_only_with_valid_receipt": true
}
```

Set stage state to sealed, write candidate atomically, then prohibit later writes to stage state, snapshots, candidate, or compact evidence.

- [x] **Step 4: Generate and validate the attempt-local receipt**

Pass exactly `required_attempt_receipt_paths(AttemptId)` to `eval.task5_decision receipt`, output to `attempts/<id>/receipt.sha256.json`, immediately run `validate-receipt`, then run `validate-selection` against a temporary pointer byte-equal to the candidate. Receipt failure leaves the attempt untouched and unselected.

- [x] **Step 5: Atomically create the root pointer**

Create a same-directory temporary file with candidate bytes, flush it, and rename without overwrite to root `selected-attempt.json`. Then run `validate-selection` against the real pointer. If pointer creation is interrupted, a rerun for the same sealed attempt may only repeat this create-if-absent step; it cannot rerun earlier stages. If the existing pointer is byte-identical and validates, return success idempotently; otherwise fail closed.

- [x] **Step 6: Complete the Step 7 end-to-end fault matrix**

The stub harness must execute full stage flow and inject: CPU-first provider order; zero/50% DML; missing/other provider nodes; missing profile; Official fallback; Lightweight partial coverage; stale score; CDM timeout; TEDS error; direct/grandchild orphan; command/log/output/G0/input/manifest drift; old-attempt reuse; compact interruption; receipt creation/validation/mutation; pointer interruption; strict UNKNOWN; AMD FAIL. Assert process exit, immutable state, decision durability, selected/receipt validity, and absence of false PASS for every row.

- [x] **Step 7: Run Task 3 GREEN and the full offline gate**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_task5_manifest.py tests\test_task5_comparison.py tests\test_directml_attestation.py tests\test_task5_decision.py tests\test_run_task5_paired_v16_script.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_run_release_evidence_v16_script.py tests\test_release_evidence.py tests\test_release_contract.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check eval\task5_decision.py tests\test_task5_decision.py tests\test_run_task5_paired_v16_script.py
.\.venv\Scripts\python.exe -m ruff format --check eval\task5_decision.py tests\test_task5_decision.py tests\test_run_task5_paired_v16_script.py
git diff --check
```

Expected: all pass with only previously documented environment skips. No real inference or external r7 mutation occurs.

- [x] **Step 8: Commit Task 3**

```powershell
git add scripts/run_task5_paired_v16.ps1 tests/test_run_task5_paired_v16_script.py
git commit -m "fix(eval): make Task 5 selection transactional"
```

### Task 4: Independent final review and real-run authorization gate

**Files:**
- Modify only when review or verification exposes a defect in Tasks 1-3.

**Interfaces:**
- Consumes all remediated Task 5 code.
- Produces an explicit offline `Approved` or `Changes requested`; it does not run the real 1,651-page workload.

- [x] **Step 1: Generate one review package from `54e05e7` through remediation HEAD**

```powershell
& 'C:\Program Files\Git\bin\bash.exe' `
  'C:/Users/rocm/.codex/plugins/cache/openai-curated-remote/superpowers/6.1.1/skills/subagent-driven-development/scripts/review-package' `
  54e05e7913676b3efa9e3174b1c6499e77f0a7be HEAD `
  '.superpowers/sdd/task5-runner-remediation-review.diff'
```

- [x] **Step 2: Require independent runtime-probe review**

The reviewer must independently reproduce input drift rejection, log mutation rejection, timeout/descendant cleanup, receipt-failure recovery with a new AttemptId, pointer idempotence, strict UNKNOWN, AMD FAIL, and 1101/150 DirectML PASS. Static token inspection alone is insufficient.

- [x] **Step 3: Close all Critical and Important findings**

Return each finding to the responsible task's TDD cycle, commit the focused correction, regenerate the review package, and repeat until no Critical or Important remains.

- [x] **Step 4: Authorize the next phase only after clean review**

Update both SDD ledgers and mark the original Task 5 runner complete only when offline verification is green, the worktree contains only the allowed `eval/.omnidocbench/`, and the independent reviewer explicitly approves. The next phase is a stubbed Preflight/selection rehearsal followed by the separately authorized real paired run.
