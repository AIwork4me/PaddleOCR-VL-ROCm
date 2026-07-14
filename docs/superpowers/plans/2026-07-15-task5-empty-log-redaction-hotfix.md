# Task 5 Empty-Log Redaction Hotfix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Task 5 command-log redaction correct for empty, whitespace-only, and JSON-null streams on PowerShell 5.1, independently approve the fix, and resume the authorized real paired run with a new immutable AttemptId.

**Architecture:** Keep redaction in the existing PowerShell runner. Treat empty/whitespace streams as text before JSON parsing, give literal JSON `null` an explicit textual representation, and execute the real native-command harness with production error semantics. Preserve failed attempt a1 and recover only through a2.

**Tech Stack:** Windows PowerShell 5.1, Python 3.10+, pytest, Ruff, SHA-256, Windows Job Objects, OmniDocBench v1.6.

## Global Constraints

- Work only in `C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm\.worktrees\top-tier-quality` on `codex/top-tier-quality`.
- Preserve `task5-20260715-paired-a1` byte-for-byte as `invalid`; never edit, delete, move, repair, or reuse it.
- Do not manually edit Task 5 stage state, manifest, receipt, candidate, compact evidence, or root pointer.
- The fix modifies only `scripts/run_task5_paired_v16.ps1` and `tests/test_run_task5_paired_v16_script.py`.
- Real recovery uses `task5-20260715-paired-a2` and the exact explicit inputs in Task 3.
- Official coverage remains exactly `1651/1650/1/0/null`; Lightweight remains exactly `1651/1651/0/0/null`.
- OmniDocBench remains v1.6; Formula uses CDM and Table uses TEDS with official page/notebook semantics.
- DirectML accepts `DmlExecutionProvider,CPUExecutionProvider`, fallback disabled, DML node share strictly greater than 0.5, CPU partitions allowed, missing/other zero.

---

### Task 1: Fix empty-stream redaction with PowerShell 5.1 TDD

**Files:**
- Modify: `tests/test_run_task5_paired_v16_script.py`
- Modify: `scripts/run_task5_paired_v16.ps1`

**Interfaces:**
- Consumes: `_native_integrity_probe(...) -> subprocess.CompletedProcess[str]`.
- Produces: `Protect-LoggedText([string]$Value)` that always returns a non-null string.

- [ ] **Step 1: Record the immutable a1 byte map**

Run:

```powershell
$A1 = 'C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-evidence\v16-2026-07-14-official-r7-score-recovery-py310\task5\attempts\task5-20260715-paired-a1'
Get-ChildItem -LiteralPath $A1 -Recurse -File | Sort-Object FullName | ForEach-Object {
  [pscustomobject]@{ Path=$_.FullName.Substring($A1.Length); Bytes=$_.Length; Sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
} | ConvertTo-Json -Depth 4
```

Expected: only the durable invalid-attempt files; save the exact output in the ignored task report and use it for the post-fix comparison.

- [ ] **Step 2: Write the failing production-semantics tests**

Change the generated `_native_integrity_probe` harness to begin with:

```powershell
$ErrorActionPreference='Stop'
```

Add:

```python
@pytest.mark.parametrize(
    ("code", "expected_stdout"),
    [
        ("", ""),
        ("import sys; sys.stderr.write('diagnostic')", ""),
        ("import sys; sys.stdout.write('ok')", "ok"),
        ("import sys; sys.stdout.write('  \\r\\n')", ""),
        ("import sys; sys.stdout.write('null')", "null"),
    ],
)
def test_logged_native_accepts_empty_whitespace_and_json_null_streams(
    tmp_path: Path, code: str, expected_stdout: str
) -> None:
    result = _native_integrity_probe(tmp_path, code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"CAPTURE={expected_stdout}" in result.stdout
    record = json.loads(
        (tmp_path / "commands" / "probe.jsonl").read_text(encoding="utf-8")
    )
    assert record["exit_code"] == 0
    assert record["orphan_audit"] == "PASS"
    persisted = (tmp_path / "commands" / "probe-native.log").read_text(
        encoding="utf-8"
    )
    assert "null-valued expression" not in persisted
```

- [ ] **Step 3: Run RED and verify the production failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_run_task5_paired_v16_script.py::test_logged_native_accepts_empty_whitespace_and_json_null_streams -q
```

Expected: FAIL on PowerShell 5.1 with `You cannot call a method on a null-valued expression` from `Protect-LoggedText`; this is the same real Preflight failure class.

- [ ] **Step 4: Implement the minimal root-cause fix**

At the start of `Protect-LoggedText`, replace the null-only branch with:

```powershell
if ([string]::IsNullOrWhiteSpace($Value)) { return [string]$Value }
```

In the whole-document success branch of `Protect-JsonFragments`, after `ConvertFrom-Json`, add:

```powershell
if ($null -eq $structured) { return "null" }
```

Do not change any other redaction rule or command lifecycle behavior.

- [ ] **Step 5: Run GREEN and focused security regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_run_task5_paired_v16_script.py -q -k "logged_native or redaction or environment"
.\.venv\Scripts\python.exe -m pytest tests\test_run_task5_paired_v16_script.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_task5_manifest.py tests\test_task5_decision.py tests\test_task5_comparison.py tests\test_directml_attestation.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_run_release_evidence_v16_script.py tests\test_release_evidence.py tests\test_release_contract.py -q
```

Expected: all pass with only documented environment/symlink skips.

- [ ] **Step 6: Run final gates and full suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check tests\test_run_task5_paired_v16_script.py
.\.venv\Scripts\python.exe -m ruff format --check tests\test_run_task5_paired_v16_script.py
git diff --check
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: Ruff/diff pass; full pytest has zero failures and only documented skips; exact probe process/marker residual is zero.

- [ ] **Step 7: Commit the focused fix**

```powershell
git add scripts/run_task5_paired_v16.ps1 tests/test_run_task5_paired_v16_script.py
git commit -m "fix(eval): handle empty Task 5 command streams"
```

Append RED/GREEN/full/residual/a1-byte-map evidence to ignored report `.superpowers/sdd/task5-empty-log-hotfix-report.md`.

### Task 2: Independently review the hotfix

**Files:**
- Modify only if review exposes a defect in Task 1.

**Interfaces:**
- Consumes: the focused Task 1 commit.
- Produces: explicit Spec/Quality Approved or Changes Requested.

- [ ] **Step 1: Generate a focused review package**

Use the pre-fix commit as base and the hotfix commit as head with the Superpowers `review-package` script. Expected: only the runner and runner test differ.

- [ ] **Step 2: Require independent executable probes**

The reviewer must independently execute PowerShell 5.1 cases for empty stdout, empty stderr, both empty, whitespace, JSON `null`, JSONL, pretty multiline JSON, and secret sentinels. Require strict records, correct log hashes, zero secret persistence, and zero residual processes.

- [ ] **Step 3: Close Critical and Important findings**

Return any Critical or Important finding to Task 1 TDD. Repeat review until Critical=0 and Important=0.

### Task 3: Resume the authorized real paired run as a2

**Files:**
- Generate external evidence only under `C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-evidence\v16-2026-07-14-official-r7-score-recovery-py310\task5\attempts\task5-20260715-paired-a2`.
- Do not modify repository files during inference/scoring.

**Interfaces:**
- Consumes: independently approved hotfix commit and healthy server `http://127.0.0.1:8111/v1`.
- Produces: sealed attempt-local compact evidence, receipt, candidate, and root selection pointer.

- [ ] **Step 1: Revalidate pre-run resources and a1 immutability**

Confirm HEAD/status, `/v1/models`, no competing evaluation clients, the exact a1 byte map, G0 snapshot, dataset/layout/config/interpreters, and absence of `task5-20260715-paired-a2` and root selection pointer.

- [ ] **Step 2: Run a2 Preflight with exact explicit inputs**

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_task5_paired_v16.ps1 `
  -Stage Preflight `
  -R7Root 'C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-evidence\v16-2026-07-14-official-r7-score-recovery-py310' `
  -AttemptId 'task5-20260715-paired-a2' `
  -PythonExe '.\.venv\Scripts\python.exe' `
  -ScorerPythonExe 'C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-scorer-v16-py310\Scripts\python.exe' `
  -ServerUrl 'http://127.0.0.1:8111/v1' `
  -ApiModelName 'PaddleOCR-VL-1.6-GGUF.gguf' `
  -DatasetDir 'C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm\data\omnidocbench\v16' `
  -LayoutModel 'C:\Users\rocm\AppData\Local\Temp\paddleocr-vl-rocm-release-gate-20260712-afaf890\models\PP-DocLayoutV3-onnx' `
  -RuntimeConfig 'C:\Users\rocm\AppData\Local\Temp\paddleocr-vl-rocm-release-gate-20260712-afaf890\config.json' `
  -G0Receipt '.\docs\releases\0.1.0-g0-evidence.md'
```

Immediately run `eval.task5_manifest validate` and inspect manifest/stage state before inference. Expected: a2 Preflight complete, manifest commit/environment/input identities valid, a1 unchanged.

- [ ] **Step 3: Run Official and monitor durable progress**

Invoke the identical command with `-Stage Official`. Monitor stage state, prediction/stats counts, service health, and process tree without modifying evidence. Expected final contract: `1651/1650/1/0/null`, only the approved peg-native failure, no fallback.

- [ ] **Step 4: Run Lightweight DirectML and monitor durable progress**

Invoke the identical command with `-Stage Lightweight`. Expected: `1651/1651/0/0/null`, requested auto, active providers exactly DML then CPU, fallback disabled, DML share >0.5, missing/other zero.

- [ ] **Step 5: Score, compare, and decide**

Invoke `-Stage Score`, then `-Stage Compare`, then `-Stage Decide` with the same arguments. Do not substitute a verdict. Record actual strict equivalence, AMD adaptation, CDM/TEDS counts/errors, metrics, and G3 decision.

- [ ] **Step 6: Verify final authority and residual state**

Run attempt-local `validate-receipt` and root `validate-selection`; rehash exact receipt paths; confirm pointer/candidate byte equality, G0 before/after equality, sealed state, exact compact topology, no forbidden root authorities, a1 byte-map identity, no command/log/lifecycle errors, and zero residual evaluation processes.

- [ ] **Step 7: Report evidence-driven outcome**

If strict is PASS, report normalized output/canonical trace equivalence. If UNKNOWN or FAIL, report it exactly. If integrity fails, preserve a2 and stop; never repair or reuse it. Do not publish or push until the verified evidence branch is reviewed.
