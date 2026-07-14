from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

import eval.task5_manifest as task5_manifest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_task5_paired_v16.ps1"
POWERSHELL = Path(os.environ["WINDIR"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_runner_uses_separate_task5_manifest_and_never_calls_g0_runner() -> None:
    text = _text()
    assert 'Join-Path $R7Root "task5"' in text
    assert "run_release_evidence_v16.ps1" not in text
    assert "Remove-Item $R7Root" not in text
    assert "snapshot-before.json" in text
    assert "snapshot-after.json" in text


def test_runner_exposes_only_approved_stages_and_attempt_id_contract() -> None:
    text = _text()
    assert '[ValidateSet("Preflight", "Official", "Lightweight", "Score", "Compare", "Decide", "All")]' in text
    assert "^[a-z0-9][a-z0-9-]{0,63}$" in text
    assert 'Join-Path (Join-Path $Task5Root "attempts") $AttemptId' in text


def test_runner_never_deletes_or_reuses_attempts() -> None:
    text = _text()
    assert "Attempt already exists; use a new AttemptId" in text
    assert "Clear-StageOutputs" not in text
    assert "Remove-Item -Recurse" not in text
    assert "old-attempt file reuse" in text


def test_both_engines_run_fresh_inference_without_cross_engine_fallback() -> None:
    text = _text()
    assert text.count('"--stage", "infer"') == 2
    assert '"--engine", "official"' in text
    assert '"--engine", "lightweight"' in text
    assert '"--trace-dir"' in text
    assert '"--layout-profile-prefix"' in text
    assert "--fallback-pred-dir" not in text


def test_lightweight_and_official_both_run_non_cdm_and_cdm_scoring() -> None:
    text = _text()
    assert text.count('"--cdm"') >= 2
    assert 'results/official/metric-cdm.json' in text
    assert 'results/lightweight/metric-cdm.json' in text
    assert 'results/official/metric.json' in text
    assert 'results/lightweight/metric.json' in text


def test_runner_pins_v16_and_explicit_page_contracts() -> None:
    text = _text()
    assert "OmniDocBench-v1.6" in text
    assert "v1.7" not in text
    assert "1651" in text and "1650" in text
    assert "peg-native" in text
    assert "Lightweight coverage integrity mismatch" in text
    assert "Official coverage integrity mismatch" in text


def test_directml_gate_is_majority_not_zero_cpu() -> None:
    text = _text()
    assert '"DmlExecutionProvider", "CPUExecutionProvider"' in text
    assert "dml_node_share -le 0.5" in text
    assert "cpu_node_events -ne 0" not in text
    assert "missing_provider_node_events" in text
    assert "other_provider_node_events" in text
    assert '"--allow-fail-verdict"' in text
    assert 'if ($measuredDecision.amd_adaptation.verdict -eq "PASS")' in text


def test_runner_invokes_comparison_attestation_decision_and_receipt() -> None:
    text = _text()
    assert "eval.task5_comparison" in text
    assert "scripts/compare_inference_traces.py" in text
    assert "eval.directml_attestation" in text
    assert '"eval.task5_decision", "decide"' in text
    assert '"eval.task5_decision", "receipt"' in text
    assert '"eval.task5_decision", "validate-receipt"' in text


def test_receipt_uses_only_task4_allowlisted_compact_files() -> None:
    text = _text()
    for relative in (
        "manifest.json",
        "selected-attempt.json",
        "attempts/$AttemptId/stage-state.json",
        "attempts/$AttemptId/snapshot-before.json",
        "attempts/$AttemptId/snapshot-after.json",
        "results/official/metric.json",
        "results/official/metric-cdm.json",
        "results/lightweight/metric.json",
        "results/lightweight/metric-cdm.json",
        "comparison/normalized-output.json",
        "comparison/trace-diff.json",
        "comparison/directml-attestation.json",
        "comparison/decision.json",
    ):
        assert relative in text
    assert "paired-official/*.md" not in text
    assert "traces/official/*.jsonl" not in text


def test_resume_integrity_binds_commit_manifest_commands_outputs_and_g0() -> None:
    text = _text()
    for token in (
        "producing_commit",
        "manifest_sha256",
        "command_log_sha256",
        "output_map_sha256",
        "snapshot-before.json",
        "snapshot-after.json",
        "G0 integrity mismatch",
        "orphan audit",
    ):
        assert token in text


def test_fault_gates_are_fail_closed_and_decisions_remain_measured_results() -> None:
    text = _text()
    for token in (
        "CPU-first provider order",
        "DirectML node share must be strictly greater than 0.5",
        "missing profile",
        "Official fallback",
        "partial coverage",
        "stale score",
        "CDM timeout",
        "TEDS error",
        "orphan process",
        "command log integrity mismatch",
        "output integrity mismatch",
        "receipt mutation",
    ):
        assert token in text
    assert "strict_equivalence=$strictVerdict" in text
    assert "amd_adaptation=$amdVerdict" in text


def test_powershell_parser_accepts_script() -> None:
    command = (
        "$errors=$null; [void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{SCRIPT}', [ref]$null, [ref]$errors); if($errors.Count){{ $errors | % ToString; exit 1 }}"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _provider_probe(tmp_path: Path, stats: dict[str, object], report: dict[str, object]) -> subprocess.CompletedProcess[str]:
    work = tmp_path / "work"
    lightweight = work / "lightweight"
    lightweight.mkdir(parents=True)
    (lightweight / "_run_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    command = rf"""
$errors=$null
$tokens=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('{SCRIPT}',[ref]$tokens,[ref]$errors)
if($errors.Count){{exit 90}}
foreach($name in @('Read-Json','Assert-ProviderMajority')){{
  $fn=$ast.Find({{param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name}},$true)
  Invoke-Expression $fn.Extent.Text
}}
$WorkRoot='{work}'
$ApprovedProviders=@('DmlExecutionProvider','CPUExecutionProvider')
$report='{json.dumps(report, separators=(",", ":"))}' | ConvertFrom-Json
Assert-ProviderMajority $report
"""
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _valid_stats() -> dict[str, object]:
    return {
        "layout_provider_requested": "auto",
        "layout_providers_active": ["DmlExecutionProvider", "CPUExecutionProvider"],
        "layout_fallback_disabled": True,
    }


def _valid_attestation() -> dict[str, object]:
    return {
        "verdict": "PASS",
        "dml_node_events": 1101,
        "cpu_node_events": 150,
        "dml_node_share": 1101 / 1251,
        "missing_provider_node_events": 0,
        "other_provider_node_events": 0,
    }


def test_directml_probe_accepts_majority_with_transparent_cpu_partitions(tmp_path: Path) -> None:
    result = _provider_probe(tmp_path, _valid_stats(), _valid_attestation())
    assert result.returncode == 0, result.stdout + result.stderr


def test_directml_probe_rejects_cpu_first_provider_order(tmp_path: Path) -> None:
    stats = _valid_stats()
    stats["layout_providers_active"] = ["CPUExecutionProvider", "DmlExecutionProvider"]
    result = _provider_probe(tmp_path, stats, _valid_attestation())
    assert result.returncode != 0
    assert "CPU-first" in result.stdout + result.stderr


def test_directml_probe_rejects_at_most_half_dml(tmp_path: Path) -> None:
    report = _valid_attestation()
    report.update({"dml_node_events": 50, "cpu_node_events": 50, "dml_node_share": 0.5})
    result = _provider_probe(tmp_path, _valid_stats(), report)
    assert result.returncode != 0
    assert "strictly greater" in result.stdout + result.stderr


def test_directml_probe_rejects_missing_or_other_provider_nodes(tmp_path: Path) -> None:
    for key in ("missing_provider_node_events", "other_provider_node_events"):
        report = _valid_attestation()
        report[key] = 1
        result = _provider_probe(tmp_path / key, _valid_stats(), report)
        assert result.returncode != 0
        assert "missing/other" in result.stdout + result.stderr


def _coverage_probe(function_name: str, stats: dict[str, object]) -> subprocess.CompletedProcess[str]:
    command = rf"""
$errors=$null
$tokens=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('{SCRIPT}',[ref]$tokens,[ref]$errors)
if($errors.Count){{exit 90}}
$fn=$ast.Find({{param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq '{function_name}'}},$true)
if($null -eq $fn){{Write-Error 'coverage function missing'; exit 91}}
Invoke-Expression $fn.Extent.Text
$stats='{json.dumps(stats, separators=(",", ":"))}' | ConvertFrom-Json
{function_name} $stats
"""
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_official_coverage_probe_rejects_fallback() -> None:
    result = _coverage_probe(
        "Assert-OfficialCoverage",
        {"count": 1651, "ok": 1650, "fail": 1, "fallback": 1, "limit_pages": None},
    )
    assert result.returncode != 0
    assert "Official coverage" in result.stdout + result.stderr


def test_lightweight_coverage_probe_requires_exact_full_corpus() -> None:
    valid = {"count": 1651, "ok": 1651, "fail": 0, "fallback": 0, "limit_pages": None}
    assert _coverage_probe("Assert-LightweightCoverage", valid).returncode == 0
    for change in ({"ok": 1650, "fail": 1}, {"fallback": 1}, {"limit_pages": 1651}):
        invalid = {**valid, **change}
        result = _coverage_probe("Assert-LightweightCoverage", invalid)
        assert result.returncode != 0
        assert "Lightweight coverage" in result.stdout + result.stderr


def test_logged_native_preserves_spaces_quotes_and_strict_stdout(tmp_path: Path) -> None:
    harness = tmp_path / "native-probe.ps1"
    command_root = tmp_path / "commands"
    command_root.mkdir()
    harness.write_text(
        rf"""
$errors=$null
$tokens=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('{SCRIPT}',[ref]$tokens,[ref]$errors)
if($errors.Count){{exit 90}}
foreach($name in @('Get-Sha256','Get-StringSha256','Write-AtomicText','Add-CommandRecord','ConvertTo-NativeArgument','Protect-LoggedText','Initialize-NativeJobRunner','Invoke-LoggedNative')){{
  $fn=$ast.Find({{param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name}},$true)
  Invoke-Expression $fn.Extent.Text
}}
$RepoRoot='{ROOT}'
$AttemptRoot='{tmp_path}'
$CommandRoot='{command_root}'
$CommandTimeoutSeconds=5
$TerminationGraceSeconds=2
$python='{ROOT / ".venv/Scripts/python.exe"}'
$code='import json; print(json.dumps({{"space value":"quoted ok"}}))'
$captured=Invoke-LoggedNative 'Probe' 'quoted' $python @('-c',$code) -Capture
if($captured -ne '{{"space value": "quoted ok"}}'){{throw "capture mismatch: $captured"}}
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_logged_native_executes_exact_argv_without_shell_wrapper(tmp_path: Path) -> None:
    arguments = [
        "%TASK5_ARGV_SENTINEL%", ">", "&", "|", "^", 'embedded"quote',
        "trailing\\", "", "space value",
    ]
    source = tmp_path / "argv-probe.cs"
    executable = tmp_path / "argv-probe.exe"
    argv_output = tmp_path / "argv-output.txt"
    source.write_text(
        "using System; using System.IO; using System.Linq; using System.Text; "
        "public static class P { public static int Main(string[] a) { "
        "File.WriteAllText(a[0], String.Join(\",\", a.Skip(1).Select(x=>Convert.ToBase64String(Encoding.UTF8.GetBytes(x)))), new UTF8Encoding(false)); "
        "return 0; } }",
        encoding="utf-8",
    )
    compile_result = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", f"Add-Type -Path '{source}' -OutputAssembly '{executable}' -OutputType WindowsApplication"],
        cwd=tmp_path, text=True, capture_output=True, check=False,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    files_before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    env = os.environ.copy()
    env["TASK5_ARGV_SENTINEL"] = "EXPANDED-BY-SHELL"
    result = _native_integrity_probe(
        tmp_path,
        "",
        executable=str(executable),
        argument_list=[str(argv_output), *arguments],
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    encoded_argv = argv_output.read_text(encoding="utf-8")
    assert [base64.b64decode(value).decode() for value in encoded_argv.split(",")] == arguments
    record = json.loads((tmp_path / "commands" / "probe.jsonl").read_text(encoding="utf-8"))
    assert record["descendant_pids"] == []
    script_text = _text()
    assert "CreateProcessW(executable,cmd" in script_text
    assert "ComSpec" not in script_text and '"cmd.exe"' not in script_text
    expected_argv = [str(argv_output), *arguments]
    assert record["arguments_sha256"] == hashlib.sha256("\0".join(expected_argv).encode()).hexdigest()
    assert not (tmp_path / "commands" / "EXPANDED-BY-SHELL").exists()
    files_after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    assert files_after - files_before == {
        Path("argv-output.txt"), Path("integrity-probe.ps1"),
        Path("commands/probe-native.log"), Path("commands/probe.jsonl"),
    }


def test_runner_binds_exact_environment_and_stage_integrity_contract() -> None:
    text = _text()
    assert "[int]$CommandTimeoutSeconds = 86400" in text
    assert "[int]$TerminationGraceSeconds = 10" in text
    for key in (
        "benchmark", "os", "machine", "gpu_devices", "python", "scorer_python",
        "onnxruntime", "available_providers", "paddleocr", "official_adapter",
        "lightweight_adapter", "server_model_runtime",
    ):
        assert f"{key} =" in text
    assert "Assert-StageStartIntegrity" in text
    assert "manifest-revalidate" in text
    assert "manifest integrity mismatch" in text


def _native_integrity_probe(
    tmp_path: Path,
    code: str,
    *,
    timeout: int = 5,
    grace: int = 2,
    executable: str | None = None,
    argument_list: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    harness = tmp_path / "integrity-probe.ps1"
    command_root = tmp_path / "commands"
    command_root.mkdir()
    function_names = (
        "Get-Sha256", "Get-StringSha256", "Write-AtomicText", "Add-CommandRecord",
        "ConvertTo-NativeArgument", "Protect-LoggedText", "Initialize-NativeJobRunner",
        "Invoke-LoggedNative",
    )
    imports = ",".join(f"'{name}'" for name in function_names)
    native_executable = executable or str(ROOT / ".venv/Scripts/python.exe")
    native_arguments = argument_list or ["-c", code]
    powershell_arguments = ",".join("'" + value.replace("'", "''") + "'" for value in native_arguments)
    harness.write_text(
        rf"""
$errors=$null
$tokens=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('{SCRIPT}',[ref]$tokens,[ref]$errors)
if($errors.Count){{exit 90}}
foreach($name in @({imports})){{
  $fn=$ast.Find({{param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name}},$true)
  if($null -eq $fn){{throw "missing function: $name"}}
  Invoke-Expression $fn.Extent.Text
}}
$RepoRoot='{ROOT}'
$AttemptRoot='{tmp_path}'
$CommandRoot='{command_root}'
$CommandTimeoutSeconds={timeout}
$TerminationGraceSeconds={grace}
$nativeExecutable='{native_executable}'
$captured=Invoke-LoggedNative 'Probe' 'native' $nativeExecutable @({powershell_arguments}) -Capture
Write-Output "CAPTURE=$captured"
""",
        encoding="utf-8",
    )
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
        cwd=ROOT, text=True, capture_output=True, check=False, timeout=30, env=env,
    )


def test_logged_native_redacts_before_disk_but_capture_remains_raw(tmp_path: Path) -> None:
    secrets = "Bearer S3CRET api_key=KEY123 token=T0K signature=SIG prompt=PRIVATE"
    result = _native_integrity_probe(tmp_path, f"print({secrets!r})")
    assert result.returncode == 0, result.stdout + result.stderr
    assert secrets in result.stdout
    disk = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "commands").iterdir())
    for sentinel in ("S3CRET", "KEY123", "T0K", "SIG", "PRIVATE"):
        assert sentinel not in disk
    assert "<redacted>" in disk and "<prompt-redacted>" in disk


def test_structured_redaction_recurses_and_scrubs_free_text(tmp_path: Path) -> None:
    sample = {
        "authorization": "Bearer AUTH-SENTINEL",
        "nested": [{"api_key": "KEY-SENTINEL", "payload": {"raw_result": "RAW-SENTINEL"}}],
        "safe": "https://host/model.bin?X-Amz-Signature=SIGNED-SENTINEL&token=QUERY-SENTINEL",
        "path": r"C:\models\private-model.gguf",
        "headers": {"Authorization": "Bearer HEADER-SENTINEL"},
        "prompt": "PROMPT-SENTINEL",
    }
    result = _native_integrity_probe(tmp_path, f"import json; print(json.dumps({sample!r}))")
    assert result.returncode == 0, result.stdout + result.stderr
    disk = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "commands").iterdir())
    for sentinel in ("AUTH-SENTINEL", "KEY-SENTINEL", "RAW-SENTINEL", "SIGNED-SENTINEL", "QUERY-SENTINEL", "private-model.gguf", "HEADER-SENTINEL", "PROMPT-SENTINEL"):
        assert sentinel not in disk


def test_redaction_handles_jsonl_embedded_json_and_scans_entire_command_tree(tmp_path: Path) -> None:
    sentinels = {
        "AUTH-JSONL", "BEARER-JSONL", "API-EMBEDDED", "TOKEN-NESTED",
        "CREDENTIAL-ARRAY", "SIG-AMZ", "SIG-SIGNED-URL", "PROMPT-TEXT",
        "PAYLOAD-NESTED", "RAW-RESULT-TEXT",
    }
    lines = [
        '{"authorization":"Bearer AUTH-JSONL","bearer":"BEARER-JSONL"}',
        'prefix {"api_key":"API-EMBEDDED","nested":[{"token":"TOKEN-NESTED"},'
        '{"credential":"CREDENTIAL-ARRAY","payload":"PAYLOAD-NESTED"}]} suffix',
        'signed=https://host/item?X-Amz-Signature=SIG-AMZ&signature=SIG-SIGNED-URL',
        'prompt: PROMPT-TEXT',
        'raw result: RAW-RESULT-TEXT',
    ]
    code = "import sys; print('\\n'.join(sys.argv[1:]))"
    result = _native_integrity_probe(tmp_path, code, argument_list=["-c", code, *lines])
    assert result.returncode == 0, result.stdout + result.stderr
    persisted = "".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (tmp_path / "commands").rglob("*") if path.is_file()
    )
    for sentinel in sentinels:
        assert sentinel not in persisted
    assert "<redacted>" in persisted
    assert "<prompt-redacted>" in persisted
    assert "<payload-redacted>" in persisted
    assert "<raw-result-redacted>" in persisted


def test_server_environment_never_persists_raw_requested_model_and_rejects_empty_gpu() -> None:
    text = _text()
    assert "requested_model = $ApiModelName" not in text
    assert "requested_model_sha256" in text
    assert "GPU environment identity is empty" in text


def _environment_contract_probe(
    tmp_path: Path, *, empty_gpu: bool = False, model_name: str = r"C:\secret\MODEL-SENTINEL.gguf",
    require_strict_server_json: bool = False,
) -> subprocess.CompletedProcess[str]:
    tmp_path.mkdir(parents=True)
    harness = tmp_path / "environment.ps1"
    gpu_body = "return @()" if empty_gpu else "return @([pscustomobject]@{Name='AMD Radeon';PNPDeviceID='PCI\\VEN_1002';DriverVersion='1.2.3'})"
    harness.write_text(
        rf"""
$ErrorActionPreference='Stop';$errors=$null;$tokens=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('{SCRIPT}',[ref]$tokens,[ref]$errors)
foreach($name in @('Get-Sha256','Get-StringSha256','Get-EnvironmentContract')){{
 $fn=$ast.Find({{param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name}},$true);Invoke-Expression $fn.Extent.Text
}}
$RepoRoot='{ROOT}';$Benchmark='OmniDocBench-v1.6';$PythonExe='python-stub.exe';$ScorerPythonExe='scorer-stub.exe';$ServerUrl='http://stub/v1';$ApiModelName='{model_name}'
function Get-CimInstance{{param([Parameter(ValueFromRemainingArguments=$true)]$rest);{gpu_body}}}
function Invoke-DirectPython{{param([string]$Code,[string[]]$Arguments)
 if($Code -like '*urllib.request*'){{
  if({'$true' if require_strict_server_json else '$false'} -and ($Code -notlike '*object_pairs_hook*' -or $Code -notlike '*parse_constant*')){{throw 'server JSON parser is not strict'}}
  return '{{"models_sha256":"{('c'*64)}"}}'
 }}
 if($Code -like '*onnxruntime*'){{return '{{"version":"1.0","available_providers":["DmlExecutionProvider","CPUExecutionProvider"]}}'}}
 if($Code -like '*importlib.metadata*'){{return '{{"version":"1.0"}}'}}
 return '{{"version":"3.10","executable_sha256":"{('a'*64)}"}}'
}}
Get-EnvironmentContract | ConvertTo-Json -Compress -Depth 20
""",
        encoding="utf-8",
    )
    return subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)], cwd=ROOT, text=True, capture_output=True, check=False)


def test_actual_environment_helper_redacts_model_and_validates_gpu(tmp_path: Path) -> None:
    valid = _environment_contract_probe(tmp_path / "valid")
    assert valid.returncode == 0, valid.stdout + valid.stderr
    assert "MODEL-SENTINEL" not in valid.stdout
    environment = json.loads(valid.stdout)
    assert set(environment) == {
        "benchmark", "os", "machine", "gpu_devices", "python", "scorer_python",
        "onnxruntime", "available_providers", "paddleocr", "official_adapter",
        "lightweight_adapter", "server_model_runtime",
    }
    assert environment["server_model_runtime"]["requested_model"] == "<redacted>"
    assert len(environment["server_model_runtime"]["requested_model_sha256"]) == 64
    empty = _environment_contract_probe(tmp_path / "empty", empty_gpu=True)
    assert empty.returncode != 0
    assert "GPU environment identity is empty" in empty.stdout + empty.stderr


def test_environment_model_identity_is_exact_trimmed_utf8_and_server_json_is_strict(tmp_path: Path) -> None:
    model = "  CaseSensitive-Model  "
    result = _environment_contract_probe(
        tmp_path / "strict", model_name=model, require_strict_server_json=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    environment = json.loads(result.stdout)
    expected = hashlib.sha256(model.strip().encode("utf-8")).hexdigest()
    lowercased = hashlib.sha256(model.strip().lower().encode("utf-8")).hexdigest()
    assert environment["server_model_runtime"]["requested_model_sha256"] == expected
    assert expected != lowercased


def test_logged_native_timeout_is_bounded_and_durable(tmp_path: Path) -> None:
    started = time.monotonic()
    result = _native_integrity_probe(tmp_path, "import time; time.sleep(30)", timeout=1, grace=2)
    elapsed = time.monotonic() - started
    assert result.returncode != 0
    assert elapsed < 15
    record = json.loads((tmp_path / "commands" / "probe.jsonl").read_text(encoding="utf-8"))
    assert record["timed_out"] is True
    assert record["orphan_audit"] == "PASS"
    assert record["termination_result"] == "terminated"


def test_logged_native_kills_surviving_grandchild_and_records_tree(tmp_path: Path) -> None:
    grandchild_path = tmp_path / "grandchild.py"
    child_path = tmp_path / "child.py"
    grandchild_path.write_text("import time; time.sleep(30)\n", encoding="utf-8")
    child_path.write_text(
        f"import subprocess,sys,time; subprocess.Popen([sys.executable,{str(grandchild_path)!r}]); time.sleep(30)\n",
        encoding="utf-8",
    )
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,{str(child_path)!r}]); time.sleep(30)"
    result = _native_integrity_probe(tmp_path, parent, timeout=1, grace=1)
    assert result.returncode != 0
    record = json.loads((tmp_path / "commands" / "probe.jsonl").read_text(encoding="utf-8"))
    assert len(record["descendant_pids"]) >= 2
    assert record["orphan_audit"] == "PASS"
    assert record["termination_result"] == "terminated"


def test_logged_native_rejects_child_surviving_normal_root_exit(tmp_path: Path) -> None:
    child_path = tmp_path / "child.ps1"
    root_path = tmp_path / "root.ps1"
    child_path.write_text("Start-Sleep -Seconds 20\n", encoding="utf-8")
    root_path.write_text(
        f"Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile','-File','{child_path}')\n",
        encoding="utf-8",
    )
    result = _native_integrity_probe(
        tmp_path,
        "",
        timeout=5,
        grace=2,
        executable=str(POWERSHELL),
        argument_list=["-NoProfile", "-File", str(root_path)],
    )
    assert result.returncode != 0
    assert "orphan process observed after normal command exit" in result.stdout + result.stderr
    record = json.loads((tmp_path / "commands" / "probe.jsonl").read_text(encoding="utf-8"))
    assert record["timed_out"] is False
    assert record["descendant_pids"]
    assert record["termination_result"] == "terminated"


def test_job_contains_fast_launcher_escape_before_root_executes(tmp_path: Path) -> None:
    marker = "task5-fast-launcher-escape-7f2c"
    grandchild = tmp_path / f"{marker}-grandchild.ps1"
    launcher = tmp_path / f"{marker}-launcher.ps1"
    root_script = tmp_path / f"{marker}-root.ps1"
    grandchild.write_text("Start-Sleep -Seconds 60\n", encoding="utf-8")
    launcher.write_text(
        f"Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile','-File','{grandchild}')\n",
        encoding="utf-8",
    )
    root_script.write_text(
        f"Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile','-File','{launcher}')\n",
        encoding="utf-8",
    )
    result = _native_integrity_probe(
        tmp_path,
        "",
        timeout=5,
        grace=2,
        executable=str(POWERSHELL),
        argument_list=["-NoProfile", "-File", str(root_script)],
    )
    assert result.returncode != 0
    assert (
        "orphan process observed after normal command exit" in result.stdout + result.stderr
        or "Command timeout" in result.stdout + result.stderr
    )
    record = json.loads((tmp_path / "commands" / "probe.jsonl").read_text(encoding="utf-8"))
    assert record["job_active_count"] == 0
    assert record["survivor_pids"] == []
    assert record["termination_result"] == "terminated"
    audit = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", f"@(Get-CimInstance Win32_Process | ? {{ $_.CommandLine -like '*{marker}*' }}).Count"],
        text=True, capture_output=True, check=False,
    )
    # The audit shell contains the marker in its own command line; no second PID may survive.
    assert audit.returncode == 0 and audit.stdout.strip() == "1"


def _command_log_probe(tmp_path: Path, record_text: str) -> subprocess.CompletedProcess[str]:
    attempt = tmp_path / "attempt"
    commands = attempt / "commands"
    commands.mkdir(parents=True)
    (commands / "probe.log").write_bytes(b"safe\n")
    jsonl = commands / "probe.jsonl"
    jsonl.write_text(record_text, encoding="utf-8")
    harness = tmp_path / "command-log-probe.ps1"
    harness.write_text(
        rf"""
$errors=$null;$tokens=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('{SCRIPT}',[ref]$tokens,[ref]$errors)
foreach($name in @('Get-Sha256','Get-StringSha256','ConvertTo-NativeArgument','Initialize-NativeJobRunner','Invoke-DirectPython','Assert-CommandLogIntegrity')){{
  $fn=$ast.Find({{param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name}},$true)
  Invoke-Expression $fn.Extent.Text
}}
$PythonExe='{ROOT / ".venv/Scripts/python.exe"}'
$RepoRoot='{ROOT}'
$AttemptRoot='{attempt}'
Assert-CommandLogIntegrity '{jsonl}'
""",
        encoding="utf-8",
    )
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )


def _valid_command_record() -> dict[str, object]:
    digest = hashlib.sha256(b"safe\n").hexdigest()
    return {
        "name": "probe", "executable_sha256": "a" * 64, "arguments_sha256": "b" * 64,
        "started_at_utc": "2026-07-14T00:00:00Z", "ended_at_utc": "2026-07-14T00:00:01Z",
        "exit_code": 0, "timed_out": False, "descendant_pids": [],
        "survivor_pids": [], "job_active_count": 0,
        "termination_result": "not-required", "orphan_audit": "PASS",
        "log_path": "commands/probe.log", "log_sha256": digest,
    }


def test_command_log_integrity_helper_accepts_exact_record(tmp_path: Path) -> None:
    result = _command_log_probe(tmp_path, json.dumps(_valid_command_record()) + "\n")
    assert result.returncode == 0, result.stdout + result.stderr


def test_command_log_integrity_helper_rejects_semantic_and_jsonl_mutations(tmp_path: Path) -> None:
    base = _valid_command_record()
    mutations = {
        "nonzero": json.dumps({**base, "exit_code": 1}) + "\n",
        "wrong_type": json.dumps({**base, "exit_code": "0"}) + "\n",
        "timeout": json.dumps({**base, "timed_out": True}) + "\n",
        "orphan": json.dumps({**base, "orphan_audit": "FAIL"}) + "\n",
        "digest": json.dumps({**base, "log_sha256": "0" * 64}) + "\n",
        "duplicate_name": json.dumps(base) + "\n" + json.dumps(base) + "\n",
        "duplicate_key": json.dumps(base)[:-1] + ',"name":"again"}\n',
        "nan": json.dumps(base)[:-1] + ',"extra":NaN}\n',
        "false_orphan_pass": json.dumps({**base, "survivor_pids": [123], "job_active_count": 1}) + "\n",
        "not_required_descendant": json.dumps({**base, "descendant_pids": [123]}) + "\n",
        "terminated_without_descendant": json.dumps({**base, "termination_result": "terminated"}) + "\n",
        "survivors_marked_pass": json.dumps({**base, "termination_result": "survivors", "survivor_pids": [123], "job_active_count": 1}) + "\n",
        "timeout_exit_zero": json.dumps({**base, "timed_out": True, "termination_result": "root-terminated"}) + "\n",
        "duplicate_pid": json.dumps({**base, "descendant_pids": [123, 123], "termination_result": "terminated"}) + "\n",
    }
    for name, text in mutations.items():
        result = _command_log_probe(tmp_path / name, text)
        assert result.returncode != 0, name


def test_next_stage_precheck_rejects_rehashed_but_semantically_corrupt_log(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    commands = attempt / "commands"
    commands.mkdir(parents=True)
    (commands / "official-probe.log").write_bytes(b"mutated\n")
    record = _valid_command_record()
    record.update({"name": "probe", "log_path": "commands/official-probe.log"})
    jsonl = commands / "official.jsonl"
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"{}\n")
    command_sha = hashlib.sha256(jsonl.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    state = {
        "producing_commit": commit,
        "manifest_sha256": manifest_sha,
        "stages": {"Official": {"command_log_sha256": command_sha, "output_roots": ["ignored"], "output_map_sha256": "ok"}},
    }
    harness = tmp_path / "next-stage.ps1"
    harness.write_text(
        rf"""
$errors=$null;$tokens=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('{SCRIPT}',[ref]$tokens,[ref]$errors)
foreach($name in @('Get-Sha256','Get-StringSha256','ConvertTo-NativeArgument','Initialize-NativeJobRunner','Invoke-DirectPython','Assert-CommandLogIntegrity','Get-GitCommit','Assert-RecordedStagesIntegrity')){{
  $fn=$ast.Find({{param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name}},$true)
  Invoke-Expression $fn.Extent.Text
}}
function Get-OutputMap([string[]]$Roots){{return @{{}}}}
function Get-OutputMapSha([object]$Map){{return 'ok'}}
$PythonExe='{ROOT / ".venv/Scripts/python.exe"}'
$RepoRoot='{ROOT}'
$AttemptRoot='{attempt}';$CommandRoot='{commands}';$ManifestPath='{manifest}'
$state='{json.dumps(state, separators=(',', ':'))}' | ConvertFrom-Json
Assert-RecordedStagesIntegrity $state
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "digest mismatch" in result.stdout + result.stderr


def test_current_stage_log_is_strictly_verified_before_state_hash(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    commands = attempt / "commands"
    commands.mkdir(parents=True)
    (commands / "probe.log").write_bytes(b"safe\n")
    record = {**_valid_command_record(), "timed_out": True, "termination_result": "not-required"}
    (commands / "probe.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    harness = tmp_path / "current-stage.ps1"
    harness.write_text(
        rf"""
$ErrorActionPreference='Stop';$errors=$null;$tokens=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('{SCRIPT}',[ref]$tokens,[ref]$errors)
foreach($name in @('Get-Sha256','Get-StringSha256','ConvertTo-NativeArgument','Initialize-NativeJobRunner','Invoke-DirectPython','Assert-CommandLogIntegrity','Invoke-DurableStage')){{
 $fn=$ast.Find({{param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name}},$true);Invoke-Expression $fn.Extent.Text
}}
$PythonExe='{ROOT / ".venv/Scripts/python.exe"}';$RepoRoot='{ROOT}';$AttemptRoot='{attempt}';$CommandRoot='{commands}';$StageStatePath='{attempt / 'state.json'}'
$script:state=[pscustomobject]@{{status='active';stages=[pscustomobject]@{{}}}}
function Read-State{{return $script:state}};function Assert-RecordedStagesIntegrity{{param($s)}};function Assert-StageStartIntegrity{{param($s,$n)}}
function Get-OutputMap{{param($r);return [ordered]@{{x='y'}}}};function Get-OutputMapSha{{param($m);return 'ok'}};function Write-AtomicJson{{param($p,$v)}}
Invoke-DurableStage 'Probe' {{}} @('ignored')
""",
        encoding="utf-8",
    )
    result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "command log" in combined and "integrity mismatch" in combined


def _make_stage_manifest(tmp_path: Path, commit: str) -> tuple[Path, dict[str, Path], dict[str, str], dict[str, object]]:
    r7 = tmp_path / "r7"
    (r7 / "results" / "official").mkdir(parents=True)
    (r7 / "task5").mkdir()
    (r7 / "manifest.json").write_text('{"sealed":true}\n', encoding="utf-8")
    authority: dict[str, str] = {}
    for relative in task5_manifest.OFFICIAL_OUTPUTS:
        path = r7 / relative
        path.write_text(json.dumps({"output": relative}) + "\n", encoding="utf-8")
        authority[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = {
        "dataset": tmp_path / "dataset.json",
        "layout_model": tmp_path / "layout.onnx",
        "runtime_config": tmp_path / "runtime-config.json",
        "python_executable": tmp_path / "python-stub.exe",
        "scorer_python_executable": tmp_path / "scorer-python-stub.exe",
    }
    for name, path in inputs.items():
        path.write_bytes((name + "\n").encode())
    environment: dict[str, object] = {
        "benchmark": "OmniDocBench-v1.6", "os": "Windows", "machine": "probe",
        "gpu_devices": [], "python": {"version": "stub"}, "scorer_python": {"version": "stub"},
        "onnxruntime": {"version": "stub"}, "available_providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
        "paddleocr": {"version": "stub"}, "official_adapter": {"sha256": "a" * 64},
        "lightweight_adapter": {"sha256": "b" * 64},
        "server_model_runtime": {"models_sha256": "c" * 64, "requested_model": "stub"},
    }
    receipt = ROOT / "docs" / "releases" / "0.1.0-g0-evidence.md"
    original = task5_manifest.APPROVED_G0_OUTPUT_SHA256
    task5_manifest.APPROVED_G0_OUTPUT_SHA256 = authority
    try:
        manifest = task5_manifest.build_task5_manifest(
            r7_root=r7, receipt_path=receipt, git_commit=commit,
            inputs=inputs, environment=environment,
            contracts={"benchmark": "OmniDocBench-v1.6"},
        )
    finally:
        task5_manifest.APPROVED_G0_OUTPUT_SHA256 = original
    manifest_path = r7 / "task5" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, inputs, authority, environment


def _stage_integrity_probe(tmp_path: Path, *, mutate_input: str | None = None, bad_commit: bool = False, environment_drift: bool = False) -> subprocess.CompletedProcess[str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest_path, inputs, authority, environment = _make_stage_manifest(tmp_path, commit)
    if mutate_input:
        inputs[mutate_input].write_bytes(b"changed\n")
    if bad_commit:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["git_commit"] = "0" * 40
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    state = {
        "producing_commit": "0" * 40 if bad_commit else commit,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    bootstrap = f"import eval.task5_manifest as m,sys;m.APPROVED_G0_OUTPUT_SHA256={authority!r};raise SystemExit(m.main())"
    bootstrap_b64 = base64.b64encode(bootstrap.encode()).decode()
    recomputed_environment = dict(environment)
    if environment_drift:
        recomputed_environment["os"] = "Windows-drifted"
    environment_json = json.dumps(recomputed_environment, separators=(",", ":"))
    harness = tmp_path / "stage-integrity.ps1"
    harness.write_text(
        rf"""
$errors=$null;$tokens=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('{SCRIPT}',[ref]$tokens,[ref]$errors)
foreach($name in @('Get-Sha256','Get-StringSha256','ConvertTo-NativeArgument','Initialize-NativeJobRunner','Invoke-DirectPython','ConvertTo-CanonicalJson','Read-StrictJson','Get-GitCommit','Assert-StageStartIntegrity')){{
 $fn=$ast.Find({{param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name}},$true); Invoke-Expression $fn.Extent.Text
}}
$PythonExe='{ROOT / ".venv/Scripts/python.exe"}';$RepoRoot='{ROOT}';$ManifestPath='{manifest_path}';$Task5Root='{manifest_path.parent}'
function Get-EnvironmentContract{{return ('{environment_json}' | ConvertFrom-Json)}}
function Invoke-LoggedNative([string]$StageName,[string]$Name,[string]$Executable,[string[]]$Arguments){{
 $bootstrap=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{bootstrap_b64}'))
 Invoke-DirectPython $bootstrap @('validate','--manifest',$ManifestPath,'--task5-root',$Task5Root) | Out-Null
}}
$state='{json.dumps(state, separators=(',', ':'))}' | ConvertFrom-Json
Assert-StageStartIntegrity $state 'Official'
""",
        encoding="utf-8",
    )
    return subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)], cwd=ROOT, text=True, capture_output=True, check=False)


@pytest.mark.parametrize("input_name", ["dataset", "layout_model", "runtime_config", "python_executable", "scorer_python_executable"])
def test_stage_start_integrity_revalidates_each_bound_input(tmp_path: Path, input_name: str) -> None:
    drift = _stage_integrity_probe(tmp_path / input_name, mutate_input=input_name)
    assert drift.returncode != 0


def test_stage_start_integrity_revalidates_environment_and_commit(tmp_path: Path) -> None:
    assert _stage_integrity_probe(tmp_path / "valid").returncode == 0
    environment = _stage_integrity_probe(tmp_path / "environment", environment_drift=True)
    assert environment.returncode != 0
    assert "environment integrity mismatch" in environment.stdout + environment.stderr
    mismatch = _stage_integrity_probe(tmp_path / "commit", bad_commit=True)
    assert mismatch.returncode != 0
    assert "producing commit integrity mismatch" in mismatch.stdout + mismatch.stderr


@pytest.mark.parametrize("drift", ["dataset", "layout", "runtime", "python", "scorer", "environment", "commit"])
def test_preflight_boundary_rejects_drift_before_first_business_command(tmp_path: Path, drift: str) -> None:
    marker = tmp_path / "business-launched.txt"
    harness = tmp_path / "preflight-order.ps1"
    harness.write_text(
        rf"""
$ErrorActionPreference='Stop';$errors=$null;$tokens=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('{SCRIPT}',[ref]$tokens,[ref]$errors)
foreach($name in @('Invoke-DurableStage','Invoke-PreflightStage')){{
 $fn=$ast.Find({{param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name}},$true);Invoke-Expression $fn.Extent.Text
}}
$ManifestPath='{tmp_path / 'manifest.json'}';$AttemptRoot='{tmp_path}';$Benchmark='v16';$script:Drift=$null
function Initialize-PreflightAttempt{{$script:Drift='{drift}'}}
function Read-State{{return [pscustomobject]@{{stages=[pscustomobject]@{{}}}}}}
function Assert-RecordedStagesIntegrity{{param($State)}}
function Assert-StageStartIntegrity{{param($State,$StageName);if($script:Drift){{throw 'manifest integrity mismatch'}}}}
function Invoke-PreflightBusiness{{[IO.File]::WriteAllText('{marker}','launched')}}
function Add-InternalCommandRecord{{param($a,$b,$c)}}
Invoke-PreflightStage
""",
        encoding="utf-8",
    )
    result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert not marker.exists(), drift
