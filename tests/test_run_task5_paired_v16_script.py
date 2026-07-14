from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_task5_paired_v16.ps1"


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
foreach($name in @('Get-Sha256','Get-StringSha256','Write-AtomicText','Add-CommandRecord','ConvertTo-NativeArgument','Invoke-LoggedNative')){{
  $fn=$ast.Find({{param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name}},$true)
  Invoke-Expression $fn.Extent.Text
}}
$RepoRoot='{ROOT}'
$CommandRoot='{command_root}'
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
