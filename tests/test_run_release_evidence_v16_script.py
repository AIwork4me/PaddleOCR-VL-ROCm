from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_release_evidence_v16.ps1"


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


def _powershell() -> str:
    executable = shutil.which("powershell")
    if executable is None:
        pytest.skip("Windows PowerShell is required")
    return executable


def _stub_python(directory: Path, *, fail_on: str = "") -> Path:
    stub = directory / "python.cmd"
    stub.write_text(
        "@echo off\n"
        "setlocal EnableDelayedExpansion\n"
        "set NEXT_OUTPUT=0\n"
        "for %%A in (%*) do (\n"
        "  echo %%~A>>\"%STUB_ARG_LOG%\"\n"
        "  if !NEXT_OUTPUT!==1 (echo {\"stub\":true}>\"%%~A\"& set NEXT_OUTPUT=0)\n"
        "  if \"%%~A\"==\"--output\" set NEXT_OUTPUT=1\n"
        ")\n"
        f"echo %*| findstr /C:\"{fail_on}\" >nul && exit /b 23\n"
        "exit /b 0\n",
        encoding="utf-8",
    )
    return stub


def _run(script: Path, *arguments: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), *arguments],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_failed_preflight_preserves_space_arguments_and_stops_official(tmp_path: Path) -> None:
    tools = tmp_path / "stub tools"
    tools.mkdir()
    argument_log = tmp_path / "argument log.txt"
    _stub_python(tools, fail_on="check_server.py")
    evidence = tmp_path / "fresh evidence root"
    dataset = tmp_path / "dataset with spaces"
    layout = tmp_path / "layout model with spaces"
    dataset.mkdir()
    layout.mkdir()
    (dataset / "manifest.json").write_text("{}", encoding="utf-8")
    (layout / "model.onnx").write_bytes(b"stub")
    env = os.environ.copy()
    env.update(
        PATH=f"{tools}{os.pathsep}{env['PATH']}",
        STUB_ARG_LOG=str(argument_log),
        RELEASE_EVIDENCE_ALLOW_DIRTY="1",
    )

    completed = _run(
        SCRIPT,
        "-Stage", "All",
        "-EvidenceRoot", str(evidence),
        "-DatasetDir", str(dataset),
        "-LayoutModel", str(layout),
        env=env,
    )

    assert completed.returncode != 0
    arguments = argument_log.read_text(encoding="utf-8").splitlines()
    assert f"dataset={dataset / 'manifest.json'}" in arguments
    assert f"layout_model={layout / 'model.onnx'}" in arguments
    assert "check_server.py" in "\n".join(arguments)
    assert "run_eval.py" not in "\n".join(arguments)
    records = [json.loads(line) for line in (evidence / "logs" / "commands.jsonl").read_text().splitlines()]
    assert records[-1]["exit_code"] == 23
    assert records[-1]["arguments"][-1] == "REDACTED_URL"
    assert "http://" not in json.dumps(records)


def test_rejects_historical_output_before_creating_it(tmp_path: Path) -> None:
    protected = ROOT / "results" / "omnidocbench" / "v16" / "new evidence"
    completed = _run(SCRIPT, "-EvidenceRoot", str(protected))
    assert completed.returncode != 0
    assert "protected historical path" in (completed.stdout + completed.stderr)
    assert not protected.exists()


def test_source_contains_resume_security_and_staging_guards() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "git status --porcelain" in text
    assert "manifest.json" in text
    assert "git_commit" in text and "sha256" in text
    assert "Authorization" in text and "REDACTED" in text
    assert "layout_fallback_disabled" in text
    assert "CPUExecutionProvider" in text
    assert "git add" not in text


def test_resume_rejects_changed_immutable_input(tmp_path: Path) -> None:
    evidence = tmp_path / "resume evidence"
    dataset = tmp_path / "dataset"
    layout = tmp_path / "layout"
    dataset.mkdir()
    layout.mkdir()
    manifest = dataset / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    (layout / "model.onnx").write_bytes(b"model")
    env = os.environ.copy()
    env["RELEASE_EVIDENCE_ALLOW_DIRTY"] = "1"

    first = _run(
        SCRIPT,
        "-Stage", "Preflight",
        "-EvidenceRoot", str(evidence),
        "-DatasetDir", str(dataset),
        "-LayoutModel", str(layout),
        "-ServerUrl", "http://127.0.0.1:1/v1",
        env=env,
    )
    assert first.returncode != 0
    assert (evidence / "manifest.json").is_file()
    manifest.write_text('{"changed": true}', encoding="utf-8")

    resumed = _run(
        SCRIPT,
        "-Stage", "Preflight",
        "-EvidenceRoot", str(evidence),
        "-DatasetDir", str(dataset),
        "-LayoutModel", str(layout),
        env=env,
    )

    assert resumed.returncode != 0
    assert "Resume refused" in (resumed.stdout + resumed.stderr)
