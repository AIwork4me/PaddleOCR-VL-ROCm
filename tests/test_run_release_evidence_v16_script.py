from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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
    program = directory / "stub.py"
    program.write_text(
        "import hashlib,json,os,pathlib,sys\n"
        "a=sys.argv[1:]\n"
        "with open(os.environ['STUB_ARG_LOG'],'a',encoding='utf-8') as f:\n"
        " [f.write(x+'\\n') for x in a]\n"
        f"fail={fail_on!r}\n"
        "if fail and any(fail in x for x in a): sys.exit(23)\n"
        "if len(a)>1 and a[1]=='manifest':\n"
        " inputs={}\n"
        " for i,x in enumerate(a):\n"
        "  if x=='--input':\n"
        "   n,p=a[i+1].split('=',1); q=pathlib.Path(p); inputs[n]={'path':str(q.resolve()),'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}\n"
        " out=pathlib.Path(a[a.index('--output')+1]); out.write_text(json.dumps({'git_commit':a[a.index('--git-commit')+1],'inputs':inputs},sort_keys=True),encoding='utf-8')\n"
        "elif len(a)>1 and a[1]=='decide': print('{}')\n"
        "elif 'run_eval.py' in ' '.join(a):\n"
        " pred=pathlib.Path(a[a.index('--predictions-dir')+1]); pred.mkdir(parents=True,exist_ok=True)\n"
        " if a[a.index('--stage')+1]=='infer': (pred/'_run_stats.json').write_text(json.dumps({'layout_provider_requested':'auto','layout_providers_active':['DmlExecutionProvider','CPUExecutionProvider'],'layout_fallback_disabled':True}),encoding='utf-8')\n"
        " else:\n"
        "  for flag in ('--copy-report','--run-summary','--provenance'):\n"
        "   p=pathlib.Path(a[a.index(flag)+1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_text('{}',encoding='utf-8')\n",
        encoding="utf-8",
    )
    stub = directory / "python.cmd"
    stub.write_text(
        "@echo off\n"
        f'"{sys.executable}" "{program}" %*\n',
        encoding="utf-8",
    )
    return stub


def _run(script: Path, *arguments: str, env: dict[str, str] | None = None, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), *arguments],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )


def _clean_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "clean repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / SCRIPT.name)
    for relative in (
        "eval/release_evidence.py",
        "eval/release_contract.py",
        "eval/benchmark_contract.py",
        "eval/configs/omnidocbench_v16.yaml",
        "src/paddleocr_vl_rocm/assets/runtime-manifest.json",
        "eval/patches/omnidocbench-v16-windows-cdm.patch",
        "eval/.omnidocbench/tools/generate_result_tables.ipynb",
        "eval/.omnidocbench/src/core/metrics.py",
        "eval/.omnidocbench/src/metrics/cal_metric.py",
        "eval/.omnidocbench/src/metrics/table_metric.py",
        "eval/.omnidocbench/src/metrics/cdm_metric.py",
        "eval/.omnidocbench/src/dataset/end2end_dataset.py",
    ):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# anchor\n", encoding="utf-8")
    (repo / "src/paddleocr_vl_rocm/assets/runtime-manifest.json").write_text(
        json.dumps({"resources": [
            {"name": "paddleocr-vl-main-gguf", "destination": "models/PaddleOCR-VL-1.6-GGUF.gguf"},
            {"name": "paddleocr-vl-mmproj", "destination": "models/PaddleOCR-VL-1.6-GGUF-mmproj.gguf"},
        ]}),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    return repo, repo / "scripts" / SCRIPT.name


def test_failed_preflight_preserves_space_arguments_and_stops_official(tmp_path: Path) -> None:
    repo, script = _clean_repo(tmp_path)
    tools = tmp_path / "stub tools"
    tools.mkdir()
    argument_log = tmp_path / "argument log.txt"
    _stub_python(tools, fail_on="check_server.py")
    evidence = tmp_path / "fresh evidence root"
    dataset = tmp_path / "dataset with spaces"
    layout = tmp_path / "layout model with spaces"
    dataset.mkdir()
    layout.mkdir()
    (dataset / "OmniDocBench.json").write_text("{}", encoding="utf-8")
    (layout / "inference.onnx").write_bytes(b"stub")
    (layout / "inference.yml").write_text("stub", encoding="utf-8")
    main = tmp_path / "model root" / "PaddleOCR-VL-1.6-GGUF.gguf"
    mmproj = tmp_path / "model root" / "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
    main.parent.mkdir()
    main.write_bytes(b"main")
    mmproj.write_bytes(b"mmproj")
    config = tmp_path / "active config.json"
    config.write_text(json.dumps({"main_gguf": str(main), "mmproj": str(mmproj), "layout_model_dir": str(layout)}), encoding="utf-8")
    env = os.environ.copy()
    env.update(
        PATH=f"{tools}{os.pathsep}{env['PATH']}",
        STUB_ARG_LOG=str(argument_log),
    )

    completed = _run(
        script,
        "-Stage", "All",
        "-EvidenceRoot", str(evidence),
        "-DatasetDir", str(dataset),
        "-LayoutModel", str(layout),
        "-RuntimeConfig", str(config),
        env=env,
        cwd=tmp_path,
    )

    assert completed.returncode != 0
    arguments = argument_log.read_text(encoding="utf-8").splitlines()
    assert f"dataset={dataset / 'OmniDocBench.json'}" in arguments
    assert f"layout_model={layout / 'inference.onnx'}" in arguments
    assert "check_server.py" in "\n".join(arguments)
    assert "run_eval.py" not in "\n".join(arguments)
    records = [json.loads(line) for line in (evidence / "logs" / "commands.jsonl").read_text().splitlines()]
    assert records[-1]["exit_code"] == 23
    assert records[-1]["arguments"][-1] == "<redacted-url>"
    assert "http://" not in json.dumps(records)
    assert str(tmp_path) not in json.dumps(records)
    assert json.loads((evidence / "logs" / "stages" / "preflight.json").read_text())["status"] == "failed"

    _stub_python(tools)
    retried = _run(
        script,
        "-Stage", "Preflight",
        "-EvidenceRoot", str(evidence),
        "-DatasetDir", str(dataset),
        "-LayoutModel", str(layout),
        "-RuntimeConfig", str(config),
        env=env,
        cwd=tmp_path,
    )
    assert retried.returncode == 0, retried.stderr
    state = json.loads((evidence / "logs" / "stages" / "preflight.json").read_text())
    assert state["status"] == "completed"
    before = argument_log.read_text().count("check_server.py")
    skipped = _run(
        script,
        "-Stage", "Preflight",
        "-EvidenceRoot", str(evidence),
        "-DatasetDir", str(dataset),
        "-LayoutModel", str(layout),
        "-RuntimeConfig", str(config),
        env=env,
        cwd=tmp_path,
    )
    assert skipped.returncode == 0
    assert argument_log.read_text().count("check_server.py") == before
    changed_url = _run(
        script, "-Stage", "Preflight", *(
            "-EvidenceRoot", str(evidence), "-DatasetDir", str(dataset),
            "-LayoutModel", str(layout), "-RuntimeConfig", str(config),
            "-ServerUrl", "http://127.0.0.1:9999/v1",
        ), env=env, cwd=tmp_path,
    )
    assert changed_url.returncode != 0
    assert "invocation fingerprint mismatch" in (changed_url.stdout + changed_url.stderr)
    changed_model = _run(
        script, "-Stage", "Preflight", *(
            "-EvidenceRoot", str(evidence), "-DatasetDir", str(dataset),
            "-LayoutModel", str(layout), "-RuntimeConfig", str(config),
            "-ApiModelName", "different-model.gguf",
        ), env=env, cwd=tmp_path,
    )
    assert changed_model.returncode != 0
    assert "invocation fingerprint mismatch" in (changed_model.stdout + changed_model.stderr)


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
    repo, script = _clean_repo(tmp_path)
    evidence = tmp_path / "resume evidence"
    dataset = tmp_path / "dataset"
    layout = tmp_path / "layout"
    dataset.mkdir()
    layout.mkdir()
    manifest = dataset / "OmniDocBench.json"
    manifest.write_text("{}", encoding="utf-8")
    (layout / "inference.onnx").write_bytes(b"model")
    (layout / "inference.yml").write_text("config", encoding="utf-8")
    main = tmp_path / "PaddleOCR-VL-1.6-GGUF.gguf"
    mmproj = tmp_path / "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
    main.write_bytes(b"main")
    mmproj.write_bytes(b"mm")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"main_gguf": str(main), "mmproj": str(mmproj), "layout_model_dir": str(layout)}), encoding="utf-8")
    tools = tmp_path / "tools"
    tools.mkdir()
    argument_log = tmp_path / "args.log"
    _stub_python(tools, fail_on="check_server.py")
    env = os.environ.copy()
    env.update(PATH=f"{tools}{os.pathsep}{env['PATH']}", STUB_ARG_LOG=str(argument_log))

    first = _run(
        script,
        "-Stage", "Preflight",
        "-EvidenceRoot", str(evidence),
        "-DatasetDir", str(dataset),
        "-LayoutModel", str(layout),
        "-RuntimeConfig", str(config),
        "-ServerUrl", "http://127.0.0.1:1/v1",
        env=env,
        cwd=tmp_path,
    )
    assert first.returncode != 0
    assert (evidence / "manifest.json").is_file()
    manifest.write_text('{"changed": true}', encoding="utf-8")

    resumed = _run(
        script,
        "-Stage", "Preflight",
        "-EvidenceRoot", str(evidence),
        "-DatasetDir", str(dataset),
        "-LayoutModel", str(layout),
        "-RuntimeConfig", str(config),
        env=env,
        cwd=tmp_path,
    )

    assert resumed.returncode != 0
    assert "Resume refused" in (resumed.stdout + resumed.stderr)


def test_runner_uses_exact_release_anchors_and_durable_stage_state() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for anchor in (
        "OmniDocBench.json",
        "inference.onnx",
        "inference.yml",
        "eval/configs/omnidocbench_v16.yaml",
        "src/paddleocr_vl_rocm/assets/runtime-manifest.json",
        "eval/benchmark_contract.py",
    ):
        assert anchor in text
    assert "Select-Object -First 1" not in text
    assert 'status = "started"' in text
    assert 'status = "completed"' in text
    assert 'status = "failed"' in text
    assert "input_manifest_sha256" in text
    assert "invocation_fingerprint" in text
    assert "command_sha256" in text
    assert "output_sha256" in text
    assert "decision.json" in text
    assert "RELEASE_EVIDENCE_ALLOW_DIRTY" not in text


def test_runner_redacts_secret_classes_and_has_directml_preflight() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "<redacted-path>" in text
    assert "--token" in text
    assert "--api-key" in text
    assert "duration_ms" in text
    assert "command_name" in text
    assert "directml-preflight" in text
    assert text.index("directml-preflight") < text.index("lightweight-infer")


def test_all_persists_decision_and_detects_changed_completed_output(tmp_path: Path) -> None:
    _, script = _clean_repo(tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    argument_log = tmp_path / "args.log"
    _stub_python(tools)
    dataset = tmp_path / "dataset"
    layout = tmp_path / "layout"
    dataset.mkdir()
    layout.mkdir()
    (dataset / "OmniDocBench.json").write_text("{}", encoding="utf-8")
    (layout / "inference.onnx").write_bytes(b"model")
    (layout / "inference.yml").write_text("config", encoding="utf-8")
    main, mmproj = tmp_path / "PaddleOCR-VL-1.6-GGUF.gguf", tmp_path / "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
    main.write_bytes(b"main")
    mmproj.write_bytes(b"mm")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"main_gguf": str(main), "mmproj": str(mmproj), "layout_model_dir": str(layout)}), encoding="utf-8")
    evidence = tmp_path / "evidence"
    env = os.environ.copy()
    env.update(PATH=f"{tools}{os.pathsep}{env['PATH']}", STUB_ARG_LOG=str(argument_log))
    common = (
        "-EvidenceRoot", str(evidence), "-DatasetDir", str(dataset),
        "-LayoutModel", str(layout), "-RuntimeConfig", str(config),
    )
    stale = evidence / "official" / "partial.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("partial", encoding="utf-8")

    completed = _run(script, "-Stage", "All", *common, env=env, cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert not stale.exists()
    assert json.loads((evidence / "decision.json").read_text()) == {}
    args = argument_log.read_text(encoding="utf-8").splitlines()
    lightweight = next(
        i for i in range(len(args) - 8)
        if args[i] == "eval/run_eval.py" and args[i + 2] == "infer" and "lightweight" in args[i : i + 10]
    )
    lightweight_args = args[lightweight : lightweight + 24]
    assert ["--server-url", "http://127.0.0.1:8111/v1"] == lightweight_args[lightweight_args.index("--server-url") : lightweight_args.index("--server-url") + 2]
    assert ["--api-model-name", "PaddleOCR-VL-1.6-GGUF.gguf"] == lightweight_args[lightweight_args.index("--api-model-name") : lightweight_args.index("--api-model-name") + 2]
    assert ["--layout-model", str(layout)] == lightweight_args[lightweight_args.index("--layout-model") : lightweight_args.index("--layout-model") + 2]
    decide_state = json.loads((evidence / "logs" / "stages" / "decide.json").read_text())
    assert decide_state["status"] == "completed"
    assert decide_state["output_sha256"]["decision.json"]
    extra = evidence / "official" / "unexpected.txt"
    extra.write_text("unexpected", encoding="utf-8")
    added = _run(script, "-Stage", "Official", *common, env=env, cwd=tmp_path)
    assert added.returncode != 0
    assert "output hash/set mismatch" in (added.stdout + added.stderr)
    extra.unlink()
    command_log = evidence / "logs" / "stages" / "official.commands.jsonl"
    original_log = command_log.read_bytes()
    command_log.write_bytes(original_log + b"tampered\n")
    log_tampered = _run(script, "-Stage", "Official", *common, env=env, cwd=tmp_path)
    assert log_tampered.returncode != 0
    assert "command log hash mismatch" in (log_tampered.stdout + log_tampered.stderr)
    decisions_before = argument_log.read_text().count("release_evidence.py")
    predecessor_tampered = _run(script, "-Stage", "Decide", *common, env=env, cwd=tmp_path)
    assert predecessor_tampered.returncode != 0
    assert "Predecessor command log hash mismatch" in (predecessor_tampered.stdout + predecessor_tampered.stderr)
    assert argument_log.read_text().count("release_evidence.py") == decisions_before
    command_log.write_bytes(original_log)
    command_log.unlink()
    log_missing = _run(script, "-Stage", "Official", *common, env=env, cwd=tmp_path)
    assert log_missing.returncode != 0
    assert "command log hash mismatch" in (log_missing.stdout + log_missing.stderr)
    command_log.write_bytes(original_log)
    metric = evidence / "results" / "official" / "metric.json"
    metric.write_text('{"tampered": true}', encoding="utf-8")

    resumed = _run(script, "-Stage", "Official", *common, env=env, cwd=tmp_path)

    assert resumed.returncode != 0
    assert "output hash/set mismatch" in (resumed.stdout + resumed.stderr)
    config.write_text(json.dumps({"main_gguf": str(main), "mmproj": str(mmproj), "layout_model_dir": str(tmp_path / "wrong-layout")}), encoding="utf-8")
    infer_before = argument_log.read_text().count("eval/run_eval.py")
    mismatch = _run(script, "-Stage", "Lightweight", *common, env=env, cwd=tmp_path)
    assert mismatch.returncode != 0
    assert "layout_model_dir does not match" in (mismatch.stdout + mismatch.stderr)
    assert argument_log.read_text().count("eval/run_eval.py") == infer_before


def test_standalone_stage_requires_preflight_before_native_commands(tmp_path: Path) -> None:
    _, script = _clean_repo(tmp_path)
    completed = _run(script, "-Stage", "Official", "-EvidenceRoot", str(tmp_path / "evidence"), cwd=tmp_path)
    assert completed.returncode != 0
    assert "Missing completed predecessor: Preflight" in (completed.stdout + completed.stderr)
    assert not (tmp_path / "evidence").exists()


def test_clean_gate_rejects_untracked_file_from_foreign_cwd(tmp_path: Path) -> None:
    repo, script = _clean_repo(tmp_path)
    (repo / "unexpected.txt").write_text("dirty", encoding="utf-8")
    completed = _run(
        script,
        "-EvidenceRoot", str(tmp_path / "external evidence"),
        cwd=tmp_path,
    )
    assert completed.returncode != 0
    assert "clean worktree" in (completed.stdout + completed.stderr)


def test_split_secret_values_are_redacted_by_argument_classifier() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index("function ConvertTo-SafeArgument")
    end = text.index("function Invoke-LoggedNative", start)
    functions = text[start:end]
    command = (
        functions
        + '\n$secretNext = $false; $safe = @("--token", "top-secret", "--api-key", "second-secret") | ForEach-Object {'
        + '\nif ($secretNext) { $secretNext = $false; "<redacted-secret>"; return }; '
        + 'if ($_ -in @("--token", "--api-key", "--authorization")) { $secretNext = $true; $_; return }; ConvertTo-SafeArgument $_ }; '
        + "$safe | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", command], text=True, capture_output=True
    )
    assert completed.returncode == 0, completed.stderr
    assert "top-secret" not in completed.stdout
    assert "second-secret" not in completed.stdout
    assert json.loads(completed.stdout).count("<redacted-secret>") == 2


def test_physical_link_into_protected_results_is_rejected(tmp_path: Path) -> None:
    protected = ROOT / "results" / "omnidocbench" / "v16"
    link = tmp_path / "evidence-link"
    try:
        link.symlink_to(protected, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory links are unavailable: {exc}")
    completed = _run(SCRIPT, "-EvidenceRoot", str(link / "ordinary-child"))
    assert completed.returncode != 0
    assert "protected historical path" in (completed.stdout + completed.stderr)
