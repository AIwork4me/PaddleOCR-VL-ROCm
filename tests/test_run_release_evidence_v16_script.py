from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_release_evidence_v16.ps1"


def test_release_runner_isolates_and_orders_stages() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert (
        'ValidateSet("Preflight", "Official", "OfficialScore", "Lightweight", "Decide", "All")'
        in text
    )
    assert '"-m", "eval.release_evidence", "manifest"' in text
    assert text.index('"Preflight"') < text.index('"Official"')
    assert '"-m", "eval.release_contract"' in text
    assert "layout_provider_requested" in text
    assert "DmlExecutionProvider" in text
    assert "--copy-report" in text
    assert "--run-summary" in text
    assert "--provenance" in text
    assert '"OfficialScore"' in text
    assert "check_omnidocbench_scorer.py" in text
    assert text.index("scorer-preflight") < text.index("official-infer")
    assert '"--cdm"' in text
    assert "[string]$ScorerPythonExe" in text
    assert '"--scorer-python", $ScorerPythonExe' in text
    assert "scorer_environment_sha256" in text
    assert "requirements-omnidocbench-v16-transitive.txt" in text


def test_score_only_source_never_invokes_inference_and_requires_both_scores() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    score_start = text.index("function Invoke-OfficialScore")
    score_end = text.index("function Assert-DirectMlEvidence", score_start)
    score_body = text[score_start:score_end]
    assert "Invoke-RecoveryAuthentication" in score_body
    assert '"--stage", "infer"' not in score_body
    assert "metric.json" in score_body
    assert "metric-cdm.json" in score_body
    assert "run-summary.json" in score_body
    assert "run-summary-cdm.json" in score_body
    assert "provenance.json" in score_body
    assert "provenance-cdm.json" in score_body


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("same-executable", "ScorerPythonExe must differ physically from PythonExe"),
        ("same-prefix", "Scorer sys.prefix must differ physically from inference sys.prefix"),
        ("non-venv", "ScorerPythonExe must be inside a real virtual environment"),
    ),
)
def test_runner_rejects_nonisolated_scorer_interpreters(
    tmp_path: Path, case: str, expected: str
) -> None:
    _, script = _clean_repo(tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    argument_log = tmp_path / "args.log"
    inference_python = _stub_python(tools)
    scorer_python = inference_python
    env = os.environ.copy()
    if case == "same-prefix":
        scorer_python = inference_python.with_name("scorer-python.cmd")
        shutil.copy2(inference_python, scorer_python)
    elif case == "non-venv":
        scorer_python = tools / "standalone" / "python.cmd"
        scorer_python.parent.mkdir()
        shutil.copy2(inference_python, scorer_python)
        env["STUB_SCORER_NON_VENV"] = "1"
    dataset = tmp_path / "dataset"
    layout = tmp_path / "layout"
    dataset.mkdir()
    layout.mkdir()
    (dataset / "OmniDocBench.json").write_text("{}", encoding="utf-8")
    (layout / "inference.onnx").write_bytes(b"model")
    (layout / "inference.yml").write_text("config", encoding="utf-8")
    main = tmp_path / "PaddleOCR-VL-1.6-GGUF.gguf"
    mmproj = tmp_path / "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
    main.write_bytes(b"main")
    mmproj.write_bytes(b"mm")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"main_gguf": str(main), "mmproj": str(mmproj), "layout_model_dir": str(layout)}
        ),
        encoding="utf-8",
    )
    env.update(
        STUB_ARG_LOG=str(argument_log),
        STUB_REPORTED_EXE=str(inference_python),
        STUB_SCORER_EXE=str(scorer_python),
    )

    completed = _run(
        script,
        "-Stage",
        "Preflight",
        "-EvidenceRoot",
        str(tmp_path / "evidence"),
        "-DatasetDir",
        str(dataset),
        "-LayoutModel",
        str(layout),
        "-RuntimeConfig",
        str(config),
        env=env,
        cwd=tmp_path,
    )

    assert completed.returncode != 0
    assert expected in (completed.stdout + completed.stderr)


@pytest.mark.parametrize(
    ("version", "schema", "omit_key", "extra_schema", "accepted", "expected"),
    (
        ("3.10", "record-v1", "", "", True, ""),
        ("3.10", "legacy-installed-files-v1", "", "", True, ""),
        ("3.11", "record-v1", "", "", False, "Scorer must report CPython 3.10"),
        (
            "3.10",
            "unverified",
            "",
            "",
            False,
            "Scorer dependency attestation schema is invalid",
        ),
        (
            "3.10",
            "legacy-installed-files-v1",
            "metadata_sha256",
            "",
            False,
            "Scorer dependency legacy attestation is invalid",
        ),
        (
            "3.10",
            "record-v1",
            "record_sha256",
            "",
            False,
            "Scorer dependency RECORD attestation is invalid",
        ),
        (
            "3.10",
            "record-v1",
            "",
            "legacy-installed-files-v1",
            False,
            "Scorer dependency attestation schema fields are mixed",
        ),
        (
            "3.10",
            "RECORD-V1",
            "",
            "",
            False,
            "Scorer dependency attestation schema is invalid",
        ),
        (
            "3.10",
            "Legacy-Installed-Files-V1",
            "",
            "",
            False,
            "Scorer dependency attestation schema is invalid",
        ),
        (
            "3.10",
            "legacy-installed-files-v1",
            "",
            "record-v1",
            False,
            "Scorer dependency attestation schema fields are mixed",
        ),
    ),
)
def test_runner_enforces_cpython_310_scorer_before_scoring(
    tmp_path: Path,
    version: str,
    schema: str,
    omit_key: str,
    extra_schema: str,
    accepted: bool,
    expected: str,
) -> None:
    _, script = _clean_repo(tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    argument_log = tmp_path / "args.log"
    inference_python = _stub_python(tools)
    dataset = tmp_path / "dataset"
    layout = tmp_path / "layout"
    dataset.mkdir()
    layout.mkdir()
    (dataset / "OmniDocBench.json").write_text("{}", encoding="utf-8")
    (layout / "inference.onnx").write_bytes(b"model")
    (layout / "inference.yml").write_text("config", encoding="utf-8")
    main = tmp_path / "PaddleOCR-VL-1.6-GGUF.gguf"
    mmproj = tmp_path / "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
    main.write_bytes(b"main")
    mmproj.write_bytes(b"mm")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"main_gguf": str(main), "mmproj": str(mmproj), "layout_model_dir": str(layout)}
        ),
        encoding="utf-8",
    )
    scorer = tools / "scorer-venv" / "Scripts" / "python.cmd"
    env = os.environ.copy()
    env.update(
        STUB_ARG_LOG=str(argument_log),
        STUB_REPORTED_EXE=str(inference_python),
        STUB_SCORER_EXE=str(scorer),
        STUB_SCORER_VERSION=version,
        STUB_ATTESTATION_SCHEMA=schema,
        STUB_OMIT_ATTESTATION_KEY=omit_key,
        STUB_EXTRA_ATTESTATION_SCHEMA=extra_schema,
    )

    completed = _run(
        script,
        "-Stage",
        "Preflight",
        "-EvidenceRoot",
        str(tmp_path / "evidence"),
        "-DatasetDir",
        str(dataset),
        "-LayoutModel",
        str(layout),
        "-RuntimeConfig",
        str(config),
        env=env,
        cwd=tmp_path,
    )

    assert (completed.returncode == 0) is accepted
    if not accepted:
        assert expected in (completed.stdout + completed.stderr)
    assert "eval.run_eval" not in argument_log.read_text(encoding="utf-8")


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
        "if '-c' in a:\n"
        " if os.environ.get('STUB_MISSING_PACKAGE')=='1': sys.exit(1)\n"
        " root=pathlib.Path.cwd(); wrong=os.environ.get('STUB_WRONG_ORIGIN')=='1'; base=pathlib.Path(os.environ.get('TEMP','.')) if wrong else root\n"
        " venv=pathlib.Path(os.environ['STUB_REPORTED_EXE']).parent.parent; records={}\n"
        " for name in ('paddleocr','paddlex','paddlepaddle'):\n"
        "  p=venv/(name+'-dist-info')/'RECORD'; p.parent.mkdir(exist_ok=True); p.write_text(name,encoding='utf-8') if not p.exists() else None; records[name]=str(p)\n"
        " versions={'paddleocr':'3.7.0','paddlex':'3.7.2','paddlepaddle':'3.2.1'}\n"
        " if os.environ.get('STUB_WRONG_VERSION')=='1': versions['paddleocr']='0.0.0'\n"
        " hashes={n:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for n,p in records.items()}\n"
        " origins={n:str(base/(n+'.py')) for n in records}; dist_origins={n:str(venv) for n in records}\n"
        " print(json.dumps({'version':'stub-version','executable':os.environ['STUB_REPORTED_EXE'],'prefix':str(venv),'base_prefix':str(venv.parent/'base-python'),'eval_origin':str(base/'eval/__init__.py'),'package_origin':str(base/'src/paddleocr_vl_rocm/__init__.py'),'core_versions':versions,'core_origins':origins,'distribution_origins':dist_origins,'record_paths':records,'record_sha256':hashes,'dependency_environment_sha256':os.environ.get('STUB_ENV_HASH','a'*64)})); sys.exit(0)\n"
        "if a and a[0].endswith('check_omnidocbench_scorer.py') and '--output' in a:\n"
        " scorer=pathlib.Path(os.environ['STUB_SCORER_EXE']).resolve(); prefix=scorer.parent.parent; base=prefix if os.environ.get('STUB_SCORER_NON_VENV')=='1' else prefix.parent/'base-python'; package=pathlib.Path(os.environ.get('STUB_SCORER_PACKAGE',str(scorer))); content=hashlib.sha256(package.read_bytes()).hexdigest()\n"
        " schema=os.environ.get('STUB_ATTESTATION_SCHEMA','record-v1'); normalized_schema=schema.casefold(); dependency={'version':'1.0','attestation_schema':schema,'origin_sha256':'c'*64,'content_sha256':content,'file_count':1}\n"
        " if normalized_schema=='record-v1': dependency['record_sha256']='d'*64\n"
        " elif normalized_schema=='legacy-installed-files-v1': dependency.update(installed_files_sha256='d'*64,metadata_sha256='e'*64)\n"
        " extra=os.environ.get('STUB_EXTRA_ATTESTATION_SCHEMA','')\n"
        " if extra=='record-v1': dependency['record_sha256']='f'*64\n"
        " elif extra=='legacy-installed-files-v1': dependency.update(installed_files_sha256='f'*64,metadata_sha256='f'*64)\n"
        " dependency.pop(os.environ.get('STUB_OMIT_ATTESTATION_KEY',''),None)\n"
        " value={'python_version':os.environ.get('STUB_SCORER_VERSION','3.10'),'python_executable':str(scorer),'python_executable_sha256':hashlib.sha256(str(scorer).encode()).hexdigest(),'python_prefix':str(prefix),'python_prefix_sha256':hashlib.sha256(str(prefix).encode()).hexdigest(),'python_base_prefix':str(base),'python_base_prefix_sha256':hashlib.sha256(str(base).encode()).hexdigest(),'python_version_sha256':'b'*64,'dependency_environment_sha256':content,'dependencies':{'demo':dependency}}\n"
        " out=pathlib.Path(a[a.index('--output')+1]); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(value,sort_keys=True),encoding='utf-8'); print(json.dumps(value)); sys.exit(0)\n"
        f"fail={fail_on!r}\n"
        "if fail and any(fail in x for x in a): sys.exit(23)\n"
        "if 'eval.release_evidence' in a and 'manifest' in a:\n"
        " inputs={}\n"
        " for i,x in enumerate(a):\n"
        "  if x=='--input':\n"
        "   n,p=a[i+1].split('=',1); q=pathlib.Path(p); inputs[n]={'path':str(q.resolve()),'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}\n"
        " out=pathlib.Path(a[a.index('--output')+1]); out.write_text(json.dumps({'git_commit':a[a.index('--git-commit')+1],'inputs':inputs},sort_keys=True),encoding='utf-8')\n"
        "elif 'eval.score_recovery' in a:\n"
        " out=pathlib.Path(a[a.index('--output')+1]); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'authenticated':True}),encoding='utf-8')\n"
        "elif 'eval.release_evidence' in a and 'decide' in a: print('{}')\n"
        "elif 'eval.run_eval' in a:\n"
        " pred=pathlib.Path(a[a.index('--predictions-dir')+1]); pred.mkdir(parents=True,exist_ok=True)\n"
        " if a[a.index('--stage')+1]=='infer': (pred/'_run_stats.json').write_text(json.dumps({'layout_provider_requested':'auto','layout_providers_active':['DmlExecutionProvider','CPUExecutionProvider'],'layout_fallback_disabled':True}),encoding='utf-8')\n"
        " else:\n"
        "  for flag in ('--copy-report','--run-summary','--provenance'):\n"
        "   p=pathlib.Path(a[a.index(flag)+1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_text('{}',encoding='utf-8')\n",
        encoding="utf-8",
    )
    stub = directory / "python.cmd"
    command = f'@echo off\n"{sys.executable}" "{program}" %*\n'
    stub.write_text(command, encoding="utf-8")
    scorer = directory / "scorer-venv" / "Scripts" / "python.cmd"
    scorer.parent.mkdir(parents=True, exist_ok=True)
    scorer.write_text(command, encoding="utf-8")
    return stub


def _run(
    script: Path, *arguments: str, env: dict[str, str] | None = None, cwd: Path = ROOT
) -> subprocess.CompletedProcess[str]:
    values = list(arguments)
    run_env = env.copy() if env else None
    if env and env.get("STUB_REPORTED_EXE") and "-PythonExe" not in values:
        values += ["-PythonExe", env["STUB_REPORTED_EXE"]]
    if env and env.get("STUB_REPORTED_EXE") and "-ScorerPythonExe" not in values:
        scorer = env.get(
            "STUB_SCORER_EXE",
            str(Path(env["STUB_REPORTED_EXE"]).parent / "scorer-venv" / "Scripts" / "python.cmd"),
        )
        values += ["-ScorerPythonExe", scorer]
        assert run_env is not None
        run_env["STUB_SCORER_EXE"] = scorer
    return subprocess.run(
        [_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), *values],
        cwd=cwd,
        env=run_env,
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
        "eval/score_recovery.py",
        "eval/requirements-omnidocbench-v16.txt",
        "eval/requirements-omnidocbench-v16-transitive.txt",
        "scripts/check_omnidocbench_scorer.py",
        "eval/configs/omnidocbench_v16.yaml",
        "src/paddleocr_vl_rocm/assets/runtime-manifest.json",
        "eval/patches/omnidocbench-v16-windows-cdm.patch",
        "eval/.omnidocbench/tools/generate_result_tables.ipynb",
        "eval/.omnidocbench/pyproject.toml",
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
        json.dumps(
            {
                "resources": [
                    {
                        "name": "paddleocr-vl-main-gguf",
                        "destination": "models/PaddleOCR-VL-1.6-GGUF.gguf",
                    },
                    {
                        "name": "paddleocr-vl-mmproj",
                        "destination": "models/PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    return repo, repo / "scripts" / SCRIPT.name


def test_failed_preflight_preserves_space_arguments_and_stops_official(tmp_path: Path) -> None:
    repo, script = _clean_repo(tmp_path)
    tools = tmp_path / "stub tools"
    tools.mkdir()
    argument_log = tmp_path / "argument log.txt"
    stub_python = _stub_python(tools, fail_on="check_server.py")
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
    config.write_text(
        json.dumps(
            {"main_gguf": str(main), "mmproj": str(mmproj), "layout_model_dir": str(layout)}
        ),
        encoding="utf-8",
    )
    scorer_package = tmp_path / "scorer-package.py"
    scorer_package.write_text("VERSION = 1\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        STUB_ARG_LOG=str(argument_log),
        STUB_REPORTED_EXE=str(stub_python),
        STUB_SCORER_PACKAGE=str(scorer_package),
    )
    env["STUB_MISSING_PACKAGE"] = "1"
    missing_package = _run(
        script,
        "-Stage",
        "Preflight",
        "-EvidenceRoot",
        str(evidence),
        "-DatasetDir",
        str(dataset),
        "-LayoutModel",
        str(layout),
        "-RuntimeConfig",
        str(config),
        env=env,
        cwd=tmp_path,
    )
    assert missing_package.returncode != 0
    env.pop("STUB_MISSING_PACKAGE")
    env["STUB_WRONG_VERSION"] = "1"
    wrong_version = _run(
        script,
        "-Stage",
        "Preflight",
        "-EvidenceRoot",
        str(evidence),
        "-DatasetDir",
        str(dataset),
        "-LayoutModel",
        str(layout),
        "-RuntimeConfig",
        str(config),
        env=env,
        cwd=tmp_path,
    )
    assert wrong_version.returncode != 0
    assert "dependency version mismatch" in (wrong_version.stdout + wrong_version.stderr)
    env.pop("STUB_WRONG_VERSION")
    env["STUB_WRONG_ORIGIN"] = "1"
    wrong_origin = _run(
        script,
        "-Stage",
        "Preflight",
        "-EvidenceRoot",
        str(evidence),
        "-DatasetDir",
        str(dataset),
        "-LayoutModel",
        str(layout),
        "-RuntimeConfig",
        str(config),
        env=env,
        cwd=tmp_path,
    )
    assert wrong_origin.returncode != 0
    assert "module origins are outside this worktree" in (wrong_origin.stdout + wrong_origin.stderr)
    env.pop("STUB_WRONG_ORIGIN")

    completed = _run(
        script,
        "-Stage",
        "All",
        "-EvidenceRoot",
        str(evidence),
        "-DatasetDir",
        str(dataset),
        "-LayoutModel",
        str(layout),
        "-RuntimeConfig",
        str(config),
        env=env,
        cwd=tmp_path,
    )

    assert completed.returncode != 0
    arguments = argument_log.read_text(encoding="utf-8").splitlines()
    assert f"dataset={dataset / 'OmniDocBench.json'}" in arguments
    assert f"layout_model={layout / 'inference.onnx'}" in arguments
    assert "check_server.py" in "\n".join(arguments)
    assert "eval.run_eval" not in arguments
    records = [
        json.loads(line) for line in (evidence / "logs" / "commands.jsonl").read_text().splitlines()
    ]
    assert records[-1]["exit_code"] == 23
    assert records[-1]["arguments"][-1] == "<redacted-url>"
    assert "http://" not in json.dumps(records)
    assert str(tmp_path) not in json.dumps(records)
    assert (
        json.loads((evidence / "logs" / "stages" / "preflight.json").read_text())["status"]
        == "failed"
    )

    _stub_python(tools)
    retried = _run(
        script,
        "-Stage",
        "Preflight",
        "-EvidenceRoot",
        str(evidence),
        "-DatasetDir",
        str(dataset),
        "-LayoutModel",
        str(layout),
        "-RuntimeConfig",
        str(config),
        env=env,
        cwd=tmp_path,
    )
    assert retried.returncode == 0, retried.stderr
    state = json.loads((evidence / "logs" / "stages" / "preflight.json").read_text())
    assert state["status"] == "completed"
    before = argument_log.read_text().count("check_server.py")
    skipped = _run(
        script,
        "-Stage",
        "Preflight",
        "-EvidenceRoot",
        str(evidence),
        "-DatasetDir",
        str(dataset),
        "-LayoutModel",
        str(layout),
        "-RuntimeConfig",
        str(config),
        env=env,
        cwd=tmp_path,
    )
    assert skipped.returncode == 0
    assert argument_log.read_text().count("check_server.py") == before
    scorer_package.write_text("VERSION = 2\n", encoding="utf-8")
    package_mutation = _run(
        script,
        "-Stage",
        "Preflight",
        "-EvidenceRoot",
        str(evidence),
        "-DatasetDir",
        str(dataset),
        "-LayoutModel",
        str(layout),
        "-RuntimeConfig",
        str(config),
        env=env,
        cwd=tmp_path,
    )
    assert package_mutation.returncode != 0
    assert "scorer interpreter, origin, RECORD, or package content differs" in (
        package_mutation.stdout + package_mutation.stderr
    )
    scorer_package.write_text("VERSION = 1\n", encoding="utf-8")
    doctor_before = argument_log.read_text().count("doctor")
    missing_official = _run(
        script,
        "-Stage",
        "Lightweight",
        *(
            "-EvidenceRoot",
            str(evidence),
            "-DatasetDir",
            str(dataset),
            "-LayoutModel",
            str(layout),
            "-RuntimeConfig",
            str(config),
        ),
        env=env,
        cwd=tmp_path,
    )
    assert missing_official.returncode != 0
    assert "Missing completed predecessor: Official" in (
        missing_official.stdout + missing_official.stderr
    )
    assert argument_log.read_text().count("doctor") == doctor_before
    official = _run(
        script,
        "-Stage",
        "Official",
        *(
            "-EvidenceRoot",
            str(evidence),
            "-DatasetDir",
            str(dataset),
            "-LayoutModel",
            str(layout),
            "-RuntimeConfig",
            str(config),
        ),
        env=env,
        cwd=tmp_path,
    )
    assert official.returncode == 0, official.stderr
    lightweight_run = _run(
        script,
        "-Stage",
        "Lightweight",
        *(
            "-EvidenceRoot",
            str(evidence),
            "-DatasetDir",
            str(dataset),
            "-LayoutModel",
            str(layout),
            "-RuntimeConfig",
            str(config),
        ),
        env=env,
        cwd=tmp_path,
    )
    assert lightweight_run.returncode == 0, lightweight_run.stderr
    assert argument_log.read_text().count("doctor") > doctor_before
    changed_url = _run(
        script,
        "-Stage",
        "Preflight",
        *(
            "-EvidenceRoot",
            str(evidence),
            "-DatasetDir",
            str(dataset),
            "-LayoutModel",
            str(layout),
            "-RuntimeConfig",
            str(config),
            "-ServerUrl",
            "http://127.0.0.1:9999/v1",
        ),
        env=env,
        cwd=tmp_path,
    )
    assert changed_url.returncode != 0
    assert "invocation fingerprint mismatch" in (changed_url.stdout + changed_url.stderr)
    changed_model = _run(
        script,
        "-Stage",
        "Preflight",
        *(
            "-EvidenceRoot",
            str(evidence),
            "-DatasetDir",
            str(dataset),
            "-LayoutModel",
            str(layout),
            "-RuntimeConfig",
            str(config),
            "-ApiModelName",
            "different-model.gguf",
        ),
        env=env,
        cwd=tmp_path,
    )
    assert changed_model.returncode != 0
    assert "invocation fingerprint mismatch" in (changed_model.stdout + changed_model.stderr)
    other_tools = tmp_path / "other interpreter"
    other_tools.mkdir()
    other_python = _stub_python(other_tools)
    changed_env = env.copy()
    changed_env["STUB_REPORTED_EXE"] = str(other_python)
    changed_interpreter = _run(
        script,
        "-Stage",
        "Preflight",
        *(
            "-EvidenceRoot",
            str(evidence),
            "-DatasetDir",
            str(dataset),
            "-LayoutModel",
            str(layout),
            "-RuntimeConfig",
            str(config),
        ),
        env=changed_env,
        cwd=tmp_path,
    )
    assert changed_interpreter.returncode != 0
    assert "Resume refused" in (changed_interpreter.stdout + changed_interpreter.stderr)
    changed_environment = env.copy()
    changed_environment["STUB_ENV_HASH"] = "b" * 64
    environment_resume = _run(
        script,
        "-Stage",
        "Preflight",
        *(
            "-EvidenceRoot",
            str(evidence),
            "-DatasetDir",
            str(dataset),
            "-LayoutModel",
            str(layout),
            "-RuntimeConfig",
            str(config),
        ),
        env=changed_environment,
        cwd=tmp_path,
    )
    assert environment_resume.returncode != 0
    assert "invocation fingerprint mismatch" in (
        environment_resume.stdout + environment_resume.stderr
    )
    record = tmp_path / "paddleocr-dist-info" / "RECORD"
    original_record = record.read_bytes()
    record.write_bytes(b"mutated")
    record_resume = _run(
        script,
        "-Stage",
        "Preflight",
        *(
            "-EvidenceRoot",
            str(evidence),
            "-DatasetDir",
            str(dataset),
            "-LayoutModel",
            str(layout),
            "-RuntimeConfig",
            str(config),
        ),
        env=env,
        cwd=tmp_path,
    )
    assert record_resume.returncode != 0
    assert "Resume refused" in (record_resume.stdout + record_resume.stderr)
    record.write_bytes(original_record)


def test_rejects_historical_output_before_creating_it(tmp_path: Path) -> None:
    protected = ROOT / "results" / "omnidocbench" / "v16" / "new evidence"
    completed = _run(
        SCRIPT,
        "-EvidenceRoot",
        str(protected),
        "-PythonExe",
        sys.executable,
    )
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
    config.write_text(
        json.dumps(
            {"main_gguf": str(main), "mmproj": str(mmproj), "layout_model_dir": str(layout)}
        ),
        encoding="utf-8",
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    argument_log = tmp_path / "args.log"
    stub_python = _stub_python(tools, fail_on="check_server.py")
    env = os.environ.copy()
    env.update(STUB_ARG_LOG=str(argument_log), STUB_REPORTED_EXE=str(stub_python))

    first = _run(
        script,
        "-Stage",
        "Preflight",
        "-EvidenceRoot",
        str(evidence),
        "-DatasetDir",
        str(dataset),
        "-LayoutModel",
        str(layout),
        "-RuntimeConfig",
        str(config),
        "-ServerUrl",
        "http://127.0.0.1:1/v1",
        env=env,
        cwd=tmp_path,
    )
    assert first.returncode != 0
    assert (evidence / "manifest.json").is_file()
    manifest.write_text('{"changed": true}', encoding="utf-8")

    resumed = _run(
        script,
        "-Stage",
        "Preflight",
        "-EvidenceRoot",
        str(evidence),
        "-DatasetDir",
        str(dataset),
        "-LayoutModel",
        str(layout),
        "-RuntimeConfig",
        str(config),
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
    assert "cdm_tool_environment_sha256" in text
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
    stub_python = _stub_python(tools)
    dataset = tmp_path / "dataset"
    layout = tmp_path / "layout"
    dataset.mkdir()
    layout.mkdir()
    (dataset / "OmniDocBench.json").write_text("{}", encoding="utf-8")
    (layout / "inference.onnx").write_bytes(b"model")
    (layout / "inference.yml").write_text("config", encoding="utf-8")
    main, mmproj = (
        tmp_path / "PaddleOCR-VL-1.6-GGUF.gguf",
        tmp_path / "PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
    )
    main.write_bytes(b"main")
    mmproj.write_bytes(b"mm")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"main_gguf": str(main), "mmproj": str(mmproj), "layout_model_dir": str(layout)}
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence"
    env = os.environ.copy()
    env.update(STUB_ARG_LOG=str(argument_log), STUB_REPORTED_EXE=str(stub_python))
    common = (
        "-EvidenceRoot",
        str(evidence),
        "-DatasetDir",
        str(dataset),
        "-LayoutModel",
        str(layout),
        "-RuntimeConfig",
        str(config),
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
        i
        for i in range(len(args) - 8)
        if args[i] == "eval.run_eval"
        and "infer" in args[i : i + 6]
        and "lightweight" in args[i : i + 12]
    )
    lightweight_args = args[lightweight : lightweight + 24]
    assert ["--server-url", "http://127.0.0.1:8111/v1"] == lightweight_args[
        lightweight_args.index("--server-url") : lightweight_args.index("--server-url") + 2
    ]
    assert ["--api-model-name", "PaddleOCR-VL-1.6-GGUF.gguf"] == lightweight_args[
        lightweight_args.index("--api-model-name") : lightweight_args.index("--api-model-name") + 2
    ]
    assert ["--layout-model", str(layout)] == lightweight_args[
        lightweight_args.index("--layout-model") : lightweight_args.index("--layout-model") + 2
    ]
    decide_state = json.loads((evidence / "logs" / "stages" / "decide.json").read_text())
    assert decide_state["status"] == "completed"
    assert decide_state["output_sha256"]["decision.json"]
    config.write_text(
        json.dumps(
            {
                "main_gguf": str(main),
                "mmproj": str(mmproj),
                "layout_model_dir": str(tmp_path / "wrong-layout"),
            }
        ),
        encoding="utf-8",
    )
    infer_before = argument_log.read_text().count("eval.run_eval")
    mismatch = _run(script, "-Stage", "Lightweight", *common, env=env, cwd=tmp_path)
    assert mismatch.returncode != 0
    assert "layout_model_dir does not match" in (mismatch.stdout + mismatch.stderr)
    assert argument_log.read_text().count("eval.run_eval") == infer_before
    config.write_text(
        json.dumps(
            {"main_gguf": str(main), "mmproj": str(mmproj), "layout_model_dir": str(layout)}
        ),
        encoding="utf-8",
    )
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
    decisions_before = argument_log.read_text().splitlines().count("decide")
    predecessor_tampered = _run(script, "-Stage", "Decide", *common, env=env, cwd=tmp_path)
    assert predecessor_tampered.returncode != 0
    assert "Predecessor command log hash mismatch" in (
        predecessor_tampered.stdout + predecessor_tampered.stderr
    )
    assert argument_log.read_text().splitlines().count("decide") == decisions_before
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


def test_failed_score_retry_preserves_inference_and_both_scores_gate_official(
    tmp_path: Path,
) -> None:
    _, script = _clean_repo(tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    argument_log = tmp_path / "args.log"
    stub_python = _stub_python(tools, fail_on="--copy-report")
    dataset = tmp_path / "dataset"
    layout = tmp_path / "layout"
    dataset.mkdir()
    layout.mkdir()
    (dataset / "OmniDocBench.json").write_text("{}", encoding="utf-8")
    (layout / "inference.onnx").write_bytes(b"model")
    (layout / "inference.yml").write_text("config", encoding="utf-8")
    main = tmp_path / "PaddleOCR-VL-1.6-GGUF.gguf"
    mmproj = tmp_path / "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
    main.write_bytes(b"main")
    mmproj.write_bytes(b"mm")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"main_gguf": str(main), "mmproj": str(mmproj), "layout_model_dir": str(layout)}
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence"
    env = os.environ.copy()
    env.update(STUB_ARG_LOG=str(argument_log), STUB_REPORTED_EXE=str(stub_python))
    common = (
        "-EvidenceRoot",
        str(evidence),
        "-DatasetDir",
        str(dataset),
        "-LayoutModel",
        str(layout),
        "-RuntimeConfig",
        str(config),
    )
    assert _run(script, "-Stage", "Preflight", *common, env=env, cwd=tmp_path).returncode == 0

    failed = _run(script, "-Stage", "Official", *common, env=env, cwd=tmp_path)

    assert failed.returncode != 0
    stats = evidence / "official" / "_run_stats.json"
    assert stats.is_file()
    stats_before = stats.read_bytes()
    infer_before = argument_log.read_text(encoding="utf-8").splitlines().count("infer")

    _stub_python(tools)
    retried = _run(script, "-Stage", "Official", *common, env=env, cwd=tmp_path)

    assert retried.returncode == 0, retried.stderr
    assert stats.read_bytes() == stats_before
    assert argument_log.read_text(encoding="utf-8").splitlines().count("infer") == infer_before
    result_dir = evidence / "results" / "official"
    for name in (
        "metric.json",
        "run-summary.json",
        "provenance.json",
        "metric-cdm.json",
        "run-summary-cdm.json",
        "provenance-cdm.json",
    ):
        assert (result_dir / name).is_file()

    (result_dir / "metric-cdm.json").unlink()
    blocked = _run(script, "-Stage", "Lightweight", *common, env=env, cwd=tmp_path)
    assert blocked.returncode != 0
    assert "Predecessor output hash/set mismatch: Official" in (blocked.stdout + blocked.stderr)


def test_score_only_recovery_binds_assets_then_continues_lightweight_and_decide(
    tmp_path: Path,
) -> None:
    _, script = _clean_repo(tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    argument_log = tmp_path / "args.log"
    stub_python = _stub_python(tools)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "OmniDocBench.json").write_text("{}", encoding="utf-8")
    layout = tmp_path / "layout"
    layout.mkdir()
    layout_onnx = layout / "inference.onnx"
    layout_onnx.write_bytes(b"layout-model")
    (layout / "inference.yml").write_text("layout-config", encoding="utf-8")
    main = tmp_path / "PaddleOCR-VL-1.6-GGUF.gguf"
    mmproj = tmp_path / "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
    main.write_bytes(b"main")
    mmproj.write_bytes(b"mmproj")
    config = tmp_path / "runtime config.json"
    config.write_text(
        json.dumps(
            {"main_gguf": str(main), "mmproj": str(mmproj), "layout_model_dir": str(layout)}
        ),
        encoding="utf-8",
    )
    source = tmp_path / "immutable source"
    for relative in (
        "manifest.json",
        "logs/stages/official.json",
        "logs/stages/official.commands.jsonl",
        "official/_run_stats.json",
        "official/_errors.log",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    evidence = tmp_path / "recovery evidence"
    env = os.environ.copy()
    env.update(STUB_ARG_LOG=str(argument_log), STUB_REPORTED_EXE=str(stub_python))
    common = (
        "-EvidenceRoot",
        str(evidence),
        "-RecoverySourceRoot",
        str(source),
        "-DatasetDir",
        str(dataset),
        "-LayoutModel",
        str(layout),
        "-RuntimeConfig",
        str(config),
    )

    completed = _run(
        script,
        "-Stage",
        "OfficialScore",
        *common,
        env=env,
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = argument_log.read_text(encoding="utf-8").splitlines()
    assert "infer" not in arguments
    assert arguments.count("eval.score_recovery") == 2
    assert (evidence / "recovery" / "source.json").is_file()
    assert (evidence / "results" / "official" / "metric.json").is_file()
    assert (evidence / "results" / "official" / "metric-cdm.json").is_file()
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    assert {
        "layout_model",
        "layout_config",
        "runtime_config",
        "runtime_manifest",
        "main_gguf",
        "mmproj",
    } <= set(manifest["inputs"])

    layout_onnx.write_bytes(b"mutated")
    rejected = _run(script, "-Stage", "Lightweight", *common, env=env, cwd=tmp_path)
    assert rejected.returncode != 0
    assert "Resume refused" in (rejected.stdout + rejected.stderr)
    assert argument_log.read_text(encoding="utf-8").splitlines().count("infer") == 0

    layout_onnx.write_bytes(b"layout-model")
    lightweight = _run(script, "-Stage", "Lightweight", *common, env=env, cwd=tmp_path)
    assert lightweight.returncode == 0, lightweight.stderr
    decided = _run(script, "-Stage", "Decide", *common, env=env, cwd=tmp_path)
    assert decided.returncode == 0, decided.stderr
    assert (evidence / "decision.json").is_file()


def test_score_only_recovery_rejects_source_output_overlap(tmp_path: Path) -> None:
    _, script = _clean_repo(tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    argument_log = tmp_path / "args.log"
    stub_python = _stub_python(tools)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "OmniDocBench.json").write_text("{}", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    sentinel = source / "sentinel.txt"
    sentinel.write_text("immutable", encoding="utf-8")
    env = os.environ.copy()
    env.update(STUB_ARG_LOG=str(argument_log), STUB_REPORTED_EXE=str(stub_python))

    completed = _run(
        script,
        "-Stage",
        "OfficialScore",
        "-EvidenceRoot",
        str(source),
        "-RecoverySourceRoot",
        str(source),
        "-DatasetDir",
        str(dataset),
        env=env,
        cwd=tmp_path,
    )

    assert completed.returncode != 0
    assert "must not overlap" in (completed.stdout + completed.stderr)
    assert sentinel.read_text(encoding="utf-8") == "immutable"


def test_standalone_stage_requires_preflight_before_native_commands(tmp_path: Path) -> None:
    _, script = _clean_repo(tmp_path)
    completed = _run(
        script,
        "-Stage",
        "Official",
        "-EvidenceRoot",
        str(tmp_path / "evidence"),
        "-PythonExe",
        sys.executable,
        cwd=tmp_path,
    )
    assert completed.returncode != 0
    assert "Missing completed predecessor: Preflight" in (completed.stdout + completed.stderr)
    assert not (tmp_path / "evidence").exists()


def test_release_evidence_module_imports_without_pythonpath_from_foreign_cwd(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    command = f"Push-Location '{ROOT}'; & '{sys.executable}' -m eval.release_evidence --help"
    completed = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", command],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "manifest" in completed.stdout and "decide" in completed.stdout


def test_worktree_venv_reports_worktree_module_origins() -> None:
    probe = (
        "import eval.release_evidence,json,paddleocr_vl_rocm,sys; "
        "print(json.dumps({'executable':sys.executable,'eval':eval.release_evidence.__file__,"
        "'package':paddleocr_vl_rocm.__file__}))"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert completed.returncode == 0, completed.stderr
    origins = json.loads(completed.stdout)
    assert Path(origins["executable"]).resolve() == Path(sys.executable).resolve()
    assert Path(origins["eval"]).resolve().is_relative_to((ROOT / "eval").resolve())
    assert Path(origins["package"]).resolve().is_relative_to((ROOT / "src").resolve())


def test_exact_origin_probe_survives_windows_powershell_5_native_quoting(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign cwd with spaces"
    foreign.mkdir()
    text = SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"\$probe = '([^']+)'", text)
    assert match is not None
    probe = match.group(1)
    assert '"' not in probe and "'" not in probe
    command = f"Push-Location '{ROOT}'; $probe = '{probe}'; & '{sys.executable}' -c $probe"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", command],
        cwd=foreign,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "NameError" not in completed.stderr
    value = json.loads(completed.stdout)
    assert {"version", "executable", "eval_origin", "package_origin"} <= set(value)
    assert value["core_versions"] == {
        "paddleocr": "3.7.0",
        "paddlex": "3.7.2",
        "paddlepaddle": "3.2.1",
    }
    assert all(len(digest) == 64 for digest in value["record_sha256"].values())
    assert Path(value["eval_origin"]).is_relative_to(ROOT / "eval")
    assert Path(value["package_origin"]).is_relative_to(ROOT / "src")


def test_clean_gate_rejects_untracked_file_from_foreign_cwd(tmp_path: Path) -> None:
    repo, script = _clean_repo(tmp_path)
    (repo / "unexpected.txt").write_text("dirty", encoding="utf-8")
    completed = _run(
        script,
        "-EvidenceRoot",
        str(tmp_path / "external evidence"),
        "-PythonExe",
        sys.executable,
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
