from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from rich.console import Console

from paddleocr_vl_rocm import doctor
from paddleocr_vl_rocm.doctor import (
    CheckResult,
    CheckSpec,
    DoctorContext,
    checks_to_json,
    doctor_exit_code,
    render_checks,
    run_doctor,
)


def test_run_doctor_continues_after_check_error_and_adds_remediation(monkeypatch):
    def broken(_context):
        raise RuntimeError("controlled failure")

    def healthy(_context):
        return CheckResult("healthy", "PASS", "ready", "", {})

    monkeypatch.setattr(
        doctor,
        "DOCTOR_CHECKS",
        (
            CheckSpec("broken", broken, "Repair the broken component."),
            CheckSpec("healthy", healthy, "Repair health."),
        ),
    )

    results = run_doctor({})

    assert [result.name for result in results] == ["broken", "healthy"]
    assert results[0].status == "FAIL"
    assert results[0].remediation == "Repair the broken component."
    assert results[1].status == "PASS"


def test_amd_adapter_check_parses_compact_powershell_json(monkeypatch):
    completed = Mock(returncode=0, stdout='[{"Name":"AMD Radeon 8060S"}]', stderr="")
    monkeypatch.setattr(doctor.subprocess, "run", Mock(return_value=completed))

    result = doctor._check_amd_gpu(DoctorContext(config={}, root=Path.cwd(), server_url=None))

    assert result.status == "PASS"
    assert result.details["adapters"] == ["AMD Radeon 8060S"]
    command = doctor.subprocess.run.call_args.args[0]
    assert "ConvertTo-Json -Compress" in command[-1]


def test_directml_check_fails_closed_when_provider_is_missing(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]),
    )
    context = DoctorContext(
        config={"layout_model_dir": "models/layout"},
        root=Path.cwd(),
        server_url=None,
    )

    result = doctor._check_directml(context)

    assert result.status == "FAIL"
    assert "DmlExecutionProvider" in result.message
    assert result.remediation


def test_resource_check_reports_hash_failure(tmp_path, monkeypatch):
    names = {
        "llama-cpp-hip-runtime": "runtime/archive.zip",
        "paddleocr-vl-main-gguf": "models/main.gguf",
        "paddleocr-vl-mmproj": "models/mmproj.gguf",
        "pp-doclayout-v3-onnx": "models/layout/inference.onnx",
        "pp-doclayout-v3-config": "models/layout/inference.yml",
    }
    manifest = {
        "resources": [
            {
                "name": name,
                "url": f"https://example.test/{name}",
                "destination": destination,
                "size": 4,
                "sha256": "0" * 64,
            }
            for name, destination in names.items()
        ]
    }
    monkeypatch.setattr(doctor, "load_runtime_manifest", lambda: manifest)

    def fake_verify(_path, resource):
        if resource.name == "paddleocr-vl-main-gguf":
            raise RuntimeError("SHA-256 mismatch")

    monkeypatch.setattr(doctor, "verify_resource", fake_verify)

    result = doctor._check_resources(DoctorContext(config={}, root=tmp_path, server_url=None))

    assert result.status == "FAIL"
    assert "paddleocr-vl-main-gguf" in result.message
    assert result.remediation


def test_server_check_redacts_credentials_and_query_secrets(monkeypatch):
    secret_url = "http://user:password@127.0.0.1:8111/v1?api_key=top-secret"
    monkeypatch.setattr(
        doctor.requests,
        "get",
        Mock(side_effect=requests.RequestException(f"failed for {secret_url}")),
    )

    result = doctor._check_server(DoctorContext(config={}, root=Path.cwd(), server_url=secret_url))

    serialized = json.dumps(result.to_dict())
    assert result.status == "FAIL"
    assert "password" not in serialized
    assert "top-secret" not in serialized
    assert "127.0.0.1:8111" in serialized
    requested = doctor.requests.get.call_args.args[0]
    assert requested == "http://user:password@127.0.0.1:8111/v1/models?api_key=top-secret"


def test_server_check_rejects_non_models_json_and_closes_response(monkeypatch):
    response = Mock(status_code=200)
    response.json.return_value = {"error": "not a models response"}
    monkeypatch.setattr(doctor.requests, "get", Mock(return_value=response))

    result = doctor._check_server(
        DoctorContext(config={}, root=Path.cwd(), server_url="http://127.0.0.1:8111/v1")
    )

    assert result.status == "FAIL"
    assert "data" in result.message
    response.close.assert_called_once()


def test_disk_check_uses_nearest_existing_parent(tmp_path, monkeypatch):
    missing = tmp_path / "not-created" / "managed-root"
    disk_usage = Mock(return_value=SimpleNamespace(free=10 * 1024**3, total=20 * 1024**3))
    monkeypatch.setattr(doctor.shutil, "disk_usage", disk_usage)

    result = doctor._check_disk(DoctorContext(config={}, root=missing, server_url=None))

    assert result.status == "PASS"
    assert disk_usage.call_args.args[0] == tmp_path


def test_hip_runtime_accepts_dll_next_to_managed_llama_server(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "amdhip64_7.dll").write_bytes(b"dll")
    monkeypatch.setenv("WINDIR", str(tmp_path / "windows"))
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("ROCM_PATH", raising=False)
    context = DoctorContext(
        config={"runtime_dir": str(runtime), "llama_server": str(runtime / "llama-server.exe")},
        root=tmp_path,
        server_url=None,
    )

    result = doctor._check_hip_runtime(context)

    assert result.status == "PASS"
    assert result.details["path"].endswith("amdhip64_7.dll")


def test_runtime_check_rejects_version_number_as_substring(tmp_path, monkeypatch):
    server = tmp_path / "llama-server.exe"
    server.write_bytes(b"exe")
    monkeypatch.setattr(doctor, "_runtime_version", lambda _path: "version: b19884")
    context = DoctorContext(
        config={"llama_server": str(server), "runtime_version": "b9884"},
        root=tmp_path,
        server_url=None,
    )

    result = doctor._check_runtime(context)

    assert result.status == "FAIL"


def test_windows_check_rejects_unsupported_release(monkeypatch):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Windows")
    monkeypatch.setattr(doctor.platform, "release", lambda: "8")
    monkeypatch.setattr(doctor.platform, "version", lambda: "6.2.9200")

    result = doctor._check_windows(DoctorContext(config={}, root=Path.cwd(), server_url=None))

    assert result.status == "FAIL"
    assert "Windows 10 or Windows 11" in result.remediation


def test_resource_check_rejects_incomplete_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "load_runtime_manifest", lambda: {"resources": []})

    result = doctor._check_resources(DoctorContext(config={}, root=tmp_path, server_url=None))

    assert result.status == "FAIL"
    assert "missing" in result.message.lower()


def test_renderers_and_exit_codes_are_stable():
    checks = [
        CheckResult("gpu", "PASS", "ready", "", {"name": "AMD"}),
        CheckResult("disk", "WARN", "low", "Free disk space.", {}),
        CheckResult("server", "FAIL", "offline", "Start the server.", {}),
    ]
    console = Console(record=True, width=120)

    render_checks(checks, console=console)

    rendered = console.export_text()
    assert "gpu" in rendered and "server" in rendered
    assert json.loads(checks_to_json(checks))[2]["status"] == "FAIL"
    assert doctor_exit_code(checks) == 2
    assert doctor_exit_code(checks[:2]) == 0


@pytest.mark.parametrize("status", ["PASS", "WARN", "FAIL"])
def test_check_result_accepts_only_stable_statuses(status):
    assert CheckResult("check", status, "message", "fix", {}).status == status


def test_check_result_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        CheckResult("check", "UNKNOWN", "message", "fix", {})
