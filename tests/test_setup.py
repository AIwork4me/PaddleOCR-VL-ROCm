from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from paddleocr_vl_rocm import setup as managed_setup
from paddleocr_vl_rocm.resources import Resource
from paddleocr_vl_rocm.setup import SetupOptions, setup_managed_runtime, start_managed_server


def _resource(name: str, destination: str, content: bytes) -> tuple[dict, bytes]:
    return (
        {
            "name": name,
            "url": f"https://example.test/{name}",
            "destination": destination,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
        content,
    )


def _runtime_zip(member: str = "llama-server.exe", content: bytes = b"runtime") -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(member, content)
    return output.getvalue()


def _manifest(runtime_bytes: bytes) -> tuple[dict, dict[str, bytes]]:
    entries = [
        _resource(
            "llama-cpp-hip-runtime",
            "runtime/llama-b9884-bin-win-hip-radeon-x64.zip",
            runtime_bytes,
        ),
        _resource("paddleocr-vl-main-gguf", "models/main.gguf", b"main"),
        _resource("paddleocr-vl-mmproj", "models/mmproj.gguf", b"mmproj"),
        _resource("pp-doclayout-v3-onnx", "models/layout/inference.onnx", b"onnx"),
        _resource("pp-doclayout-v3-config", "models/layout/inference.yml", b"config"),
    ]
    return (
        {
            "schema": 1,
            "runtime": {"version": "b9884", "commit": "86961efd5"},
            "resources": [entry for entry, _content in entries],
        },
        {entry["name"]: content for entry, content in entries},
    )


def _install_fakes(monkeypatch, manifest, contents, calls):
    monkeypatch.setattr(managed_setup, "load_runtime_manifest", lambda: manifest)

    def fake_download(resource: Resource, root: Path, **_kwargs) -> Path:
        calls.append(resource.name)
        destination = root.joinpath(*Path(resource.destination).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents[resource.name])
        return destination

    monkeypatch.setattr(managed_setup, "download_resource", fake_download)
    monkeypatch.setattr(managed_setup, "_runtime_version", lambda _path: "llama.cpp b9884")


def test_setup_installs_resources_and_writes_local_config(tmp_path, monkeypatch):
    manifest, contents = _manifest(_runtime_zip())
    calls = []
    _install_fakes(monkeypatch, manifest, contents, calls)

    result = setup_managed_runtime(SetupOptions(root=tmp_path))

    assert set(calls) == set(contents)
    assert result.llama_server == tmp_path / "runtime" / "llama-server.exe"
    assert result.main_gguf.read_bytes() == b"main"
    assert result.mmproj.read_bytes() == b"mmproj"
    assert (result.layout_model_dir / "inference.onnx").read_bytes() == b"onnx"
    config = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config["manifest_schema"] == 1
    assert config["runtime_version"] == "b9884"
    assert "token" not in result.config_path.read_text(encoding="utf-8").lower()


def test_setup_reuses_complete_install_without_downloading(tmp_path, monkeypatch):
    manifest, contents = _manifest(_runtime_zip())
    calls = []
    _install_fakes(monkeypatch, manifest, contents, calls)
    setup_managed_runtime(SetupOptions(root=tmp_path))
    monkeypatch.setattr(managed_setup, "download_resource", pytest.fail)

    result = setup_managed_runtime(SetupOptions(root=tmp_path))

    assert result.llama_server.is_file()


def test_setup_extraction_failure_preserves_previous_runtime(tmp_path, monkeypatch):
    manifest, contents = _manifest(_runtime_zip())
    calls = []
    _install_fakes(monkeypatch, manifest, contents, calls)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    old_server = runtime / "llama-server.exe"
    old_server.write_bytes(b"old runtime")
    monkeypatch.setattr(managed_setup, "_extract_runtime", lambda *_args: (_ for _ in ()).throw(RuntimeError("extract")))

    with pytest.raises(RuntimeError, match="extract"):
        setup_managed_runtime(SetupOptions(root=tmp_path, force=True))

    assert old_server.read_bytes() == b"old runtime"


def test_extract_runtime_rejects_zip_path_traversal(tmp_path):
    archive = tmp_path / "runtime.zip"
    archive.write_bytes(_runtime_zip("../outside.exe"))

    with pytest.raises(RuntimeError, match="unsafe path"):
        managed_setup._extract_runtime(archive, tmp_path / "staging")


@pytest.mark.parametrize("member", ["C:/evil.exe", "C:evil.exe", "//server/share/evil.exe"])
def test_extract_runtime_rejects_windows_drive_paths(tmp_path, member):
    archive = tmp_path / "runtime.zip"
    archive.write_bytes(_runtime_zip(member))

    with pytest.raises(RuntimeError, match="unsafe path"):
        managed_setup._extract_runtime(archive, tmp_path / "staging")


def test_start_managed_server_uses_argument_list_and_health_check(tmp_path, monkeypatch):
    manifest, contents = _manifest(_runtime_zip())
    calls = []
    _install_fakes(monkeypatch, manifest, contents, calls)
    result = setup_managed_runtime(SetupOptions(root=tmp_path))
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(managed_setup.subprocess, "Popen", Mock(return_value=process))
    response = Mock(status_code=200)
    response.json.return_value = {"data": []}
    monkeypatch.setattr(managed_setup.requests, "get", Mock(return_value=response))

    returned = start_managed_server(result, port=8111, timeout=1.0)

    assert returned is process
    args = managed_setup.subprocess.Popen.call_args.args[0]
    assert args == [
        str(result.llama_server),
        "-m",
        str(result.main_gguf),
        "--mmproj",
        str(result.mmproj),
        "--host",
        "127.0.0.1",
        "--port",
        "8111",
        "-ngl",
        "99",
    ]
    assert managed_setup.subprocess.Popen.call_args.kwargs["shell"] is False
    response.close.assert_called_once()


def test_start_managed_server_reports_early_process_exit(tmp_path, monkeypatch):
    result = managed_setup.SetupResult(
        root=tmp_path,
        runtime_dir=tmp_path / "runtime",
        llama_server=tmp_path / "runtime" / "llama-server.exe",
        main_gguf=tmp_path / "models" / "main.gguf",
        mmproj=tmp_path / "models" / "mmproj.gguf",
        layout_model_dir=tmp_path / "models" / "layout",
        config_path=tmp_path / "config.json",
    )
    process = Mock()
    process.poll.return_value = 7
    monkeypatch.setattr(managed_setup.subprocess, "Popen", Mock(return_value=process))
    get = Mock()
    monkeypatch.setattr(managed_setup.requests, "get", get)

    with pytest.raises(RuntimeError, match="exited with code 7.*server.log"):
        start_managed_server(result, timeout=60.0)

    get.assert_not_called()
    process.terminate.assert_not_called()


def test_start_managed_server_rechecks_process_after_successful_health_response(
    tmp_path, monkeypatch
):
    result = managed_setup.SetupResult(
        root=tmp_path,
        runtime_dir=tmp_path / "runtime",
        llama_server=tmp_path / "runtime" / "llama-server.exe",
        main_gguf=tmp_path / "models" / "main.gguf",
        mmproj=tmp_path / "models" / "mmproj.gguf",
        layout_model_dir=tmp_path / "models" / "layout",
        config_path=tmp_path / "config.json",
    )
    process = Mock()
    process.poll.side_effect = [None, 7]
    monkeypatch.setattr(managed_setup.subprocess, "Popen", Mock(return_value=process))
    response = Mock(status_code=200)
    response.json.return_value = {"data": []}
    monkeypatch.setattr(managed_setup.requests, "get", Mock(return_value=response))

    with pytest.raises(RuntimeError, match="exited with code 7.*server.log"):
        start_managed_server(result, timeout=1.0)

    response.close.assert_called_once()


def test_start_managed_server_timeout_terminates_process(tmp_path, monkeypatch):
    result = managed_setup.SetupResult(
        root=tmp_path,
        runtime_dir=tmp_path / "runtime",
        llama_server=tmp_path / "runtime" / "llama-server.exe",
        main_gguf=tmp_path / "models" / "main.gguf",
        mmproj=tmp_path / "models" / "mmproj.gguf",
        layout_model_dir=tmp_path / "models" / "layout",
        config_path=tmp_path / "config.json",
    )
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(managed_setup.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(
        managed_setup.requests,
        "get",
        Mock(side_effect=requests.RequestException("not ready")),
    )
    monkeypatch.setattr(managed_setup.time, "monotonic", Mock(side_effect=[0.0, 2.0]))
    monkeypatch.setattr(managed_setup.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="server.log"):
        start_managed_server(result, timeout=1.0)

    process.terminate.assert_called_once()


def test_start_managed_server_kill_path_waits_for_exit(tmp_path, monkeypatch):
    result = managed_setup.SetupResult(
        root=tmp_path,
        runtime_dir=tmp_path / "runtime",
        llama_server=tmp_path / "runtime" / "llama-server.exe",
        main_gguf=tmp_path / "models" / "main.gguf",
        mmproj=tmp_path / "models" / "mmproj.gguf",
        layout_model_dir=tmp_path / "models" / "layout",
        config_path=tmp_path / "config.json",
    )
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("llama-server", 5.0), 0]
    monkeypatch.setattr(managed_setup.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(
        managed_setup.requests,
        "get",
        Mock(side_effect=requests.RequestException("not ready")),
    )
    monkeypatch.setattr(managed_setup.time, "monotonic", Mock(side_effect=[0.0, 2.0]))
    monkeypatch.setattr(managed_setup.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="server.log"):
        start_managed_server(result, timeout=1.0)

    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2


def test_start_managed_server_preserves_diagnostic_when_terminate_races(tmp_path, monkeypatch):
    result = managed_setup.SetupResult(
        root=tmp_path,
        runtime_dir=tmp_path / "runtime",
        llama_server=tmp_path / "runtime" / "llama-server.exe",
        main_gguf=tmp_path / "models" / "main.gguf",
        mmproj=tmp_path / "models" / "mmproj.gguf",
        layout_model_dir=tmp_path / "models" / "layout",
        config_path=tmp_path / "config.json",
    )
    process = Mock()
    process.poll.return_value = None
    process.terminate.side_effect = ProcessLookupError("already exited")
    monkeypatch.setattr(managed_setup.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(
        managed_setup.requests,
        "get",
        Mock(side_effect=requests.RequestException("not ready")),
    )
    monkeypatch.setattr(managed_setup.time, "monotonic", Mock(side_effect=[0.0, 2.0]))

    with pytest.raises(RuntimeError, match="server.log"):
        start_managed_server(result, timeout=1.0)


def test_start_managed_server_preserves_diagnostic_when_kill_wait_times_out(
    tmp_path, monkeypatch
):
    result = managed_setup.SetupResult(
        root=tmp_path,
        runtime_dir=tmp_path / "runtime",
        llama_server=tmp_path / "runtime" / "llama-server.exe",
        main_gguf=tmp_path / "models" / "main.gguf",
        mmproj=tmp_path / "models" / "mmproj.gguf",
        layout_model_dir=tmp_path / "models" / "layout",
        config_path=tmp_path / "config.json",
    )
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = subprocess.TimeoutExpired("llama-server", 5.0)
    monkeypatch.setattr(managed_setup.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(
        managed_setup.requests,
        "get",
        Mock(side_effect=requests.RequestException("not ready")),
    )
    monkeypatch.setattr(managed_setup.time, "monotonic", Mock(side_effect=[0.0, 2.0]))

    with pytest.raises(RuntimeError, match="server.log"):
        start_managed_server(result, timeout=1.0)

    process.kill.assert_called_once()
