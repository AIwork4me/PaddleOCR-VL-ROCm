from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import requests

from .resources import Resource, download_resource, load_runtime_manifest, verify_resource


@dataclass(frozen=True)
class SetupOptions:
    root: Path | None = None
    force: bool = False


@dataclass(frozen=True)
class SetupResult:
    root: Path
    runtime_dir: Path
    llama_server: Path
    main_gguf: Path
    mmproj: Path
    layout_model_dir: Path
    config_path: Path


def _default_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "PaddleOCR-VL-ROCm"
    return Path.home() / "AppData" / "Local" / "PaddleOCR-VL-ROCm"


def _resource_map(manifest: dict) -> dict[str, Resource]:
    resources = {
        str(value["name"]): Resource.from_mapping(value) for value in manifest["resources"]
    }
    required = {
        "llama-cpp-hip-runtime",
        "paddleocr-vl-main-gguf",
        "paddleocr-vl-mmproj",
        "pp-doclayout-v3-onnx",
        "pp-doclayout-v3-config",
    }
    missing = required.difference(resources)
    if missing:
        raise RuntimeError(f"Runtime manifest is missing resources: {sorted(missing)}")
    return resources


def _installed_path(root: Path, resource: Resource) -> Path:
    return root.joinpath(*PurePosixPath(resource.destination).parts)


def _result(root: Path, resources: dict[str, Resource]) -> SetupResult:
    main = _installed_path(root, resources["paddleocr-vl-main-gguf"])
    mmproj = _installed_path(root, resources["paddleocr-vl-mmproj"])
    layout_onnx = _installed_path(root, resources["pp-doclayout-v3-onnx"])
    runtime_dir = root / "runtime"
    return SetupResult(
        root=root,
        runtime_dir=runtime_dir,
        llama_server=runtime_dir / "llama-server.exe",
        main_gguf=main,
        mmproj=mmproj,
        layout_model_dir=layout_onnx.parent,
        config_path=root / "config.json",
    )


def _runtime_version(llama_server: Path) -> str:
    completed = subprocess.run(
        [str(llama_server), "--version"],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    if completed.returncode != 0:
        raise RuntimeError(f"llama-server --version failed ({completed.returncode}): {output}")
    return output


def _verify_runtime(llama_server: Path, version: str) -> None:
    if not llama_server.is_file():
        raise RuntimeError(f"llama-server.exe not found: {llama_server}")
    output = _runtime_version(llama_server)
    expected_build = version.removeprefix("b")
    if version not in output and expected_build not in output:
        raise RuntimeError(f"Unexpected llama-server version at {llama_server}: {output}")


def _extract_runtime(archive_path: Path, staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=False)
    resolved_staging = staging.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            name = member.filename
            relative = PurePosixPath(name)
            windows_path = PureWindowsPath(name)
            mode = member.external_attr >> 16
            if (
                not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or "\\" in name
                or bool(windows_path.drive)
                or stat.S_ISLNK(mode)
            ):
                raise RuntimeError(f"Runtime archive contains unsafe path: {name}")
            destination = staging.joinpath(*relative.parts)
            if not destination.resolve().is_relative_to(resolved_staging):
                raise RuntimeError(f"Runtime archive contains unsafe path: {name}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def _existing_install(
    result: SetupResult,
    resources: dict[str, Resource],
    manifest: dict,
) -> bool:
    if not result.config_path.is_file():
        return False
    try:
        config = json.loads(result.config_path.read_text(encoding="utf-8"))
        if config.get("manifest_schema") != manifest["schema"]:
            return False
        if config.get("runtime_version") != manifest["runtime"]["version"]:
            return False
        _verify_runtime(result.llama_server, manifest["runtime"]["version"])
        for name in (
            "paddleocr-vl-main-gguf",
            "paddleocr-vl-mmproj",
            "pp-doclayout-v3-onnx",
            "pp-doclayout-v3-config",
        ):
            resource = resources[name]
            verify_resource(_installed_path(result.root, resource), resource)
    except (KeyError, OSError, RuntimeError, ValueError):
        return False
    return True


def _activate_runtime(staging: Path, runtime_dir: Path) -> None:
    backup = runtime_dir.parent / f".runtime-backup-{uuid.uuid4().hex}"
    had_runtime = runtime_dir.exists()
    if had_runtime:
        runtime_dir.replace(backup)
    try:
        staging.replace(runtime_dir)
    except Exception:
        if had_runtime and backup.exists() and not runtime_dir.exists():
            backup.replace(runtime_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _write_config(result: SetupResult, manifest: dict) -> None:
    config = {
        "manifest_schema": manifest["schema"],
        "runtime_version": manifest["runtime"]["version"],
        "runtime_dir": str(result.runtime_dir),
        "llama_server": str(result.llama_server),
        "main_gguf": str(result.main_gguf),
        "mmproj": str(result.mmproj),
        "layout_model_dir": str(result.layout_model_dir),
        "server_port": 8111,
    }
    temporary = result.config_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(result.config_path)


def setup_managed_runtime(options: SetupOptions | None = None) -> SetupResult:
    options = options or SetupOptions()
    root = (options.root or _default_root()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "cache").mkdir(exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    manifest = load_runtime_manifest()
    resources = _resource_map(manifest)
    result = _result(root, resources)
    if not options.force and _existing_install(result, resources, manifest):
        return result

    runtime_archive = download_resource(
        resources["llama-cpp-hip-runtime"], root / "cache"
    )
    for name in (
        "paddleocr-vl-main-gguf",
        "paddleocr-vl-mmproj",
        "pp-doclayout-v3-onnx",
        "pp-doclayout-v3-config",
    ):
        download_resource(resources[name], root)

    staging = root / f".runtime-staging-{uuid.uuid4().hex}"
    try:
        _extract_runtime(runtime_archive, staging)
        _verify_runtime(staging / "llama-server.exe", manifest["runtime"]["version"])
        _activate_runtime(staging, result.runtime_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    _write_config(result, manifest)
    return result


def start_managed_server(
    result: SetupResult,
    *,
    port: int = 8111,
    timeout: float = 60.0,
) -> subprocess.Popen:
    logs_dir = result.root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "server.log"
    args = [
        str(result.llama_server),
        "-m",
        str(result.main_gguf),
        "--mmproj",
        str(result.mmproj),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-ngl",
        "99",
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            args,
            stdout=log,
            stderr=subprocess.STDOUT,
            shell=False,
            creationflags=creationflags,
        )

    deadline = time.monotonic() + timeout
    models_url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Managed llama-server exited with code {return_code}; inspect {log_path}"
            )
        response = None
        try:
            response = requests.get(models_url, timeout=min(2.0, timeout))
            if response.status_code < 400:
                response.json()
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        f"Managed llama-server exited with code {return_code}; "
                        f"inspect {log_path}"
                    )
                return process
        except (requests.RequestException, ValueError):
            pass
        finally:
            if response is not None:
                response.close()
        time.sleep(0.25)

    if process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
        else:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                else:
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        pass
    raise RuntimeError(f"Managed llama-server did not become ready; inspect {log_path}")
