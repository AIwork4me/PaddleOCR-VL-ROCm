from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import requests
from rich.console import Console
from rich.table import Table

from .layout import PPDocLayoutV3Onnx, resolve_layout_providers
from .resources import Resource, load_runtime_manifest, verify_resource
from .setup import _default_root, _runtime_version

CheckStatus = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    remediation: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "WARN", "FAIL"}:
            raise ValueError(f"Invalid check status: {self.status}")
        if self.status == "FAIL" and not self.remediation.strip():
            raise ValueError(f"FAIL check requires remediation: {self.name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "remediation": self.remediation,
            "details": self.details,
        }


@dataclass(frozen=True)
class DoctorContext:
    config: Mapping[str, Any]
    root: Path
    server_url: str | None


@dataclass(frozen=True)
class CheckSpec:
    name: str
    check: Callable[[DoctorContext], CheckResult]
    remediation: str


def _pass(name: str, message: str, details: dict[str, Any] | None = None) -> CheckResult:
    return CheckResult(name, "PASS", message, "", details or {})


def _warn(
    name: str,
    message: str,
    remediation: str,
    details: dict[str, Any] | None = None,
) -> CheckResult:
    return CheckResult(name, "WARN", message, remediation, details or {})


def _fail(
    name: str,
    message: str,
    remediation: str,
    details: dict[str, Any] | None = None,
) -> CheckResult:
    return CheckResult(name, "FAIL", message, remediation, details or {})


def _check_config(context: DoctorContext) -> CheckResult:
    error = context.config.get("_config_error")
    if error:
        return _fail("config", str(error), "Run managed setup again or repair config.json.")
    required = {
        "runtime_version",
        "llama_server",
        "main_gguf",
        "mmproj",
        "layout_model_dir",
        "server_port",
    }
    missing = sorted(required.difference(context.config))
    if missing:
        return _fail(
            "config",
            f"Managed configuration is missing: {', '.join(missing)}",
            "Run managed setup to regenerate config.json.",
            {"missing": missing},
        )
    return _pass("config", "Managed configuration is complete.")


def _check_windows(_context: DoctorContext) -> CheckResult:
    system = platform.system()
    version = platform.version()
    details = {"system": system, "version": version, "release": platform.release()}
    if system != "Windows":
        return _fail(
            "windows",
            f"Managed AMD runtime requires Windows; detected {system}.",
            "Run the managed workflow on Windows 10 or Windows 11.",
            details,
        )
    if platform.release() not in {"10", "11"}:
        return _fail(
            "windows",
            f"Unsupported Windows release: {platform.release()}.",
            "Upgrade to Windows 10 or Windows 11.",
            details,
        )
    return _pass("windows", f"Windows {platform.release()} detected.", details)


def _check_amd_gpu(_context: DoctorContext) -> CheckResult:
    script = "Get-CimInstance Win32_VideoController | Select-Object Name | ConvertTo-Json -Compress"
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=20.0,
        check=False,
    )
    if completed.returncode != 0:
        return _fail(
            "amd_gpu",
            f"Display adapter query failed with code {completed.returncode}.",
            "Run PowerShell Get-CimInstance Win32_VideoController and repair WMI access.",
        )
    payload = json.loads(completed.stdout or "[]")
    values = payload if isinstance(payload, list) else [payload]
    adapters = [str(value.get("Name", "")) for value in values if isinstance(value, dict)]
    amd = [name for name in adapters if "amd" in name.lower() or "radeon" in name.lower()]
    if not amd:
        return _fail(
            "amd_gpu",
            "No AMD Radeon display adapter was detected.",
            "Install or enable a supported AMD GPU and its current Windows driver.",
            {"adapters": adapters},
        )
    return _pass("amd_gpu", f"AMD adapter detected: {amd[0]}", {"adapters": adapters})


def _check_hip_runtime(context: DoctorContext) -> CheckResult:
    windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidates = [
        Path(str(context.config.get("runtime_dir", ""))) / "amdhip64_7.dll",
        Path(str(context.config.get("llama_server", ""))).parent / "amdhip64_7.dll",
        windows_dir / "System32" / "amdhip64_7.dll",
    ]
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path:
        candidates.append(Path(rocm_path) / "bin" / "amdhip64_7.dll")
    candidates.extend(
        Path(value) / "amdhip64_7.dll"
        for value in os.environ.get("PATH", "").split(os.pathsep)
        if value
    )
    for dll in dict.fromkeys(candidates):
        if dll.is_file():
            return _pass("hip_runtime", f"HIP runtime found: {dll}", {"path": str(dll)})
    return _fail(
        "hip_runtime",
        "amdhip64_7.dll was not found in the managed runtime, System32, ROCM_PATH, or PATH.",
        "Install the AMD HIP runtime or reinstall the pinned llama.cpp HIP bundle.",
        {"searched": [str(path) for path in dict.fromkeys(candidates)]},
    )


def _check_disk(context: DoctorContext) -> CheckResult:
    probe = context.root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    minimum = 5 * 1024**3
    details = {"free_bytes": usage.free, "total_bytes": usage.total, "minimum_bytes": minimum}
    if usage.free < minimum:
        return _fail(
            "disk",
            f"Only {usage.free / 1024**3:.1f} GiB is free.",
            "Free at least 5 GiB on the managed installation drive.",
            details,
        )
    return _pass("disk", f"{usage.free / 1024**3:.1f} GiB free.", details)


def _check_runtime(context: DoctorContext) -> CheckResult:
    path = Path(str(context.config.get("llama_server", "")))
    expected = str(context.config.get("runtime_version", "b9884"))
    if not path.is_file():
        return _fail(
            "runtime",
            f"llama-server.exe is missing: {path}",
            "Run managed setup to reinstall the pinned llama.cpp runtime.",
        )
    version = _runtime_version(path)
    expected_build = re.escape(expected.removeprefix("b"))
    if re.search(rf"(?<!\d){expected_build}(?!\d)", version) is None:
        return _fail(
            "runtime",
            f"Unexpected llama.cpp version: {version}",
            "Reinstall the pinned b9884 runtime with managed setup --force.",
            {"path": str(path), "version": version},
        )
    return _pass("runtime", f"llama.cpp {expected} verified.", {"version": version})


def _resource_path(context: DoctorContext, resource: Resource) -> Path:
    base = context.root / "cache" if resource.name == "llama-cpp-hip-runtime" else context.root
    return base.joinpath(*PurePosixPath(resource.destination).parts)


def _check_resources(context: DoctorContext) -> CheckResult:
    manifest = load_runtime_manifest()
    expected = {
        "llama-cpp-hip-runtime",
        "paddleocr-vl-main-gguf",
        "paddleocr-vl-mmproj",
        "pp-doclayout-v3-onnx",
        "pp-doclayout-v3-config",
    }
    values = manifest.get("resources", [])
    names = [str(value.get("name", "")) for value in values if isinstance(value, dict)]
    if len(names) != len(set(names)) or set(names) != expected:
        missing = sorted(expected.difference(names))
        unexpected = sorted(set(names).difference(expected))
        return _fail(
            "resources",
            "Runtime manifest resource set is invalid; "
            f"missing={missing}, unexpected={unexpected}.",
            "Reinstall the package containing the authenticated runtime manifest.",
            {"names": names, "missing": missing, "unexpected": unexpected},
        )
    failures: list[str] = []
    verified: list[str] = []
    for value in values:
        resource = Resource.from_mapping(value)
        path = _resource_path(context, resource)
        try:
            verify_resource(path, resource)
        except (OSError, RuntimeError) as exc:
            failures.append(f"{resource.name}: {exc}")
        else:
            verified.append(resource.name)
    if failures:
        return _fail(
            "resources",
            "; ".join(failures),
            "Run managed setup again to download and verify the pinned resources.",
            {"verified": verified, "failures": failures},
        )
    return _pass("resources", f"Verified {len(verified)} pinned resources.", {"verified": verified})


def _check_directml(context: DoctorContext) -> CheckResult:
    import onnxruntime

    available = list(onnxruntime.get_available_providers())
    try:
        providers = resolve_layout_providers(available, "auto", "Windows")
    except RuntimeError as exc:
        return _fail(
            "directml",
            str(exc),
            "Install onnxruntime-directml and a compatible AMD Windows display driver.",
            {"available_providers": available},
        )
    layout_dir = Path(str(context.config.get("layout_model_dir", "")))
    try:
        model = PPDocLayoutV3Onnx(
            layout_dir,
            providers=providers,
            requested_provider="auto",
        )
    except Exception as exc:
        return _fail(
            "directml",
            f"DirectML layout session failed: {exc}",
            "Reinstall the layout model and verify DirectML driver/runtime availability.",
            {"available_providers": available, "layout_model_dir": str(layout_dir)},
        )
    active = list(model.layout_providers_active)
    if not active or active[0] != "DmlExecutionProvider":
        return _fail(
            "directml",
            f"DirectML is not the active first provider: {active}",
            "Repair DirectML activation; CPU fallback is not accepted for the managed path.",
            {"available_providers": available, "active_providers": active},
        )
    return _pass(
        "directml",
        "PP-DocLayoutV3 activated with DirectML first and fallback disabled.",
        {"available_providers": available, "active_providers": active},
    )


def _check_port(context: DoctorContext) -> CheckResult:
    port = int(context.config.get("server_port", 8111))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        return _warn(
            "port",
            f"Port {port} is already in use: {exc}",
            "Stop the conflicting process or confirm it is the intended llama-server.",
            {"port": port},
        )
    finally:
        sock.close()
    return _pass("port", f"Port {port} is available.", {"port": port})


def _redacted_url(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or "<host>"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme or 'http'}://{host}{path}"


def _models_url(server_url: str) -> str:
    parsed = urlsplit(server_url)
    path = parsed.path.rstrip("/")
    if not path.endswith("/models"):
        path = f"{path}/models"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


def _redact_error(text: str, server_url: str) -> str:
    redacted = text.replace(server_url, _redacted_url(server_url))
    redacted = re.sub(r"(?i)(api[_-]?key|token|password)=([^\s&]+)", r"\1=<redacted>", redacted)
    redacted = re.sub(r"://[^/@\s]+@", "://<redacted>@", redacted)
    return redacted


def _check_server(context: DoctorContext) -> CheckResult:
    if context.server_url:
        server_url = context.server_url.rstrip("/")
    else:
        port = int(context.config.get("server_port", 8111))
        server_url = f"http://127.0.0.1:{port}/v1"
    models_url = _models_url(server_url)
    response = None
    try:
        response = requests.get(models_url, timeout=5.0)
        if response.status_code >= 400:
            return _fail(
                "server",
                f"GET {_redacted_url(models_url)} returned HTTP {response.status_code}.",
                "Start the managed llama-server or correct the configured endpoint.",
            )
        payload = response.json()
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return _fail(
                "server",
                "The /v1/models response must be a JSON object with a data list.",
                "Start a compatible llama.cpp server or correct the configured endpoint.",
                {"url": _redacted_url(models_url)},
            )
    except (requests.RequestException, ValueError) as exc:
        return _fail(
            "server",
            _redact_error(str(exc), models_url),
            "Start the managed llama-server or correct the configured endpoint.",
            {"url": _redacted_url(models_url)},
        )
    finally:
        if response is not None:
            response.close()
    return _pass("server", "OpenAI-compatible /v1/models is reachable.", {"payload": payload})


DOCTOR_CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec("config", _check_config, "Run managed setup to regenerate config.json."),
    CheckSpec("windows", _check_windows, "Use Windows 10 or Windows 11."),
    CheckSpec("amd_gpu", _check_amd_gpu, "Install or enable a supported AMD GPU."),
    CheckSpec("hip_runtime", _check_hip_runtime, "Install the AMD HIP runtime."),
    CheckSpec("disk", _check_disk, "Free space on the managed installation drive."),
    CheckSpec("runtime", _check_runtime, "Reinstall the pinned runtime."),
    CheckSpec("resources", _check_resources, "Reinstall and verify managed resources."),
    CheckSpec("directml", _check_directml, "Install and activate ONNX Runtime DirectML."),
    CheckSpec("port", _check_port, "Resolve the configured port conflict."),
    CheckSpec("server", _check_server, "Start or repair the configured VLM server."),
)


def _load_config(config: Mapping[str, Any] | str | Path | None) -> tuple[dict[str, Any], Path]:
    if isinstance(config, Mapping):
        value = dict(config)
        runtime_dir = Path(str(value.get("runtime_dir", _default_root() / "runtime")))
        return value, runtime_dir.parent
    path = Path(config).expanduser() if config is not None else _default_root() / "config.json"
    root = path.parent.resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("config root must be an object")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        value = {"_config_error": f"Unable to load managed config {path}: {exc}"}
    return value, root


def run_doctor(
    config: Mapping[str, Any] | str | Path | None,
    server_url: str | None = None,
) -> list[CheckResult]:
    config_value, root = _load_config(config)
    context = DoctorContext(config=config_value, root=root, server_url=server_url)
    results: list[CheckResult] = []
    specs = (
        tuple(spec for spec in DOCTOR_CHECKS if spec.name == "server")
        if config is None and server_url is not None
        else DOCTOR_CHECKS
    )
    for spec in specs:
        try:
            result = spec.check(context)
        except Exception as exc:
            result = _fail(spec.name, str(exc), spec.remediation)
        results.append(result)
    return results


def doctor_exit_code(checks: Sequence[CheckResult]) -> int:
    return 2 if any(check.status == "FAIL" for check in checks) else 0


def checks_to_json(checks: Sequence[CheckResult]) -> str:
    return json.dumps([check.to_dict() for check in checks], ensure_ascii=False, indent=2)


def render_checks(checks: Sequence[CheckResult], console: Console | None = None) -> None:
    output = console or Console(highlight=False)
    table = Table(title="PaddleOCR-VL-ROCm Doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Message")
    table.add_column("Remediation")
    colors = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}
    for check in checks:
        table.add_row(
            check.name,
            f"[{colors[check.status]}]{check.status}[/{colors[check.status]}]",
            check.message,
            check.remediation,
        )
    output.print(table)
