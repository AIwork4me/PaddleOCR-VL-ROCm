from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib
from packaging import markers


def _load_module():
    path = Path("scripts/check_omnidocbench_scorer.py")
    spec = importlib.util.spec_from_file_location("check_omnidocbench_scorer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cpython_310() -> Path:
    configured = os.environ.get("SCORER_PYTHON_310")
    candidates = [
        Path(sys.executable) if sys.version_info[:2] == (3, 10) else None,
        Path(configured) if configured else None,
        Path(discovered) if (discovered := shutil.which("python3.10")) else None,
        Path(
            r"C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-scorer-v16-py310"
            r"\Scripts\python.exe"
        ),
    ]
    launcher = shutil.which("py")
    if launcher:
        completed = subprocess.run(
            [launcher, "-3.10", "-c", "import sys; print(sys.executable)"],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            candidates.append(Path(completed.stdout.strip()))
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        completed = subprocess.run(
            [
                str(candidate),
                "-c",
                "import packaging,sys; raise SystemExit(sys.version_info[:2] != (3, 10))",
            ],
            check=False,
        )
        if completed.returncode == 0:
            return candidate
    pytest.fail("A real CPython 3.10 interpreter is required for the import-boundary gate")


def test_checker_imports_and_parses_toml_at_real_cpython_310_boundary(tmp_path: Path) -> None:
    python = _cpython_310()
    fallback = tmp_path / "tomli.py"
    fallback.write_text(
        "def loads(value):\n    assert value == \"answer = 42\\n\"\n    return {'answer': 42}\n",
        encoding="utf-8",
    )
    script = Path("scripts/check_omnidocbench_scorer.py").resolve()
    program = (
        "import importlib.util,json,pathlib; "
        f"p=pathlib.Path({str(script)!r}); "
        "s=importlib.util.spec_from_file_location('checker',p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "print(json.dumps(m.tomllib.loads('answer = 42\\n'),sort_keys=True))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), value] if (value := env.get("PYTHONPATH")) else [str(tmp_path)]
    )

    completed = subprocess.run(
        [str(python), "-c", program],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"answer": 42}


def test_complete_exact_scorer_dependency_contract_includes_pylatexenc() -> None:
    module = _load_module()

    assert module.DIRECT_DISTRIBUTIONS["pylatexenc"] == "2.10"
    assert {"apted", "evaluate", "lxml", "numpy", "pandas", "Pillow", "scipy"} <= set(
        module.DIRECT_DISTRIBUTIONS
    )
    assert all(
        version and not any(char in version for char in "<>=~*")
        for version in module.DIRECT_DISTRIBUTIONS.values()
    )
    lock = {
        name.lower(): version
        for name, version in (
            line.split("==", 1)
            for line in Path("eval/requirements-omnidocbench-v16.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        )
    }
    assert lock == {name.lower(): version for name, version in module.DIRECT_DISTRIBUTIONS.items()}

    checkout = tomllib.loads(Path("eval/.omnidocbench/pyproject.toml").read_text(encoding="utf-8"))
    assert checkout["project"]["requires-python"] == ">=3.10,<3.12"
    assert {
        name.lower(): version
        for name, version in (
            dependency.split("==", 1) for dependency in checkout["project"]["dependencies"]
        )
    } == lock
    transitive = Path("eval/requirements-omnidocbench-v16-transitive.txt")
    assert transitive.is_file()
    transitive_lock = module._read_lock(transitive)
    assert transitive_lock["tomli"] == ("tomli", "2.2.1")
    assert "targeted only at CPython 3.10 on Windows" in transitive.read_text(encoding="utf-8")
    assert all(
        "==" in line
        for line in transitive.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )


@pytest.mark.parametrize("version", ((3, 9, 19), (3, 11, 15), (3, 13, 0)))
def test_scorer_rejects_any_python_other_than_cpython_310(monkeypatch, version):
    module = _load_module()
    monkeypatch.setattr(module.sys, "version_info", version)

    with pytest.raises(RuntimeError, match=r"CPython 3\.10.*lxml 4\.9\.1"):
        module.check_scorer(Path("checkout"), require_cdm_tools=False)


def test_scorer_accepts_cpython_310(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module.sys, "version_info", (3, 10, 14))
    monkeypatch.setattr(module, "_validate_checkout_dependency_contract", lambda *args: None)
    monkeypatch.setattr(module.metadata, "version", lambda name: module.DIRECT_DISTRIBUTIONS[name])
    monkeypatch.setattr(module, "_attest_locked_distributions", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_import_registries", lambda checkout: {})

    result = module.check_scorer(Path("checkout"), require_cdm_tools=False)

    assert result["python_version"] == "3.10"


def _distribution(version: str, *requirements: str):
    package_metadata = Message()
    for requirement in requirements:
        package_metadata["Requires-Dist"] = requirement
    return SimpleNamespace(version=version, metadata=package_metadata)


def test_dependency_closure_propagates_requested_extras() -> None:
    module = _load_module()
    locked = {
        "owner": ("owner", "1.0"),
        "transport": ("transport", "1.0"),
    }
    distributions = {
        "owner": _distribution("1.0", "transport[http]==1.0"),
        "transport": _distribution("1.0", 'http-helper==2.0; extra == "http"'),
    }

    with pytest.raises(RuntimeError, match="transport requires http-helper"):
        module._validate_locked_dependency_closure(distributions, locked)


def test_dependency_closure_rejects_locked_specifier_conflict() -> None:
    module = _load_module()
    locked = {
        "owner": ("owner", "1.0"),
        "transport": ("transport", "1.0"),
    }
    distributions = {
        "owner": _distribution("1.0", "transport>=2.0"),
        "transport": _distribution("1.0"),
    }

    with pytest.raises(RuntimeError, match=r"owner requires transport>=2.0; found 1.0"):
        module._validate_locked_dependency_closure(distributions, locked)


def test_dependency_closure_includes_real_aiohttp_cpython_310_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    locked = module._read_lock(Path("eval/requirements-omnidocbench-v16-transitive.txt"))
    environment = markers.default_environment()
    environment.update(python_version="3.10", python_full_version="3.10.20")
    monkeypatch.setattr(markers, "default_environment", lambda: environment)
    distributions = {
        "aiohttp": _distribution("3.10.11", 'async-timeout <6.0,>=4.0 ; python_version < "3.11"'),
    }
    if "async-timeout" in locked:
        distributions["async-timeout"] = _distribution(locked["async-timeout"][1])

    module._validate_locked_dependency_closure(distributions, locked)
    assert locked["async-timeout"] == ("async-timeout", "5.0.1")


def test_distribution_attestation_hashes_record_listed_content_and_rejects_mutation(tmp_path):
    module = _load_module()
    site = tmp_path / "Lib" / "site-packages"
    package = site / "demo.py"
    dist_info = site / "demo-1.0.dist-info"
    dist_info.mkdir(parents=True)
    package.write_text("VALUE = 1\n", encoding="utf-8")
    (dist_info / "METADATA").write_text("Name: demo\nVersion: 1.0\n", encoding="utf-8")
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(package.read_bytes()).digest()).decode().rstrip("=")
    )
    (dist_info / "RECORD").write_text(
        f"demo.py,sha256={digest},{package.stat().st_size}\n"
        "demo-1.0.dist-info/METADATA,,\n"
        "demo-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    distribution = module.metadata.Distribution.at(dist_info)

    attestation = module._attest_distribution(
        "demo", "1.0", distribution, environment_root=tmp_path
    )
    assert (
        attestation["record_sha256"]
        == hashlib.sha256((dist_info / "RECORD").read_bytes()).hexdigest()
    )
    assert attestation["file_count"] == 3

    package.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="RECORD hash mismatch"):
        module._attest_distribution("demo", "1.0", distribution, environment_root=tmp_path)


def test_missing_dependency_fails_before_registry_import(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    imported = False
    monkeypatch.setattr(module.sys, "version_info", (3, 10, 0))
    monkeypatch.setattr(module, "_validate_checkout_dependency_contract", lambda *args: None)

    def version(name: str) -> str:
        if name == "pylatexenc":
            raise module.metadata.PackageNotFoundError(name)
        return module.DIRECT_DISTRIBUTIONS[name]

    def import_registries(checkout: Path):
        nonlocal imported
        imported = True
        return {}

    monkeypatch.setattr(module.metadata, "version", version)
    monkeypatch.setattr(module, "_import_registries", import_registries)

    with pytest.raises(RuntimeError, match="pylatexenc==2.10"):
        module.check_scorer(Path("checkout"), require_cdm_tools=False)
    assert imported is False


def test_version_mutation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module.sys, "version_info", (3, 10, 0))
    monkeypatch.setattr(module, "_validate_checkout_dependency_contract", lambda *args: None)
    monkeypatch.setattr(
        module.metadata,
        "version",
        lambda name: "0.0.0" if name == "pylatexenc" else module.DIRECT_DISTRIBUTIONS[name],
    )

    with pytest.raises(RuntimeError, match="pylatexenc==2.10"):
        module.check_scorer(Path("checkout"), require_cdm_tools=False)


def test_cdm_preflight_exercises_tex_cjk_and_imagemagick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    exercised: list[Path] = []
    monkeypatch.setattr(module.sys, "version_info", (3, 10, 0))
    monkeypatch.setattr(module, "_validate_checkout_dependency_contract", lambda *args: None)
    monkeypatch.setattr(
        module.metadata,
        "version",
        lambda name: module.DIRECT_DISTRIBUTIONS[name],
    )
    monkeypatch.setattr(module, "_import_registries", lambda checkout: {})
    monkeypatch.setattr(module, "_attest_locked_distributions", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        module,
        "_exercise_cdm_runtime",
        lambda checkout: exercised.append(checkout) or {"smoke": "ok"},
    )

    result = module.check_scorer(Path("checkout"), require_cdm_tools=True)

    assert exercised == [Path("checkout")]
    assert result["cdm_runtime"] == {"smoke": "ok"}
