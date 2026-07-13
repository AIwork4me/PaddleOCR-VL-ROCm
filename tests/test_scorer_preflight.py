from __future__ import annotations

import base64
import hashlib
import importlib.util
from pathlib import Path

import pytest
import tomllib


def _load_module():
    path = Path("scripts/check_omnidocbench_scorer.py")
    spec = importlib.util.spec_from_file_location("check_omnidocbench_scorer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert lock == {
        name.lower(): version for name, version in module.DIRECT_DISTRIBUTIONS.items()
    }

    checkout = tomllib.loads(Path("eval/.omnidocbench/pyproject.toml").read_text(encoding="utf-8"))
    assert checkout["project"]["requires-python"] == ">=3.10,<3.12"
    assert {
        name.lower(): version
        for name, version in (dependency.split("==", 1) for dependency in checkout["project"]["dependencies"])
    } == lock
    transitive = Path("eval/requirements-omnidocbench-v16-transitive.txt")
    assert transitive.is_file()
    assert all("==" in line for line in transitive.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#"))


def test_scorer_rejects_python_outside_checkout_supported_range(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.sys, "version_info", (3, 13, 0))

    with pytest.raises(RuntimeError, match="Python 3.10 or 3.11"):
        module.check_scorer(Path("checkout"), require_cdm_tools=False)


def test_distribution_attestation_hashes_record_listed_content_and_rejects_mutation(tmp_path):
    module = _load_module()
    site = tmp_path / "Lib" / "site-packages"
    package = site / "demo.py"
    dist_info = site / "demo-1.0.dist-info"
    dist_info.mkdir(parents=True)
    package.write_text("VALUE = 1\n", encoding="utf-8")
    (dist_info / "METADATA").write_text("Name: demo\nVersion: 1.0\n", encoding="utf-8")
    digest = base64.urlsafe_b64encode(hashlib.sha256(package.read_bytes()).digest()).decode().rstrip("=")
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
    assert attestation["record_sha256"] == hashlib.sha256(
        (dist_info / "RECORD").read_bytes()
    ).hexdigest()
    assert attestation["file_count"] == 3

    package.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="RECORD hash mismatch"):
        module._attest_distribution("demo", "1.0", distribution, environment_root=tmp_path)


def test_missing_dependency_fails_before_registry_import(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    imported = False
    monkeypatch.setattr(module.sys, "version_info", (3, 11, 0))
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
    monkeypatch.setattr(module.sys, "version_info", (3, 11, 0))
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
    monkeypatch.setattr(module.sys, "version_info", (3, 11, 0))
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
