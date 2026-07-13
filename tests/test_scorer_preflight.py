from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path("scripts/check_omnidocbench_scorer.py")
    spec = importlib.util.spec_from_file_location("check_omnidocbench_scorer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_exact_scorer_dependency_contract_includes_pylatexenc() -> None:
    module = _load_module()

    assert module.EXPECTED_DISTRIBUTIONS["pylatexenc"] == "2.10"
    assert {"apted", "evaluate", "lxml", "numpy", "pandas", "Pillow", "scipy"} <= set(
        module.EXPECTED_DISTRIBUTIONS
    )
    assert all(
        version and not any(char in version for char in "<>=~*")
        for version in module.EXPECTED_DISTRIBUTIONS.values()
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
        name.lower(): version for name, version in module.EXPECTED_DISTRIBUTIONS.items()
    }


def test_missing_dependency_fails_before_registry_import(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    imported = False

    def version(name: str) -> str:
        if name == "pylatexenc":
            raise module.metadata.PackageNotFoundError(name)
        return module.EXPECTED_DISTRIBUTIONS[name]

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
    monkeypatch.setattr(
        module.metadata,
        "version",
        lambda name: "0.0.0" if name == "pylatexenc" else module.EXPECTED_DISTRIBUTIONS[name],
    )

    with pytest.raises(RuntimeError, match="pylatexenc==2.10"):
        module.check_scorer(Path("checkout"), require_cdm_tools=False)


def test_cdm_preflight_exercises_tex_cjk_and_imagemagick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    exercised: list[Path] = []
    monkeypatch.setattr(
        module.metadata,
        "version",
        lambda name: module.EXPECTED_DISTRIBUTIONS[name],
    )
    monkeypatch.setattr(module, "_import_registries", lambda checkout: {})
    monkeypatch.setattr(
        module,
        "_exercise_cdm_runtime",
        lambda checkout: exercised.append(checkout) or {"smoke": "ok"},
    )

    result = module.check_scorer(Path("checkout"), require_cdm_tools=True)

    assert exercised == [Path("checkout")]
    assert result["cdm_runtime"] == {"smoke": "ok"}
