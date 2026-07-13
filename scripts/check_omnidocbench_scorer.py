"""Attest and import-start the pinned OmniDocBench v1.6 scorer environment."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata as metadata
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import tomllib
from packaging.requirements import InvalidRequirement, Requirement

DIRECT_DISTRIBUTIONS = {
    "apted": "1.0.3",
    "beautifulsoup4": "4.11.1",
    "evaluate": "0.4.3",
    "func-timeout": "4.3.5",
    "Levenshtein": "0.25.1",
    "loguru": "0.7.2",
    "lxml": "4.9.1",
    "matplotlib": "3.7.5",
    "nltk": "3.9.1",
    "numpy": "1.24.4",
    "pandas": "2.0.3",
    "Pillow": "10.4.0",
    "pylatexenc": "2.10",
    "PyYAML": "6.0.2",
    "scipy": "1.10.1",
    "tabulate": "0.9.0",
    "tqdm": "4.67.1",
}
REQUIRED_REGISTRATIONS = {
    "datasets": {"end2end_dataset"},
    "metrics": {"Edit_dist", "TEDS", "CDM"},
    "tasks": {"end2end_eval"},
}


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _read_lock(path: Path) -> dict[str, tuple[str, str]]:
    locked: dict[str, tuple[str, str]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1 or any(marker in line for marker in (";", " @ ", " --")):
            raise RuntimeError(
                f"Lock entry must be one unconditional exact pin: {path}:{line_number}"
            )
        name, version = (part.strip() for part in line.split("==", 1))
        normalized = _normalized_name(name)
        if not name or not version or normalized in locked:
            raise RuntimeError(f"Invalid or duplicate lock entry: {path}:{line_number}")
        locked[normalized] = (name, version)
    return locked


def _validate_checkout_dependency_contract(
    checkout: Path, direct: dict[str, tuple[str, str]]
) -> None:
    pyproject_path = checkout / "pyproject.toml"
    if not pyproject_path.is_file():
        raise RuntimeError(f"Pinned scorer pyproject is missing: {pyproject_path}")
    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]
    if project.get("requires-python") != ">=3.10,<3.12":
        raise RuntimeError("Pinned scorer Python range is not >=3.10,<3.12.")
    checkout_direct: dict[str, tuple[str, str]] = {}
    for raw in project.get("dependencies", ()):
        if raw.count("==") != 1:
            raise RuntimeError("Pinned scorer dependency is not an exact requirement.")
        name, version = (part.strip() for part in raw.split("==", 1))
        checkout_direct[_normalized_name(name)] = (name, version)
    if checkout_direct != direct:
        raise RuntimeError("Direct dependency lock diverges from the pinned scorer pyproject.")


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _attest_distribution(
    name: str,
    expected: str,
    distribution: metadata.Distribution,
    *,
    environment_root: Path,
) -> dict[str, object]:
    if distribution.version != expected:
        raise RuntimeError(
            f"Required scorer dependency version mismatch: {name}=={expected}; "
            f"found {distribution.version}"
        )
    root = environment_root.resolve()
    origin = Path(distribution.locate_file(".")).resolve()
    if not _is_within(origin, root):
        raise RuntimeError(f"Scorer dependency origin is outside its interpreter: {name}")
    record_entries = [path for path in distribution.files or () if Path(path).name == "RECORD"]
    if len(record_entries) != 1:
        raise RuntimeError(f"Scorer dependency RECORD is absent or ambiguous: {name}")
    record_path = Path(distribution.locate_file(record_entries[0])).resolve()
    if not record_path.is_file() or not _is_within(record_path, root):
        raise RuntimeError(f"Scorer dependency RECORD is outside its interpreter: {name}")

    content_records: list[str] = []
    with record_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise RuntimeError(f"Scorer dependency RECORD is empty: {name}")
    for row in rows:
        if len(row) != 3 or not row[0]:
            raise RuntimeError(f"Malformed scorer dependency RECORD row: {name}")
        listed = Path(distribution.locate_file(row[0])).resolve()
        if not listed.is_file() or not _is_within(listed, root):
            raise RuntimeError(f"RECORD-listed file is missing or outside interpreter: {name}")
        payload = listed.read_bytes()
        actual_digest = hashlib.sha256(payload).hexdigest()
        if row[1]:
            try:
                algorithm, encoded = row[1].split("=", 1)
                expected_digest = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
                actual_declared = hashlib.new(algorithm, payload).digest()
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"Unsupported RECORD hash for scorer dependency: {name}"
                ) from exc
            if actual_declared != expected_digest:
                raise RuntimeError(f"RECORD hash mismatch for scorer dependency: {name}")
        if row[2]:
            try:
                expected_size = int(row[2])
            except ValueError as exc:
                raise RuntimeError(f"Malformed RECORD size for scorer dependency: {name}") from exc
            if expected_size != len(payload):
                raise RuntimeError(f"RECORD size mismatch for scorer dependency: {name}")
        relative = listed.relative_to(root).as_posix()
        content_records.append(f"{relative}\0{actual_digest}\0{len(payload)}")
    canonical = "\n".join(sorted(content_records))
    return {
        "version": distribution.version,
        "origin_sha256": _hash_text(origin.as_posix().casefold()),
        "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
        "content_sha256": _hash_text(canonical),
        "file_count": len(content_records),
    }


def _attest_locked_distributions(
    locked: dict[str, tuple[str, str]], *, environment_root: Path
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    distributions: dict[str, metadata.Distribution] = {}
    for normalized in sorted(locked):
        name, expected = locked[normalized]
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required scorer dependency is missing: {name}=={expected}"
            ) from exc
        distributions[normalized] = distribution
        result[normalized] = _attest_distribution(
            name, expected, distribution, environment_root=environment_root
        )
    _validate_locked_dependency_closure(distributions, locked)
    return result


def _validate_locked_dependency_closure(
    distributions: dict[str, metadata.Distribution],
    locked: dict[str, tuple[str, str]],
) -> None:
    activated_extras = {name: set() for name in distributions}
    changed = True
    while changed:
        changed = False
        for owner, distribution in distributions.items():
            contexts = {"", *activated_extras[owner]}
            for raw_requirement in distribution.metadata.get_all("Requires-Dist") or ():
                try:
                    requirement = Requirement(raw_requirement)
                except InvalidRequirement as exc:
                    raise RuntimeError(
                        f"Malformed installed dependency metadata: {owner}"
                    ) from exc
                if requirement.marker and not any(
                    requirement.marker.evaluate({"extra": extra}) for extra in contexts
                ):
                    continue
                required = _normalized_name(requirement.name)
                if required not in locked or required not in distributions:
                    raise RuntimeError(
                        f"Transitive lock is incomplete: {owner} requires {requirement.name}"
                    )
                actual = distributions[required].version
                if requirement.specifier and actual not in requirement.specifier:
                    raise RuntimeError(
                        f"Transitive lock conflict: {owner} requires {requirement}; "
                        f"found {actual}"
                    )
                new_extras = set(requirement.extras) - activated_extras[required]
                if new_extras:
                    activated_extras[required].update(new_extras)
                    changed = True


def _import_registries(checkout: Path) -> dict[str, list[str]]:
    checkout = checkout.resolve()
    sys.path.insert(0, str(checkout))
    try:
        from src.core.registry import describe_registries

        registries = describe_registries()
    finally:
        sys.path.pop(0)
    for group, required in REQUIRED_REGISTRATIONS.items():
        missing = required - set(registries.get(group, []))
        if missing:
            raise RuntimeError(f"Missing default scorer registrations: {group}={sorted(missing)}")
    return registries


def _tool_identity(name: str) -> dict[str, str]:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required CDM tool is missing: {name}")
    completed = subprocess.run(
        [path, "--version" if name == "magick" else "-version"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Required CDM tool failed to start: {name}")
    rendered = (completed.stdout + completed.stderr).strip()
    return {
        "path_sha256": _hash_text(str(Path(path).resolve())),
        "version_sha256": _hash_text(rendered),
    }


def _exercise_cdm_runtime(checkout: Path) -> dict[str, object]:
    tools = {name: _tool_identity(name) for name in ("pdflatex", "kpsewhich", "magick")}
    checkout = checkout.resolve()
    sys.path.insert(0, str(checkout))
    try:
        from src.metrics.cdm.modules.texlive_env import describe_tex_runtime

        tex_runtime = describe_tex_runtime()
    finally:
        sys.path.pop(0)
    for key in ("cjk_sty", "cjk_font_fd"):
        value = str(tex_runtime.get(key, ""))
        if not value or value.startswith(("exit_code=", "unavailable")):
            raise RuntimeError(f"Required CDM TeX resource is unavailable: {key}")

    source = r"""\documentclass{article}
\usepackage{CJKutf8}
\pagestyle{empty}
\begin{document}
\begin{CJK}{UTF8}{gkai}中文\end{CJK}
\end{document}
"""
    with tempfile.TemporaryDirectory(prefix="omnidocbench_cdm_preflight_") as temporary:
        directory = Path(temporary)
        tex = directory / "smoke.tex"
        pdf = directory / "smoke.pdf"
        png = directory / "smoke.png"
        tex.write_text(source, encoding="utf-8")
        commands = (
            [
                shutil.which("pdflatex") or "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex.name,
            ],
            [shutil.which("magick") or "magick", "-density", "72", str(pdf), str(png)],
        )
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=directory,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "CDM TeX/ImageMagick smoke test failed: "
                    f"{Path(command[0]).name} exit {completed.returncode}"
                )
        if not png.is_file() or png.stat().st_size == 0:
            raise RuntimeError("CDM ImageMagick smoke test produced no PNG")
    return {
        "tools": tools,
        "cjk_sty_sha256": _hash_text(str(tex_runtime["cjk_sty"])),
        "cjk_font_fd_sha256": _hash_text(str(tex_runtime["cjk_font_fd"])),
        "smoke": "ok",
    }


def check_scorer(
    checkout: Path,
    *,
    require_cdm_tools: bool,
    direct_lock: Path = Path("eval/requirements-omnidocbench-v16.txt"),
    transitive_lock: Path = Path("eval/requirements-omnidocbench-v16-transitive.txt"),
    attest_only: bool = False,
) -> dict[str, object]:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "OmniDocBench scorer requires isolated CPython 3.10 on Windows because "
            "official lxml 4.9.1 Windows wheels do not support CPython 3.11."
        )
    direct = _read_lock(direct_lock)
    expected_direct = {
        _normalized_name(name): (name, version) for name, version in DIRECT_DISTRIBUTIONS.items()
    }
    if direct != expected_direct:
        raise RuntimeError("Direct dependency lock diverges from the pinned OmniDocBench checkout.")
    _validate_checkout_dependency_contract(checkout, direct)
    transitive = _read_lock(transitive_lock)
    overlap = set(direct) & set(transitive)
    if overlap:
        raise RuntimeError(f"Direct dependencies repeated in transitive lock: {sorted(overlap)}")
    locked = {**direct, **transitive}
    versions: dict[str, str] = {}
    for name, expected in DIRECT_DISTRIBUTIONS.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required scorer dependency is missing: {name}=={expected}"
            ) from exc
        if actual != expected:
            raise RuntimeError(
                f"Required scorer dependency version mismatch: {name}=={expected}; found {actual}"
            )
        versions[name] = actual
    attestations = _attest_locked_distributions(locked, environment_root=Path(sys.prefix))
    installed = json.dumps(attestations, sort_keys=True, separators=(",", ":"))
    result: dict[str, object] = {
        "python_version": f"{sys.version_info[0]}.{sys.version_info[1]}",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": _hash_text(str(Path(sys.executable).resolve())),
        "python_file_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_prefix_sha256": _hash_text(str(Path(sys.prefix).resolve())),
        "python_base_prefix": str(Path(sys.base_prefix).resolve()),
        "python_base_prefix_sha256": _hash_text(str(Path(sys.base_prefix).resolve())),
        "python_version_sha256": _hash_text(sys.version),
        "dependencies": attestations,
        "dependency_environment_sha256": _hash_text(installed),
    }
    if attest_only:
        return result
    result["registries"] = _import_registries(checkout)
    if require_cdm_tools:
        result["cdm_runtime"] = _exercise_cdm_runtime(checkout)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--direct-lock", type=Path, default=Path("eval/requirements-omnidocbench-v16.txt")
    )
    parser.add_argument(
        "--transitive-lock",
        type=Path,
        default=Path("eval/requirements-omnidocbench-v16-transitive.txt"),
    )
    parser.add_argument("--attest-only", action="store_true")
    parser.add_argument("--require-cdm-tools", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = check_scorer(
            args.checkout,
            require_cdm_tools=args.require_cdm_tools,
            direct_lock=args.direct_lock,
            transitive_lock=args.transitive_lock,
            attest_only=args.attest_only,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
