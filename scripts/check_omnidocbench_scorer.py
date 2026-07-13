"""Attest and import-start the pinned OmniDocBench v1.6 scorer environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

EXPECTED_DISTRIBUTIONS = {
    "apted": "1.0.3",
    "beautifulsoup4": "4.15.0",
    "evaluate": "0.4.6",
    "func-timeout": "4.3.5",
    "Levenshtein": "0.27.3",
    "loguru": "0.7.3",
    "lxml": "6.1.1",
    "matplotlib": "3.11.0",
    "nltk": "3.9.4",
    "numpy": "2.3.5",
    "pandas": "3.0.3",
    "Pillow": "12.3.0",
    "pylatexenc": "2.10",
    "PyYAML": "6.0.2",
    "scipy": "1.17.1",
    "tabulate": "0.10.0",
    "tqdm": "4.68.3",
    "pip": "26.1.2",
    "setuptools": "79.0.1",
}
REQUIRED_REGISTRATIONS = {
    "datasets": {"end2end_dataset"},
    "metrics": {"Edit_dist", "TEDS", "CDM"},
    "tasks": {"end2end_eval"},
}


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def check_scorer(checkout: Path, *, require_cdm_tools: bool) -> dict[str, object]:
    versions: dict[str, str] = {}
    for name, expected in EXPECTED_DISTRIBUTIONS.items():
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
    installed = "\n".join(
        f"{name.lower()}=={versions[name]}" for name in sorted(versions, key=str.lower)
    )
    result: dict[str, object] = {
        "python_executable_sha256": _hash_text(str(Path(sys.executable).resolve())),
        "python_version_sha256": _hash_text(sys.version),
        "dependencies": versions,
        "dependency_environment_sha256": _hash_text(installed),
    }
    result["registries"] = _import_registries(checkout)
    if require_cdm_tools:
        result["cdm_runtime"] = _exercise_cdm_runtime(checkout)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-cdm-tools", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = check_scorer(args.checkout, require_cdm_tools=args.require_cdm_tools)
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
