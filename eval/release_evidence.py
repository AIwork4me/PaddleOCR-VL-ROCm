from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.artifact_utils import analyze_metric_quality, extract_notebook_metrics, sha256_file
from eval.release_contract import (
    validate_approved_failure_predictions,
    validate_release_run_stats,
)

# Policy source: docs/releases/0.1.0-g3-attestation.md.
G3_MINIMUM_OVERALL = 95.99
LOGICAL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def build_input_manifest(paths: Mapping[str, Path], *, git_commit: str) -> dict[str, object]:
    if not isinstance(git_commit, str) or not git_commit.strip():
        raise ValueError("Manifest git_commit must be a non-empty string")
    if not paths:
        raise ValueError("Manifest inputs must be a non-empty object")
    inputs: dict[str, object] = {}
    for name, path in sorted(paths.items()):
        if not isinstance(name, str) or LOGICAL_NAME.fullmatch(name) is None:
            raise ValueError(f"Invalid manifest input logical name: {name!r}")
        if not path.is_file():
            raise ValueError(f"Immutable input must be an existing regular file: {path}")
        digest = sha256_file(path)
        if SHA256.fullmatch(digest) is None:
            raise ValueError(f"Input hash must be a 64-character SHA-256 value: {path}")
        inputs[name] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    return {"git_commit": git_commit, "inputs": inputs}


def validate_isolated_output_paths(paths: Iterable[Path], protected: Iterable[Path]) -> None:
    protected_paths = [path.resolve(strict=False) for path in protected]
    for path in paths:
        resolved = path.resolve(strict=False)
        if any(resolved == root or root in resolved.parents for root in protected_paths):
            raise ValueError(f"Output is inside a protected historical path: {path}")


def notebook_overall(text: float, formula: float, table: float) -> tuple[dict[str, float], float]:
    components = {
        "text_edit_dist": round(text, 3),
        "formula_cdm_percent": round(formula * 100, 3),
        "table_teds_percent": round(table * 100, 3),
    }
    overall = (
        (1.0 - components["text_edit_dist"]) * 100.0
        + components["formula_cdm_percent"]
        + components["table_teds_percent"]
    ) / 3.0
    return components, overall


def decide_release_gates(
    official_stats: dict[str, Any], lightweight_metric: dict[str, Any]
) -> dict[str, object]:
    failures = validate_release_run_stats(official_stats, version="v16", engine="official")
    predictions_dir = Path(str(official_stats.get("predictions_dir", ".")))
    validate_approved_failure_predictions(predictions_dir, failures)

    raw_components = {
        "Text Edit": _metric_number(
            lightweight_metric, "text_block", "all", "Edit_dist", "ALL_page_avg"
        ),
        "Formula CDM": _metric_number(lightweight_metric, "display_formula", "page", "CDM", "ALL"),
        "Table TEDS": _metric_number(lightweight_metric, "table", "page", "TEDS", "ALL"),
    }
    for name, value in raw_components.items():
        if value is None:
            raise ValueError("Lightweight metric is missing a required notebook component")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and within 0..1")

    extracted = extract_notebook_metrics(lightweight_metric)
    text = extracted["text_edit_dist"]
    formula_percent = extracted["formula_cdm_percent"]
    table_percent = extracted["table_teds_percent"]
    if not (
        isinstance(text, float)
        and isinstance(formula_percent, float)
        and isinstance(table_percent, float)
    ):
        raise ValueError("Lightweight metric is missing a required notebook component")
    components, overall = notebook_overall(text, formula_percent / 100.0, table_percent / 100.0)
    if not math.isfinite(overall):
        raise ValueError("Overall must be finite")
    quality = analyze_metric_quality(lightweight_metric)
    quality_pass = all(bool(item["valid"]) for item in quality.values())
    return {
        "g0": True,
        "g3": quality_pass and overall >= G3_MINIMUM_OVERALL,
        "components": components,
        "overall": overall,
        "metric_quality": quality,
        "approved_known_failures": failures,
    }


def _metric_number(metric: dict[str, Any], *keys: str) -> float | None:
    value: object = metric
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _validate_manifest(value: dict[str, Any]) -> None:
    if set(value) != {"git_commit", "inputs"}:
        raise ValueError("Manifest must contain exactly git_commit and inputs")
    git_commit = value["git_commit"]
    inputs = value["inputs"]
    if not isinstance(git_commit, str) or not git_commit.strip():
        raise ValueError("Manifest git_commit must be a non-empty string")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Manifest inputs must be a non-empty object")
    for name, record in inputs.items():
        if not isinstance(name, str) or LOGICAL_NAME.fullmatch(name) is None:
            raise ValueError(f"Invalid manifest input logical name: {name!r}")
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise ValueError(
                f"Manifest input {name!r} must contain exactly path, bytes, and sha256"
            )
        path = record["path"]
        byte_count = record["bytes"]
        digest = record["sha256"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"Manifest input {name!r} path must be a non-empty string")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            raise ValueError(f"Manifest input {name!r} bytes must be a nonnegative integer")
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise ValueError(
                f"Manifest input {name!r} sha256 must be 64 lowercase hexadecimal characters"
            )
        input_path = Path(path)
        if not input_path.is_absolute():
            raise ValueError(f"Manifest input {name!r} path must be absolute")
        if not input_path.is_file():
            raise ValueError(f"Manifest input {name!r} must exist as a regular file")
        resolved = input_path.resolve(strict=True)
        if resolved != input_path:
            raise ValueError(f"Manifest input {name!r} resolved path has changed")
        if input_path.stat().st_size != byte_count:
            raise ValueError(f"Manifest input {name!r} byte size has changed")
        if sha256_file(input_path) != digest:
            raise ValueError(f"Manifest input {name!r} SHA-256 hash has changed")


def _input(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("inputs must use NAME=PATH")
    return name, Path(raw_path)


def _unique_inputs(values: list[tuple[str, Path]]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, path in values:
        if name in paths:
            raise ValueError(f"Duplicate manifest input logical name: {name!r}")
        paths[name] = path
    return paths


def _find_metric(evidence_root: Path) -> Path:
    candidates = sorted((evidence_root / "results" / "lightweight").glob("*metric*.json"))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one lightweight metric JSON file")
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and validate immutable release evidence.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--git-commit", required=True)
    manifest_parser.add_argument("--input", action="append", type=_input, required=True)
    manifest_parser.add_argument("--output", type=Path)

    decide_parser = subparsers.add_parser("decide")
    decide_parser.add_argument("--evidence-root", type=Path, required=True)
    decide_parser.add_argument("--official-stats", type=Path)
    decide_parser.add_argument("--lightweight-metric", type=Path)
    decide_parser.add_argument("--official-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            manifest = build_input_manifest(_unique_inputs(args.input), git_commit=args.git_commit)
            rendered = json.dumps(manifest, indent=2)
            if args.output:
                args.output.write_text(rendered + "\n", encoding="utf-8")
            else:
                print(rendered)
            return 0

        manifest_path = args.evidence_root / "manifest.json"
        _validate_manifest(_load_object(manifest_path))
        stats_path = args.official_stats or args.evidence_root / "official" / "_run_stats.json"
        official_stats = _load_object(stats_path)
        official_stats["predictions_dir"] = str(stats_path.parent)
        if args.official_only:
            failures = validate_release_run_stats(official_stats, version="v16", engine="official")
            validate_approved_failure_predictions(stats_path.parent, failures)
            decision: dict[str, object] = {
                "g0": True,
                "approved_known_failures": failures,
            }
        else:
            metric_path = args.lightweight_metric or _find_metric(args.evidence_root)
            decision = decide_release_gates(official_stats, _load_object(metric_path))
        print(json.dumps(decision, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
