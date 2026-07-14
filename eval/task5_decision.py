"""Independent Task 5 equivalence, AMD-adaptation, and G3 decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from eval.artifact_utils import analyze_metric_quality, extract_notebook_metrics

EXPECTED_PAIRED_PAGES = 1650
G3_MINIMUM_OVERALL = 96.13
RECEIPT_NAME = "receipt.sha256.json"

_ROOT_RECEIPT_FILES = {
    "manifest.json",
    "selected-attempt.json",
    "decision.json",
    "provider-attestation.json",
}
_ATTEMPT_FILES = {
    "manifest.json",
    "stage-state.json",
    "snapshot-before.json",
    "snapshot-after.json",
    "selected-stage-state.json",
    "provider-attestation.json",
    "decision.json",
}
_RESULT_FILES = {
    "metric.json",
    "metric-cdm.json",
    "run-summary.json",
    "run-summary-cdm.json",
    "provenance.json",
    "provenance-cdm.json",
}
_COMPARISON_FILES = {
    "input.json",
    "input-comparison.json",
    "output.json",
    "output-comparison.json",
    "trace.json",
    "trace-comparison.json",
    "provider-attestation.json",
    "decision.json",
}
_MANIFEST_FILES = {
    "predictions-manifest.json",
    "prediction-manifest.json",
    "traces-manifest.json",
    "trace-manifest.json",
    "predictions.manifest.sha256.json",
    "traces.manifest.sha256.json",
}


def extract_paired_scores(
    non_cdm: Mapping[str, object], cdm: Mapping[str, object]
) -> dict[str, object]:
    """Extract one score set from matching non-CDM and CDM reports."""
    if not isinstance(non_cdm, Mapping) or not isinstance(cdm, Mapping):
        raise ValueError("Metric reports must be JSON objects")
    non_cdm_metric = dict(non_cdm)
    cdm_metric = dict(cdm)
    non_cdm_values = extract_notebook_metrics(non_cdm_metric)
    cdm_values = extract_notebook_metrics(cdm_metric)

    shared_bounds = {
        "text_edit_dist": (0.0, 1.0),
        "table_teds_percent": (0.0, 100.0),
        "reading_order_edit_dist": (0.0, 1.0),
    }
    selected: dict[str, float] = {}
    for name, (minimum, maximum) in shared_bounds.items():
        left = _required_score(non_cdm_values, name, minimum, maximum)
        right = _required_score(cdm_values, name, minimum, maximum)
        if left != right:
            raise ValueError(f"Non-CDM and CDM {name} must agree after approved rounding")
        selected[name] = left

    formula = _required_score(cdm_values, "formula_cdm_percent", 0.0, 100.0)
    selected["formula_cdm_percent"] = formula
    selected["overall"] = (
        (1.0 - selected["text_edit_dist"]) * 100.0
        + formula
        + selected["table_teds_percent"]
    ) / 3.0

    non_cdm_quality = analyze_metric_quality(non_cdm_metric)
    cdm_quality = analyze_metric_quality(cdm_metric)
    formula_quality = dict(cdm_quality["formula_cdm"])
    non_cdm_table = dict(non_cdm_quality["table_teds"])
    cdm_table = dict(cdm_quality["table_teds"])
    table_quality = {
        "valid": bool(non_cdm_table.get("valid")) and bool(cdm_table.get("valid")),
        "non_cdm": non_cdm_table,
        "cdm": cdm_table,
    }
    return {
        "text_edit_dist": selected["text_edit_dist"],
        "formula_cdm_percent": selected["formula_cdm_percent"],
        "table_teds_percent": selected["table_teds_percent"],
        "reading_order_edit_dist": selected["reading_order_edit_dist"],
        "overall": selected["overall"],
        "metric_quality": {
            "formula_cdm": formula_quality,
            "table_teds": table_quality,
        },
    }


def strict_equivalence_decision(
    output_report: Mapping[str, object], trace_report: Mapping[str, object]
) -> dict[str, object]:
    """Render strict equivalence with output differences taking priority."""
    paired = _nonnegative_int(output_report.get("paired_pages"), "paired_pages")
    different = _nonnegative_int(output_report.get("different_pages"), "different_pages")
    official_only = _optional_nonnegative_int(output_report, "official_only_pages")
    lightweight_only = _optional_nonnegative_int(output_report, "lightweight_only_pages")
    trace_verdict = trace_report.get("verdict")
    if trace_verdict not in {"PASS", "UNKNOWN", "FAIL"}:
        raise ValueError("Trace report verdict must be PASS, UNKNOWN, or FAIL")

    output_pass = (
        paired == EXPECTED_PAIRED_PAGES
        and different == 0
        and official_only == 0
        and lightweight_only == 0
    )
    if not output_pass or trace_verdict == "FAIL":
        verdict = "FAIL"
    elif trace_verdict == "UNKNOWN":
        verdict = "UNKNOWN"
    else:
        verdict = "PASS"
    return {
        "verdict": verdict,
        "output_equivalent": output_pass,
        "trace_verdict": trace_verdict,
        "expected_paired_pages": EXPECTED_PAIRED_PAGES,
        "paired_pages": paired,
        "different_pages": different,
        "official_only_pages": official_only,
        "lightweight_only_pages": lightweight_only,
    }


def component_not_worse(
    official: Mapping[str, float], lightweight: Mapping[str, float]
) -> bool:
    return (
        lightweight["text_edit_dist"] <= official["text_edit_dist"]
        and lightweight["formula_cdm_percent"] >= official["formula_cdm_percent"]
        and lightweight["table_teds_percent"] >= official["table_teds_percent"]
    )


def amd_adaptation_decision(
    *,
    official_scores: Mapping[str, object],
    lightweight_scores: Mapping[str, object],
    provider_attestation: Mapping[str, object],
    lightweight_stats: Mapping[str, object],
    public_contracts_pass: bool,
) -> dict[str, object]:
    """Render AMD-adaptation and G3 without consulting strict equivalence."""
    official = _validated_scores(official_scores, "official")
    lightweight = _validated_scores(lightweight_scores, "lightweight")
    components_pass = component_not_worse(official, lightweight)
    threshold_pass = lightweight["overall"] >= G3_MINIMUM_OVERALL
    quality_pass = _metric_quality_passes(official_scores) and _metric_quality_passes(
        lightweight_scores
    )
    attestation_pass, provider_evidence = _provider_attestation_passes(
        provider_attestation, lightweight_stats
    )
    stats_pass = _lightweight_stats_pass(lightweight_stats)
    contracts_pass = public_contracts_pass is True
    g3 = all(
        (
            components_pass,
            threshold_pass,
            quality_pass,
            attestation_pass,
            stats_pass,
            contracts_pass,
        )
    )
    checks = {
        "component_not_worse": components_pass,
        "overall_at_least_96_13": threshold_pass,
        "metric_quality": quality_pass,
        "provider_attestation": attestation_pass,
        "lightweight_coverage": stats_pass,
        "public_contracts": contracts_pass,
    }
    return {
        "verdict": "PASS" if g3 else "FAIL",
        "g3": g3,
        "checks": checks,
        "minimum_overall": G3_MINIMUM_OVERALL,
        "lightweight_overall": lightweight["overall"],
        "provider_evidence": provider_evidence,
    }


def build_task5_receipt(
    task5_root: Path, relative_paths: Sequence[str]
) -> dict[str, object]:
    """Hash an explicit allowlist of small Task 5 evidence files."""
    if not isinstance(relative_paths, Sequence) or isinstance(relative_paths, (str, bytes)):
        raise ValueError("Receipt paths must be a sequence")
    names = list(relative_paths)
    if RECEIPT_NAME in names:
        raise ValueError("Receipt cannot hash itself")
    if not names:
        raise ValueError("Receipt must hash at least one evidence file")
    if len(set(names)) != len(names):
        raise ValueError("Receipt paths must be unique")
    root = _safe_root(task5_root)
    files: dict[str, object] = {}
    for name in sorted(names):
        normalized, path = _contained_file(root, name)
        files[normalized] = _relative_file_identity(root, normalized, path)
    return {"schema": 1, "algorithm": "sha256", "files": files}


def validate_task5_receipt(
    task5_root: Path, receipt: Mapping[str, object]
) -> dict[str, object]:
    """Re-hash every receipt input and reject any identity change."""
    if set(receipt) != {"schema", "algorithm", "files"}:
        raise ValueError("Receipt must contain exactly schema, algorithm, and files")
    if receipt.get("schema") != 1 or receipt.get("algorithm") != "sha256":
        raise ValueError("Receipt schema or algorithm is unsupported")
    files = receipt.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("Receipt files must be a non-empty object")
    expected_names = list(files)
    if expected_names != sorted(expected_names):
        raise ValueError("Receipt files must be sorted")
    rebuilt = build_task5_receipt(task5_root, expected_names)
    if rebuilt != dict(receipt):
        raise ValueError("Receipt input identity has changed")
    return rebuilt


def _required_score(
    values: Mapping[str, object], name: str, minimum: float, maximum: float
) -> float:
    value = values.get(name)
    if value is None:
        raise ValueError(f"Metric report is missing required {name}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if not minimum <= numeric <= maximum:
        raise ValueError(f"{name} must be within {minimum:g}..{maximum:g}")
    return numeric


def _validated_scores(scores: Mapping[str, object], label: str) -> dict[str, float]:
    bounds = {
        "text_edit_dist": (0.0, 1.0),
        "formula_cdm_percent": (0.0, 100.0),
        "table_teds_percent": (0.0, 100.0),
        "reading_order_edit_dist": (0.0, 1.0),
        "overall": (0.0, 100.0),
    }
    try:
        return {
            name: _required_score(scores, name, minimum, maximum)
            for name, (minimum, maximum) in bounds.items()
        }
    except ValueError as error:
        raise ValueError(f"Invalid {label} scores: {error}") from error


def _metric_quality_passes(scores: Mapping[str, object]) -> bool:
    quality = scores.get("metric_quality")
    if not isinstance(quality, Mapping):
        return False
    if set(quality) != {"formula_cdm", "table_teds"}:
        return False
    return all(
        isinstance(item, Mapping) and item.get("valid") is True for item in quality.values()
    )


def _lightweight_stats_pass(stats: Mapping[str, object]) -> bool:
    raw = (stats.get("count"), stats.get("ok"), stats.get("fail"), stats.get("fallback"))
    summary = (
        stats.get("prediction_count"),
        stats.get("ok_pages"),
        stats.get("failed_pages"),
        stats.get("fallback_pages"),
    )
    expected = (1651, 1650, 1, 0)
    return (raw == expected or summary == expected) and _provider_runtime_passes(stats)


def _provider_runtime_passes(stats: Mapping[str, object]) -> bool:
    return (
        type(stats.get("layout_provider_requested")) is str
        and stats.get("layout_provider_requested") == "auto"
        and type(stats.get("layout_providers_active")) is list
        and stats.get("layout_providers_active")
        == ["DmlExecutionProvider", "CPUExecutionProvider"]
        and type(stats.get("layout_fallback_disabled")) is bool
        and stats.get("layout_fallback_disabled") is True
    )


def _provider_attestation_passes(
    attestation: Mapping[str, object], stats: Mapping[str, object]
) -> tuple[bool, dict[str, object]]:
    dml = attestation.get("dml_node_events")
    cpu = attestation.get("cpu_node_events")
    dml_share = attestation.get("dml_node_share")
    cpu_share = attestation.get("cpu_node_share")
    missing = attestation.get("missing_provider_node_events")
    other = attestation.get("other_provider_node_events")
    valid_counts = all(
        type(value) is int and value >= 0 for value in (dml, cpu, missing, other)
    )
    valid_shares = all(
        type(value) is float and math.isfinite(value) and 0.0 <= value <= 1.0
        for value in (dml_share, cpu_share)
    )
    counts_match_shares = False
    majority = False
    if valid_counts and valid_shares:
        assert isinstance(dml, int)
        assert isinstance(cpu, int)
        assert isinstance(dml_share, float)
        assert isinstance(cpu_share, float)
        provider_nodes = dml + cpu
        if provider_nodes > 0:
            expected_dml_share = dml / provider_nodes
            expected_cpu_share = cpu / provider_nodes
            counts_match_shares = math.isclose(
                dml_share, expected_dml_share, rel_tol=0.0, abs_tol=1e-12
            ) and math.isclose(
                cpu_share, expected_cpu_share, rel_tol=0.0, abs_tol=1e-12
            )
            majority = dml > 0 and expected_dml_share > 0.5
    runtime_pass = _provider_runtime_passes(stats)
    passed = (
        attestation.get("verdict") == "PASS"
        and valid_counts
        and valid_shares
        and counts_match_shares
        and majority
        and missing == 0
        and other == 0
        and runtime_pass
    )
    evidence = {
        "dml_node_events": dml,
        "cpu_node_events": cpu,
        "dml_node_share": dml_share,
        "cpu_node_share": cpu_share,
        "missing_provider_node_events": missing,
        "other_provider_node_events": other,
        "layout_provider_requested": stats.get("layout_provider_requested"),
        "layout_providers_active": stats.get("layout_providers_active"),
        "layout_fallback_disabled": stats.get("layout_fallback_disabled"),
        "counts_match_shares": counts_match_shares,
    }
    return passed, evidence


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _optional_nonnegative_int(report: Mapping[str, object], name: str) -> int:
    return _nonnegative_int(report.get(name, 0), name)


def _safe_root(task5_root: Path) -> Path:
    if task5_root.is_symlink():
        raise ValueError("Task 5 root cannot be a symlink")
    root = task5_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Task 5 root must be a directory")
    return root


def _contained_file(root: Path, name: str) -> tuple[str, Path]:
    if not isinstance(name, str) or not name:
        raise ValueError("Receipt path must be a non-empty relative string")
    if "\\" in name:
        raise ValueError("Receipt path must use root-relative POSIX separators")
    relative = PurePosixPath(name)
    if relative.is_absolute():
        raise ValueError("Receipt path must be relative")
    if any(part in {"", ".", ".."} for part in relative.parts):
        if ".." in relative.parts:
            raise ValueError("Receipt path cannot escape the Task 5 root")
        raise ValueError("Receipt path must be normalized")
    normalized = relative.as_posix()
    if normalized == RECEIPT_NAME:
        raise ValueError("Receipt cannot hash itself")
    if not _receipt_path_allowed(relative):
        raise ValueError(f"Receipt path is not in the Task 5 allowlist: {normalized}")
    path = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Receipt input cannot be a symlink: {normalized}")
    resolved = path.resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError("Receipt path cannot escape the Task 5 root")
    if not resolved.is_file():
        raise ValueError(f"Receipt input must be a regular file: {normalized}")
    return normalized, resolved


def _receipt_path_allowed(relative: PurePosixPath) -> bool:
    parts = relative.parts
    if len(parts) == 1:
        return parts[0] in _ROOT_RECEIPT_FILES
    if len(parts) < 3 or parts[0] != "attempts" or not parts[1]:
        return False
    tail = parts[2:]
    if len(tail) == 1:
        return tail[0] in _ATTEMPT_FILES or tail[0] in _MANIFEST_FILES
    if len(tail) == 3 and tail[0] == "results" and tail[1] in {"official", "lightweight"}:
        return tail[2] in _RESULT_FILES
    if len(tail) == 2 and tail[0] in {"comparison", "comparisons"}:
        return tail[1] in _COMPARISON_FILES
    if len(tail) == 2 and tail[0] in {"manifests", "raw-manifests"}:
        return tail[1] in _MANIFEST_FILES
    return False


def _relative_file_identity(root: Path, name: str, path: Path) -> dict[str, object]:
    before = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Receipt input must be a non-symlink regular file: {name}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened_before = os.fstat(descriptor)
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)
    identities = (before, opened_before, opened_after, after)
    if any(
        (item.st_dev, item.st_ino, item.st_size) != (before.st_dev, before.st_ino, before.st_size)
        for item in identities[1:]
    ):
        raise ValueError(f"Receipt input changed while hashing: {name}")
    if byte_count != before.st_size:
        raise ValueError(f"Receipt input changed while hashing: {name}")
    if path.resolve(strict=True).parent != root and root not in path.resolve(strict=True).parents:
        raise ValueError("Receipt path cannot escape the Task 5 root")
    return {"path": name, "bytes": byte_count, "sha256": digest.hexdigest()}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key is not allowed: {key}")
        value[key] = item
    return value


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _decide(args: argparse.Namespace) -> int:
    paths = {
        "official_non_cdm": args.official_non_cdm,
        "official_cdm": args.official_cdm,
        "lightweight_non_cdm": args.lightweight_non_cdm,
        "lightweight_cdm": args.lightweight_cdm,
        "output_report": args.output_report,
        "trace_report": args.trace_report,
        "provider_attestation": args.provider_attestation,
        "lightweight_stats": args.lightweight_stats,
    }
    loaded = {name: _load_json_object(path) for name, path in paths.items()}
    official = extract_paired_scores(loaded["official_non_cdm"], loaded["official_cdm"])
    lightweight = extract_paired_scores(
        loaded["lightweight_non_cdm"], loaded["lightweight_cdm"]
    )
    strict = strict_equivalence_decision(loaded["output_report"], loaded["trace_report"])
    amd = amd_adaptation_decision(
        official_scores=official,
        lightweight_scores=lightweight,
        provider_attestation=loaded["provider_attestation"],
        lightweight_stats=loaded["lightweight_stats"],
        public_contracts_pass=args.public_contracts_pass,
    )
    decision = {
        "schema": 1,
        "benchmark": "OmniDocBench-v1.6",
        "coverage": {
            "expected_paired_pages": EXPECTED_PAIRED_PAGES,
            "paired_pages": strict["paired_pages"],
        },
        "scores": {"official": official, "lightweight": lightweight},
        "strict_equivalence": strict,
        "amd_adaptation": amd,
        "g3": amd["g3"],
        "evidence": {
            name: {"sha256": _sha256(path)} for name, path in sorted(paths.items())
        },
    }
    _write_json(args.output, decision)
    return 0


def _build_receipt_cli(args: argparse.Namespace) -> int:
    receipt = build_task5_receipt(args.task5_root, args.path)
    _write_json(args.output, receipt)
    return 0


def _validate_receipt_cli(args: argparse.Namespace) -> int:
    validate_task5_receipt(args.task5_root, _load_json_object(args.receipt))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    decide = commands.add_parser("decide")
    for name in (
        "official-non-cdm",
        "official-cdm",
        "lightweight-non-cdm",
        "lightweight-cdm",
        "output-report",
        "trace-report",
        "provider-attestation",
        "lightweight-stats",
    ):
        decide.add_argument(f"--{name}", type=Path, required=True)
    decide.add_argument("--public-contracts-pass", action="store_true")
    decide.add_argument("--output", type=Path, required=True)
    decide.set_defaults(handler=_decide)

    receipt = commands.add_parser("receipt")
    receipt.add_argument("--task5-root", type=Path, required=True)
    receipt.add_argument("--path", action="append", required=True)
    receipt.add_argument("--output", type=Path, required=True)
    receipt.set_defaults(handler=_build_receipt_cli)

    validate = commands.add_parser("validate-receipt")
    validate.add_argument("--task5-root", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, required=True)
    validate.set_defaults(handler=_validate_receipt_cli)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
