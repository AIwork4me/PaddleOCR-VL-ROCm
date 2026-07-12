import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.release_contract import (
    validate_approved_failure_predictions,
    validate_release_run_stats,
)

OFFICIAL_LOCAL_STEM = "paddleocr_official_local_llamacpp_gguf"


@dataclass(frozen=True)
class ArtifactPaths:
    predictions_dir: Path
    metric_result: Path
    run_summary: Path
    provenance: Path


def official_local_paths(version: str, *, cdm: bool = False) -> ArtifactPaths:
    results_dir = Path("results/omnidocbench") / version
    suffix = "_cdm" if cdm else ""
    return ArtifactPaths(
        predictions_dir=Path("predictions") / f"{OFFICIAL_LOCAL_STEM}_{version}",
        metric_result=results_dir / f"{OFFICIAL_LOCAL_STEM}_quick_match_metric_result{suffix}.json",
        run_summary=results_dir / f"{OFFICIAL_LOCAL_STEM}_quick_match_run_summary{suffix}.json",
        provenance=results_dir / f"{OFFICIAL_LOCAL_STEM}_provenance.json",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_manifest_sha256(predictions_dir: Path) -> str:
    rows = [
        f"{path.name}\t{sha256_file(path)}"
        for path in sorted(predictions_dir.glob("*.md"), key=lambda item: item.name)
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def copy_metric_report(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Metric report not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def _nested(metric: dict[str, Any], *keys: str) -> Any:
    value: Any = metric
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _nested_number(metric: dict[str, Any], *keys: str) -> float | None:
    value = _nested(metric, *keys)
    return value if isinstance(value, (int, float)) else None


def _debug_quality(
    metric: dict[str, Any], element: str, name: str, error_key: str
) -> dict[str, Any]:
    debug = _nested(metric, element, "metric_debug", name) or {}
    sample_count = debug.get("sample_count")
    timeout_count = int(debug.get("timeout_case_count") or 0)
    error_count = int(debug.get(error_key) or 0)
    valid = (
        isinstance(sample_count, int)
        and sample_count > 0
        and timeout_count == 0
        and error_count == 0
    )
    return {
        "valid": valid,
        "sample_count": sample_count,
        "timeout_case_count": timeout_count,
        error_key: error_count,
        "reason": ""
        if valid
        else (
            f"{name} requires samples>0, timeouts=0, errors=0; "
            f"samples={sample_count}, timeouts={timeout_count}, errors={error_count}"
        ),
    }


def analyze_metric_quality(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "formula_cdm": _debug_quality(metric, "display_formula", "CDM", "exception_case_count"),
        "table_teds": _debug_quality(metric, "table", "TEDS", "error_case_count"),
    }


def _rounded(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def extract_notebook_metrics(metric: dict[str, Any]) -> dict[str, float | None]:
    text = _nested_number(metric, "text_block", "all", "Edit_dist", "ALL_page_avg")
    formula_raw = _nested_number(metric, "display_formula", "page", "CDM", "ALL")
    table_raw = _nested_number(metric, "table", "page", "TEDS", "ALL")
    table_s_raw = _nested_number(metric, "table", "page", "TEDS_structure_only", "ALL")
    reading = _nested_number(metric, "reading_order", "all", "Edit_dist", "ALL_page_avg")

    text_value = _rounded(text)
    formula_value = _rounded(None if formula_raw is None else formula_raw * 100.0)
    table_value = _rounded(None if table_raw is None else table_raw * 100.0)
    overall = None
    if text_value is not None and formula_value is not None and table_value is not None:
        overall = ((1.0 - text_value) * 100.0 + formula_value + table_value) / 3.0

    return {
        "text_edit_dist": text_value,
        "formula_cdm_percent": formula_value,
        "table_teds_percent": table_value,
        "table_teds_structure_only_percent": _rounded(
            None if table_s_raw is None else table_s_raw * 100.0
        ),
        "reading_order_edit_dist": _rounded(reading),
        "overall": overall,
    }


def extract_readme_metrics(metric: dict[str, Any]) -> dict[str, float | None]:
    quality = analyze_metric_quality(metric)
    values = extract_notebook_metrics(metric)
    if not quality["formula_cdm"]["valid"]:
        values["formula_cdm_percent"] = None
        values["overall"] = None
    if not quality["table_teds"]["valid"]:
        values["table_teds_percent"] = None
        values["overall"] = None
    return values


def approved_known_failures(
    run_stats: dict[str, Any], predictions_dir: Path
) -> list[dict[str, str]]:
    if run_stats.get("count") != 1651 or run_stats.get("engine") != "official":
        return []
    try:
        failures = validate_release_run_stats(run_stats, version="v16", engine="official")
    except ValueError:
        return []
    validate_approved_failure_predictions(predictions_dir, failures)
    return failures


def write_run_summary(
    *,
    save_name: str,
    run_stats_path: Path,
    metric_result_path: Path,
    destination: Path,
    cdm: bool,
) -> Path:
    run_stats = load_json(run_stats_path)
    metric_result = load_json(metric_result_path)
    failures = [
        {
            key: (value[:200] if isinstance(value, str) else value)
            for key, value in item.items()
            if key in {"image", "status", "error", "seconds", "attempts"}
        }
        for item in run_stats.get("stats", [])
        if isinstance(item, dict) and str(item.get("status", "")).startswith(("fail", "fallback"))
    ][:20]
    run_stats_summary = {
        "count": run_stats.get("count"),
        "ok": run_stats.get("ok"),
        "fail": run_stats.get("fail"),
        "fallback": run_stats.get("fallback"),
        "limit_pages": run_stats.get("limit_pages"),
        "failure_samples": failures,
    }
    summary = {
        "save_name": save_name,
        "engine": run_stats.get("engine"),
        "layout_provider_requested": run_stats.get("layout_provider_requested"),
        "layout_providers_active": list(run_stats.get("layout_providers_active") or []),
        "cdm": cdm,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_count": run_stats.get("count"),
        "ok_pages": run_stats.get("ok"),
        "failed_pages": run_stats.get("fail"),
        "fallback_pages": run_stats.get("fallback"),
        "approved_known_failures": approved_known_failures(run_stats, run_stats_path.parent),
        "metric_result_path": str(metric_result_path),
        "run_stats_path": str(run_stats_path),
        "notebook_metrics": extract_notebook_metrics(metric_result),
        "readme_metrics": extract_readme_metrics(metric_result),
        "metric_quality": analyze_metric_quality(metric_result),
        "run_stats_summary": run_stats_summary,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def write_provenance(
    *,
    destination: Path,
    git_commit: str,
    engine: str,
    server_url: str,
    api_model_name: str,
    adapter_command: str,
    scoring_config_path: Path,
    dataset_manifest_path: Path,
    predictions_dir: Path,
    metric_result_paths: list[Path],
    run_summary_paths: list[Path],
    run_stats_path: Path,
    omnidocbench: dict[str, Any],
    dataset_sha256: str,
    config_sha256: str,
    prediction_manifest_sha256: str,
) -> Path:
    run_stats = load_json(run_stats_path)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "engine": engine,
        "layout_provider_requested": run_stats.get("layout_provider_requested"),
        "layout_providers_active": list(run_stats.get("layout_providers_active") or []),
        "vlm_server_url": server_url,
        "api_model_name": api_model_name,
        "adapter_command": adapter_command,
        "omnidocbench": omnidocbench,
        "scoring_config_path": str(scoring_config_path),
        "config_sha256": config_sha256,
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_sha256": dataset_sha256,
        "prediction_dir": str(predictions_dir),
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "page_count": run_stats.get("count"),
        "ok_pages": run_stats.get("ok"),
        "failed_pages": run_stats.get("fail"),
        "fallback_pages": run_stats.get("fallback"),
        "approved_known_failures": approved_known_failures(run_stats, predictions_dir),
        "metric_result_paths": [str(path) for path in metric_result_paths],
        "run_summary_paths": [str(path) for path in run_summary_paths],
        "run_stats_path": str(run_stats_path),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
