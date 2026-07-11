import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def analyze_metric_quality(metric: dict[str, Any]) -> dict[str, Any]:
    cdm_debug = _nested(metric, "display_formula", "metric_debug", "CDM") or {}
    sample_count = cdm_debug.get("sample_count") if isinstance(cdm_debug, dict) else None
    exception_count = cdm_debug.get("exception_case_count") if isinstance(cdm_debug, dict) else None
    formula_cdm = {
        "valid": True,
        "reason": "",
        "sample_count": sample_count,
        "exception_case_count": exception_count,
    }
    if (
        isinstance(sample_count, int)
        and sample_count > 0
        and isinstance(exception_count, int)
        and exception_count >= sample_count
    ):
        first_reason = ""
        exception_cases = cdm_debug.get("exception_cases", [])
        if exception_cases and isinstance(exception_cases[0], dict):
            first_reason = str(exception_cases[0].get("reason", ""))
        formula_cdm["valid"] = False
        formula_cdm["reason"] = (
            f"all CDM samples raised exceptions ({exception_count}/{sample_count}); "
            f"first_reason={first_reason}"
        )
    return {"formula_cdm": formula_cdm}


def extract_readme_metrics(metric: dict[str, Any]) -> dict[str, float | None]:
    quality = analyze_metric_quality(metric)
    cdm_value = _nested_number(metric, "display_formula", "page", "CDM", "ALL")
    if not quality["formula_cdm"]["valid"]:
        cdm_value = None

    return {
        "text_edit_dist": _nested_number(metric, "text_block", "page", "Edit_dist", "ALL"),
        "reading_order_edit_dist": _nested_number(
            metric, "reading_order", "page", "Edit_dist", "ALL"
        ),
        "table_teds_percent": (
            _nested_number(metric, "table", "page", "TEDS", "ALL") * 100
            if _nested_number(metric, "table", "page", "TEDS", "ALL") is not None
            else None
        ),
        "formula_cdm_percent": cdm_value * 100 if cdm_value is not None else None,
    }


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
        "cdm": cdm,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_count": run_stats.get("count"),
        "ok_pages": run_stats.get("ok"),
        "failed_pages": run_stats.get("fail"),
        "fallback_pages": run_stats.get("fallback"),
        "metric_result_path": str(metric_result_path),
        "run_stats_path": str(run_stats_path),
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
) -> Path:
    run_stats = load_json(run_stats_path)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "engine": engine,
        "vlm_server_url": server_url,
        "api_model_name": api_model_name,
        "adapter_command": adapter_command,
        "scoring_config_path": str(scoring_config_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "prediction_dir": str(predictions_dir),
        "page_count": run_stats.get("count"),
        "ok_pages": run_stats.get("ok"),
        "failed_pages": run_stats.get("fail"),
        "fallback_pages": run_stats.get("fallback"),
        "metric_result_paths": [str(path) for path in metric_result_paths],
        "run_summary_paths": [str(path) for path in run_summary_paths],
        "run_stats_path": str(run_stats_path),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
