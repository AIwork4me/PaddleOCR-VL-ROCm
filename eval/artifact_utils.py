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


def extract_readme_metrics(metric: dict[str, Any]) -> dict[str, float | None]:
    def nested(*keys: str) -> float | None:
        value: Any = metric
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value if isinstance(value, (int, float)) else None

    return {
        "text_edit_dist": nested("text_block", "page", "Edit_dist", "ALL"),
        "reading_order_edit_dist": nested("reading_order", "page", "Edit_dist", "ALL"),
        "table_teds_percent": (
            nested("table", "page", "TEDS", "ALL") * 100
            if nested("table", "page", "TEDS", "ALL") is not None
            else None
        ),
        "formula_cdm_percent": (
            nested("display_formula", "page", "CDM", "ALL") * 100
            if nested("display_formula", "page", "CDM", "ALL") is not None
            else None
        ),
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
        "run_stats": run_stats,
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
