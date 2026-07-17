"""Independent final G4 decision over performance and quality evidence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from eval.artifact_utils import sha256_file
from eval.g4_performance import build_receipt, decide_g4
from eval.g4_quality import build_quality_receipt, decide_g4_quality

SCHEMA = 1
FINAL_RECEIPT_FILES = {
    "sample_manifest",
    "performance_artifact",
    "performance_decision",
    "performance_receipt",
    "quality_artifact",
    "quality_decision",
    "quality_receipt",
    "final_decision",
}


def decide_g4_release(
    *,
    manifest: Mapping[str, object],
    performance_artifact: Mapping[str, object],
    performance_decision: Mapping[str, object],
    quality_artifact: Mapping[str, object],
    quality_decision: Mapping[str, object],
) -> dict[str, object]:
    """Recompute both decisions and accept quality evidence in place of raw byte equality."""
    recomputed_performance = decide_g4(manifest, performance_artifact)
    if performance_decision != recomputed_performance:
        raise ValueError("G4 performance decision does not match recomputation")
    recomputed_quality = decide_g4_quality(manifest, quality_artifact)
    if quality_decision != recomputed_quality:
        raise ValueError("G4 quality decision does not match recomputation")
    performance_checks = recomputed_performance["checks"]
    if not isinstance(performance_checks, Mapping):
        raise ValueError("G4 performance checks are invalid")
    numerical_performance = all(
        performance_checks.get(name) is True
        for name in (
            "sample_contract",
            "zero_failures",
            "mean_at_most_13_00",
            "p95_at_most_34_82",
        )
    )
    quality_preserved = recomputed_quality["g4_quality"] is True
    accepted = numerical_performance and quality_preserved
    return {
        "schema": SCHEMA,
        "verdict": "PASS" if accepted else "FAIL",
        "g4": accepted,
        "checks": {
            "numerical_performance": numerical_performance,
            "quality_preserved": quality_preserved,
            "raw_output_equivalent": performance_checks.get("output_equivalent") is True,
        },
        "performance": {
            "pages": recomputed_performance["pages"],
            "failures": 0 if performance_checks.get("zero_failures") is True else None,
            "wall_seconds": recomputed_performance["wall_seconds"],
            "pages_per_minute": recomputed_performance["pages_per_minute"],
            "timing": recomputed_performance["timing"],
        },
        "quality": {
            "exact_pages": recomputed_quality["exact_pages"],
            "normalized_pages": recomputed_quality["normalized_pages"],
            "scored_pages": recomputed_quality["scored_pages"],
            "compared_metrics": recomputed_quality["compared_metrics"],
            "published_accuracy": recomputed_quality["published_accuracy"],
            "metric_regressions": len(recomputed_quality["regressions"]),
        },
    }


def validate_source_receipts(
    *,
    manifest_path: Path,
    performance_artifact_path: Path,
    performance_decision_path: Path,
    performance_receipt: Mapping[str, object],
    quality_artifact_path: Path,
    quality_decision_path: Path,
    quality_receipt: Mapping[str, object],
    quality_contract_path: Path,
    subset_gt_path: Path,
) -> None:
    expected_performance = build_receipt(
        {
            "sample_manifest": manifest_path,
            "run_artifact": performance_artifact_path,
            "decision": performance_decision_path,
        }
    )
    if performance_receipt != expected_performance:
        raise ValueError("G4 performance receipt does not match its source files")
    expected_quality = build_quality_receipt(
        {
            "sample_manifest": manifest_path,
            "performance_artifact": performance_artifact_path,
            "quality_artifact": quality_artifact_path,
            "quality_decision": quality_decision_path,
            "scorer_contract": quality_contract_path,
            "subset_gt": subset_gt_path,
        }
    )
    if quality_receipt != expected_quality:
        raise ValueError("G4 quality receipt does not match its source files")


def build_final_receipt(paths: Mapping[str, Path]) -> dict[str, object]:
    if set(paths) != FINAL_RECEIPT_FILES:
        raise ValueError("Final G4 receipt requires the exact evidence set")
    files: dict[str, object] = {}
    for name, path in sorted(paths.items()):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Final G4 receipt input is missing or unsafe: {name}")
        files[name] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {"schema": SCHEMA, "files": files}
