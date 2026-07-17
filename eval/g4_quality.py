"""Fail-closed quality-preservation decision for G4 performance candidates."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from eval.g4_performance import EXPECTED_PAGES, validate_sample_manifest

SCHEMA = 1
LOWER_IS_BETTER = {
    "text_edit",
    "formula_edit",
    "table_edit",
    "reading_order_edit",
}
HIGHER_IS_BETTER = {"cdm", "teds", "teds_structure_only"}
METRICS = LOWER_IS_BETTER | HIGHER_IS_BETTER


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _score(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return number


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def decide_g4_quality(
    manifest: Mapping[str, object], artifact: Mapping[str, object]
) -> dict[str, object]:
    """Validate a paired targeted score artifact and decide quality preservation."""
    expected = validate_sample_manifest(manifest)
    required = {
        "schema",
        "benchmark",
        "project_commit",
        "sample_manifest_sha256",
        "dataset_sha256",
        "scorer_commit",
        "scorer_tree_sha256",
        "scorer_config_sha256",
        "performance_artifact_sha256",
        "normalization",
        "samples",
    }
    if set(artifact) != required:
        raise ValueError("G4 quality artifact schema is invalid")
    if artifact["schema"] != SCHEMA or artifact["benchmark"] != "OmniDocBench-v1.6":
        raise ValueError("G4 quality artifact identity is invalid")
    commit = artifact["project_commit"]
    scorer_commit = artifact["scorer_commit"]
    for value, name in ((commit, "project_commit"), (scorer_commit, "scorer_commit")):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError(f"{name} must be a full Git SHA")
    for name in (
        "sample_manifest_sha256",
        "dataset_sha256",
        "scorer_tree_sha256",
        "scorer_config_sha256",
        "performance_artifact_sha256",
    ):
        _hash(artifact[name], name)
    if artifact["normalization"] != "task5-scorer-markdown-v1":
        raise ValueError("G4 quality normalization is invalid")
    rows = artifact["samples"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_PAGES:
        raise ValueError(f"G4 quality artifact requires exactly {EXPECTED_PAGES} samples")

    exact_pages = 0
    normalized_pages = 0
    scored_pages = 0
    regressions: list[dict[str, object]] = []
    compared_metrics = 0
    for expected_sample, raw in zip(expected, rows, strict=True):
        row = _object(raw, "G4 quality sample")
        if set(row) != {
            "category",
            "image",
            "relation",
            "reference_sha256",
            "candidate_sha256",
            "reference_normalized_sha256",
            "candidate_normalized_sha256",
            "metrics",
        }:
            raise ValueError("G4 quality sample schema is invalid")
        if (
            row["category"] != expected_sample["category"]
            or row["image"] != expected_sample["image"]
        ):
            raise ValueError("G4 quality sample order does not match the frozen manifest")
        reference_sha = _hash(row["reference_sha256"], "reference_sha256")
        candidate_sha = _hash(row["candidate_sha256"], "candidate_sha256")
        reference_normalized = _hash(
            row["reference_normalized_sha256"], "reference_normalized_sha256"
        )
        candidate_normalized = _hash(
            row["candidate_normalized_sha256"], "candidate_normalized_sha256"
        )
        metrics = _object(row["metrics"], "metrics")
        relation = row["relation"]
        if relation == "exact":
            if reference_sha != candidate_sha or metrics:
                raise ValueError("Exact G4 quality samples require equal hashes and no metrics")
            exact_pages += 1
            continue
        if relation == "normalized":
            if (
                reference_sha == candidate_sha
                or reference_normalized != candidate_normalized
                or metrics
            ):
                raise ValueError(
                    "Normalized G4 quality samples require distinct raw hashes, "
                    "equal normalized hashes, and no metrics"
                )
            normalized_pages += 1
            continue
        if relation != "scored":
            raise ValueError("Unknown G4 quality relation")
        if reference_normalized == candidate_normalized:
            raise ValueError("Scored samples must differ after normalization")
        if not metrics or not set(metrics).issubset(METRICS):
            raise ValueError("Scored samples require at least one recognized metric")
        scored_pages += 1
        for metric, raw_pair in metrics.items():
            pair = _object(raw_pair, f"{metric} pair")
            if set(pair) != {"reference", "candidate"}:
                raise ValueError(f"{metric} pair schema is invalid")
            reference = _score(pair["reference"], f"{metric}.reference")
            candidate = _score(pair["candidate"], f"{metric}.candidate")
            compared_metrics += 1
            passed = candidate <= reference if metric in LOWER_IS_BETTER else candidate >= reference
            if not passed:
                regressions.append(
                    {
                        "image": row["image"],
                        "metric": metric,
                        "reference": reference,
                        "candidate": candidate,
                    }
                )

    quality_preserved = scored_pages > 0 and not regressions
    return {
        "schema": SCHEMA,
        "verdict": "PASS" if quality_preserved else "FAIL",
        "g4_quality": quality_preserved,
        "checks": {
            "sample_contract": True,
            "all_differences_scored": scored_pages > 0,
            "zero_metric_regressions": not regressions,
        },
        "pages": EXPECTED_PAGES,
        "exact_pages": exact_pages,
        "normalized_pages": normalized_pages,
        "scored_pages": scored_pages,
        "compared_metrics": compared_metrics,
        "regressions": regressions,
    }
