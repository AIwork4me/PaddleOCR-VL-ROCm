from __future__ import annotations

import hashlib

import pytest

from eval.g4_quality import decide_g4_quality


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def manifest() -> dict[str, object]:
    categories = (
        "book",
        "PPT2PDF",
        "academic_literature",
        "note",
        "colorful_textbook",
        "exam_paper",
        "magazine",
        "research_report",
        "newspaper",
    )
    samples = []
    for category in categories:
        for index in range(3):
            samples.append(
                {
                    "category": category,
                    "image": f"{category}-{index}.png",
                    "sha256": digest(f"{category}-{index}"),
                }
            )
    return {
        "schema": 1,
        "benchmark": "OmniDocBench-v1.6",
        "selection": "sha256-filename-first-3-per-category",
        "dataset_sha256": digest("dataset"),
        "samples": samples,
    }


def artifact(sample_manifest: dict[str, object]) -> dict[str, object]:
    rows = []
    for index, sample in enumerate(sample_manifest["samples"]):  # type: ignore[index]
        relation = "scored" if index == 0 else "exact"
        rows.append(
            {
                "category": sample["category"],
                "image": sample["image"],
                "relation": relation,
                "reference_sha256": digest(f"reference-{index}"),
                "candidate_sha256": (
                    digest(f"candidate-{index}")
                    if relation == "scored"
                    else digest(f"reference-{index}")
                ),
                "reference_normalized_sha256": digest(f"normalized-reference-{index}"),
                "candidate_normalized_sha256": (
                    digest(f"normalized-candidate-{index}")
                    if relation == "scored"
                    else digest(f"normalized-reference-{index}")
                ),
                "metrics": (
                    {
                        "text_edit": {"reference": 0.2, "candidate": 0.1},
                        "cdm": {"reference": 0.8, "candidate": 0.9},
                    }
                    if relation == "scored"
                    else {}
                ),
            }
        )
    return {
        "schema": 1,
        "benchmark": "OmniDocBench-v1.6",
        "project_commit": "a" * 40,
        "sample_manifest_sha256": digest("manifest"),
        "dataset_sha256": digest("dataset"),
        "scorer_commit": "b" * 40,
        "scorer_tree_sha256": digest("scorer-tree"),
        "scorer_config_sha256": digest("config"),
        "performance_artifact_sha256": digest("performance"),
        "normalization": "task5-scorer-markdown-v1",
        "samples": rows,
    }


def test_g4_quality_accepts_metric_non_regression() -> None:
    sample_manifest = manifest()
    decision = decide_g4_quality(sample_manifest, artifact(sample_manifest))
    assert decision["g4_quality"] is True
    assert decision["compared_metrics"] == 2


@pytest.mark.parametrize(
    ("metric", "reference", "candidate"),
    [
        ("text_edit", 0.1, 0.2),
        ("formula_edit", 0.1, 0.2),
        ("table_edit", 0.1, 0.2),
        ("reading_order_edit", 0.1, 0.2),
        ("cdm", 0.9, 0.8),
        ("teds", 0.9, 0.8),
        ("teds_structure_only", 0.9, 0.8),
    ],
)
def test_g4_quality_rejects_each_metric_regression(
    metric: str, reference: float, candidate: float
) -> None:
    sample_manifest = manifest()
    run = artifact(sample_manifest)
    run["samples"][0]["metrics"] = {  # type: ignore[index]
        metric: {"reference": reference, "candidate": candidate}
    }
    decision = decide_g4_quality(sample_manifest, run)
    assert decision["g4_quality"] is False
    assert decision["regressions"][0]["metric"] == metric  # type: ignore[index]


def test_g4_quality_rejects_unscored_difference() -> None:
    sample_manifest = manifest()
    run = artifact(sample_manifest)
    run["samples"][0]["metrics"] = {}  # type: ignore[index]
    with pytest.raises(ValueError, match="at least one"):
        decide_g4_quality(sample_manifest, run)


def test_g4_quality_accepts_normalized_only_difference() -> None:
    sample_manifest = manifest()
    run = artifact(sample_manifest)
    row = run["samples"][1]  # type: ignore[index]
    row["relation"] = "normalized"
    row["candidate_sha256"] = digest("different")
    decision = decide_g4_quality(sample_manifest, run)
    assert decision["g4_quality"] is True
    assert decision["normalized_pages"] == 1


def test_g4_quality_requires_at_least_one_scored_page() -> None:
    sample_manifest = manifest()
    run = artifact(sample_manifest)
    row = run["samples"][0]  # type: ignore[index]
    row["relation"] = "exact"
    row["candidate_sha256"] = row["reference_sha256"]
    row["candidate_normalized_sha256"] = row["reference_normalized_sha256"]
    row["metrics"] = {}
    assert decide_g4_quality(sample_manifest, run)["g4_quality"] is False
