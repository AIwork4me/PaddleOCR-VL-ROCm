from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.analyze_omnidocbench_deltas import (
    load_component_samples,
    rank_deltas,
    validate_v16_component_coverage,
)


def _write_result_dir(root: Path, formula: list[dict], table: list[dict]) -> None:
    root.mkdir()
    (root / "end2end_quick_match_display_formula_result.json").write_text(
        json.dumps(formula), encoding="utf-8"
    )
    (root / "end2end_quick_match_table_result.json").write_text(json.dumps(table), encoding="utf-8")


def _sample(page: str, gt_idx: int, metric: str, score: float, pred: str) -> dict:
    return {
        "img_id": page,
        "gt_idx": [gt_idx],
        "gt_position": [gt_idx * 10, 0, gt_idx * 10 + 5, 5],
        "gt": f"gt-{page}-{gt_idx}",
        "pred": pred,
        "metric": {metric: score},
    }


def _page_mean(rows: list[dict], metric: str) -> float:
    by_page: dict[str, list[float]] = {}
    for row in rows:
        by_page.setdefault(row["img_id"], []).append(row["metric"][metric])
    return sum(sum(scores) / len(scores) for scores in by_page.values()) / len(by_page)


def _write_full_v16_result_dir(root: Path, primary_formula: list[dict]) -> None:
    formula = list(primary_formula)
    formula.extend(
        _sample(f"formula-fill-{index % 311}.png", index, "CDM", 1.0, "same")
        for index in range(2352 - len(primary_formula))
    )
    table = [
        _sample(f"table-{index % 458}.png", index, "TEDS", 1.0, "same") for index in range(665)
    ]
    _write_result_dir(root, formula, table)
    (root / "paired_metric_result.json").write_text(
        json.dumps(
            {
                "display_formula": {"page": {"CDM": {"ALL": _page_mean(formula, "CDM")}}},
                "table": {"page": {"TEDS": {"ALL": _page_mean(table, "TEDS")}}},
            }
        ),
        encoding="utf-8",
    )


def test_load_component_samples_uses_immutable_position_and_gt_fingerprint(tmp_path: Path):
    result_dir = tmp_path / "official"
    _write_result_dir(
        result_dir,
        [_sample("page-b.png", 1, "CDM", 0.9, "formula")],
        [_sample("page-b.png", 1, "TEDS", 0.8, "table")],
    )

    samples = load_component_samples(result_dir)

    assert [(item["component"], item["img_id"], item["gt_position"]) for item in samples] == [
        ("Formula CDM", "page-b.png", [10, 0, 15, 5]),
        ("Table TEDS", "page-b.png", [10, 0, 15, 5]),
    ]
    assert all(len(item["gt_fingerprint"]) == 64 for item in samples)
    assert [item["score"] for item in samples] == [0.9, 0.8]
    assert samples[0]["gt"] == "gt-page-b.png-1"
    assert samples[0]["pred"] == "formula"


def test_load_component_samples_attaches_metric_error_metadata(tmp_path: Path):
    result_dir = tmp_path / "official"
    _write_result_dir(
        result_dir,
        [_sample("page-a.png", 3, "CDM", 0.0, "")],
        [],
    )
    (result_dir / "run_metric_result_cdm.json").write_text(
        json.dumps(
            {
                "display_formula": {
                    "metric_debug": {
                        "CDM": {
                            "timeout_cases": [
                                {"img_id": "page-a.png", "gt_idx": [3], "reason": "timeout"}
                            ]
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    samples = load_component_samples(result_dir)

    assert samples[0]["error_metadata"] == [
        {"kind": "timeout", "img_id": "page-a.png", "gt_idx": [3], "reason": "timeout"}
    ]


def test_rank_deltas_ranks_losses_and_keeps_metrics_separate():
    reference = [
        {
            "component": "Formula CDM",
            "img_id": "page-a.png",
            "gt_idx": 0,
            "gt_position": [0, 0, 5, 5],
            "gt_fingerprint": "formula-a",
            "score": 1.0,
            "gt": "fa",
            "pred": "official-fa",
            "error_metadata": [],
        },
        {
            "component": "Formula CDM",
            "img_id": "page-b.png",
            "gt_idx": 0,
            "gt_position": [0, 0, 5, 5],
            "gt_fingerprint": "formula-b0",
            "score": 0.9,
            "gt": "fb0",
            "pred": "official-fb0",
            "error_metadata": [],
        },
        {
            "component": "Formula CDM",
            "img_id": "page-b.png",
            "gt_idx": 1,
            "gt_position": [10, 0, 15, 5],
            "gt_fingerprint": "formula-b1",
            "score": 0.7,
            "gt": "fb1",
            "pred": "official-fb1",
            "error_metadata": [],
        },
        {
            "component": "Table TEDS",
            "img_id": "page-a.png",
            "gt_idx": 0,
            "gt_position": [0, 0, 5, 5],
            "gt_fingerprint": "table-a",
            "score": 0.8,
            "gt": "ta",
            "pred": "official-ta",
            "error_metadata": [],
        },
    ]
    candidate = [
        {
            "component": "Formula CDM",
            "img_id": "page-a.png",
            "gt_idx": 0,
            "gt_position": [0, 0, 5, 5],
            "gt_fingerprint": "formula-a",
            "score": 0.2,
            "gt": "fa",
            "pred": "light-fa",
            "error_metadata": [{"kind": "timeout"}],
        },
        {
            "component": "Formula CDM",
            "img_id": "page-b.png",
            "gt_idx": 0,
            "gt_position": [0, 0, 5, 5],
            "gt_fingerprint": "formula-b0",
            "score": 0.5,
            "gt": "fb0",
            "pred": "light-fb0",
            "error_metadata": [],
        },
        {
            "component": "Formula CDM",
            "img_id": "page-b.png",
            "gt_idx": 1,
            "gt_position": [10, 0, 15, 5],
            "gt_fingerprint": "formula-b1",
            "score": 0.7,
            "gt": "fb1",
            "pred": "light-fb1",
            "error_metadata": [],
        },
        {
            "component": "Table TEDS",
            "img_id": "page-a.png",
            "gt_idx": 0,
            "gt_position": [0, 0, 5, 5],
            "gt_fingerprint": "table-a",
            "score": 0.1,
            "gt": "ta",
            "pred": "light-ta",
            "error_metadata": [],
        },
    ]

    report = rank_deltas(reference, candidate)

    assert report["delta_definition"] == "official_score - lightweight_score"
    assert [(row["component"], row["page"], row["delta"]) for row in report["ranked_samples"]] == [
        ("Formula CDM", "page-a.png", 0.8),
        ("Table TEDS", "page-a.png", 0.7),
        ("Formula CDM", "page-b.png", 0.4),
        ("Formula CDM", "page-b.png", 0.0),
    ]
    assert report["ranked_samples"][0]["official_prediction"] == "official-fa"
    assert report["ranked_samples"][0]["lightweight_prediction"] == "light-fa"
    assert report["ranked_samples"][0]["lightweight_error_metadata"] == [{"kind": "timeout"}]
    assert [(row["component"], row["page"], row["delta"]) for row in report["ranked_pages"]] == [
        ("Formula CDM", "page-a.png", 0.8),
        ("Table TEDS", "page-a.png", 0.7),
        ("Formula CDM", "page-b.png", 0.2),
    ]
    assert report["components"] == [
        {"component": "Formula CDM", "delta": 0.5, "page_count": 2, "sample_count": 3},
        {"component": "Table TEDS", "delta": 0.7, "page_count": 1, "sample_count": 1},
    ]


def test_rank_deltas_aligns_changed_scorer_indices_by_source_identity(tmp_path: Path):
    official = tmp_path / "official"
    lightweight = tmp_path / "lightweight"
    official.mkdir()
    lightweight.mkdir()
    official_row = _sample("formula.png", 21, "CDM", 1.0, "official")
    lightweight_row = _sample("formula.png", 0, "CDM", 0.25, "lightweight")
    lightweight_row["gt"] = official_row["gt"]
    lightweight_row["gt_position"] = official_row["gt_position"]
    (official / "paired_display_formula_result.json").write_text(
        json.dumps([official_row]), encoding="utf-8"
    )
    (lightweight / "paired_display_formula_result.json").write_text(
        json.dumps([lightweight_row]), encoding="utf-8"
    )

    report = rank_deltas(load_component_samples(official), load_component_samples(lightweight))

    assert report["matched_sample_count"] == 1
    assert report["reference_only_keys"] == []
    assert report["lightweight_only_keys"] == []
    assert report["ranked_samples"][0]["delta"] == 0.75


def test_v16_coverage_rejects_consistently_incomplete_sample_and_metric(tmp_path: Path):
    result_dir = tmp_path / "incomplete"
    _write_result_dir(
        result_dir,
        [_sample("only-formula-page.png", 0, "CDM", 1.0, "formula")],
        [_sample("only-table-page.png", 0, "TEDS", 1.0, "table")],
    )
    (result_dir / "paired_metric_result.json").write_text(
        json.dumps(
            {
                "display_formula": {"page": {"CDM": {"ALL": 1.0}}},
                "table": {"page": {"TEDS": {"ALL": 1.0}}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Formula CDM coverage mismatch"):
        validate_v16_component_coverage(result_dir, load_component_samples(result_dir))


def test_v16_coverage_rejects_missing_formula_page_with_full_sample_count(tmp_path: Path):
    samples = []
    for index in range(2352):
        samples.append(
            {
                "component": "Formula CDM",
                "img_id": f"formula-{index % 312}.png",
                "score": 1.0,
            }
        )
    for index in range(665):
        samples.append(
            {
                "component": "Table TEDS",
                "img_id": f"table-{index % 458}.png",
                "score": 1.0,
            }
        )

    with pytest.raises(ValueError, match="pages=312, expected 313"):
        validate_v16_component_coverage(tmp_path, samples)


def test_v16_coverage_rejects_missing_table_page_with_full_sample_count(tmp_path: Path):
    samples = []
    for index in range(2352):
        samples.append(
            {
                "component": "Formula CDM",
                "img_id": f"formula-{index % 313}.png",
                "score": 1.0,
            }
        )
    for index in range(665):
        samples.append(
            {
                "component": "Table TEDS",
                "img_id": f"table-{index % 457}.png",
                "score": 1.0,
            }
        )

    with pytest.raises(ValueError, match="pages=457, expected 458"):
        validate_v16_component_coverage(tmp_path, samples)


def test_cli_top_limit_and_output_are_deterministic(tmp_path: Path):
    official = tmp_path / "official"
    lightweight = tmp_path / "lightweight"
    _write_full_v16_result_dir(
        official,
        [
            _sample("page-z.png", 2, "CDM", 0.9, "official-z"),
            _sample("page-a.png", 1, "CDM", 0.8, "official-a"),
        ],
    )
    _write_full_v16_result_dir(
        lightweight,
        [
            _sample("page-z.png", 2, "CDM", 0.1, "light-z"),
            _sample("page-a.png", 1, "CDM", 0.2, "light-a"),
        ],
    )
    output = tmp_path / "report.json"
    script = Path(__file__).parents[1] / "scripts" / "analyze_omnidocbench_deltas.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--official-result-dir",
            str(official),
            "--lightweight-result-dir",
            str(lightweight),
            "--out-json",
            str(output),
            "--top",
            "1",
        ],
        check=True,
    )
    first = output.read_bytes()
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--official-result-dir",
            str(official),
            "--lightweight-result-dir",
            str(lightweight),
            "--out-json",
            str(output),
            "--top",
            "1",
        ],
        check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert output.read_bytes() == first
    assert len(report["ranked_samples"]) == 1
    assert report["ranked_samples"][0]["page"] == "page-z.png"
