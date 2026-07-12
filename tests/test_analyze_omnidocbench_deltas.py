from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.analyze_omnidocbench_deltas import load_component_samples, rank_deltas


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
        "gt": f"gt-{page}-{gt_idx}",
        "pred": pred,
        "metric": {metric: score},
    }


def test_load_component_samples_uses_component_page_and_gt_index_keys(tmp_path: Path):
    result_dir = tmp_path / "official"
    _write_result_dir(
        result_dir,
        [_sample("page-b.png", 1, "CDM", 0.9, "formula")],
        [_sample("page-b.png", 1, "TEDS", 0.8, "table")],
    )

    samples = load_component_samples(result_dir)

    assert [(item["component"], item["img_id"], item["gt_idx"]) for item in samples] == [
        ("Formula CDM", "page-b.png", 1),
        ("Table TEDS", "page-b.png", 1),
    ]
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
            "score": 1.0,
            "gt": "fa",
            "pred": "official-fa",
            "error_metadata": [],
        },
        {
            "component": "Formula CDM",
            "img_id": "page-b.png",
            "gt_idx": 0,
            "score": 0.9,
            "gt": "fb0",
            "pred": "official-fb0",
            "error_metadata": [],
        },
        {
            "component": "Formula CDM",
            "img_id": "page-b.png",
            "gt_idx": 1,
            "score": 0.7,
            "gt": "fb1",
            "pred": "official-fb1",
            "error_metadata": [],
        },
        {
            "component": "Table TEDS",
            "img_id": "page-a.png",
            "gt_idx": 0,
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
            "score": 0.2,
            "gt": "fa",
            "pred": "light-fa",
            "error_metadata": [{"kind": "timeout"}],
        },
        {
            "component": "Formula CDM",
            "img_id": "page-b.png",
            "gt_idx": 0,
            "score": 0.5,
            "gt": "fb0",
            "pred": "light-fb0",
            "error_metadata": [],
        },
        {
            "component": "Formula CDM",
            "img_id": "page-b.png",
            "gt_idx": 1,
            "score": 0.7,
            "gt": "fb1",
            "pred": "light-fb1",
            "error_metadata": [],
        },
        {
            "component": "Table TEDS",
            "img_id": "page-a.png",
            "gt_idx": 0,
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


def test_cli_top_limit_and_output_are_deterministic(tmp_path: Path):
    official = tmp_path / "official"
    lightweight = tmp_path / "lightweight"
    _write_result_dir(
        official,
        [
            _sample("page-z.png", 2, "CDM", 0.9, "official-z"),
            _sample("page-a.png", 1, "CDM", 0.8, "official-a"),
        ],
        [],
    )
    _write_result_dir(
        lightweight,
        [
            _sample("page-z.png", 2, "CDM", 0.1, "light-z"),
            _sample("page-a.png", 1, "CDM", 0.2, "light-a"),
        ],
        [],
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
