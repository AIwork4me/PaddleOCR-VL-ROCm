from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_formula_cdm_cases import load_formula_scores, summarize_cases


def test_load_formula_scores_accepts_per_sample_mapping(tmp_path: Path):
    sample_path = tmp_path / "per_sample.json"
    sample_path.write_text(
        json.dumps(
            {
                "page_a.png": [
                    {"sample_id": "a-1", "CDM": 1.0, "gt": "x", "pred": "x"},
                    {"sample_id": "a-2", "CDM": 0.25, "gt": "\\frac{1}{2}", "pred": ""},
                ],
                "page_b.png": [{"sample_id": "b-1", "CDM": 0.0, "gt": "y", "pred": "bad"}],
            }
        ),
        encoding="utf-8",
    )

    cases = load_formula_scores(sample_path)

    assert [case["page"] for case in cases] == ["page_a.png", "page_a.png", "page_b.png"]
    assert [case["cdm"] for case in cases] == [1.0, 0.25, 0.0]
    assert cases[1]["pred"] == ""


def test_load_formula_scores_accepts_scalar_sample_mapping(tmp_path: Path):
    sample_path = tmp_path / "scalar_per_sample.json"
    sample_path.write_text(
        json.dumps({"page-d156.png_[3]": 1.0, "page-other.png": 0.25}),
        encoding="utf-8",
    )

    cases = load_formula_scores(sample_path)

    assert cases == [
        {"page": "page-d156.png", "sample_id": "3", "cdm": 1.0, "gt": "", "pred": ""},
        {"page": "page-other.png", "sample_id": "", "cdm": 0.25, "gt": "", "pred": ""},
    ]


def test_summarize_cases_ranks_lowest_cases():
    cases = [
        {"page": "a", "sample_id": "1", "cdm": 0.5},
        {"page": "b", "sample_id": "2", "cdm": 0.0},
        {"page": "c", "sample_id": "3", "cdm": 0.9},
    ]

    summary = summarize_cases(cases, threshold=0.8)

    assert summary["count"] == 3
    assert summary["below_threshold_count"] == 2
    assert summary["zero_count"] == 1
    assert [case["page"] for case in summary["lowest_cases"]] == ["b", "a", "c"]
