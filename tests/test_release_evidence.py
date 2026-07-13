from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.release_contract import KNOWN_V16_OFFICIAL_FAILURE
from eval.release_evidence import (
    build_input_manifest,
    decide_release_gates,
    validate_isolated_output_paths,
)


def accepted_known_failure_stats() -> dict[str, object]:
    failed_image = KNOWN_V16_OFFICIAL_FAILURE["image"]
    details = [{"image": f"page-{index:04d}.png", "status": "ok"} for index in range(1650)]
    details.append(
        {
            "image": failed_image,
            "status": "fail-http",
            "error": "HTTP 500: peg-native",
        }
    )
    return {
        "count": 1651,
        "ok": 1650,
        "fail": 1,
        "fallback": 0,
        "limit_pages": None,
        "engine": "official",
        "stats": details,
    }


def metric(*, text: float, formula: float, table: float) -> dict[str, object]:
    return {
        "text_block": {"all": {"Edit_dist": {"ALL_page_avg": text}}},
        "display_formula": {
            "page": {"CDM": {"ALL": formula}},
            "metric_debug": {
                "CDM": {
                    "sample_count": 1,
                    "timeout_case_count": 0,
                    "exception_case_count": 0,
                }
            },
        },
        "table": {
            "page": {"TEDS": {"ALL": table}},
            "metric_debug": {
                "TEDS": {
                    "sample_count": 1,
                    "timeout_case_count": 0,
                    "error_case_count": 0,
                }
            },
        },
    }


def test_manifest_hashes_every_immutable_input(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    manifest = build_input_manifest({"main_model": model}, git_commit="ac77c5b")
    assert manifest["git_commit"] == "ac77c5b"
    assert manifest["inputs"]["main_model"]["sha256"] == hashlib.sha256(b"model").hexdigest()


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_manifest_rejects_any_input_that_is_not_a_file(tmp_path: Path, kind: str) -> None:
    path = tmp_path / kind
    if kind == "directory":
        path.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        build_input_manifest({kind: path}, git_commit="ac77c5b")


def test_isolation_rejects_historical_result_path() -> None:
    with pytest.raises(ValueError, match="protected historical path"):
        validate_isolated_output_paths(
            [Path("results/omnidocbench/v16/paddleocrvl_rocm_quick_match_metric_result.json")],
            [Path("results/omnidocbench/v16")],
        )


def test_isolation_rejects_protected_root_but_not_prefix_sibling(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "history"
    validate_isolated_output_paths([tmp_path / "history-new"], [protected])
    with pytest.raises(ValueError, match="protected historical path"):
        validate_isolated_output_paths([protected], [protected])


def test_gate_decision_requires_notebook_rounded_overall() -> None:
    decision = decide_release_gates(
        official_stats=accepted_known_failure_stats(),
        lightweight_metric=metric(text=0.0344, formula=0.969224, table=0.943224),
    )
    assert decision["components"] == {
        "text_edit_dist": 0.034,
        "formula_cdm_percent": 96.922,
        "table_teds_percent": 94.322,
    }
    assert decision["overall"] == 95.948
    assert decision["g0"] is True
    assert decision["g3"] is False


def test_gate_decision_fails_closed_on_invalid_metric_quality() -> None:
    value = metric(text=0.01, formula=0.99, table=0.99)
    value["table"]["metric_debug"]["TEDS"]["error_case_count"] = 1
    decision = decide_release_gates(accepted_known_failure_stats(), value)
    assert decision["overall"] == 99.0
    assert decision["g3"] is False


def test_decide_cli_writes_json_and_rejects_bad_hash(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    (evidence / "official").mkdir(parents=True)
    (evidence / "results" / "lightweight").mkdir(parents=True)
    (evidence / "official" / "_run_stats.json").write_text(
        json.dumps(accepted_known_failure_stats()), encoding="utf-8"
    )
    (evidence / "results" / "lightweight" / "metric.json").write_text(
        json.dumps(metric(text=0.01, formula=0.99, table=0.99)), encoding="utf-8"
    )
    (evidence / "manifest.json").write_text(
        json.dumps({"inputs": {"model": {"sha256": "short"}}}), encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "eval/release_evidence.py",
            "decide",
            "--evidence-root",
            str(evidence),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "64-character" in result.stderr
