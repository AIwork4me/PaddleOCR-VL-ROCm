from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.attribute_accuracy_deltas import attribute_case

BOUNDARIES = ("crop", "payload", "raw_vlm", "final_output")


def test_formula_attributes_first_positive_cdm_effect_to_crop() -> None:
    official = {name: f"official-{name}" for name in BOUNDARIES}
    lightweight = {name: f"lightweight-{name}" for name in BOUNDARIES}

    def replay(boundary: str, value: object) -> float:
        assert isinstance(value, str)
        return 0.965 if boundary == "crop" and value.startswith("official") else 0.061

    result = attribute_case(official, lightweight, replay)

    assert result["status"] == "proven"
    assert result["boundary"] == "crop"
    assert result["contribution"] == pytest.approx(0.904)


def test_table_attributes_content_aware_teds_effect_to_raw_vlm() -> None:
    official = {name: {"source": "official", "html": name} for name in BOUNDARIES}
    lightweight = {name: {"source": "lightweight", "html": name} for name in BOUNDARIES}

    def content_aware_teds_replay(boundary: str, value: object) -> float:
        assert isinstance(value, dict)
        if boundary == "raw_vlm" and value["source"] == "official":
            return 0.975
        return 0.6340579710144927

    result = attribute_case(official, lightweight, content_aware_teds_replay)

    assert result["boundary"] == "raw_vlm"
    assert result["contribution"] == pytest.approx(0.3409420289855073)


def test_each_oracle_is_restored_before_the_next_boundary() -> None:
    official = {name: f"official-{name}" for name in BOUNDARIES}
    lightweight = {name: f"lightweight-{name}" for name in BOUNDARIES}
    calls: list[tuple[str, object]] = []

    def replay(boundary: str, value: object) -> float:
        calls.append((boundary, value))
        return 0.5

    attribute_case(official, lightweight, replay)

    assert calls == [
        call
        for boundary in BOUNDARIES
        for call in (
            (boundary, lightweight[boundary]),
            (boundary, official[boundary]),
            (boundary, lightweight[boundary]),
        )
    ]


def test_negative_and_zero_effects_are_recorded_but_not_called_causal() -> None:
    official = {name: f"official-{name}" for name in BOUNDARIES}
    lightweight = {name: f"lightweight-{name}" for name in BOUNDARIES}

    def replay(boundary: str, value: object) -> float:
        baseline = 0.7
        if value == f"official-{boundary}":
            return {"crop": 0.6, "payload": 0.7, "raw_vlm": 0.9, "final_output": 1.0}[boundary]
        return baseline

    result = attribute_case(official, lightweight, replay)

    assert result["boundary"] == "raw_vlm"
    assert [entry["contribution"] for entry in result["contributions"]] == pytest.approx(
        [-0.1, 0.0, 0.2, 0.3]
    )


@pytest.mark.parametrize("missing", BOUNDARIES)
def test_unobservable_official_input_is_unproven_not_zero(missing: str) -> None:
    official: dict[str, Any] = {name: f"official-{name}" for name in BOUNDARIES}
    official[missing] = {"status": "unobservable"}
    lightweight = {name: f"lightweight-{name}" for name in BOUNDARIES}

    result = attribute_case(official, lightweight, lambda _boundary, _value: 0.5)

    entry = next(item for item in result["contributions"] if item["boundary"] == missing)
    assert entry == {"boundary": missing, "status": "unproven"}
    assert "contribution" not in entry


def test_fingerprint_only_observation_is_not_a_replayable_official_input() -> None:
    official = {
        "boundaries": {
            name: {"status": "observable", "fingerprint": "a" * 64} for name in BOUNDARIES
        }
    }
    lightweight = {
        "boundaries": {
            name: {"status": "observable", "fingerprint": "b" * 64} for name in BOUNDARIES
        }
    }

    result = attribute_case(official, lightweight, lambda *_args: pytest.fail("must not replay"))

    assert result["status"] == "unproven"
    assert all(entry["status"] == "unproven" for entry in result["contributions"])


def test_real_twenty_case_summary_has_no_authenticated_oracle_inputs() -> None:
    summary_path = Path("tests/fixtures/accuracy/v16-trace-capture-summary.json")
    cases = json.loads(summary_path.read_text(encoding="utf-8"))["cases"]

    results = [
        attribute_case(case["official"], case["lightweight"], lambda *_: pytest.fail("replay"))
        for case in cases
    ]

    assert len(results) == 20
    assert {result["status"] for result in results} == {"unproven"}
    assert all("contribution" not in result for result in results)
