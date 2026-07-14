from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval.directml_attestation import attest_directml_profile


def _write_profile(tmp_path: Path, providers: list[str | None]) -> Path:
    events = []
    for index, provider in enumerate(providers):
        args = {} if provider is None else {"provider": provider}
        events.append({"cat": "Node", "name": f"node-{index}", "args": args})
    events.append({"cat": "Session", "name": "model_run", "args": {}})
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(events), encoding="utf-8")
    return profile


def _valid_directml_stats() -> dict[str, object]:
    return {
        "layout_provider_requested": "auto",
        "layout_providers_active": ["DmlExecutionProvider", "CPUExecutionProvider"],
        "layout_fallback_disabled": True,
    }


def test_attestation_requires_positive_dml_and_zero_cpu_nodes(tmp_path: Path) -> None:
    profile = _write_profile(tmp_path, ["DmlExecutionProvider"] * 7)

    report = attest_directml_profile(profile, _valid_directml_stats())

    assert report["verdict"] == "PASS"
    assert report["dml_node_events"] == 7
    assert report["cpu_node_events"] == 0
    assert report["missing_provider_node_events"] == 0
    assert report["other_provider_node_events"] == 0
    assert report["node_providers"] == ["DmlExecutionProvider"]
    assert report["other_providers"] == []
    assert report["profile_sha256"] == hashlib.sha256(profile.read_bytes()).hexdigest()
    assert report["profile_bytes"] == profile.stat().st_size


@pytest.mark.parametrize(
    "providers",
    [[], ["CPUExecutionProvider"], ["DmlExecutionProvider", "CPUExecutionProvider"], [None]],
)
def test_attestation_fails_missing_or_cpu_node_execution(
    tmp_path: Path, providers: list[str | None]
) -> None:
    report = attest_directml_profile(_write_profile(tmp_path, providers), _valid_directml_stats())

    assert report["verdict"] == "FAIL"


@pytest.mark.parametrize(
    ("stats_update", "expected"),
    [
        ({"layout_provider_requested": "directml"}, "FAIL"),
        ({"layout_providers_active": ["DmlExecutionProvider"]}, "FAIL"),
        ({"layout_fallback_disabled": False}, "FAIL"),
    ],
)
def test_attestation_fails_closed_on_invalid_runtime_stats(
    tmp_path: Path, stats_update: dict[str, object], expected: str
) -> None:
    stats = _valid_directml_stats()
    stats.update(stats_update)

    report = attest_directml_profile(
        _write_profile(tmp_path, ["DmlExecutionProvider"]), stats
    )

    assert report["verdict"] == expected


def test_attestation_counts_other_providers_without_copying_raw_profile(tmp_path: Path) -> None:
    profile = _write_profile(
        tmp_path, ["DmlExecutionProvider", "CUDAExecutionProvider", "CUDAExecutionProvider"]
    )

    report = attest_directml_profile(profile, _valid_directml_stats())

    assert report["verdict"] == "FAIL"
    assert report["other_provider_node_events"] == 2
    assert report["node_providers"] == ["CUDAExecutionProvider", "DmlExecutionProvider"]
    assert report["other_providers"] == ["CUDAExecutionProvider"]
    assert set(report) == {
        "dml_node_events",
        "cpu_node_events",
        "missing_provider_node_events",
        "other_provider_node_events",
        "node_providers",
        "other_providers",
        "profile_sha256",
        "profile_bytes",
        "verdict",
    }
