from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
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


def _run_cli(profile: Path, stats: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.directml_attestation",
            "--profile",
            str(profile),
            "--stats",
            str(stats),
            *extra,
        ],
        capture_output=True,
        check=False,
        text=True,
    )


def _write_valid_stats(tmp_path: Path) -> Path:
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps(_valid_directml_stats()), encoding="utf-8")
    return stats


def test_attestation_accepts_directml_majority_with_cpu_graph_partitions(tmp_path: Path) -> None:
    profile = _write_profile(
        tmp_path,
        ["DmlExecutionProvider"] * 7 + ["CPUExecutionProvider"] * 3,
    )

    report = attest_directml_profile(profile, _valid_directml_stats())

    assert report["verdict"] == "PASS"
    assert report["dml_node_events"] == 7
    assert report["cpu_node_events"] == 3
    assert report["dml_node_share"] == 0.7
    assert report["cpu_node_share"] == 0.3
    assert report["missing_provider_node_events"] == 0
    assert report["other_provider_node_events"] == 0
    assert report["node_providers"] == ["CPUExecutionProvider", "DmlExecutionProvider"]
    assert report["other_providers"] == []
    assert report["profile_sha256"] == hashlib.sha256(profile.read_bytes()).hexdigest()
    assert report["profile_bytes"] == profile.stat().st_size


@pytest.mark.parametrize(
    "providers",
    [
        [],
        ["CPUExecutionProvider"],
        ["DmlExecutionProvider", "CPUExecutionProvider"],
        ["DmlExecutionProvider", "CPUExecutionProvider", "CPUExecutionProvider"],
        [None],
    ],
)
def test_attestation_fails_without_strict_dml_majority(
    tmp_path: Path, providers: list[str | None]
) -> None:
    report = attest_directml_profile(_write_profile(tmp_path, providers), _valid_directml_stats())

    assert report["verdict"] == "FAIL"
    assert math.isfinite(report["dml_node_share"])
    assert math.isfinite(report["cpu_node_share"])
    assert 0.0 <= report["dml_node_share"] <= 1.0
    assert 0.0 <= report["cpu_node_share"] <= 1.0


def test_attestation_node_shares_sum_to_one_when_provider_nodes_exist(tmp_path: Path) -> None:
    report = attest_directml_profile(
        _write_profile(
            tmp_path,
            ["DmlExecutionProvider"] * 2 + ["CPUExecutionProvider"],
        ),
        _valid_directml_stats(),
    )

    assert report["dml_node_share"] + report["cpu_node_share"] == pytest.approx(1.0)


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

    report = attest_directml_profile(_write_profile(tmp_path, ["DmlExecutionProvider"]), stats)

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
        "dml_node_share",
        "cpu_node_share",
        "missing_provider_node_events",
        "other_provider_node_events",
        "node_providers",
        "other_providers",
        "profile_sha256",
        "profile_bytes",
        "verdict",
    }


@pytest.mark.parametrize(
    "raw_profile",
    [
        '[{"cat":"Node","args":{"provider":"DmlExecutionProvider"}},NaN]',
        '[{"cat":"Node","args":{"provider":"DmlExecutionProvider"}},42]',
        '[{"cat":"Node","args":{"provider":"DmlExecutionProvider"}},[]]',
        '[{"cat":"Node","args":{"provider":"DmlExecutionProvider"}},null]',
        (
            '[{"cat":"Node","args":{"provider":"CPUExecutionProvider",'
            '"provider":"DmlExecutionProvider"}}]'
        ),
    ],
)
def test_profile_strict_json_rejects_nonfinite_nonobject_and_duplicate_keys(
    tmp_path: Path, raw_profile: str
) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(raw_profile, encoding="utf-8")

    with pytest.raises(ValueError):
        attest_directml_profile(profile, _valid_directml_stats())


@pytest.mark.parametrize(
    "raw_stats",
    [
        (
            '{"layout_provider_requested":"auto",'
            '"layout_providers_active":["DmlExecutionProvider","CPUExecutionProvider"],'
            '"layout_fallback_disabled":false,"layout_fallback_disabled":true}'
        ),
        (
            '{"layout_provider_requested":"auto",'
            '"layout_providers_active":["CPUExecutionProvider"],'
            '"layout_providers_active":["DmlExecutionProvider","CPUExecutionProvider"],'
            '"layout_fallback_disabled":true}'
        ),
        (
            '{"layout_provider_requested":"auto",'
            '"layout_providers_active":["DmlExecutionProvider","CPUExecutionProvider"],'
            '"layout_fallback_disabled":true,"duration":Infinity}'
        ),
    ],
)
def test_cli_rejects_ambiguous_or_nonfinite_stats(tmp_path: Path, raw_stats: str) -> None:
    profile = _write_profile(tmp_path, ["DmlExecutionProvider"])
    stats = tmp_path / "stats.json"
    stats.write_text(raw_stats, encoding="utf-8")

    result = _run_cli(profile, stats)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_fail_verdict_is_nonzero_unless_explicitly_allowed(tmp_path: Path) -> None:
    profile = _write_profile(tmp_path, ["CPUExecutionProvider"])
    stats = _write_valid_stats(tmp_path)

    strict = _run_cli(profile, stats)
    allowed = _run_cli(profile, stats, "--allow-fail-verdict")

    assert strict.returncode == 1
    assert allowed.returncode == 0
    assert json.loads(strict.stdout) == json.loads(allowed.stdout)
    assert json.loads(strict.stdout)["verdict"] == "FAIL"
    assert strict.stderr == allowed.stderr == ""


def test_cli_pass_verdict_returns_zero(tmp_path: Path) -> None:
    result = _run_cli(
        _write_profile(tmp_path, ["DmlExecutionProvider"]), _write_valid_stats(tmp_path)
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["verdict"] == "PASS"
    assert result.stderr == ""


def test_cli_malformed_profile_has_stable_error_without_traceback(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text("{", encoding="utf-8")

    result = _run_cli(profile, _write_valid_stats(tmp_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("error: invalid JSON in ")
    assert "Traceback" not in result.stderr
