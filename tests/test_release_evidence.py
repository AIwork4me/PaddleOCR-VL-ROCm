from __future__ import annotations

import hashlib
import json
import os
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
    assert manifest["inputs"]["main_model"]["bytes"] == 5


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


def test_isolation_normalizes_parent_segments(tmp_path: Path) -> None:
    protected = tmp_path / "history"
    with pytest.raises(ValueError, match="protected historical path"):
        validate_isolated_output_paths([protected / "staging" / ".." / "result.json"], [protected])


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")
def test_isolation_case_folds_and_accepts_mixed_separators(tmp_path: Path) -> None:
    protected = tmp_path / "History"
    mixed = Path(str(tmp_path / "history" / "nested").replace("\\", "/"))
    with pytest.raises(ValueError, match="protected historical path"):
        validate_isolated_output_paths([mixed], [protected])


def test_isolation_resolves_symlinked_output(tmp_path: Path) -> None:
    protected = tmp_path / "history"
    protected.mkdir()
    link = tmp_path / "evidence-link"
    try:
        link.symlink_to(protected, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(ValueError, match="protected historical path"):
        validate_isolated_output_paths([link / "result.json"], [protected])


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


def test_gate_decision_accepts_9599_boundary() -> None:
    decision = decide_release_gates(
        official_stats=accepted_known_failure_stats(),
        lightweight_metric=metric(text=0.034, formula=0.9698, table=0.9439),
    )
    assert decision["overall"] == pytest.approx(95.99)
    assert decision["g3"] is True


def test_gate_decision_fails_closed_on_invalid_metric_quality() -> None:
    value = metric(text=0.01, formula=0.99, table=0.99)
    value["table"]["metric_debug"]["TEDS"]["error_case_count"] = 1
    decision = decide_release_gates(accepted_known_failure_stats(), value)
    assert decision["overall"] == 99.0
    assert decision["g3"] is False


def write_evidence(tmp_path: Path) -> Path:
    evidence = tmp_path / "evidence"
    (evidence / "official").mkdir(parents=True)
    (evidence / "results" / "lightweight").mkdir(parents=True)
    (evidence / "official" / "_run_stats.json").write_text(
        json.dumps(accepted_known_failure_stats()), encoding="utf-8"
    )
    (evidence / "results" / "lightweight" / "metric.json").write_text(
        json.dumps(metric(text=0.01, formula=0.99, table=0.99)), encoding="utf-8"
    )
    model = evidence / "model.gguf"
    model.write_bytes(b"model")
    (evidence / "manifest.json").write_text(
        json.dumps(build_input_manifest({"model": model}, git_commit="abc123")),
        encoding="utf-8",
    )
    return evidence


def run_decide(evidence: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
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


@pytest.mark.parametrize(
    "manifest",
    [
        {"foo": 1},
        {"git_commit": "", "inputs": {"model": {}}},
        {"git_commit": "abc123", "inputs": {}},
        {
            "git_commit": "abc123",
            "inputs": {
                "model": {
                    "path": "model.gguf",
                    "bytes": 5,
                    "sha256": "z" * 64,
                }
            },
        },
        {
            "git_commit": "abc123",
            "inputs": {"model": {"path": "model.gguf", "sha256": "a" * 64}},
        },
        {
            "git_commit": "abc123",
            "inputs": {"bad name": {"path": "model.gguf", "bytes": 5, "sha256": "a" * 64}},
        },
        {
            "git_commit": "abc123",
            "inputs": {"model": {"path": "", "bytes": 5, "sha256": "a" * 64}},
        },
        {
            "git_commit": "abc123",
            "inputs": {"model": {"path": "model.gguf", "bytes": True, "sha256": "a" * 64}},
        },
        {
            "git_commit": "abc123",
            "inputs": {
                "model": {
                    "path": "model.gguf",
                    "bytes": 5,
                    "sha256": "a" * 64,
                    "extra": True,
                }
            },
        },
    ],
)
def test_decide_cli_rejects_invalid_manifest_schema(
    tmp_path: Path, manifest: dict[str, object]
) -> None:
    evidence = write_evidence(tmp_path)
    (evidence / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = run_decide(evidence)
    assert result.returncode != 0
    assert "manifest" in result.stderr.lower()


def test_decide_cli_rejects_non_object_manifest(tmp_path: Path) -> None:
    evidence = write_evidence(tmp_path)
    (evidence / "manifest.json").write_text("[]", encoding="utf-8")
    result = run_decide(evidence)
    assert result.returncode != 0
    assert "must be an object" in result.stderr


def test_decide_cli_rejects_malformed_metric_structure(tmp_path: Path) -> None:
    evidence = write_evidence(tmp_path)
    (evidence / "results" / "lightweight" / "metric.json").write_text(
        json.dumps({"display_formula": {}}), encoding="utf-8"
    )
    result = run_decide(evidence)
    assert result.returncode != 0
    assert "required notebook component" in result.stderr


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_decide_cli_rejects_nonstandard_json_constants(tmp_path: Path, constant: str) -> None:
    evidence = write_evidence(tmp_path)
    (evidence / "results" / "lightweight" / "metric.json").write_text(
        '{"value": ' + constant + "}", encoding="utf-8"
    )
    result = run_decide(evidence)
    assert result.returncode != 0
    assert "non-standard JSON constant" in result.stderr


@pytest.mark.parametrize(
    ("component", "value"),
    [
        ("text", float("nan")),
        ("text", -0.001),
        ("text", 1.001),
        ("formula", float("inf")),
        ("formula", -0.001),
        ("formula", 1.001),
        ("table", float("-inf")),
        ("table", -0.001),
        ("table", 1.001),
    ],
)
def test_gate_decision_rejects_nonfinite_or_out_of_domain_components(
    component: str, value: float
) -> None:
    values = {"text": 0.01, "formula": 0.99, "table": 0.99}
    values[component] = value
    with pytest.raises(ValueError, match="finite.*0.*1"):
        decide_release_gates(accepted_known_failure_stats(), metric(**values))


@pytest.mark.parametrize(
    "mutation",
    ["missing", "directory", "relative", "changed_size", "changed_content", "fake_hash"],
)
def test_decide_cli_revalidates_manifest_files(tmp_path: Path, mutation: str) -> None:
    evidence = write_evidence(tmp_path)
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["inputs"]["model"]
    model = Path(record["path"])
    if mutation == "missing":
        model.unlink()
    elif mutation == "directory":
        model.unlink()
        model.mkdir()
    elif mutation == "relative":
        record["path"] = "model.gguf"
    elif mutation == "changed_size":
        model.write_bytes(b"model-expanded")
    elif mutation == "changed_content":
        model.write_bytes(b"MODEL")
    else:
        record["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = run_decide(evidence)
    assert result.returncode != 0
    assert "manifest input" in result.stderr.lower()


def test_decide_cli_accepts_representative_persisted_evidence(tmp_path: Path) -> None:
    evidence = write_evidence(tmp_path)
    result = run_decide(evidence)
    assert result.returncode == 0, result.stderr
    decision = json.loads(result.stdout)
    assert decision["g0"] is True
    assert decision["g3"] is True


def test_manifest_cli_rejects_duplicate_logical_names(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    result = subprocess.run(
        [
            sys.executable,
            "eval/release_evidence.py",
            "manifest",
            "--git-commit",
            "abc123",
            "--input",
            f"model={first}",
            "--input",
            f"model={second}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "duplicate" in result.stderr.lower()
