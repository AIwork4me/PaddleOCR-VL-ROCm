from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

import eval.task5_decision as task5_decision
from eval.task5_decision import (
    amd_adaptation_decision,
    build_task5_receipt,
    extract_paired_scores,
    strict_equivalence_decision,
    validate_task5_receipt,
)


def scores(
    *,
    text_edit: float = 0.030,
    formula: float = 96.0,
    table: float = 94.0,
    overall: float = 96.20,
) -> dict[str, object]:
    return {
        "text_edit_dist": text_edit,
        "formula_cdm_percent": formula,
        "table_teds_percent": table,
        "reading_order_edit_dist": 0.02,
        "overall": overall,
        "metric_quality": {
            "formula_cdm": {"valid": True},
            "table_teds": {"valid": True},
        },
    }


def valid_lightweight_stats() -> dict[str, object]:
    return {
        "count": 1651,
        "ok": 1651,
        "fail": 0,
        "fallback": 0,
        "layout_provider_requested": "auto",
        "layout_providers_active": [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
        "layout_fallback_disabled": True,
    }


def valid_output_report(**updates: object) -> dict[str, object]:
    report: dict[str, object] = {
        "verdict": "PASS",
        "expected_paired_pages": 1650,
        "paired_pages": 1650,
        "equal_pages": 1650,
        "different_pages": 0,
        "official_only_pages": 0,
        "lightweight_only_pages": 0,
        "approved_exclusion": {
            "stem": "newspaper_The Times UK_0801@magazinesclubnew_page_031",
            "official_present": True,
            "lightweight_present": True,
        },
    }
    report.update(updates)
    return report


def valid_provider_attestation(
    *, dml: int = 1101, cpu: int = 150
) -> dict[str, object]:
    provider_nodes = dml + cpu
    return {
        "verdict": "PASS",
        "dml_node_events": dml,
        "cpu_node_events": cpu,
        "dml_node_share": dml / provider_nodes,
        "cpu_node_share": cpu / provider_nodes,
        "missing_provider_node_events": 0,
        "other_provider_node_events": 0,
        "node_providers": ["CPUExecutionProvider", "DmlExecutionProvider"],
        "other_providers": [],
    }


def decide_amd(
    official: dict[str, object], lightweight: dict[str, object]
) -> dict[str, object]:
    return amd_adaptation_decision(
        official_scores=official,
        lightweight_scores=lightweight,
        provider_attestation=valid_provider_attestation(),
        lightweight_stats=valid_lightweight_stats(),
        public_contracts_pass=True,
    )


def decide_with_lightweight_overall(overall: float) -> dict[str, object]:
    return decide_amd(scores(overall=96.20), scores(overall=overall))


def metric(
    *,
    text: float = 0.0304,
    formula: float = 0.96,
    table: float = 0.94,
    reading: float = 0.0204,
    formula_samples: int = 3,
    table_samples: int = 4,
) -> dict[str, object]:
    return {
        "text_block": {"all": {"Edit_dist": {"ALL_page_avg": text}}},
        "display_formula": {
            "page": {"CDM": {"ALL": formula}},
            "metric_debug": {
                "CDM": {
                    "sample_count": formula_samples,
                    "timeout_case_count": 0,
                    "exception_case_count": 0,
                }
            },
        },
        "table": {
            "page": {
                "TEDS": {"ALL": table},
                "TEDS_structure_only": {"ALL": table},
            },
            "metric_debug": {
                "TEDS": {
                    "sample_count": table_samples,
                    "timeout_case_count": 0,
                    "error_case_count": 0,
                }
            },
        },
        "reading_order": {"all": {"Edit_dist": {"ALL_page_avg": reading}}},
    }


def test_strict_fail_beats_unknown() -> None:
    decision = strict_equivalence_decision(
        valid_output_report(verdict="FAIL", equal_pages=1649, different_pages=1),
        {"verdict": "UNKNOWN", "unobservable_count": 8},
    )
    assert decision["verdict"] == "FAIL"


def test_strict_verdict_requires_complete_equal_outputs_and_trace() -> None:
    complete = valid_output_report()
    assert strict_equivalence_decision(complete, {"verdict": "PASS"})["verdict"] == "PASS"
    assert (
        strict_equivalence_decision(complete, {"verdict": "UNKNOWN"})["verdict"]
        == "UNKNOWN"
    )
    assert (
        strict_equivalence_decision(
            {
                **complete,
                "verdict": "FAIL",
                "paired_pages": 1649,
                "equal_pages": 1649,
            },
            {"verdict": "PASS"},
        )["verdict"]
        == "FAIL"
    )


def test_strict_rejects_missing_only_fields_and_inconsistent_arithmetic() -> None:
    missing_only = valid_output_report()
    del missing_only["official_only_pages"]
    with pytest.raises(ValueError, match="official_only_pages"):
        strict_equivalence_decision(missing_only, {"verdict": "PASS"})

    with pytest.raises(ValueError, match="equal_pages.*different_pages"):
        strict_equivalence_decision(
            valid_output_report(equal_pages=1650, different_pages=1),
            {"verdict": "PASS"},
        )


@pytest.mark.parametrize(
    "contradiction",
    [
        {"verdict": "PASS", "equal_pages": 1649, "different_pages": 1},
        {"verdict": "FAIL"},
        {"expected_paired_pages": 1649},
    ],
)
def test_strict_rejects_report_verdict_or_coverage_contradiction(
    contradiction: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="report|expected_paired_pages"):
        strict_equivalence_decision(
            valid_output_report(**contradiction), {"verdict": "PASS"}
        )


def test_strict_unknown_does_not_block_independent_amd_pass() -> None:
    amd = amd_adaptation_decision(
        official_scores=scores(overall=96.20),
        lightweight_scores=scores(overall=96.20),
        provider_attestation=valid_provider_attestation(),
        lightweight_stats=valid_lightweight_stats(),
        public_contracts_pass=True,
    )
    assert amd["verdict"] == "PASS"
    assert amd["g3"] is True


@pytest.mark.parametrize("overall", [96.129, 95.743])
def test_g3_fails_below_9613(overall: float) -> None:
    assert decide_with_lightweight_overall(overall)["g3"] is False


def test_component_regression_fails_even_when_overall_is_high() -> None:
    official = scores(text_edit=0.030, formula=96.0, table=94.0, overall=96.2)
    lightweight = scores(text_edit=0.031, formula=97.0, table=95.0, overall=96.5)
    assert decide_amd(official, lightweight)["verdict"] == "FAIL"


@pytest.mark.parametrize(
    ("change", "value"),
    [("formula_cdm_percent", 95.999), ("table_teds_percent", 93.999)],
)
def test_higher_is_better_components_cannot_regress(change: str, value: float) -> None:
    lightweight = scores(overall=96.5)
    lightweight[change] = value
    assert decide_amd(scores(), lightweight)["verdict"] == "FAIL"


@pytest.mark.parametrize(
    ("attestation", "stats", "contracts"),
    [
        ({"verdict": "FAIL"}, valid_lightweight_stats(), True),
        (valid_provider_attestation(), {**valid_lightweight_stats(), "fallback": 1}, True),
        (valid_provider_attestation(), valid_lightweight_stats(), False),
    ],
)
def test_amd_requires_attestation_complete_stats_and_contracts(
    attestation: dict[str, object], stats: dict[str, object], contracts: bool
) -> None:
    decision = amd_adaptation_decision(
        official_scores=scores(),
        lightweight_scores=scores(),
        provider_attestation=attestation,
        lightweight_stats=stats,
        public_contracts_pass=contracts,
    )
    assert decision["verdict"] == "FAIL"
    assert decision["g3"] is False


def test_lightweight_coverage_requires_all_1651_pages_successful() -> None:
    assert decide_amd(scores(), scores())["verdict"] == "PASS"
    old_partial = {**valid_lightweight_stats(), "ok": 1650, "fail": 1}
    decision = amd_adaptation_decision(
        official_scores=scores(),
        lightweight_scores=scores(),
        provider_attestation=valid_provider_attestation(),
        lightweight_stats=old_partial,
        public_contracts_pass=True,
    )
    assert decision["checks"]["lightweight_coverage"] is False
    assert decision["verdict"] == "FAIL"


def test_lightweight_coverage_rejects_conflicting_raw_and_summary_aliases() -> None:
    both_valid = {
        **valid_lightweight_stats(),
        "prediction_count": 1651,
        "ok_pages": 1651,
        "failed_pages": 0,
        "fallback_pages": 0,
    }
    valid_decision = amd_adaptation_decision(
        official_scores=scores(),
        lightweight_scores=scores(),
        provider_attestation=valid_provider_attestation(),
        lightweight_stats=both_valid,
        public_contracts_pass=True,
    )
    assert valid_decision["checks"]["lightweight_coverage"] is True

    conflicting = {
        **valid_lightweight_stats(),
        "prediction_count": 1651,
        "ok_pages": 1650,
        "failed_pages": 1,
        "fallback_pages": 0,
    }
    decision = amd_adaptation_decision(
        official_scores=scores(),
        lightweight_scores=scores(),
        provider_attestation=valid_provider_attestation(),
        lightweight_stats=conflicting,
        public_contracts_pass=True,
    )
    assert decision["checks"]["lightweight_coverage"] is False


@pytest.mark.parametrize(
    "invalid",
    [
        {
            key: value
            for key, value in valid_lightweight_stats().items()
            if key not in {"count", "ok", "fail", "fallback"}
        },
        {key: value for key, value in valid_lightweight_stats().items() if key != "ok"},
        {**valid_lightweight_stats(), "fallback": False},
    ],
)
def test_lightweight_coverage_rejects_missing_partial_or_noninteger_counts(
    invalid: dict[str, object],
) -> None:
    decision = amd_adaptation_decision(
        official_scores=scores(),
        lightweight_scores=scores(),
        provider_attestation=valid_provider_attestation(),
        lightweight_stats=invalid,
        public_contracts_pass=True,
    )
    assert decision["checks"]["lightweight_coverage"] is False


def test_extract_paired_scores_uses_cdm_formula_and_approved_rounding() -> None:
    non_cdm = metric(text=0.03044, formula=0.10, table=0.94444, reading=0.02044)
    cdm = metric(text=0.03045, formula=0.96777, table=0.94444, reading=0.02045)

    extracted = extract_paired_scores(non_cdm, cdm)

    assert extracted["text_edit_dist"] == 0.03
    assert extracted["formula_cdm_percent"] == 96.777
    assert extracted["table_teds_percent"] == 94.444
    assert extracted["reading_order_edit_dist"] == 0.02
    assert extracted["overall"] == pytest.approx((97.0 + 96.777 + 94.444) / 3)


def test_extract_paired_scores_rejects_rounded_disagreement_and_bad_values() -> None:
    with pytest.raises(ValueError, match="agree"):
        extract_paired_scores(metric(text=0.030), metric(text=0.032))
    with pytest.raises(ValueError, match="finite"):
        extract_paired_scores(metric(), metric(formula=float("nan")))
    with pytest.raises(ValueError, match="0..100"):
        extract_paired_scores(metric(), metric(formula=1.01))
    missing = metric()
    del missing["reading_order"]
    with pytest.raises(ValueError, match="missing"):
        extract_paired_scores(missing, metric())


def test_failed_metric_quality_blocks_g3() -> None:
    lightweight = scores(overall=96.5)
    lightweight["metric_quality"] = {
        "formula_cdm": {"valid": False},
        "table_teds": {"valid": True},
    }
    assert decide_amd(scores(), lightweight)["g3"] is False


def test_provider_condition_allows_cpu_nodes_when_dml_has_strict_majority() -> None:
    decision = amd_adaptation_decision(
        official_scores=scores(),
        lightweight_scores=scores(),
        provider_attestation=valid_provider_attestation(dml=1101, cpu=150),
        lightweight_stats=valid_lightweight_stats(),
        public_contracts_pass=True,
    )
    assert decision["checks"]["provider_attestation"] is True
    assert decision["provider_evidence"]["dml_node_events"] == 1101
    assert decision["provider_evidence"]["cpu_node_events"] == 150
    assert decision["verdict"] == "PASS"


def test_provider_condition_rejects_equal_dml_and_cpu_even_if_verdict_claims_pass() -> None:
    decision = amd_adaptation_decision(
        official_scores=scores(),
        lightweight_scores=scores(),
        provider_attestation=valid_provider_attestation(dml=50, cpu=50),
        lightweight_stats=valid_lightweight_stats(),
        public_contracts_pass=True,
    )
    assert decision["checks"]["provider_attestation"] is False
    assert decision["verdict"] == "FAIL"


def test_provider_condition_rejects_share_that_disagrees_with_counts() -> None:
    attestation = valid_provider_attestation(dml=1101, cpu=150)
    attestation["dml_node_share"] = 0.6
    attestation["cpu_node_share"] = 0.4
    decision = amd_adaptation_decision(
        official_scores=scores(),
        lightweight_scores=scores(),
        provider_attestation=attestation,
        lightweight_stats=valid_lightweight_stats(),
        public_contracts_pass=True,
    )
    assert decision["checks"]["provider_attestation"] is False
    assert decision["verdict"] == "FAIL"


def _write_receipt_inputs(root: Path) -> list[str]:
    names = ["manifest.json", "selected-attempt.json", "comparison/decision.json"]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    return names


def test_receipt_is_sorted_root_relative_and_detects_mutation(tmp_path: Path) -> None:
    names = _write_receipt_inputs(tmp_path)
    receipt = build_task5_receipt(tmp_path, list(reversed(names)))

    assert list(receipt["files"]) == sorted(names)
    assert receipt["files"]["manifest.json"]["path"] == "manifest.json"
    validate_task5_receipt(tmp_path, receipt)

    (tmp_path / "comparison" / "decision.json").write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        validate_task5_receipt(tmp_path, receipt)


def test_receipt_rejects_self_hash_escape_absolute_unallowlisted_and_symlink(
    tmp_path: Path,
) -> None:
    _write_receipt_inputs(tmp_path)
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="itself"):
        build_task5_receipt(tmp_path, ["receipt.sha256.json"])
    with pytest.raises(ValueError, match="relative"):
        build_task5_receipt(tmp_path, [str(outside)])
    with pytest.raises(ValueError, match="escape"):
        build_task5_receipt(tmp_path, ["../outside-secret.txt"])
    with pytest.raises(ValueError, match="allowlist"):
        build_task5_receipt(tmp_path, ["secrets.env"])
    comparison = tmp_path / "comparison"
    comparison.mkdir(exist_ok=True)
    link = comparison / "directml-attestation.json"
    try:
        link.symlink_to(comparison / "decision.json")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="symlink"):
        build_task5_receipt(tmp_path, ["comparison/directml-attestation.json"])


def test_receipt_accepts_real_results_and_comparison_tree(tmp_path: Path) -> None:
    names = [
        "results/official/metric.json",
        "results/official/metric-cdm.json",
        "results/official/run-summary.json",
        "results/official/run-summary-cdm.json",
        "results/official/provenance.json",
        "results/official/provenance-cdm.json",
        "results/lightweight/metric.json",
        "results/lightweight/metric-cdm.json",
        "results/lightweight/run-summary.json",
        "results/lightweight/run-summary-cdm.json",
        "results/lightweight/provenance.json",
        "results/lightweight/provenance-cdm.json",
        "comparison/input-contract.json",
        "comparison/normalized-output.json",
        "comparison/trace-diff.json",
        "comparison/directml-attestation.json",
        "comparison/decision.json",
    ]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    assert set(build_task5_receipt(tmp_path, names)["files"]) == set(names)


@pytest.mark.parametrize(
    "name",
    [
        "comparison/unknown.json",
        "lightweight/page-0001.md",
        "traces/lightweight/page-0001.jsonl",
        "results/lightweight/raw-response.json",
    ],
)
def test_receipt_rejects_unknown_and_raw_members(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("sensitive", encoding="utf-8")
    with pytest.raises(ValueError, match="allowlist"):
        build_task5_receipt(tmp_path, [name])


def test_receipt_rejects_same_length_in_place_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("AAAA", encoding="utf-8")
    original_read = os.read
    mutated = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            path.write_text("BBBB", encoding="utf-8")
        return chunk

    monkeypatch.setattr(os, "read", mutating_read)
    with pytest.raises(ValueError, match="changed"):
        build_task5_receipt(tmp_path, ["manifest.json"])


def test_cli_decide_writes_fail_decision_without_infrastructure_error(tmp_path: Path) -> None:
    inputs = {
        "official-non-cdm": metric(table=0.96),
        "official-cdm": metric(table=0.96),
        "lightweight-non-cdm": metric(table=0.96),
        "lightweight-cdm": metric(table=0.96),
        "output-report": valid_output_report(
            verdict="FAIL", equal_pages=1649, different_pages=1
        ),
        "trace-report": {"verdict": "PASS"},
        "provider-attestation": valid_provider_attestation(),
        "lightweight-stats": valid_lightweight_stats(),
    }
    args: list[str] = []
    for name, value in inputs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        args.extend([f"--{name}", str(path)])
    output = tmp_path / "decision.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.task5_decision",
            "decide",
            *args,
            "--public-contracts-pass",
            "--output",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert set(decision) == {
        "schema",
        "benchmark",
        "coverage",
        "scores",
        "strict_equivalence",
        "amd_adaptation",
        "g3",
        "evidence",
    }
    assert decision["strict_equivalence"]["verdict"] == "FAIL"
    assert decision["amd_adaptation"]["verdict"] == "PASS"
    assert decision["g3"] is True


def _decision_cli_namespace(tmp_path: Path) -> Namespace:
    values = {
        "official_non_cdm": metric(table=0.96),
        "official_cdm": metric(table=0.96),
        "lightweight_non_cdm": metric(table=0.96),
        "lightweight_cdm": metric(table=0.96),
        "output_report": valid_output_report(),
        "trace_report": {"verdict": "PASS"},
        "provider_attestation": valid_provider_attestation(),
        "lightweight_stats": valid_lightweight_stats(),
    }
    paths: dict[str, Path] = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        paths[name] = path
    return Namespace(
        **paths,
        public_contracts_pass=True,
        output=tmp_path / "decision.json",
    )


def test_decision_rejects_parse_hash_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _decision_cli_namespace(tmp_path)
    target = args.official_non_cdm
    original_read = os.read
    mutated = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            raw = target.read_bytes()
            target.write_bytes(raw.replace(b"0.0304", b"0.0305"))
        return chunk

    monkeypatch.setattr(os, "read", mutating_read)
    with pytest.raises(ValueError, match="changed"):
        task5_decision._decide(args)


def test_cli_rejects_decision_and_receipt_output_input_collisions(tmp_path: Path) -> None:
    args = _decision_cli_namespace(tmp_path)
    args.output = args.official_non_cdm
    original = args.output.read_bytes()
    with pytest.raises(ValueError, match="output.*input"):
        task5_decision._decide(args)
    assert args.output.read_bytes() == original

    root = tmp_path / "receipt-root"
    root.mkdir()
    (root / "manifest.json").write_text("evidence", encoding="utf-8")
    receipt_args = Namespace(
        task5_root=root,
        path=["manifest.json"],
        output=root / "manifest.json",
    )
    with pytest.raises(ValueError, match="output.*input"):
        task5_decision._build_receipt_cli(receipt_args)
    assert (root / "manifest.json").read_text(encoding="utf-8") == "evidence"


def test_cli_rejects_duplicate_or_nonfinite_json_without_traceback(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"paired_pages":1650,"paired_pages":1,"x":NaN}', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.task5_decision",
            "validate-receipt",
            "--task5-root",
            str(tmp_path),
            "--receipt",
            str(bad),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr
