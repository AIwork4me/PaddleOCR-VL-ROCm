from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

import eval.task5_decision as task5_decision
from eval.task5_comparison import (
    compare_boundary_documents,
    observation,
    unobservable,
)
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


TRACE_BOUNDARIES = (
    "request_order",
    "label",
    "bbox",
    "crop_pixels",
    "prompt",
    "payload",
    "raw_result",
    "postprocess",
)


def valid_trace_report(verdict: str = "PASS") -> dict[str, object]:
    different_records = 1 if verdict == "FAIL" else 0
    unobservable_records = 1 if verdict == "UNKNOWN" else 0
    first_divergence_counts = {"event_structure": 0}
    first_divergence_counts.update({name: 0 for name in TRACE_BOUNDARIES})
    first_divergence_counts["page_postprocess"] = 0
    if different_records:
        first_divergence_counts["postprocess"] = different_records
    unobservable_counts = {"block_structure": 0}
    unobservable_counts.update({name: 0 for name in TRACE_BOUNDARIES})
    if unobservable_records:
        unobservable_counts["raw_result"] = unobservable_records
    return {
        "verdict": verdict,
        "expected_paired_pages": 1650,
        "paired_pages": 1650,
        "official_only_pages": 0,
        "lightweight_only_pages": 0,
        "empty_page_traces": 0,
        "different_records": different_records,
        "unobservable_records": unobservable_records,
        "first_divergence_counts": first_divergence_counts,
        "unobservable_counts": unobservable_counts,
        "approved_exclusion": {
            "stem": "newspaper_The Times UK_0801@magazinesclubnew_page_031",
            "official_present": True,
            "lightweight_present": True,
        },
    }


def _comparison_event(
    *,
    boundaries: dict[str, dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "page": "page-1",
        "block_index": 0,
        "boundaries": boundaries
        or {name: observation(f"same-{name}") for name in TRACE_BOUNDARIES},
        "page_postprocess": observation("same-page"),
    }


def _with_trace_coverage(report: dict[str, object]) -> dict[str, object]:
    report.update(
        {
            "expected_paired_pages": 1650,
            "paired_pages": 1650,
            "official_only_pages": 0,
            "lightweight_only_pages": 0,
            "empty_page_traces": 0,
            "approved_exclusion": {
                "stem": "newspaper_The Times UK_0801@magazinesclubnew_page_031",
                "official_present": True,
                "lightweight_present": True,
            },
        }
    )
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
        valid_trace_report("UNKNOWN"),
    )
    assert decision["verdict"] == "FAIL"


def test_strict_verdict_requires_complete_equal_outputs_and_trace() -> None:
    complete = valid_output_report()
    assert strict_equivalence_decision(complete, valid_trace_report())["verdict"] == "PASS"
    assert (
        strict_equivalence_decision(complete, valid_trace_report("UNKNOWN"))["verdict"]
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
            valid_trace_report(),
        )["verdict"]
        == "FAIL"
    )


def test_strict_rejects_missing_only_fields_and_inconsistent_arithmetic() -> None:
    missing_only = valid_output_report()
    del missing_only["official_only_pages"]
    with pytest.raises(ValueError, match="official_only_pages"):
        strict_equivalence_decision(missing_only, valid_trace_report())

    with pytest.raises(ValueError, match="equal_pages.*different_pages"):
        strict_equivalence_decision(
            valid_output_report(equal_pages=1650, different_pages=1),
            valid_trace_report(),
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
            valid_output_report(**contradiction), valid_trace_report()
        )


@pytest.mark.parametrize("verdict", ["PASS", "UNKNOWN", "FAIL"])
def test_strict_accepts_complete_trace_pass_unknown_and_fail(verdict: str) -> None:
    decision = strict_equivalence_decision(
        valid_output_report(), valid_trace_report(verdict)
    )
    assert decision["verdict"] == verdict


def test_strict_accepts_complete_trace_coverage_fail() -> None:
    report = valid_trace_report()
    report.update({"verdict": "FAIL", "paired_pages": 1649})
    decision = strict_equivalence_decision(valid_output_report(), report)
    assert decision["verdict"] == "FAIL"


def test_strict_accepts_real_trace_with_multiple_unobservable_boundaries() -> None:
    official_boundaries = {
        name: unobservable() if name in {"prompt", "raw_result"} else observation(f"same-{name}")
        for name in TRACE_BOUNDARIES
    }
    report = _with_trace_coverage(
        compare_boundary_documents(
            [_comparison_event(boundaries=official_boundaries)],
            [_comparison_event()],
        )
    )
    assert report["verdict"] == "UNKNOWN"
    assert report["unobservable_records"] == 1
    assert sum(report["unobservable_counts"].values()) == 2  # type: ignore[union-attr]
    assert strict_equivalence_decision(valid_output_report(), report)["verdict"] == "UNKNOWN"


def test_strict_accepts_real_different_trace_with_unobservable_boundaries() -> None:
    official_boundaries = {
        name: (
            unobservable()
            if name in {"prompt", "raw_result"}
            else observation("official-postprocess")
            if name == "postprocess"
            else observation(f"same-{name}")
        )
        for name in TRACE_BOUNDARIES
    }
    report = _with_trace_coverage(
        compare_boundary_documents(
            [_comparison_event(boundaries=official_boundaries)],
            [_comparison_event()],
        )
    )
    assert report["verdict"] == "FAIL"
    assert report["different_records"] == 1
    assert report["unobservable_records"] == 0
    assert sum(report["unobservable_counts"].values()) == 2  # type: ignore[union-attr]
    assert strict_equivalence_decision(valid_output_report(), report)["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "case",
    [
        "verdict_only",
        "missing_field",
        "first_extra_key",
        "first_missing_key",
        "first_bool",
        "first_sum",
        "unobservable_extra_key",
        "unobservable_missing_key",
        "unobservable_bool",
        "unobservable_sum",
        "unobservable_below_records",
        "coverage_verdict",
        "forged_verdict",
    ],
)
def test_strict_rejects_incomplete_or_inconsistent_trace_report(case: str) -> None:
    if case == "verdict_only":
        report: dict[str, object] = {"verdict": "PASS"}
    else:
        report = valid_trace_report()
        if case == "missing_field":
            del report["paired_pages"]
        elif case == "first_extra_key":
            report["first_divergence_counts"] = {
                **report["first_divergence_counts"],  # type: ignore[arg-type]
                "extra": 0,
            }
        elif case == "first_missing_key":
            counts = dict(report["first_divergence_counts"])  # type: ignore[arg-type]
            del counts["payload"]
            report["first_divergence_counts"] = counts
        elif case == "first_bool":
            counts = dict(report["first_divergence_counts"])  # type: ignore[arg-type]
            counts["payload"] = False
            report["first_divergence_counts"] = counts
        elif case == "first_sum":
            counts = dict(report["first_divergence_counts"])  # type: ignore[arg-type]
            counts["payload"] = 1
            report["first_divergence_counts"] = counts
        elif case == "unobservable_extra_key":
            report["unobservable_counts"] = {
                **report["unobservable_counts"],  # type: ignore[arg-type]
                "extra": 0,
            }
        elif case == "unobservable_missing_key":
            counts = dict(report["unobservable_counts"])  # type: ignore[arg-type]
            del counts["bbox"]
            report["unobservable_counts"] = counts
        elif case == "unobservable_bool":
            counts = dict(report["unobservable_counts"])  # type: ignore[arg-type]
            counts["bbox"] = False
            report["unobservable_counts"] = counts
        elif case == "unobservable_sum":
            counts = dict(report["unobservable_counts"])  # type: ignore[arg-type]
            counts["bbox"] = 1
            report["unobservable_counts"] = counts
        elif case == "unobservable_below_records":
            report = valid_trace_report("UNKNOWN")
            counts = dict(report["unobservable_counts"])  # type: ignore[arg-type]
            counts["raw_result"] = 0
            report["unobservable_counts"] = counts
        elif case == "coverage_verdict":
            report["paired_pages"] = 1649
        elif case == "forged_verdict":
            report["verdict"] = "UNKNOWN"
        else:
            raise AssertionError(case)
    with pytest.raises(ValueError):
        strict_equivalence_decision(valid_output_report(), report)


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
        "trace-report": valid_trace_report(),
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
        "trace_report": valid_trace_report(),
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
