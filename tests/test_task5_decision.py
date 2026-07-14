from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

import eval.task5_decision as task5_decision
import eval.task5_manifest as task5_manifest
from eval.artifact_utils import sha256_file, write_run_summary
from eval.task5_comparison import (
    compare_boundary_documents,
    observation,
    unobservable,
)
from eval.task5_decision import (
    amd_adaptation_decision,
    build_task5_receipt,
    extract_paired_scores,
    required_attempt_receipt_paths,
    strict_equivalence_decision,
    validate_task5_receipt,
    validate_task5_selection,
)
from eval.task5_manifest import OFFICIAL_OUTPUTS, build_task5_manifest

ROOT = Path(__file__).parents[1]
G0_RECEIPT = ROOT / "docs/releases/0.1.0-g0-evidence.md"
TEST_G0_OUTPUT_DIGESTS = {
    relative: task5_decision.hashlib.sha256(
        (json.dumps({"output": relative}) + "\n").encode()
    ).hexdigest()
    for relative in OFFICIAL_OUTPUTS
}


@pytest.fixture(autouse=True)
def use_test_g0_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        task5_manifest, "APPROVED_G0_OUTPUT_SHA256", TEST_G0_OUTPUT_DIGESTS
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


def valid_input_contract() -> dict[str, object]:
    return {
        "benchmark": "OmniDocBench-v1.6",
        "pages": 1651,
        "paired_pages": 1650,
        "approved_exclusion": task5_decision.APPROVED_EXCLUDED_STEM,
        "formula": "CDM",
        "table": "TEDS",
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


def valid_provider_attestation(*, dml: int = 1101, cpu: int = 150) -> dict[str, object]:
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
    assert (
        strict_equivalence_decision(complete, valid_trace_report())["verdict"] == "PASS"
    )
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
        name: unobservable()
        if name in {"prompt", "raw_result"}
        else observation(f"same-{name}")
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
    assert (
        strict_equivalence_decision(valid_output_report(), report)["verdict"]
        == "UNKNOWN"
    )


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
    assert (
        strict_equivalence_decision(valid_output_report(), report)["verdict"] == "FAIL"
    )


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
        (
            valid_provider_attestation(),
            {**valid_lightweight_stats(), "fallback": 1},
            True,
        ),
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


def test_provider_condition_rejects_equal_dml_and_cpu_even_if_verdict_claims_pass() -> (
    None
):
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


def _write_receiptable(root: Path, name: str, value: object | None = None) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if value is None:
        path.write_text(name, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _make_complete_selection(root: Path, attempt_id: str = "a1") -> dict[str, Path]:
    r7 = root.parent
    (r7 / "results/official").mkdir(parents=True, exist_ok=True)
    (r7 / "manifest.json").write_text('{"sealed":true}\n', encoding="utf-8")
    for relative in OFFICIAL_OUTPUTS:
        (r7 / relative).write_bytes((json.dumps({"output": relative}) + "\n").encode())
    dataset = r7 / "dataset.json"
    dataset.write_text("{}\n", encoding="utf-8")
    root.mkdir()
    manifest = build_task5_manifest(
        r7_root=r7,
        receipt_path=G0_RECEIPT,
        git_commit="a" * 40,
        inputs={"dataset": dataset},
        environment={"os": "Windows"},
        contracts={"benchmark": "OmniDocBench-v1.6"},
    )
    _write_receiptable(root, "manifest.json", manifest)
    manifest_sha = task5_decision.hashlib.sha256(
        (root / "manifest.json").read_bytes()
    ).hexdigest()
    base = f"attempts/{attempt_id}"
    stage = {
        "schema": 1,
        "attempt_id": attempt_id,
        "status": "sealed",
        "producing_commit": "a" * 40,
        "manifest_sha256": manifest_sha,
        "stages": {},
        "started_at_utc": "2026-07-14T00:00:00Z",
    }
    snapshot = manifest["g0"]
    candidate = {
        "schema": 1,
        "attempt_id": attempt_id,
        "manifest_sha256": manifest_sha,
        "strict_equivalence": "PASS",
        "amd_adaptation": "PASS",
        "g0_closure": "PASS",
        "effective_only_with_valid_receipt": True,
    }
    metric_value = metric(formula=0.98, table=0.98)
    paired_scores = extract_paired_scores(metric_value, metric_value)
    strict = strict_equivalence_decision(valid_output_report(), valid_trace_report())
    amd = decide_amd(paired_scores, paired_scores)
    decision = {
        "schema": 1,
        "benchmark": "OmniDocBench-v1.6",
        "coverage": {"expected_paired_pages": 1650, "paired_pages": 1650},
        "scores": {"official": paired_scores, "lightweight": paired_scores},
        "strict_equivalence": strict,
        "amd_adaptation": amd,
        "g3": True,
        "evidence": {},
    }
    _write_receiptable(root, f"{base}/stage-state.json", stage)
    _write_receiptable(root, f"{base}/snapshot-before.json", snapshot)
    _write_receiptable(root, f"{base}/snapshot-after.json", snapshot)
    _write_receiptable(root, f"{base}/selected-attempt.json", candidate)
    compact_values = {
        f"{base}/compact/results/official/metric.json": metric_value,
        f"{base}/compact/results/official/metric-cdm.json": metric_value,
        f"{base}/compact/results/lightweight/metric.json": metric_value,
        f"{base}/compact/results/lightweight/metric-cdm.json": metric_value,
        f"{base}/compact/comparison/normalized-output.json": valid_output_report(),
        f"{base}/compact/comparison/trace-diff.json": valid_trace_report(),
        f"{base}/compact/comparison/directml-attestation.json": valid_provider_attestation(),
        f"{base}/compact/comparison/input-contract.json": valid_input_contract(),
    }
    attempt_root = root / base
    raw_stats_path = attempt_root / "work/lightweight/_run_stats.json"
    raw_stats_path.parent.mkdir(parents=True, exist_ok=True)
    raw_stats_path.write_text(
        json.dumps(
            {
                **valid_lightweight_stats(),
                "engine": "lightweight",
                "limit_pages": None,
                "stats": [],
            }
        ),
        encoding="utf-8",
    )
    metric_source = attempt_root / "work/lightweight/metric.json"
    metric_source.write_text(json.dumps(metric_value), encoding="utf-8")
    write_run_summary(
        save_name="task5-lightweight",
        run_stats_path=raw_stats_path,
        metric_result_path=metric_source,
        destination=attempt_root / "compact/results/lightweight/run-summary.json",
        cdm=False,
    )
    for name in required_attempt_receipt_paths(attempt_id):
        path = root / name
        if path.exists():
            continue
        _write_receiptable(root, name, compact_values.get(name, {}))
    evidence_paths = {
        "official_non_cdm": f"{base}/compact/results/official/metric.json",
        "official_cdm": f"{base}/compact/results/official/metric-cdm.json",
        "lightweight_non_cdm": f"{base}/compact/results/lightweight/metric.json",
        "lightweight_cdm": f"{base}/compact/results/lightweight/metric-cdm.json",
        "output_report": f"{base}/compact/comparison/normalized-output.json",
        "trace_report": f"{base}/compact/comparison/trace-diff.json",
        "provider_attestation": f"{base}/compact/comparison/directml-attestation.json",
        "lightweight_stats": f"{base}/compact/results/lightweight/run-summary.json",
    }
    decision["evidence"] = {
        name: {"sha256": sha256_file(root / relative)}
        for name, relative in evidence_paths.items()
    }
    _write_receiptable(root, f"{base}/compact/comparison/decision.json", decision)
    receipt = build_task5_receipt(root, required_attempt_receipt_paths(attempt_id))
    _write_receiptable(root, f"{base}/receipt.sha256.json", receipt)
    (root / "selected-attempt.json").write_bytes(
        (root / base / "selected-attempt.json").read_bytes()
    )
    return {
        "pointer": root / "selected-attempt.json",
        "candidate": root / base / "selected-attempt.json",
        "receipt": root / base / "receipt.sha256.json",
        "stage": root / base / "stage-state.json",
        "before": root / base / "snapshot-before.json",
        "after": root / base / "snapshot-after.json",
        "decision": root / base / "compact/comparison/decision.json",
        "manifest": root / "manifest.json",
        "metric": root / base / "compact/results/official/metric.json",
        "input_contract": root / base / "compact/comparison/input-contract.json",
        "lightweight_stats": root
        / base
        / "compact/results/lightweight/run-summary.json",
    }


def _refresh_selection_receipt(root: Path, paths: dict[str, Path]) -> None:
    receipt = build_task5_receipt(root, required_attempt_receipt_paths("a1"))
    paths["receipt"].write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )


def _refresh_lightweight_stats_evidence(root: Path, paths: dict[str, Path]) -> None:
    decision = json.loads(paths["decision"].read_text(encoding="utf-8"))
    decision["evidence"]["lightweight_stats"]["sha256"] = sha256_file(
        paths["lightweight_stats"]
    )
    paths["decision"].write_text(json.dumps(decision) + "\n", encoding="utf-8")
    _refresh_selection_receipt(root, paths)


def test_receipt_is_sorted_root_relative_and_detects_mutation(tmp_path: Path) -> None:
    tmp_path.joinpath("root").mkdir()
    root = tmp_path / "root"
    names = ["manifest.json", "attempts/a1/stage-state.json"]
    for name in names:
        _write_receiptable(root, name)
    receipt = build_task5_receipt(root, list(reversed(names)))

    assert list(receipt["files"]) == sorted(names)
    assert receipt["files"]["manifest.json"]["path"] == "manifest.json"
    validate_task5_receipt(root, receipt)

    (root / "attempts/a1/stage-state.json").write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        validate_task5_receipt(root, receipt)


def test_receipt_rejects_self_hash_escape_absolute_unallowlisted_and_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _write_receiptable(root, "manifest.json")
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="itself"):
        build_task5_receipt(root, ["receipt.sha256.json"])
    with pytest.raises(ValueError, match="relative"):
        build_task5_receipt(root, [str(outside)])
    with pytest.raises(ValueError, match="escape"):
        build_task5_receipt(root, ["../outside-secret.txt"])
    with pytest.raises(ValueError, match="allowlist"):
        build_task5_receipt(root, ["secrets.env"])
    comparison = root / "attempts/a1/compact/comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    link = comparison / "directml-attestation.json"
    try:
        link.symlink_to(comparison / "decision.json")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="symlink"):
        build_task5_receipt(
            root, ["attempts/a1/compact/comparison/directml-attestation.json"]
        )


def test_receipt_accepts_only_attempt_local_compact_authority(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    names = list(required_attempt_receipt_paths("a1"))
    for name in names:
        _write_receiptable(root, name)
    assert set(build_task5_receipt(root, names)["files"]) == set(names)

    for legacy in (
        "selected-attempt.json",
        "results/official/metric.json",
        "comparison/decision.json",
    ):
        _write_receiptable(root, legacy)
        with pytest.raises(ValueError, match="allowlist"):
            build_task5_receipt(root, [legacy])

    _write_receiptable(root, "attempts/a2/stage-state.json")
    with pytest.raises(ValueError, match="attempt"):
        build_task5_receipt(
            root,
            ["attempts/a1/stage-state.json", "attempts/a2/stage-state.json"],
        )


@pytest.mark.parametrize(
    "name",
    [
        "comparison/unknown.json",
        "lightweight/page-0001.md",
        "traces/lightweight/page-0001.jsonl",
        "results/lightweight/raw-response.json",
        "attempts/a1/work/page-0001.md",
        "attempts/a1/compact/comparison/unknown.json",
    ],
)
def test_receipt_rejects_unknown_and_raw_members(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("sensitive", encoding="utf-8")
    with pytest.raises(ValueError, match="allowlist"):
        build_task5_receipt(tmp_path, [name])


def test_validate_complete_attempt_local_selection(tmp_path: Path) -> None:
    root = tmp_path / "task5"
    _make_complete_selection(root)

    validated = validate_task5_selection(root)

    assert validated == {
        "attempt_id": "a1",
        "strict_equivalence": "PASS",
        "amd_adaptation": "PASS",
        "g0_closure": "PASS",
    }


@pytest.mark.parametrize(
    "contract",
    [
        {},
        {**valid_input_contract(), "extra": True},
        {**valid_input_contract(), "pages": True},
        {**valid_input_contract(), "paired_pages": False},
        {**valid_input_contract(), "benchmark": "OmniDocBench-v1.7"},
        {**valid_input_contract(), "pages": 1650},
        {**valid_input_contract(), "paired_pages": 1651},
        {**valid_input_contract(), "approved_exclusion": "wrong"},
        {**valid_input_contract(), "formula": "Edit_dist"},
        {**valid_input_contract(), "table": "TEDS_structure_only"},
    ],
)
def test_selection_rejects_invalid_public_input_contract(
    tmp_path: Path, contract: dict[str, object]
) -> None:
    root = tmp_path / "task5"
    paths = _make_complete_selection(root)
    paths["input_contract"].write_text(json.dumps(contract) + "\n", encoding="utf-8")
    _refresh_selection_receipt(root, paths)

    with pytest.raises(ValueError, match="contract|Contract"):
        validate_task5_selection(root)


@pytest.mark.parametrize(
    ("case", "value"),
    [
        ("missing-fallback-disabled", None),
        ("fallback-enabled", False),
        (
            "cpu-first",
            ["CPUExecutionProvider", "DmlExecutionProvider"],
        ),
        ("partial-coverage", 1650),
    ],
)
def test_selection_recomputes_amd_from_real_lightweight_run_summary(
    tmp_path: Path, case: str, value: object
) -> None:
    root = tmp_path / "task5"
    paths = _make_complete_selection(root)
    summary = json.loads(paths["lightweight_stats"].read_text(encoding="utf-8"))
    if case == "missing-fallback-disabled":
        summary.pop("layout_fallback_disabled")
    elif case == "fallback-enabled":
        summary["layout_fallback_disabled"] = value
    elif case == "cpu-first":
        summary["layout_providers_active"] = value
    else:
        summary["prediction_count"] = value
    paths["lightweight_stats"].write_text(
        json.dumps(summary) + "\n", encoding="utf-8"
    )
    _refresh_lightweight_stats_evidence(root, paths)

    with pytest.raises(ValueError, match="AMD adaptation"):
        validate_task5_selection(root)


def test_selection_rejects_lightweight_run_summary_digest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "task5"
    paths = _make_complete_selection(root)
    decision = json.loads(paths["decision"].read_text(encoding="utf-8"))
    decision["evidence"]["lightweight_stats"]["sha256"] = "0" * 64
    paths["decision"].write_text(json.dumps(decision) + "\n", encoding="utf-8")
    _refresh_selection_receipt(root, paths)

    with pytest.raises(ValueError, match="evidence"):
        validate_task5_selection(root)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("pointer-bytes", "byte"),
        ("bad-attempt", "AttemptId"),
        ("missing-receipt", "receipt"),
        ("omitted-receipt-path", "exact"),
        ("extra-receipt-path", "exact"),
        ("manifest-digest", "manifest"),
        ("g0-mismatch", "G0"),
        ("decision-verdict", "decision"),
        ("stage-attempt", "stage"),
        ("stage-active", "sealed"),
        ("absolute-disclosure", "absolute"),
    ],
)
def test_selection_validation_fails_closed_on_mutation(
    tmp_path: Path, case: str, match: str
) -> None:
    root = tmp_path / "task5"
    paths = _make_complete_selection(root)
    pointer = json.loads(paths["pointer"].read_text(encoding="utf-8"))

    if case == "pointer-bytes":
        paths["pointer"].write_text(
            json.dumps(pointer, separators=(",", ":")), encoding="utf-8"
        )
    elif case == "bad-attempt":
        pointer["attempt_id"] = "Bad/Attempt"
        paths["pointer"].write_text(json.dumps(pointer), encoding="utf-8")
    elif case == "missing-receipt":
        paths["receipt"].unlink()
    elif case in {"omitted-receipt-path", "extra-receipt-path"}:
        receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
        if case == "omitted-receipt-path":
            receipt["files"].pop(next(iter(receipt["files"])))
        else:
            receipt["files"]["attempts/a1/work/raw.json"] = {}
        paths["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    elif case == "manifest-digest":
        pointer["manifest_sha256"] = "0" * 64
        paths["candidate"].write_text(json.dumps(pointer), encoding="utf-8")
        paths["pointer"].write_text(json.dumps(pointer), encoding="utf-8")
    elif case == "g0-mismatch":
        paths["after"].write_text('{"different":true}\n', encoding="utf-8")
    elif case == "decision-verdict":
        decision = json.loads(paths["decision"].read_text(encoding="utf-8"))
        decision["strict_equivalence"]["verdict"] = "FAIL"
        paths["decision"].write_text(json.dumps(decision), encoding="utf-8")
    elif case in {"stage-attempt", "stage-active"}:
        stage = json.loads(paths["stage"].read_text(encoding="utf-8"))
        if case == "stage-attempt":
            stage["attempt_id"] = "a2"
        else:
            stage["status"] = "active"
        paths["stage"].write_text(json.dumps(stage), encoding="utf-8")
    else:
        pointer["extra"] = str(tmp_path.resolve())
        paths["candidate"].write_text(json.dumps(pointer), encoding="utf-8")
        paths["pointer"].write_text(json.dumps(pointer), encoding="utf-8")

    if case in {
        "manifest-digest",
        "g0-mismatch",
        "decision-verdict",
        "stage-attempt",
        "stage-active",
        "absolute-disclosure",
    }:
        receipt = build_task5_receipt(root, required_attempt_receipt_paths("a1"))
        paths["receipt"].write_text(
            json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
        )

    with pytest.raises((OSError, ValueError), match=match):
        validate_task5_selection(root)


def test_validate_selection_rejects_symlinked_pointer_or_attempt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "task5"
    paths = _make_complete_selection(root)
    saved_pointer = paths["pointer"].read_bytes()
    paths["pointer"].unlink()
    try:
        paths["pointer"].symlink_to(paths["candidate"])
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="symlink"):
        validate_task5_selection(root)

    paths["pointer"].unlink()
    paths["pointer"].write_bytes(saved_pointer)
    attempt = root / "attempts/a1"
    moved = root / "attempts/real-a1"
    attempt.rename(moved)
    attempt.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validate_task5_selection(root)


@pytest.mark.parametrize(
    "target_name", ["manifest", "stage", "before", "after", "decision", "metric"]
)
def test_selection_final_receipt_rehash_detects_post_semantic_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    root = tmp_path / "task5"
    paths = _make_complete_selection(root)
    original = task5_decision._read_stable_file
    pointer_reads = 0

    def mutate_after_semantic_read(path: Path, *, label: str):
        nonlocal pointer_reads
        if path.resolve() == paths["pointer"].resolve():
            pointer_reads += 1
            if pointer_reads == 2:
                target = paths[target_name]
                target.write_bytes(target.read_bytes() + b" ")
        return original(path, label=label)

    monkeypatch.setattr(task5_decision, "_read_stable_file", mutate_after_semantic_read)
    with pytest.raises(ValueError, match="changed|receipt|identity"):
        validate_task5_selection(root)


@pytest.mark.parametrize("target_name", ["pointer", "candidate", "receipt"])
def test_selection_rechecks_authority_bytes_after_final_exact22_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    root = tmp_path / "task5"
    paths = _make_complete_selection(root)
    original = task5_decision.validate_task5_receipt
    calls = 0

    def mutate_during_final_rehash(task5_root: Path, receipt: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            target = paths[target_name]
            target.write_bytes(target.read_bytes() + b" ")
        return original(task5_root, receipt)

    monkeypatch.setattr(
        task5_decision, "validate_task5_receipt", mutate_during_final_rehash
    )
    with pytest.raises(ValueError, match="changed|identity"):
        validate_task5_selection(root)


@pytest.mark.parametrize(
    "snapshot", [{}, {"receipt": {}, "official_outputs": {}}, {"extra": True}]
)
def test_selection_rejects_equal_snapshots_that_do_not_match_manifest_g0(
    tmp_path: Path, snapshot: dict[str, object]
) -> None:
    root = tmp_path / "task5"
    paths = _make_complete_selection(root)
    for name in ("before", "after"):
        paths[name].write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
    receipt = build_task5_receipt(root, required_attempt_receipt_paths("a1"))
    paths["receipt"].write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="G0|manifest"):
        validate_task5_selection(root)


@pytest.mark.parametrize(
    "case", ["empty-scores", "pass-g3-false", "wrong-evidence", "wrong-overall"]
)
def test_selection_independently_recomputes_compact_decision(
    tmp_path: Path, case: str
) -> None:
    root = tmp_path / "task5"
    paths = _make_complete_selection(root)
    decision = json.loads(paths["decision"].read_text(encoding="utf-8"))
    if case == "empty-scores":
        decision["scores"] = {"official": {}, "lightweight": {}}
    elif case == "pass-g3-false":
        decision["g3"] = False
        decision["amd_adaptation"]["g3"] = False
    elif case == "wrong-evidence":
        decision["evidence"]["official_non_cdm"]["sha256"] = "0" * 64
    else:
        decision["scores"]["lightweight"]["overall"] = 0.0
    paths["decision"].write_text(json.dumps(decision) + "\n", encoding="utf-8")
    receipt = build_task5_receipt(root, required_attempt_receipt_paths("a1"))
    paths["receipt"].write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="decision|evidence|score|g3"):
        validate_task5_selection(root)


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


def test_cli_decide_writes_fail_decision_without_infrastructure_error(
    tmp_path: Path,
) -> None:
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


def test_cli_rejects_decision_and_receipt_output_input_collisions(
    tmp_path: Path,
) -> None:
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


def test_cli_rejects_duplicate_or_nonfinite_json_without_traceback(
    tmp_path: Path,
) -> None:
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
