from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.task5_comparison import (
    APPROVED_EXCLUDED_STEM,
    BOUNDARIES,
    compare_boundary_documents,
    compare_canonical_traces,
    compare_prediction_dirs,
    normalize_scorer_markdown,
    observation,
    unobservable,
)

APPROVED_STEM = "newspaper_The Times UK_0801@magazinesclubnew_page_031"


def _write_prediction_pairs(tmp_path: Path, count: int) -> tuple[Path, Path]:
    official = tmp_path / "official"
    lightweight = tmp_path / "lightweight"
    official.mkdir()
    lightweight.mkdir()
    for index in range(count):
        name = f"page-{index:04d}.md"
        (official / name).write_bytes(b"same\r\n")
        (lightweight / name).write_bytes(b"same\n")
    return official, lightweight


def _event(
    *,
    page: str = "page-0000",
    block_index: int = 0,
    raw_result: dict[str, str] | None = None,
    postprocess: dict[str, str] | None = None,
    page_postprocess: dict[str, str] | None = None,
) -> dict[str, object]:
    boundaries = {name: observation(name) for name in BOUNDARIES}
    if raw_result is not None:
        boundaries["raw_result"] = raw_result
    if postprocess is not None:
        boundaries["postprocess"] = postprocess
    return {
        "page": page,
        "block_index": block_index,
        "boundaries": boundaries,
        "page_postprocess": page_postprocess or observation("page markdown"),
    }


def test_normalization_ignores_only_transport_newlines() -> None:
    assert normalize_scorer_markdown("a  \r\nformula\r\n\r\n") == "a  \nformula"
    assert normalize_scorer_markdown("a \n") != normalize_scorer_markdown("a\n")


def test_output_comparison_requires_exactly_1650_pairs(tmp_path: Path) -> None:
    official, lightweight = _write_prediction_pairs(tmp_path, count=1650)

    report = compare_prediction_dirs(official, lightweight, APPROVED_STEM)

    assert report["verdict"] == "PASS"
    assert report["paired_pages"] == 1650
    assert report["equal_pages"] == 1650
    assert report["different_pages"] == 0
    assert "same" not in json.dumps(report)


def test_output_comparison_fails_a_non_1650_denominator(tmp_path: Path) -> None:
    official, lightweight = _write_prediction_pairs(tmp_path, count=2)

    report = compare_prediction_dirs(official, lightweight, APPROVED_STEM)

    assert report["verdict"] == "FAIL"
    assert report["paired_pages"] == 2


def test_output_comparison_rejects_an_arbitrary_excluded_stem(tmp_path: Path) -> None:
    official, lightweight = _write_prediction_pairs(tmp_path, count=1650)
    (official / "page-0000.md").write_text("different", encoding="utf-8")

    with pytest.raises(ValueError, match=APPROVED_EXCLUDED_STEM):
        compare_prediction_dirs(official, lightweight, "page-0000")


def test_output_report_records_fixed_exclusion_presence_on_each_side(tmp_path: Path) -> None:
    official, lightweight = _write_prediction_pairs(tmp_path, count=1650)
    (lightweight / f"{APPROVED_STEM}.md").write_text("extra", encoding="utf-8")

    report = compare_prediction_dirs(official, lightweight, APPROVED_STEM)

    assert report["approved_exclusion"] == {
        "stem": APPROVED_STEM,
        "official_present": False,
        "lightweight_present": True,
    }


def test_prediction_evidence_hash_binds_fixed_exclusion_presence(tmp_path: Path) -> None:
    official, lightweight = _write_prediction_pairs(tmp_path, count=1650)
    absent = compare_prediction_dirs(official, lightweight, APPROVED_STEM)
    (lightweight / f"{APPROVED_STEM}.md").write_text("excluded", encoding="utf-8")
    present = compare_prediction_dirs(official, lightweight, APPROVED_STEM)

    assert absent["verdict"] == present["verdict"] == "PASS"
    assert absent["evidence_fingerprint"] != present["evidence_fingerprint"]


def test_observation_records_only_status_and_redacted_fingerprint() -> None:
    record = observation({"prompt": "private", "token": "secret"})

    assert set(record) == {"status", "fingerprint"}
    assert record["status"] == "observable"
    assert "private" not in json.dumps(record)
    assert unobservable() == {"status": "unobservable"}


def test_proven_difference_beats_unobservable() -> None:
    report = compare_boundary_documents(
        official=[_event(raw_result=unobservable(), postprocess=observation("a"))],
        lightweight=[_event(raw_result=observation("x"), postprocess=observation("b"))],
    )

    assert report["verdict"] == "FAIL"
    assert report["first_divergence_counts"]["postprocess"] == 1


def test_only_unobservable_boundaries_yield_unknown() -> None:
    report = compare_boundary_documents(
        official=[_event(raw_result=unobservable())],
        lightweight=[_event(raw_result=observation("x"))],
    )

    assert report["verdict"] == "UNKNOWN"


def test_events_pair_by_page_and_block_index_not_position() -> None:
    first = _event(page="a", block_index=0)
    second = _event(page="b", block_index=1)

    report = compare_boundary_documents(official=[first, second], lightweight=[second, first])

    assert report["verdict"] == "PASS"


def test_missing_event_is_a_proven_structural_difference() -> None:
    report = compare_boundary_documents(
        official=[_event(page="a"), _event(page="b")],
        lightweight=[_event(page="a")],
    )

    assert report["verdict"] == "FAIL"
    assert report["first_divergence_counts"]["event_structure"] == 1


def test_zero_evidence_never_passes() -> None:
    report = compare_boundary_documents(official=[], lightweight=[])

    assert report["verdict"] == "FAIL"
    assert report["first_divergence_counts"]["event_structure"] == 1


def test_page_level_unobservable_official_trace_yields_unknown() -> None:
    page_record = {
        "page": "a",
        "block_index": None,
        "block_structure": unobservable(),
        "boundaries": {name: unobservable() for name in BOUNDARIES},
        "page_postprocess": observation("same markdown"),
    }

    report = compare_boundary_documents(
        official=[page_record],
        lightweight=[_event(page="a", page_postprocess=observation("same markdown"))],
    )

    assert report["verdict"] == "UNKNOWN"


def test_fully_observable_page_postprocess_difference_is_fail() -> None:
    report = compare_boundary_documents(
        official=[_event(page_postprocess=observation("official markdown"))],
        lightweight=[_event(page_postprocess=observation("lightweight markdown"))],
    )

    assert report["verdict"] == "FAIL"
    assert report["first_divergence_counts"]["page_postprocess"] == 1


def test_page_level_unknown_cannot_mask_proven_page_postprocess_difference() -> None:
    page_record = {
        "page": "a",
        "block_index": None,
        "block_structure": unobservable(),
        "boundaries": {name: unobservable() for name in BOUNDARIES},
        "page_postprocess": observation("official markdown"),
    }
    lightweight = _event(page="a", page_postprocess=observation("lightweight markdown"))

    report = compare_boundary_documents([page_record], [lightweight])

    assert report["verdict"] == "FAIL"
    assert report["unobservable_records"] == 1
    assert report["first_divergence_counts"]["page_postprocess"] == 1


def test_page_postprocess_is_required_and_consistent_within_each_page() -> None:
    missing = _event()
    missing.pop("page_postprocess")
    with pytest.raises(ValueError, match="page_postprocess"):
        compare_boundary_documents([missing], [_event()])

    with pytest.raises(ValueError, match="consistent"):
        compare_boundary_documents(
            [
                _event(block_index=0, page_postprocess=observation("first")),
                _event(block_index=1, page_postprocess=observation("second")),
            ],
            [_event()],
        )


def test_page_level_record_missing_from_other_side_is_structural_fail() -> None:
    page_record = {
        "page": "a",
        "block_index": None,
        "block_structure": unobservable(),
        "boundaries": {name: unobservable() for name in BOUNDARIES},
        "page_postprocess": observation("markdown"),
    }

    report = compare_boundary_documents(official=[page_record], lightweight=[])

    assert report["verdict"] == "FAIL"
    assert report["first_divergence_counts"]["event_structure"] == 1


def test_page_level_and_block_records_cannot_mix_for_one_page() -> None:
    page_record = {
        "page": "a",
        "block_index": None,
        "block_structure": unobservable(),
        "boundaries": {name: unobservable() for name in BOUNDARIES},
        "page_postprocess": observation("page markdown"),
    }

    with pytest.raises(ValueError, match="mix"):
        compare_boundary_documents(
            official=[page_record, _event(page="a")], lightweight=[_event(page="a")]
        )


def test_trace_schema_rejects_null_block_index_without_unobservable_structure() -> None:
    invalid = {
        "page": "a",
        "block_index": None,
        "block_structure": observation(1),
        "boundaries": {name: unobservable() for name in BOUNDARIES},
        "page_postprocess": observation("page markdown"),
    }

    with pytest.raises(ValueError, match="block_structure"):
        compare_boundary_documents(official=[invalid], lightweight=[])


def test_canonical_trace_directory_comparison_counts_all_but_bounds_details(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official-traces"
    lightweight = tmp_path / "lightweight-traces"
    official.mkdir()
    lightweight.mkdir()
    for index in range(105):
        page = f"page-{index:04d}"
        reference = _event(page=page, postprocess=observation("a"))
        candidate = _event(page=page, postprocess=observation("b"))
        (official / f"{page}.jsonl").write_text(
            json.dumps(reference, sort_keys=True) + "\n", encoding="utf-8"
        )
        (lightweight / f"{page}.jsonl").write_text(
            json.dumps(candidate, sort_keys=True) + "\n", encoding="utf-8"
        )

    report = compare_canonical_traces(official, lightweight)

    assert report["verdict"] == "FAIL"
    assert report["different_records"] == 105
    assert len(report["details"]) == 100
    assert report["details_truncated"] is True
    rendered = json.dumps(report)
    assert "same markdown" not in rendered
    assert '"a"' not in rendered


def test_canonical_trace_directories_require_exactly_1650_nonempty_paired_pages(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official"
    lightweight = tmp_path / "lightweight"
    official.mkdir()
    lightweight.mkdir()
    for index in range(1649):
        page = f"page-{index:04d}"
        rendered = json.dumps(_event(page=page), sort_keys=True) + "\n"
        (official / f"{page}.jsonl").write_text(rendered, encoding="utf-8")
        (lightweight / f"{page}.jsonl").write_text(rendered, encoding="utf-8")

    report = compare_canonical_traces(official, lightweight)

    assert report["verdict"] == "FAIL"
    assert report["paired_pages"] == 1649
    assert report["expected_paired_pages"] == 1650

    final_page = "page-1649"
    rendered = json.dumps(_event(page=final_page), sort_keys=True) + "\n"
    (official / f"{final_page}.jsonl").write_text(rendered, encoding="utf-8")
    (lightweight / f"{final_page}.jsonl").write_text(rendered, encoding="utf-8")

    complete_report = compare_canonical_traces(official, lightweight)

    assert complete_report["verdict"] == "PASS"
    assert complete_report["paired_pages"] == 1650

    approved_event = json.dumps(_event(page=APPROVED_STEM), sort_keys=True) + "\n"
    (lightweight / f"{APPROVED_STEM}.jsonl").write_text(approved_event, encoding="utf-8")
    lightweight_only_exclusion = compare_canonical_traces(official, lightweight)

    assert lightweight_only_exclusion["verdict"] == "PASS"
    assert lightweight_only_exclusion["approved_exclusion"] == {
        "stem": APPROVED_STEM,
        "official_present": False,
        "lightweight_present": True,
    }

    (official / f"{APPROVED_STEM}.jsonl").write_text(approved_event, encoding="utf-8")
    both_present = compare_canonical_traces(official, lightweight)

    assert both_present["verdict"] == "PASS"
    assert both_present["approved_exclusion"]["official_present"] is True
    assert both_present["approved_exclusion"]["lightweight_present"] is True

    orphan = "unexpected-orphan"
    (lightweight / f"{orphan}.jsonl").write_text(
        json.dumps(_event(page=orphan), sort_keys=True) + "\n", encoding="utf-8"
    )
    orphan_report = compare_canonical_traces(official, lightweight)

    assert orphan_report["verdict"] == "FAIL"
    assert orphan_report["lightweight_only_pages"] == 1


def test_empty_trace_files_are_zero_evidence_failures(tmp_path: Path) -> None:
    official = tmp_path / "official"
    lightweight = tmp_path / "lightweight"
    official.mkdir()
    lightweight.mkdir()
    for directory in (official, lightweight):
        (directory / "page.jsonl").write_text("", encoding="utf-8")

    report = compare_canonical_traces(official, lightweight)

    assert report["verdict"] == "FAIL"
    assert report["empty_page_traces"] == 2


def test_evidence_hash_covers_later_boundaries_not_only_first_divergence() -> None:
    reference = _event(postprocess=observation("reference-post"))
    first_candidate = _event(
        raw_result=observation("different-raw"), postprocess=observation("candidate-one")
    )
    second_candidate = _event(
        raw_result=observation("different-raw"), postprocess=observation("candidate-two")
    )

    first = compare_boundary_documents([reference], [first_candidate])
    second = compare_boundary_documents([reference], [second_candidate])

    assert first["first_divergence_counts"] == second["first_divergence_counts"]
    assert first["evidence_fingerprint"] != second["evidence_fingerprint"]


def test_evidence_hash_covers_page_postprocess_fingerprint() -> None:
    first = compare_boundary_documents(
        [_event(page_postprocess=observation("same"))],
        [_event(page_postprocess=observation("candidate-one"))],
    )
    second = compare_boundary_documents(
        [_event(page_postprocess=observation("same"))],
        [_event(page_postprocess=observation("candidate-two"))],
    )

    assert first["first_divergence_counts"] == second["first_divergence_counts"]
    assert first["evidence_fingerprint"] != second["evidence_fingerprint"]


def test_normalize_strips_br_separators():
    from eval.task5_comparison import normalize_scorer_markdown

    text = "block1\n<br>\nblock2"
    assert normalize_scorer_markdown(text) == "block1\n\nblock2"


def test_normalize_does_not_touch_inline_br():
    from eval.task5_comparison import normalize_scorer_markdown

    assert normalize_scorer_markdown("line1<br>line2") == "line1<br>line2"
