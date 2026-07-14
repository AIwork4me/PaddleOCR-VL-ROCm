from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.task5_comparison import (
    BOUNDARIES,
    compare_boundary_documents,
    compare_canonical_traces,
    compare_prediction_dirs,
    normalize_scorer_markdown,
    observation,
    unobservable,
)

APPROVED_STEM = "approved-failed-page"


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
) -> dict[str, object]:
    boundaries = {name: observation(name) for name in BOUNDARIES}
    if raw_result is not None:
        boundaries["raw_result"] = raw_result
    if postprocess is not None:
        boundaries["postprocess"] = postprocess
    return {"page": page, "block_index": block_index, "boundaries": boundaries}


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

    report = compare_boundary_documents(
        official=[first, second], lightweight=[second, first]
    )

    assert report["verdict"] == "PASS"


def test_missing_event_is_a_proven_structural_difference() -> None:
    report = compare_boundary_documents(
        official=[_event(page="a"), _event(page="b")],
        lightweight=[_event(page="a")],
    )

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
        official=[page_record], lightweight=[_event(page="a")]
    )

    assert report["verdict"] == "UNKNOWN"


def test_trace_schema_rejects_null_block_index_without_unobservable_structure() -> None:
    invalid = {
        "page": "a",
        "block_index": None,
        "block_structure": observation(1),
        "boundaries": {name: unobservable() for name in BOUNDARIES},
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
