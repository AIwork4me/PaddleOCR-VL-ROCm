"""Bounded, fingerprint-only comparison evidence for Task 5."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from paddleocr_vl_rocm.contracts import fingerprint, redact

BOUNDARIES = (
    "request_order",
    "label",
    "bbox",
    "crop_pixels",
    "prompt",
    "payload",
    "raw_result",
    "postprocess",
)
EXPECTED_PAIRED_PAGES = 1650
DETAIL_LIMIT = 100
APPROVED_EXCLUDED_STEM = "newspaper_The Times UK_0801@magazinesclubnew_page_031"


def observation(value: object) -> dict[str, str]:
    """Represent a value without retaining the value itself."""
    return {"status": "observable", "fingerprint": fingerprint(redact(value))}


def unobservable() -> dict[str, str]:
    """Represent a boundary that the authenticated source did not expose."""
    return {"status": "unobservable"}


def normalize_scorer_markdown(text: str) -> str:
    """Normalize transport newlines while preserving Markdown-significant bytes."""
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def boundary_relation(reference: Mapping[str, str], candidate: Mapping[str, str]) -> str:
    _validate_observation(reference)
    _validate_observation(candidate)
    if reference["status"] == "unobservable" or candidate["status"] == "unobservable":
        return "unobservable"
    return "equal" if reference["fingerprint"] == candidate["fingerprint"] else "different"


def compare_prediction_dirs(
    official_dir: Path, lightweight_dir: Path, approved_excluded_stem: str
) -> dict[str, object]:
    """Compare the exact 1,650 scorer-facing Markdown pairs by filename stem."""
    if approved_excluded_stem != APPROVED_EXCLUDED_STEM:
        raise ValueError(f"approved_excluded_stem must be {APPROVED_EXCLUDED_STEM!r}")
    official_exclusion_present = (official_dir / f"{APPROVED_EXCLUDED_STEM}.md").is_file()
    lightweight_exclusion_present = (
        lightweight_dir / f"{APPROVED_EXCLUDED_STEM}.md"
    ).is_file()
    official = _markdown_files(official_dir, approved_excluded_stem)
    lightweight = _markdown_files(lightweight_dir, approved_excluded_stem)
    common = sorted(official.keys() & lightweight.keys())
    official_only = sorted(official.keys() - lightweight.keys())
    lightweight_only = sorted(lightweight.keys() - official.keys())
    equal = 0
    different = 0
    details: list[dict[str, str]] = []
    evidence: list[dict[str, str]] = []
    for stem in common:
        reference = normalize_scorer_markdown(official[stem].read_text(encoding="utf-8"))
        candidate = normalize_scorer_markdown(lightweight[stem].read_text(encoding="utf-8"))
        reference_fingerprint = fingerprint(reference)
        candidate_fingerprint = fingerprint(candidate)
        relation = "equal" if reference_fingerprint == candidate_fingerprint else "different"
        row = {
            "stem": stem,
            "relation": relation,
            "official_fingerprint": reference_fingerprint,
            "lightweight_fingerprint": candidate_fingerprint,
        }
        evidence.append(row)
        if relation == "equal":
            equal += 1
        else:
            different += 1
            if len(details) < DETAIL_LIMIT:
                details.append(row)
    for side, stems in (("official_only", official_only), ("lightweight_only", lightweight_only)):
        for stem in stems:
            row = {"stem": stem, "relation": side}
            evidence.append(row)
            if len(details) < DETAIL_LIMIT:
                details.append(row)
    structural_differences = len(official_only) + len(lightweight_only)
    verdict = (
        "PASS"
        if len(common) == EXPECTED_PAIRED_PAGES
        and different == 0
        and structural_differences == 0
        else "FAIL"
    )
    total_details = different + structural_differences
    return {
        "verdict": verdict,
        "expected_paired_pages": EXPECTED_PAIRED_PAGES,
        "paired_pages": len(common),
        "equal_pages": equal,
        "different_pages": different,
        "official_only_pages": len(official_only),
        "lightweight_only_pages": len(lightweight_only),
        "approved_exclusion": {
            "stem": APPROVED_EXCLUDED_STEM,
            "official_present": official_exclusion_present,
            "lightweight_present": lightweight_exclusion_present,
        },
        "evidence_fingerprint": fingerprint(
            {
                "approved_exclusion": {
                    "stem": APPROVED_EXCLUDED_STEM,
                    "official_present": official_exclusion_present,
                    "lightweight_present": lightweight_exclusion_present,
                },
                "pages": evidence,
            }
        ),
        "details": details,
        "details_truncated": total_details > len(details),
    }


def compare_boundary_documents(
    official: list[dict[str, object]], lightweight: list[dict[str, object]]
) -> dict[str, object]:
    """Compare validated canonical events with FAIL > UNKNOWN > PASS precedence."""
    official_events, official_page_records = _index_events(official, "official")
    lightweight_events, lightweight_page_records = _index_events(lightweight, "lightweight")
    official_page_postprocess = _page_postprocesses(official, "official")
    lightweight_page_postprocess = _page_postprocesses(lightweight, "lightweight")
    official_pages = {key[0] for key in official_events} | set(official_page_records)
    lightweight_pages = {key[0] for key in lightweight_events} | set(lightweight_page_records)
    missing_pages = official_pages ^ lightweight_pages
    shared_pages = official_pages & lightweight_pages
    unknown_pages = {
        page
        for page in shared_pages
        if page in official_page_records or page in lightweight_page_records
    }
    all_official_events = official_events
    all_lightweight_events = lightweight_events
    official_events = {
        key: value
        for key, value in official_events.items()
        if key[0] not in unknown_pages and key[0] not in missing_pages
    }
    lightweight_events = {
        key: value
        for key, value in lightweight_events.items()
        if key[0] not in unknown_pages and key[0] not in missing_pages
    }

    counts: Counter[str] = Counter()
    unobservable_counts: Counter[str] = Counter()
    details: list[dict[str, object]] = []
    all_evidence: list[dict[str, object]] = []
    different_records = len(missing_pages)
    unobservable_records = len(unknown_pages)

    def record(row: dict[str, object], *, detail: bool = True) -> None:
        all_evidence.append(row)
        if detail and len(details) < DETAIL_LIMIT:
            details.append(row)

    if not official_pages and not lightweight_pages:
        different_records = 1
        counts["event_structure"] += 1
        record({"relation": "different", "boundary": "event_structure", "reason": "zero_evidence"})

    for page in sorted(shared_pages):
        reference_page = official_page_postprocess[page]
        candidate_page = lightweight_page_postprocess[page]
        relation = boundary_relation(reference_page, candidate_page)
        record(
            {
                "page": page,
                "relation": relation,
                "boundary": "page_postprocess",
                "official": dict(reference_page),
                "lightweight": dict(candidate_page),
            },
            detail=False,
        )
        if relation == "different":
            different_records += 1
            counts["page_postprocess"] += 1
            if len(details) < DETAIL_LIMIT:
                details.append(
                    {
                        "page": page,
                        "relation": "different",
                        "boundary": "page_postprocess",
                    }
                )

    for page in sorted(missing_pages):
        counts["event_structure"] += 1
        source_events = _page_evidence(
            page,
            all_official_events if page in official_pages else all_lightweight_events,
            official_page_records if page in official_pages else lightweight_page_records,
        )
        record(
            {
                "page": page,
                "relation": "different",
                "boundary": "event_structure",
                "missing_from": "lightweight" if page in official_pages else "official",
                "available_evidence": source_events,
            }
        )

    for page in sorted(unknown_pages):
        unobservable_counts["block_structure"] += 1
        record(
            {
                "page": page,
                "relation": "unobservable",
                "boundary": "block_structure",
                "official_evidence": _page_evidence(
                    page, all_official_events, official_page_records
                ),
                "lightweight_evidence": _page_evidence(
                    page, all_lightweight_events, lightweight_page_records
                ),
            }
        )

    all_keys = sorted(set(official_events) | set(lightweight_events))
    for key in all_keys:
        reference = official_events.get(key)
        candidate = lightweight_events.get(key)
        page, block_index = key
        if reference is None or candidate is None:
            different_records += 1
            counts["event_structure"] += 1
            record(
                {
                    "page": page,
                    "block_index": block_index,
                    "relation": "different",
                    "boundary": "event_structure",
                    "missing_from": "official" if reference is None else "lightweight",
                    "available_evidence": _safe_event(candidate or reference),
                }
            )
            continue

        first_difference: str | None = None
        event_unobservable = False
        relations: list[dict[str, str]] = []
        reference_boundaries = reference["boundaries"]
        candidate_boundaries = candidate["boundaries"]
        assert isinstance(reference_boundaries, Mapping)
        assert isinstance(candidate_boundaries, Mapping)
        for boundary in BOUNDARIES:
            relation = boundary_relation(
                reference_boundaries[boundary], candidate_boundaries[boundary]
            )
            relations.append({"boundary": boundary, "relation": relation})
            if relation == "different" and first_difference is None:
                first_difference = boundary
            elif relation == "unobservable":
                unobservable_counts[boundary] += 1
                event_unobservable = True
        record(
            {
                "page": page,
                "block_index": block_index,
                "relation": (
                    "different"
                    if first_difference is not None
                    else "unobservable"
                    if event_unobservable
                    else "equal"
                ),
                "boundaries": {
                    item["boundary"]: {
                        "relation": item["relation"],
                        "official": dict(reference_boundaries[item["boundary"]]),
                        "lightweight": dict(candidate_boundaries[item["boundary"]]),
                    }
                    for item in relations
                },
            },
            detail=False,
        )
        if first_difference is not None:
            different_records += 1
            counts[first_difference] += 1
            if len(details) < DETAIL_LIMIT:
                details.append(
                {
                    "page": page,
                    "block_index": block_index,
                    "relation": "different",
                    "boundary": first_difference,
                })
        elif event_unobservable:
            unobservable_records += 1
            if len(details) < DETAIL_LIMIT:
                details.append(
                {
                    "page": page,
                    "block_index": block_index,
                    "relation": "unobservable",
                    "boundaries": [
                        item["boundary"] for item in relations if item["relation"] == "unobservable"
                    ],
                })

    verdict = "FAIL" if different_records else "UNKNOWN" if unobservable_records else "PASS"
    ordered_counts = {"event_structure": counts["event_structure"]}
    ordered_counts.update({name: counts[name] for name in BOUNDARIES})
    ordered_counts["page_postprocess"] = counts["page_postprocess"]
    return {
        "verdict": verdict,
        "official_records": len(official),
        "lightweight_records": len(lightweight),
        "different_records": different_records,
        "unobservable_records": unobservable_records,
        "first_divergence_counts": ordered_counts,
        "unobservable_counts": {
            "block_structure": unobservable_counts["block_structure"],
            **{name: unobservable_counts[name] for name in BOUNDARIES},
        },
        "evidence_fingerprint": fingerprint(all_evidence),
        "details": details,
        "details_truncated": different_records + unobservable_records > len(details),
    }


def compare_canonical_traces(official_dir: Path, lightweight_dir: Path) -> dict[str, object]:
    official_files = _trace_files(official_dir)
    lightweight_files = _trace_files(lightweight_dir)
    official_exclusion_present = APPROVED_EXCLUDED_STEM in official_files
    lightweight_exclusion_present = APPROVED_EXCLUDED_STEM in lightweight_files
    official_files.pop(APPROVED_EXCLUDED_STEM, None)
    lightweight_files.pop(APPROVED_EXCLUDED_STEM, None)
    official_pages = set(official_files)
    lightweight_pages = set(lightweight_files)
    common_pages = official_pages & lightweight_pages
    official: list[dict[str, object]] = []
    lightweight: list[dict[str, object]] = []
    official_empty_pages: list[str] = []
    lightweight_empty_pages: list[str] = []
    for page, path in official_files.items():
        events = _read_trace_file(path)
        if not events:
            official_empty_pages.append(page)
        _validate_trace_page(events, page, path)
        official.extend(events)
    for page, path in lightweight_files.items():
        events = _read_trace_file(path)
        if not events:
            lightweight_empty_pages.append(page)
        _validate_trace_page(events, page, path)
        lightweight.extend(events)
    report = compare_boundary_documents(official, lightweight)
    empty_page_traces = len(official_empty_pages) + len(lightweight_empty_pages)
    coverage = {
        "expected_paired_pages": EXPECTED_PAIRED_PAGES,
        "paired_pages": len(common_pages),
        "official_only_pages": len(official_pages - lightweight_pages),
        "lightweight_only_pages": len(lightweight_pages - official_pages),
        "empty_page_traces": empty_page_traces,
    }
    coverage_fail = (
        len(common_pages) != EXPECTED_PAIRED_PAGES
        or official_pages != lightweight_pages
        or empty_page_traces > 0
    )
    if coverage_fail:
        report["verdict"] = "FAIL"
    report["evidence_fingerprint"] = fingerprint(
        {
            "approved_exclusion": {
                "stem": APPROVED_EXCLUDED_STEM,
                "official_present": official_exclusion_present,
                "lightweight_present": lightweight_exclusion_present,
            },
            "coverage": {
                "official_pages": sorted(official_pages),
                "lightweight_pages": sorted(lightweight_pages),
                "official_empty_pages": official_empty_pages,
                "lightweight_empty_pages": lightweight_empty_pages,
            },
            "boundary_evidence": report["evidence_fingerprint"],
        }
    )
    report.update(coverage)
    report["approved_exclusion"] = {
        "stem": APPROVED_EXCLUDED_STEM,
        "official_present": official_exclusion_present,
        "lightweight_present": lightweight_exclusion_present,
    }
    return report


def _markdown_files(directory: Path, excluded_stem: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"Prediction directory not found: {directory}")
    files = {path.stem: path for path in directory.glob("*.md") if path.stem != excluded_stem}
    if len(files) != sum(1 for path in directory.glob("*.md") if path.stem != excluded_stem):
        raise ValueError(f"Duplicate prediction stem in {directory}")
    return files


def _validate_observation(value: Mapping[str, str]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("Boundary observation must be an object")
    status = value.get("status")
    if status == "observable":
        if set(value) != {"status", "fingerprint"} or not isinstance(
            value.get("fingerprint"), str
        ):
            raise ValueError("Observable boundary requires only status and fingerprint")
    elif status == "unobservable":
        if set(value) != {"status"}:
            raise ValueError("Unobservable boundary requires only status")
    else:
        raise ValueError("Boundary status must be observable or unobservable")


def _validate_event(event: dict[str, object]) -> tuple[str, int | None]:
    page = event.get("page")
    block_index = event.get("block_index")
    boundaries = event.get("boundaries")
    if not isinstance(page, str) or not page:
        raise ValueError("Trace event requires a non-empty page")
    if block_index is not None and (not isinstance(block_index, int) or block_index < 0):
        raise ValueError("Trace event block_index must be a nonnegative integer")
    if not isinstance(boundaries, Mapping) or set(boundaries) != set(BOUNDARIES):
        raise ValueError("Trace event requires exactly the eight canonical boundaries")
    for value in boundaries.values():
        _validate_observation(value)
    page_postprocess = event.get("page_postprocess")
    if not isinstance(page_postprocess, Mapping):
        raise ValueError("Trace event requires observable page_postprocess")
    _validate_observation(page_postprocess)
    if page_postprocess.get("status") != "observable":
        raise ValueError("Trace event page_postprocess must be observable")
    if block_index is None:
        block_structure = event.get("block_structure")
        if not isinstance(block_structure, Mapping):
            raise ValueError("Page-level trace requires block_structure")
        _validate_observation(block_structure)
        if block_structure.get("status") != "unobservable":
            raise ValueError("Page-level trace block_structure must be unobservable")
    return page, block_index


def _page_postprocesses(
    events: list[dict[str, object]], side: str
) -> dict[str, Mapping[str, str]]:
    page_values: dict[str, Mapping[str, str]] = {}
    for event in events:
        page = event["page"]
        value = event["page_postprocess"]
        assert isinstance(page, str)
        assert isinstance(value, Mapping)
        existing = page_values.get(page)
        if existing is not None and dict(existing) != dict(value):
            raise ValueError(f"{side} page_postprocess must be consistent within page {page!r}")
        page_values[page] = value
    return page_values


def _index_events(
    events: list[dict[str, object]], side: str
) -> tuple[dict[tuple[str, int], dict[str, object]], dict[str, dict[str, object]]]:
    indexed: dict[tuple[str, int], dict[str, object]] = {}
    page_records: dict[str, dict[str, object]] = {}
    for event in events:
        if not isinstance(event, dict):
            raise ValueError(f"{side} trace event must be an object")
        page, block_index = _validate_event(event)
        if block_index is None:
            if page in page_records:
                raise ValueError(f"Duplicate {side} page-level trace: {page}")
            if any(key[0] == page for key in indexed):
                raise ValueError(f"Cannot mix {side} page-level and block records: {page}")
            page_records[page] = event
            continue
        if page in page_records:
            raise ValueError(f"Cannot mix {side} page-level and block records: {page}")
        key = (page, block_index)
        if key in indexed:
            raise ValueError(f"Duplicate {side} trace key: {key}")
        indexed[key] = event
    return indexed, page_records


def _read_trace_dir(directory: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for path in _trace_files(directory).values():
        events.extend(_read_trace_file(path))
    return events


def _trace_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"Trace directory not found: {directory}")
    return {path.stem: path for path in sorted(directory.glob("*.jsonl"))}


def _read_trace_file(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: trace event must be an object")
            events.append(value)
    return events


def _validate_trace_page(events: list[dict[str, object]], page: str, path: Path) -> None:
    for event in events:
        if event.get("page") != page:
            raise ValueError(f"{path}: trace page must match filename stem {page!r}")


def _safe_event(event: dict[str, object] | None) -> dict[str, object] | None:
    if event is None:
        return None
    safe = {
        "page": event["page"],
        "block_index": event["block_index"],
        "boundaries": {
            name: dict(value)
            for name, value in event["boundaries"].items()
        },
    }
    for name in ("block_structure", "page_postprocess"):
        if name in event:
            safe[name] = dict(event[name])
    return safe


def _page_evidence(
    page: str,
    events: dict[tuple[str, int], dict[str, object]],
    page_records: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    if page in page_records:
        safe = _safe_event(page_records[page])
        return [safe] if safe is not None else []
    return [
        _safe_event(event)
        for key, event in sorted(events.items())
        if key[0] == page
    ]
