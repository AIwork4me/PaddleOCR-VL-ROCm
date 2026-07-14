"""Compare two block-level inference trace JSONL files.

The first divergence for each shared request is classified at the earliest
inference boundary that differs. This makes later payload and result changes
subordinate to an earlier layout or crop difference.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eval.task5_comparison import (
    compare_boundary_documents as compare_boundary_documents,
)
from eval.task5_comparison import (
    compare_canonical_traces,
)
from paddleocr_vl_rocm.contracts import fingerprint, redact

TraceEvent = dict[str, Any]


def _field(*names: str) -> Callable[[TraceEvent], Any]:
    def read(event: TraceEvent) -> Any:
        for name in names:
            if name in event:
                return event[name]
        return None

    return read


def _payload_fingerprint(event: TraceEvent) -> Any:
    if "payload_fingerprint" in event:
        return event["payload_fingerprint"]
    if "payload" in event:
        return fingerprint(redact(event["payload"]))
    return None


_ORDERED_BOUNDARIES: tuple[tuple[str, Callable[[TraceEvent], Any]], ...] = (
    ("request_order", _field("request_order")),
    ("label", _field("label", "block_label")),
    ("bbox", _field("bbox", "block_bbox")),
    ("crop_pixels", _field("image_sha256")),
    ("prompt", _field("prompt")),
    ("payload", _payload_fingerprint),
    ("raw_result", _field("raw_result_sha256")),
    ("postprocess", _field("final_result_sha256")),
)


def _first_divergence(reference: TraceEvent, candidate: TraceEvent) -> str | None:
    for name, value in _ORDERED_BOUNDARIES:
        if value(reference) != value(candidate):
            return name
    return None


def compare_traces(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, object]:
    """Compare trace events and return ordered summary and detailed differences."""

    differences: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    total_differences = 0

    def record(difference: dict[str, Any]) -> None:
        nonlocal total_differences
        total_differences += 1
        counts[difference["first_divergence"]] += 1
        if len(differences) < 100:
            differences.append(difference)

    if len(reference) != len(candidate):
        record(
            {
                "first_divergence": "request_count",
                "reference_count": len(reference),
                "candidate_count": len(candidate),
            }
        )

    for index, (reference_event, candidate_event) in enumerate(
        zip(reference, candidate, strict=False)
    ):
        first_divergence = _first_divergence(reference_event, candidate_event)
        if first_divergence is not None:
            record(
                {
                    "index": index,
                    "first_divergence": first_divergence,
                    "reference_fingerprint": fingerprint(redact(reference_event)),
                    "candidate_fingerprint": fingerprint(redact(candidate_event)),
                }
            )

    summary_order = ("request_count",) + tuple(name for name, _ in _ORDERED_BOUNDARIES)
    summary = {name: counts[name] for name in summary_order}
    return {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "difference_count": total_differences,
        "summary": summary,
        "differences": differences,
        "details_truncated": total_differences > len(differences),
    }


def _read_jsonl(path: Path) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: trace event must be an object")
            events.append(value)
    return events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference trace JSONL")
    parser.add_argument("candidate", type=Path, help="Candidate trace JSONL")
    parser.add_argument("--output", type=Path, help="Write report JSON to this path")
    return parser


def _trace_schema(events: list[TraceEvent]) -> str:
    kinds = {"canonical" if "boundaries" in event else "legacy" for event in events}
    if len(kinds) > 1:
        raise SystemExit("mixed canonical and legacy events are not allowed")
    return next(iter(kinds), "empty")


def main() -> None:
    args = build_parser().parse_args()
    if args.reference.is_dir() and args.candidate.is_dir():
        report = compare_canonical_traces(args.reference, args.candidate)
    elif args.reference.is_file() and args.candidate.is_file():
        reference = _read_jsonl(args.reference)
        candidate = _read_jsonl(args.candidate)
        reference_schema = _trace_schema(reference)
        candidate_schema = _trace_schema(candidate)
        if reference_schema != candidate_schema:
            raise SystemExit("reference and candidate trace schemas do not match")
        report = (
            compare_boundary_documents(reference, candidate)
            if reference_schema in {"canonical", "empty"}
            else compare_traces(reference, candidate)
        )
    else:
        raise SystemExit("reference and candidate must both be files or both be directories")
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
