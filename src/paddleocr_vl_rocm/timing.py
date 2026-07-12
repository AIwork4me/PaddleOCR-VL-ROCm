from __future__ import annotations

import math
from collections.abc import Sequence


def _nearest_rank(values: list[float], percentile: float) -> float:
    index = max(0, math.ceil(percentile * len(values)) - 1)
    return values[index]


def _covered_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    ordered = sorted((float(start), float(end)) for start, end in intervals)
    if not ordered:
        return 0.0
    covered = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        covered += max(0.0, current_end - current_start)
        current_start, current_end = start, end
    return covered + max(0.0, current_end - current_start)


def summarize_seconds(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": _nearest_rank(ordered, 0.50),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "max": ordered[-1],
    }
