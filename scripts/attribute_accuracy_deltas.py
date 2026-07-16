"""Attribute accuracy changes using ordered, single-boundary oracle replays.

The caller owns scoring.  In particular, formula and table callers must pass a
replay callback backed by Formula CDM and content-aware TEDS respectively.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

BOUNDARIES = ("crop", "payload", "raw_vlm", "final_output")
_MISSING = object()


def _replayable_value(case: Mapping[str, Any], boundary: str) -> object:
    nested = "boundaries" in case
    boundaries = case.get("boundaries", case)
    if not isinstance(boundaries, Mapping) or boundary not in boundaries:
        return _MISSING

    observation = boundaries[boundary]
    if not isinstance(observation, Mapping) or "status" not in observation:
        return observation
    if observation.get("status") != "observable":
        return _MISSING

    # A trace fingerprint proves observation, not possession of the input
    # needed for an oracle replay. Nested summaries therefore need an explicit
    # replayable value.
    for key in ("value", "input", "artifact"):
        if key in observation:
            return observation[key]
    if nested:
        return _MISSING
    return _MISSING


def attribute_case(
    official: dict[str, Any],
    lightweight: dict[str, Any],
    replay: Callable[[str, object], float],
) -> dict[str, object]:
    """Measure independent official-input swaps and report the earliest gain.

    Every observable boundary is measured from the lightweight baseline and
    restored before proceeding. Missing or metadata-only inputs are reported
    as unproven and never represented by a synthetic zero contribution.
    """

    contributions: list[dict[str, object]] = []
    first_positive: dict[str, object] | None = None
    all_prior_observable = True

    for boundary in BOUNDARIES:
        baseline_value = _replayable_value(lightweight, boundary)
        official_value = _replayable_value(official, boundary)
        if baseline_value is _MISSING or official_value is _MISSING:
            contributions.append({"boundary": boundary, "status": "unproven"})
            all_prior_observable = False
            continue

        before_score = float(replay(boundary, baseline_value))
        try:
            after_score = float(replay(boundary, official_value))
        finally:
            replay(boundary, baseline_value)

        contribution = after_score - before_score
        entry: dict[str, object] = {
            "boundary": boundary,
            "status": "proven",
            "before_score": before_score,
            "after_score": after_score,
            "contribution": contribution,
        }
        contributions.append(entry)
        if first_positive is None and contribution > 0.0 and all_prior_observable:
            first_positive = entry

    if first_positive is None:
        return {"status": "unproven", "contributions": contributions}
    return {
        "status": "proven",
        "boundary": first_positive["boundary"],
        "contribution": first_positive["contribution"],
        "contributions": contributions,
    }
