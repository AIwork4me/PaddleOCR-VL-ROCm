from __future__ import annotations

from copy import deepcopy

import pytest

from eval.g4_performance import decide_g4
from eval.g4_quality import decide_g4_quality
from eval.g4_release import decide_g4_release
from tests.test_g4_performance import artifact as performance_artifact
from tests.test_g4_performance import manifest
from tests.test_g4_quality import artifact as quality_artifact


def inputs() -> dict[str, object]:
    sample_manifest = manifest()
    performance = performance_artifact(sample_manifest)
    quality = quality_artifact(sample_manifest)
    return {
        "manifest": sample_manifest,
        "performance_artifact": performance,
        "performance_decision": decide_g4(sample_manifest, performance),
        "quality_artifact": quality,
        "quality_decision": decide_g4_quality(sample_manifest, quality),
    }


def test_final_g4_accepts_numerical_performance_and_projected_quality() -> None:
    values = inputs()
    performance = values["performance_artifact"]
    performance["samples"][0]["output_sha256"] = "0" * 64  # type: ignore[index]
    values["performance_decision"] = decide_g4(values["manifest"], performance)  # type: ignore[arg-type]
    decision = decide_g4_release(**values)  # type: ignore[arg-type]
    assert decision["g4"] is True
    assert decision["checks"]["raw_output_equivalent"] is False  # type: ignore[index]


def test_final_g4_rejects_latency_failure() -> None:
    values = inputs()
    performance = values["performance_artifact"]
    for row in performance["samples"]:  # type: ignore[index]
        row["total_seconds"] = 40.0
    values["performance_decision"] = decide_g4(values["manifest"], performance)  # type: ignore[arg-type]
    assert decide_g4_release(**values)["g4"] is False  # type: ignore[arg-type]


def test_final_g4_rejects_quality_failure() -> None:
    values = inputs()
    quality = values["quality_artifact"]
    quality["samples"][0]["metrics"] = {  # type: ignore[index]
        "text_edit": {"reference": 0.0, "candidate": 1.0}
    }
    values["quality_decision"] = decide_g4_quality(values["manifest"], quality)  # type: ignore[arg-type]
    assert decide_g4_release(**values)["g4"] is False  # type: ignore[arg-type]


def test_final_g4_rejects_tampered_decision() -> None:
    values = inputs()
    values["performance_decision"] = deepcopy(values["performance_decision"])
    values["performance_decision"]["verdict"] = "FAIL"  # type: ignore[index]
    with pytest.raises(ValueError, match="recomputation"):
        decide_g4_release(**values)  # type: ignore[arg-type]
