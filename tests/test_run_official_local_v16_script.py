from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_official_local_v16.ps1"


def test_official_v16_script_supports_isolated_dataset_and_prediction_paths() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert '[string]$DatasetDir = "data/omnidocbench/v16"' in text
    assert (
        '[string]$PredictionsDir = "predictions/paddleocr_official_local_llamacpp_gguf_v16"' in text
    )
    assert "--dataset-dir $DatasetDir --predictions-dir $PredictionsDir" in text


def test_official_v16_script_requires_clean_complete_stats_before_cdm() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "$Stats.count -ne 1651" in text
    assert "$Stats.ok -ne 1651" in text
    assert "$Stats.fail -ne 0" in text
    assert "$Stats.fallback -ne 0" in text
