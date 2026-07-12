from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

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

    assert "python eval/release_contract.py" in text
    assert "--stats $StatsPath --version v16 --engine official" in text
    assert "clean 1651-page official run" not in text


def test_release_contract_power_shell_call_preserves_stats_path_with_spaces(tmp_path: Path) -> None:
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is required")
    stats_path = tmp_path / "predictions with spaces" / "_run_stats.json"
    stats_path.parent.mkdir()
    stats_path.write_text(
        json.dumps(
            {
                "count": 1651,
                "ok": 1651,
                "fail": 0,
                "fallback": 0,
                "limit_pages": None,
                "engine": "official",
                "stats": [
                    {"image": f"page-{index:04d}.png", "status": "ok"} for index in range(1651)
                ],
            }
        ),
        encoding="utf-8",
    )
    command = (
        f"$StatsPath = '{stats_path}'; "
        "python eval/release_contract.py --stats $StatsPath --version v16 --engine official; "
        "exit $LASTEXITCODE"
    )

    completed = subprocess.run(
        [powershell, "-NoProfile", "-Command", command],
        cwd=SCRIPT.parents[1],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "complete success coverage" in completed.stdout
