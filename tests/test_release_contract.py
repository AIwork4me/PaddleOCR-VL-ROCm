import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.release_contract import (
    KNOWN_V16_OFFICIAL_FAILURE,
    validate_approved_failure_predictions,
    validate_release_run_stats,
)


def _stats(**overrides):
    successful = [{"image": f"page-{index:04d}.png", "status": "ok"} for index in range(1650)]
    values = {
        "count": 1651,
        "ok": 1650,
        "fail": 1,
        "fallback": 0,
        "limit_pages": None,
        "engine": "official",
        "stats": successful
        + [
            {
                "image": KNOWN_V16_OFFICIAL_FAILURE["image"],
                "status": (
                    "failed: The model produced output that does not match the expected "
                    "peg-native format"
                ),
            }
        ],
    }
    values.update(overrides)
    return values


def test_accepts_exact_known_v16_official_failure() -> None:
    assert validate_release_run_stats(_stats(), version="v16", engine="official") == [
        KNOWN_V16_OFFICIAL_FAILURE
    ]


def test_accepts_future_clean_v16_official_run() -> None:
    stats = _stats(
        ok=1651,
        fail=0,
        stats=[{"image": f"page-{index:04d}.png", "status": "ok"} for index in range(1651)],
    )

    assert validate_release_run_stats(stats, version="v16", engine="official") == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ok": 1649, "fail": 2}, "aggregate"),
        ({"fallback": 1}, "fallback=0"),
        ({"limit_pages": 16}, "unbounded"),
        ({"count": 1650, "ok": 1649}, "count=1651"),
        ({"stats": []}, "1651 per-page"),
        (
            {
                "stats": _stats()["stats"][:-1]
                + [
                    {
                        "image": "different.png",
                        "status": "fail",
                        "error": "peg-native",
                    }
                ]
            },
            "approved image",
        ),
        (
            {
                "stats": _stats()["stats"][:-1]
                + [
                    {
                        "image": KNOWN_V16_OFFICIAL_FAILURE["image"],
                        "status": "fail",
                        "error": "timeout",
                    }
                ]
            },
            "peg-native",
        ),
    ],
)
def test_rejects_any_expansion_of_known_failure(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_release_run_stats(_stats(**overrides), version="v16", engine="official")


def test_rejects_known_exception_for_non_paired_engine() -> None:
    with pytest.raises(ValueError, match="official or lightweight"):
        validate_release_run_stats(_stats(engine="other"), version="v16", engine="other")


def test_accepts_exact_known_v16_lightweight_symmetric_failure() -> None:
    stats = _stats(
        engine="lightweight",
        stats=_stats()["stats"][:-1]
        + [
            {
                "image": KNOWN_V16_OFFICIAL_FAILURE["image"],
                "status": "failed: 500 Server Error: Internal Server Error",
            }
        ],
    )

    assert validate_release_run_stats(stats, version="v16", engine="lightweight") == [
        KNOWN_V16_OFFICIAL_FAILURE
    ]


def test_rejects_lightweight_known_page_without_symmetric_500_signature() -> None:
    stats = _stats(
        engine="lightweight",
        stats=_stats()["stats"][:-1]
        + [
            {
                "image": KNOWN_V16_OFFICIAL_FAILURE["image"],
                "status": "failed: timeout",
            }
        ],
    )

    with pytest.raises(ValueError, match="500 Server Error"):
        validate_release_run_stats(stats, version="v16", engine="lightweight")


def test_rejects_stats_engine_that_does_not_match_requested_engine() -> None:
    with pytest.raises(ValueError, match="stats engine"):
        validate_release_run_stats(_stats(engine="lightweight"), version="v16", engine="official")


def test_rejects_clean_aggregate_with_failure_detail() -> None:
    with pytest.raises(ValueError, match="aggregate"):
        validate_release_run_stats(_stats(ok=1651, fail=0), version="v16", engine="official")


def test_rejects_duplicate_page_detail() -> None:
    details = _stats()["stats"]
    details[1] = dict(details[0])

    with pytest.raises(ValueError, match="unique image"):
        validate_release_run_stats(_stats(stats=details), version="v16", engine="official")


def test_cli_validates_stats_file(tmp_path: Path) -> None:
    stats_path = tmp_path / "_run_stats.json"
    stats_path.write_text(json.dumps(_stats()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "eval/release_contract.py",
            "--stats",
            str(stats_path),
            "--version",
            "v16",
            "--engine",
            "official",
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "approved known failure" in completed.stdout


def test_approved_failure_requires_missing_prediction_markdown(tmp_path: Path) -> None:
    validate_approved_failure_predictions(tmp_path, [KNOWN_V16_OFFICIAL_FAILURE])

    prediction = tmp_path / f"{Path(KNOWN_V16_OFFICIAL_FAILURE['image']).stem}.md"
    prediction.write_text("synthetic fallback", encoding="utf-8")

    with pytest.raises(ValueError, match="must not exist"):
        validate_approved_failure_predictions(tmp_path, [KNOWN_V16_OFFICIAL_FAILURE])
