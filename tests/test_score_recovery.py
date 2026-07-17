from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval.score_recovery import APPROVED_FAILED_PAGE, authenticate_inference_bundle

OLD_COMMIT = "d7fd1809568eb80818e88f674b56844d03c2de81"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    official = root / "official"
    stages = root / "logs" / "stages"
    official.mkdir(parents=True)
    stages.mkdir(parents=True)
    manifest = {"git_commit": OLD_COMMIT, "inputs": {"dataset": {"sha256": "a" * 64}}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    failed = {
        "image": APPROVED_FAILED_PAGE,
        "status": "failed: model output does not match peg-native format",
        "seconds": 1.0,
        "attempts": 2,
        "traceback": "peg-native",
    }
    stats = {
        "count": 1651,
        "ok": 1650,
        "fail": 1,
        "fallback": 0,
        "engine": "official",
        "limit_pages": None,
        "stats": [
            {"image": f"page_{index}.png", "status": "ok", "attempts": 1} for index in range(1649)
        ]
        + [{"image": "中文页面.png", "status": "ok", "attempts": 1}, failed],
    }
    (official / "_run_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False), encoding="utf-8"
    )
    (official / "_errors.log").write_text(
        f"{APPROVED_FAILED_PAGE}: peg-native (attempts=2)\n", encoding="utf-8"
    )
    for index in range(1649):
        (official / f"page_{index}.md").write_text(str(index), encoding="utf-8")
    (official / "中文页面.md").write_text("内容", encoding="utf-8")
    command_log = stages / "official.commands.jsonl"
    commands = [
        {"stage": "Official", "command_name": "official-infer", "exit_code": 0},
        {"stage": "Official", "command_name": "official-contract", "exit_code": 0},
        {"stage": "Official", "command_name": "official-score", "exit_code": 1},
    ]
    command_log.write_text("".join(json.dumps(item) + "\n" for item in commands), encoding="utf-8")
    outputs = {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(official.iterdir())
        if path.is_file()
    }
    state = {
        "stage": "Official",
        "status": "failed",
        "producing_commit": OLD_COMMIT,
        "input_manifest_sha256": _sha(root / "manifest.json"),
        "command_sha256": _sha(command_log),
        "output_sha256": outputs,
    }
    (stages / "official.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return root


def _authenticate(root: Path) -> dict[str, object]:
    return authenticate_inference_bundle(
        root,
        expected_manifest_sha256=_sha(root / "manifest.json"),
        expected_state_sha256=_sha(root / "logs/stages/official.json"),
    )


def test_known_1650_1_bundle_with_non_ascii_prediction_is_accepted(tmp_path: Path) -> None:
    root = _bundle(tmp_path)

    result = _authenticate(root)

    assert result["source_git_commit"] == OLD_COMMIT
    assert result["prediction_count"] == 1650
    assert len(result["prediction_bundle_sha256"]) == 64
    assert result["failed_page"] == APPROVED_FAILED_PAGE


def test_source_state_identity_is_authenticated(tmp_path: Path) -> None:
    root = _bundle(tmp_path)

    with pytest.raises(ValueError, match="Source state hash mismatch"):
        authenticate_inference_bundle(root, expected_manifest_sha256=_sha(root / "manifest.json"))


@pytest.mark.parametrize("target", ["prediction", "stats", "errors", "commands"])
def test_bundle_tampering_is_rejected(tmp_path: Path, target: str) -> None:
    root = _bundle(tmp_path)
    paths = {
        "prediction": root / "official" / "中文页面.md",
        "stats": root / "official" / "_run_stats.json",
        "errors": root / "official" / "_errors.log",
        "commands": root / "logs" / "stages" / "official.commands.jsonl",
    }
    paths[target].write_bytes(paths[target].read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="hash|JSON|command"):
        _authenticate(root)


def test_arbitrary_missing_page_identity_is_rejected(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    stats_path = root / "official" / "_run_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["stats"][-1]["image"] = "arbitrary.png"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    state_path = root / "logs" / "stages" / "official.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["output_sha256"]["official/_run_stats.json"] = _sha(stats_path)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="approved failed page"):
        _authenticate(root)


def test_failed_page_prediction_is_rejected(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    failed_prediction = root / "official" / Path(APPROVED_FAILED_PAGE).with_suffix(".md").name
    failed_prediction.write_text("fabricated", encoding="utf-8")

    with pytest.raises(ValueError, match="failed-page prediction"):
        _authenticate(root)
