"""Authenticate an immutable official inference bundle for score-only recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

SOURCE_COMMIT = "d7fd1809568eb80818e88f674b56844d03c2de81"
SOURCE_MANIFEST_SHA256 = "4c86bc66705b8f86b3374ed24477b7f99b238e4d0c1c1567ce758fe92aae8a4a"
SOURCE_STATE_SHA256 = "0c259d01e461138fc7021e7cb5e299520c6dc9376bc233bb378115e689f742ce"
APPROVED_FAILED_PAGE = "newspaper_The Times UK_0801@magazinesclubnew_page_031.png"
EXPECTED_OUTPUT_COUNT = 1652


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON: {path.name}: {exc}") from exc


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_commands(path: Path) -> None:
    commands = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            commands.append(_require_mapping(json.loads(line), "command record"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid command log: {exc}") from exc
    expected = [
        ("official-infer", 0),
        ("official-contract", 0),
        ("official-score", 1),
    ]
    actual = [(record.get("command_name"), record.get("exit_code")) for record in commands]
    if actual != expected:
        raise ValueError(f"Official command integrity mismatch: {actual!r}")


def authenticate_inference_bundle(
    root: Path,
    *,
    expected_manifest_sha256: str = SOURCE_MANIFEST_SHA256,
    expected_state_sha256: str = SOURCE_STATE_SHA256,
    expected_output_count: int = EXPECTED_OUTPUT_COUNT,
) -> dict[str, object]:
    """Validate a historical 1650/1 inference bundle without modifying it."""
    root = root.resolve()
    manifest_path = root / "manifest.json"
    state_path = root / "logs" / "stages" / "official.json"
    command_path = root / "logs" / "stages" / "official.commands.jsonl"
    official = root / "official"
    stats_path = official / "_run_stats.json"
    errors_path = official / "_errors.log"
    for required in (manifest_path, state_path, command_path, stats_path, errors_path):
        if not required.is_file():
            raise ValueError(f"Required recovery input is missing: {required.name}")

    manifest_sha = _sha256(manifest_path)
    if manifest_sha != expected_manifest_sha256:
        raise ValueError("Source manifest hash mismatch")
    manifest = _require_mapping(_read_json(manifest_path), "manifest")
    if manifest.get("git_commit") != SOURCE_COMMIT:
        raise ValueError("Source manifest producing commit mismatch")

    if _sha256(state_path) != expected_state_sha256:
        raise ValueError("Source state hash mismatch")
    state = _require_mapping(_read_json(state_path), "official state")
    if state.get("stage") != "Official" or state.get("status") != "failed":
        raise ValueError("Historical Official state must record the failed scoring attempt")
    if state.get("producing_commit") != SOURCE_COMMIT:
        raise ValueError("Official state producing commit mismatch")
    if state.get("input_manifest_sha256") != manifest_sha:
        raise ValueError("Official state manifest hash mismatch")
    if state.get("command_sha256") != _sha256(command_path):
        raise ValueError("Official command log hash mismatch")
    _validate_commands(command_path)

    failed_prediction = official / Path(APPROVED_FAILED_PAGE).with_suffix(".md").name
    if failed_prediction.exists():
        raise ValueError("Unexpected failed-page prediction exists")

    expected_outputs = _require_mapping(state.get("output_sha256"), "output_sha256")
    if len(expected_outputs) != expected_output_count:
        raise ValueError(f"Official output hash count mismatch: {len(expected_outputs)}")
    current_files = sorted(path for path in official.rglob("*") if path.is_file())
    current_names = {path.relative_to(root).as_posix() for path in current_files}
    if current_names != set(expected_outputs):
        raise ValueError("Official output hash set mismatch")
    for relative, expected in expected_outputs.items():
        path = root / Path(str(relative))
        if _sha256(path) != expected:
            raise ValueError(f"Official output hash mismatch: {relative}")

    stats = _require_mapping(_read_json(stats_path), "run stats")
    contract = {"count": 1651, "ok": 1650, "fail": 1, "fallback": 0, "limit_pages": None}
    if any(stats.get(key) != value for key, value in contract.items()):
        raise ValueError("Official run stats contract mismatch")
    if stats.get("engine") != "official":
        raise ValueError("Official run stats engine mismatch")
    page_stats = stats.get("stats")
    if not isinstance(page_stats, list) or len(page_stats) != 1651:
        raise ValueError("Official per-page stats count mismatch")
    failures = [
        item for item in page_stats if isinstance(item, dict) and item.get("status") != "ok"
    ]
    if len(failures) != 1 or failures[0].get("image") != APPROVED_FAILED_PAGE:
        raise ValueError("Official stats do not contain the sole approved failed page")
    if "peg-native" not in str(failures[0].get("status", "")):
        raise ValueError("Approved failed page lacks peg-native error identity")
    error_text = errors_path.read_text(encoding="utf-8")
    if APPROVED_FAILED_PAGE not in error_text or "peg-native" not in error_text:
        raise ValueError("Official error log identity mismatch")

    predictions = sorted(official.glob("*.md"))
    if len(predictions) != 1650:
        raise ValueError(f"Official prediction count mismatch: {len(predictions)}")
    bundle_lines = [f"{path.name}\0{_sha256(path)}" for path in predictions]
    bundle_sha = hashlib.sha256("\n".join(bundle_lines).encode("utf-8")).hexdigest()
    return {
        "source_git_commit": SOURCE_COMMIT,
        "source_manifest_sha256": manifest_sha,
        "source_state_sha256": _sha256(state_path),
        "source_command_sha256": _sha256(command_path),
        "prediction_bundle_sha256": bundle_sha,
        "prediction_count": len(predictions),
        "failed_page": APPROVED_FAILED_PAGE,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = authenticate_inference_bundle(args.source_root)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
