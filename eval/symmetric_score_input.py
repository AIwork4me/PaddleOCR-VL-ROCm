"""Prepare immutable scorer inputs for a paired v1.6 symmetric exclusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from eval.release_contract import (
    KNOWN_V16_OFFICIAL_FAILURE,
    validate_approved_failure_predictions,
    validate_release_run_stats,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_stats(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Stats must be a JSON object: {path}")
    return value


def _apply_path_repair(
    source: dict[str, Any], repair: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    source_details = source.get("stats")
    repair_details = repair.get("stats")
    if not isinstance(source_details, list) or not isinstance(repair_details, list):
        raise ValueError("Source and repair stats require per-page details")
    repaired = {
        item.get("image"): item
        for item in repair_details
        if isinstance(item, dict)
        and item.get("status") == "ok"
        and isinstance(item.get("image"), str)
    }
    if not repaired:
        raise ValueError("Repair stats must contain successful pages")

    result = dict(source)
    details: list[dict[str, Any]] = []
    applied: set[str] = set()
    for item in source_details:
        if not isinstance(item, dict):
            raise ValueError("Source per-page details must be objects")
        image = item.get("image")
        status = item.get("status")
        if (
            isinstance(image, str)
            and image in repaired
            and isinstance(status, str)
            and "No such file or directory" in status
        ):
            details.append(dict(repaired[image]))
            applied.add(image)
        else:
            details.append(dict(item))
    if applied != set(repaired):
        raise ValueError("Repair pages must exactly match source path failures")
    result["stats"] = details
    result["ok"] = sum(item.get("status") == "ok" for item in details)
    result["fail"] = sum(
        isinstance(item.get("status"), str) and item["status"].startswith("fail")
        for item in details
    )
    result["fallback"] = sum(
        isinstance(item.get("status"), str) and item["status"].startswith("fallback")
        for item in details
    )
    return result, len(applied)


def prepare_score_input(
    *,
    source_dir: Path,
    destination_dir: Path,
    engine: str,
    repair_stats_path: Path | None = None,
) -> Path:
    """Hard-link scorer Markdown and write a fully attributable effective stats file."""
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    source_stats_path = source_dir / "_run_stats.json"
    source = _load_stats(source_stats_path)
    if source.get("engine") != engine:
        raise ValueError("Source stats engine must match the requested engine")

    effective = source
    repaired_pages = 0
    repair_sha256: str | None = None
    if repair_stats_path is not None:
        repair_stats_path = Path(repair_stats_path)
        effective, repaired_pages = _apply_path_repair(source, _load_stats(repair_stats_path))
        repair_sha256 = _sha256(repair_stats_path)

    approved = validate_release_run_stats(effective, version="v16", engine=engine)
    markdown = sorted(source_dir.glob("*.md"), key=lambda path: path.name)
    if len(markdown) != effective.get("ok"):
        raise ValueError("Scorer Markdown count must equal effective successful-page count")
    if destination_dir.exists():
        raise FileExistsError(f"Score input destination already exists: {destination_dir}")
    destination_dir.mkdir(parents=True)
    for source_markdown in markdown:
        os.link(source_markdown, destination_dir / source_markdown.name)
    validate_approved_failure_predictions(destination_dir, approved)

    effective_path = destination_dir / "_run_stats.json"
    effective_path.write_text(json.dumps(effective, ensure_ascii=False, indent=2), encoding="utf-8")
    prediction_rows = [f"{path.name}\t{_sha256(path)}" for path in markdown]
    receipt = {
        "schema": 1,
        "benchmark": "OmniDocBench-v1.6",
        "engine": engine,
        "approved_symmetric_exclusion": KNOWN_V16_OFFICIAL_FAILURE,
        "source_run_stats_sha256": _sha256(source_stats_path),
        "repair_run_stats_sha256": repair_sha256,
        "effective_run_stats_sha256": _sha256(effective_path),
        "repaired_pages": repaired_pages,
        "prediction_count": len(markdown),
        "prediction_manifest_sha256": hashlib.sha256(
            "\n".join(prediction_rows).encode("utf-8")
        ).hexdigest(),
    }
    (destination_dir / "score-input-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--destination-dir", type=Path, required=True)
    parser.add_argument("--engine", choices=("official", "lightweight"), required=True)
    parser.add_argument("--repair-stats", type=Path)
    args = parser.parse_args()
    prepared = prepare_score_input(
        source_dir=args.source_dir,
        destination_dir=args.destination_dir,
        engine=args.engine,
        repair_stats_path=args.repair_stats,
    )
    print(prepared)


if __name__ == "__main__":
    main()
