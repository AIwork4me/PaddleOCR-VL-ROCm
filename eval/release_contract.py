from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

KNOWN_V16_OFFICIAL_FAILURE = {
    "image": "newspaper_The Times UK_0801@magazinesclubnew_page_031.png",
    "issue_url": "https://github.com/PaddlePaddle/PaddleOCR/issues/18248",
    "error_signature": "peg-native",
}

KNOWN_V16_FAILURE_SIGNATURES = {
    "official": "peg-native",
    "lightweight": "500 Server Error",
}


def _detail_kind(item: dict[str, Any]) -> str:
    status = str(item.get("status", ""))
    if status == "ok":
        return "ok"
    if status.startswith("fail"):
        return "fail"
    if status.startswith("fallback"):
        return "fallback"
    raise ValueError(f"Unknown per-page release status: {status!r}")


def _validated_details(run_stats: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    count = run_stats.get("count")
    details = run_stats.get("stats")
    if not isinstance(count, int) or not isinstance(details, list) or len(details) != count:
        raise ValueError(f"Release evidence requires {count} per-page stats details")
    if not all(isinstance(item, dict) for item in details):
        raise ValueError("Every per-page stats detail must be an object")
    typed_details: list[dict[str, Any]] = details
    images = [item.get("image") for item in typed_details]
    if not all(isinstance(image, str) and image for image in images):
        raise ValueError("Every per-page stats detail requires an image")
    if len(set(images)) != len(images):
        raise ValueError("Per-page stats details require a unique image per row")
    counts = {"ok": 0, "fail": 0, "fallback": 0}
    for item in typed_details:
        counts[_detail_kind(item)] += 1
    if any(run_stats.get(name) != value for name, value in counts.items()):
        raise ValueError("Per-page stats counts do not match the aggregate run stats")
    return typed_details, counts


def validate_release_run_stats(
    run_stats: dict[str, Any], *, version: str, engine: str
) -> list[dict[str, str]]:
    if run_stats.get("limit_pages") is not None:
        raise ValueError("Release evidence requires an unbounded run with limit_pages=null")
    if run_stats.get("engine") != engine:
        raise ValueError("The stats engine must exist and match the requested engine")
    if version == "v16" and run_stats.get("count") != 1651:
        raise ValueError("OmniDocBench v1.6 release evidence requires count=1651")
    if run_stats.get("fallback") != 0:
        raise ValueError("Release evidence requires fallback=0")

    details, counts = _validated_details(run_stats)

    count = run_stats.get("count")
    ok = counts["ok"]
    fail = counts["fail"]
    if ok == count and fail == 0:
        return []

    if version != "v16" or engine not in KNOWN_V16_FAILURE_SIGNATURES:
        raise ValueError(
            "The approved known failure applies only to the v1.6 official or lightweight engines"
        )
    if ok != 1650 or fail != 1:
        raise ValueError("The approved v1.6 exception permits exactly one failed page")

    failures = [item for item in details if _detail_kind(item) == "fail"]
    if len(failures) != 1:
        raise ValueError("The approved v1.6 exception requires exactly one failure detail")
    failure = failures[0]
    if failure.get("image") != KNOWN_V16_OFFICIAL_FAILURE["image"]:
        raise ValueError("The failed page is not the approved image")
    failure_message = f"{failure.get('status', '')}\n{failure.get('error', '')}"
    signature = KNOWN_V16_FAILURE_SIGNATURES[engine]
    if signature not in failure_message:
        raise ValueError(
            f"The approved {engine} failure must contain the {signature} error signature"
        )
    return [KNOWN_V16_OFFICIAL_FAILURE]


def validate_approved_failure_predictions(
    predictions_dir: Path, approved_failures: list[dict[str, str]]
) -> None:
    for failure in approved_failures:
        prediction = predictions_dir / f"{Path(failure['image']).stem}.md"
        if prediction.exists():
            raise ValueError(
                f"Approved failed-page prediction must not exist in scorer input: {prediction}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate release-grade inference run stats.")
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--engine", required=True)
    args = parser.parse_args(argv)
    run_stats = json.loads(args.stats.read_text(encoding="utf-8"))
    try:
        exceptions = validate_release_run_stats(run_stats, version=args.version, engine=args.engine)
        validate_approved_failure_predictions(args.stats.parent, exceptions)
    except ValueError as exc:
        parser.error(str(exc))
    if exceptions:
        print(f"Release stats accepted with approved known failure: {exceptions[0]['image']}")
    else:
        print("Release stats accepted with complete success coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
