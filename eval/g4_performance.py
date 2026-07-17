"""G4 performance artifact validation and receipt generation."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.artifact_utils import sha256_file
from paddleocr_vl_rocm.timing import summarize_seconds

SCHEMA = 1
EXPECTED_PAGES = 27
PAGES_PER_CATEGORY = 3
EXPECTED_CATEGORIES = {
    "PPT2PDF",
    "academic_literature",
    "book",
    "colorful_textbook",
    "exam_paper",
    "magazine",
    "newspaper",
    "note",
    "research_report",
}
MAX_MEAN_SECONDS = 13.00
MAX_P95_SECONDS = 34.82
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def validate_sample_manifest(manifest: Mapping[str, object]) -> list[dict[str, str]]:
    if set(manifest) != {"schema", "benchmark", "selection", "dataset_sha256", "samples"}:
        raise ValueError("G4 sample manifest schema is invalid")
    if manifest["schema"] != SCHEMA or manifest["benchmark"] != "OmniDocBench-v1.6":
        raise ValueError("G4 sample manifest identity is invalid")
    if manifest["selection"] != "sha256-filename-first-3-per-category":
        raise ValueError("G4 sample selection rule is invalid")
    _hash(manifest["dataset_sha256"], "dataset_sha256")
    samples = manifest["samples"]
    if not isinstance(samples, list) or len(samples) != EXPECTED_PAGES:
        raise ValueError(f"G4 requires exactly {EXPECTED_PAGES} samples")
    result: list[dict[str, str]] = []
    for index, raw in enumerate(samples):
        sample = _object(raw, f"samples[{index}]")
        if set(sample) != {"category", "image", "sha256"}:
            raise ValueError("G4 sample entry schema is invalid")
        category, image = sample["category"], sample["image"]
        if category not in EXPECTED_CATEGORIES:
            raise ValueError(f"Unexpected G4 sample category: {category!r}")
        if (
            not isinstance(image, str)
            or not image
            or Path(image).name != image
            or Path(image).suffix.lower() not in {".png", ".jpg", ".jpeg"}
        ):
            raise ValueError("G4 sample image must be a safe basename")
        result.append(
            {"category": category, "image": image, "sha256": _hash(sample["sha256"], "sha256")}
        )
    if len({item["image"] for item in result}) != EXPECTED_PAGES:
        raise ValueError("G4 sample images must be unique")
    counts = Counter(item["category"] for item in result)
    if set(counts) != EXPECTED_CATEGORIES or set(counts.values()) != {PAGES_PER_CATEGORY}:
        raise ValueError("G4 requires three samples from every frozen category")
    return result


def verify_sample_files(
    manifest: Mapping[str, object], *, dataset_json: Path, images_dir: Path
) -> None:
    samples = validate_sample_manifest(manifest)
    if sha256_file(dataset_json) != manifest["dataset_sha256"]:
        raise ValueError("OmniDocBench dataset hash does not match the G4 manifest")
    for sample in samples:
        path = images_dir / sample["image"]
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"G4 sample is missing or unsafe: {sample['image']}")
        if sha256_file(path) != sample["sha256"]:
            raise ValueError(f"G4 sample hash mismatch: {sample['image']}")


def decide_g4(manifest: Mapping[str, object], artifact: Mapping[str, object]) -> dict[str, object]:
    expected = validate_sample_manifest(manifest)
    required = {
        "schema",
        "benchmark",
        "mode",
        "project_commit",
        "sample_manifest_sha256",
        "environment",
        "runtime",
        "config",
        "wall_seconds",
        "samples",
    }
    if set(artifact) != required:
        raise ValueError("G4 run artifact schema is invalid")
    if (
        artifact["schema"] != SCHEMA
        or artifact["benchmark"] != "OmniDocBench-v1.6"
        or artifact["mode"] != "warm-corpus"
    ):
        raise ValueError("G4 run artifact identity is invalid")
    commit = artifact["project_commit"]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("G4 project commit must be a full Git SHA")
    _hash(artifact["sample_manifest_sha256"], "sample_manifest_sha256")
    environment = _object(artifact["environment"], "environment")
    runtime = _object(artifact["runtime"], "runtime")
    config = _object(artifact["config"], "config")
    for key in ("os", "gpu", "driver", "python"):
        if not isinstance(environment.get(key), str) or not environment[key].strip():
            raise ValueError(f"G4 environment requires {key}")
    for key in ("model_sha256", "mmproj_sha256", "llama_server_sha256", "layout_sha256"):
        _hash(runtime.get(key), key)
    required_config = {
        "cache",
        "warmup_pages",
        "vlm_max_workers",
        "n_gpu_layers",
        "server_slots",
        "server_threads",
        "context_size",
        "temperature",
        "seed",
        "top_k",
        "top_p",
        "min_p",
        "repeat_penalty",
        "flash_attention",
    }
    if set(config) != required_config:
        raise ValueError("G4 run config schema is invalid")
    if config["cache"] is not False or config["warmup_pages"] != 1:
        raise ValueError("G4 requires cache disabled and exactly one warm-up page")
    workers = config["vlm_max_workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("G4 requires a positive vlm_max_workers")
    for key in ("n_gpu_layers", "server_slots", "server_threads", "context_size", "seed", "top_k"):
        value = config[key]
        minimum = 0 if key in {"n_gpu_layers", "seed"} else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"G4 requires {key} to be an integer >= {minimum}")
    for key in ("temperature", "top_p", "min_p", "repeat_penalty"):
        _finite_number(config[key], key)
    if config["flash_attention"] is not True:
        raise ValueError("G4 requires flash attention")
    wall_seconds = _finite_number(artifact["wall_seconds"], "wall_seconds")
    rows = artifact["samples"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_PAGES:
        raise ValueError(f"G4 run requires exactly {EXPECTED_PAGES} page records")
    totals: list[float] = []
    stages = ("decode", "layout", "crop_encode", "vlm", "finalize")
    failures = 0
    output_equivalent = True
    for expected_sample, raw in zip(expected, rows, strict=True):
        row = _object(raw, "sample record")
        if set(row) != {
            "category",
            "image",
            "status",
            "total_seconds",
            "stages",
            "output_sha256",
            "baseline_sha256",
        }:
            raise ValueError("G4 page record schema is invalid")
        if (
            row["category"] != expected_sample["category"]
            or row["image"] != expected_sample["image"]
        ):
            raise ValueError("G4 page order does not match the frozen manifest")
        if row["status"] != "ok":
            failures += 1
        total = _finite_number(row["total_seconds"], "total_seconds")
        stage_values = _object(row["stages"], "stages")
        if set(stage_values) != set(stages):
            raise ValueError("G4 stage timing schema is invalid")
        for stage in stages:
            _finite_number(stage_values[stage], stage)
        output_sha = _hash(row["output_sha256"], "output_sha256")
        baseline_sha = _hash(row["baseline_sha256"], "baseline_sha256")
        output_equivalent = output_equivalent and output_sha == baseline_sha
        totals.append(total)
    summary = summarize_seconds(totals)
    mean = float(summary["mean"])
    p95 = float(summary["p95"])
    checks = {
        "sample_contract": True,
        "zero_failures": failures == 0,
        "output_equivalent": output_equivalent,
        "mean_at_most_13_00": mean <= MAX_MEAN_SECONDS,
        "p95_at_most_34_82": p95 <= MAX_P95_SECONDS,
    }
    return {
        "schema": SCHEMA,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "g4": all(checks.values()),
        "checks": checks,
        "pages": EXPECTED_PAGES,
        "wall_seconds": wall_seconds,
        "pages_per_minute": EXPECTED_PAGES / wall_seconds * 60 if wall_seconds else 0.0,
        "timing": summary,
    }


def build_receipt(paths: Mapping[str, Path]) -> dict[str, object]:
    if set(paths) != {"sample_manifest", "run_artifact", "decision"}:
        raise ValueError("G4 receipt requires the exact evidence set")
    files: dict[str, object] = {}
    for name, path in sorted(paths.items()):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"G4 receipt input is missing or unsafe: {name}")
        files[name] = {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"schema": SCHEMA, "files": files}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    decide = sub.add_parser("decide")
    decide.add_argument("--manifest", type=Path, required=True)
    decide.add_argument("--artifact", type=Path, required=True)
    decide.add_argument("--output", type=Path, required=True)
    receipt = sub.add_parser("receipt")
    receipt.add_argument("--manifest", type=Path, required=True)
    receipt.add_argument("--artifact", type=Path, required=True)
    receipt.add_argument("--decision", type=Path, required=True)
    receipt.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "decide":
        result = decide_g4(_read(args.manifest), _read(args.artifact))
    else:
        result = build_receipt(
            {
                "sample_manifest": args.manifest,
                "run_artifact": args.artifact,
                "decision": args.decision,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
