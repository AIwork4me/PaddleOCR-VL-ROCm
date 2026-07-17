"""Run the frozen 27-page G4 performance benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.artifact_utils import sha256_file
from eval.g4_performance import build_receipt, decide_g4, verify_sample_files


def _adapter_module():
    path = Path("eval/PaddleOCRVLROCm_img2md.py")
    spec = importlib.util.spec_from_file_location("g4_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_samples(samples: list[dict[str, str]], images_dir: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for sample in samples:
        shutil.copyfile(images_dir / sample["image"], destination / sample["image"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("eval/g4-v1.6-samples.json"))
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--layout-model", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8111/v1")
    parser.add_argument("--api-model-name", default="PaddleOCR-VL-1.6-GGUF.gguf")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = _read(args.manifest)
    verify_sample_files(manifest, dataset_json=args.dataset_json, images_dir=args.images_dir)
    samples: list[dict[str, str]] = manifest["samples"]
    for path in (args.model, args.mmproj, args.llama_server, args.layout_model / "inference.onnx"):
        if not path.is_file():
            raise FileNotFoundError(path)

    root = args.output_root.resolve()
    inputs = root / "inputs"
    warmup_inputs = root / "warmup-input"
    _copy_samples(samples, args.images_dir, inputs)
    _copy_samples(samples[:1], args.images_dir, warmup_inputs)
    adapter = _adapter_module()
    adapter.run_lightweight_folder(
        warmup_inputs,
        root / "warmup-output",
        layout_model=str(args.layout_model),
        server_url=args.server_url,
        api_model_name=args.api_model_name,
    )
    start = time.perf_counter()
    summary = adapter.run_lightweight_folder(
        inputs,
        root / "outputs",
        layout_model=str(args.layout_model),
        server_url=args.server_url,
        api_model_name=args.api_model_name,
    )
    wall_seconds = time.perf_counter() - start
    by_image = {item["image"]: item for item in summary["stats"]}
    records = []
    for sample in samples:
        item = by_image[sample["image"]]
        output = root / "outputs" / f"{Path(sample['image']).stem}.md"
        baseline = args.baseline_dir / output.name
        if not baseline.is_file():
            raise FileNotFoundError(f"Baseline output is missing: {output.name}")
        timing = item.get("timing") or {}
        records.append(
            {
                "category": sample["category"],
                "image": sample["image"],
                "status": item["status"],
                "total_seconds": timing.get("total_seconds", item["seconds"]),
                "stages": {
                    "decode": timing.get("decode_seconds", 0.0),
                    "layout": timing.get("layout_seconds", 0.0),
                    "crop_encode": timing.get("crop_encode_seconds", 0.0),
                    "vlm": timing.get("vlm_seconds", 0.0),
                    "finalize": timing.get("finalize_seconds", 0.0),
                },
                "output_sha256": sha256_file(output),
                "baseline_sha256": sha256_file(baseline),
            }
        )
    artifact = {
        "schema": 1,
        "benchmark": "OmniDocBench-v1.6",
        "mode": "warm-corpus",
        "project_commit": _git_commit(),
        "sample_manifest_sha256": sha256_file(args.manifest),
        "environment": {
            "os": platform.platform(),
            "gpu": args.gpu,
            "driver": args.driver,
            "python": platform.python_version(),
        },
        "runtime": {
            "model_sha256": sha256_file(args.model),
            "mmproj_sha256": sha256_file(args.mmproj),
            "llama_server_sha256": sha256_file(args.llama_server),
            "layout_sha256": sha256_file(args.layout_model / "inference.onnx"),
        },
        "config": {"cache": False, "warmup_pages": 1, "vlm_max_workers": 8},
        "wall_seconds": wall_seconds,
        "samples": records,
    }
    artifact_path = root / "g4-run-artifact.json"
    decision_path = root / "g4-decision.json"
    receipt_path = root / "g4-receipt.json"
    _write(artifact_path, artifact)
    decision = decide_g4(manifest, artifact)
    _write(decision_path, decision)
    _write(
        receipt_path,
        build_receipt(
            {
                "sample_manifest": args.manifest,
                "run_artifact": artifact_path,
                "decision": decision_path,
            }
        ),
    )
    print(json.dumps(decision, indent=2))
    return 0 if decision["g4"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
