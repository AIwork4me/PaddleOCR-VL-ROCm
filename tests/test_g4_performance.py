from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from eval.g4_performance import build_receipt, decide_g4


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def manifest() -> dict[str, object]:
    categories = (
        "PPT2PDF",
        "academic_literature",
        "book",
        "colorful_textbook",
        "exam_paper",
        "magazine",
        "newspaper",
        "note",
        "research_report",
    )
    samples = [
        {
            "category": category,
            "image": f"{category}-{index}.png",
            "sha256": digest(f"{category}-{index}"),
        }
        for category in categories
        for index in range(3)
    ]
    return {
        "schema": 1,
        "benchmark": "OmniDocBench-v1.6",
        "selection": "sha256-filename-first-3-per-category",
        "dataset_sha256": digest("dataset"),
        "samples": samples,
    }


def artifact(sample_manifest: dict[str, object], *, seconds: float = 10.0) -> dict[str, object]:
    rows = []
    for sample in sample_manifest["samples"]:  # type: ignore[index]
        rows.append(
            {
                "category": sample["category"],
                "image": sample["image"],
                "status": "ok",
                "total_seconds": seconds,
                "stages": {
                    "decode": 0.1,
                    "layout": 0.2,
                    "crop_encode": 0.3,
                    "vlm": 9.0,
                    "finalize": 0.4,
                },
                "output_sha256": digest(str(sample["image"])),
                "baseline_sha256": digest(str(sample["image"])),
            }
        )
    return {
        "schema": 1,
        "benchmark": "OmniDocBench-v1.6",
        "mode": "warm-corpus",
        "project_commit": "a" * 40,
        "sample_manifest_sha256": digest("manifest"),
        "environment": {"os": "Windows", "gpu": "AMD", "driver": "1", "python": "3.11"},
        "runtime": {
            "model_sha256": digest("model"),
            "mmproj_sha256": digest("mmproj"),
            "llama_server_sha256": digest("server"),
            "layout_sha256": digest("layout"),
        },
        "config": {
            "cache": False,
            "warmup_pages": 1,
            "vlm_max_workers": 8,
            "n_gpu_layers": 99,
            "server_slots": 8,
            "server_threads": 8,
            "context_size": 32768,
            "temperature": 0.0,
            "seed": 1,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "flash_attention": True,
        },
        "wall_seconds": 270.0,
        "samples": rows,
    }


def test_g4_accepts_exact_boundary() -> None:
    sample_manifest = manifest()
    run = artifact(sample_manifest, seconds=13.0)
    decision = decide_g4(sample_manifest, run)
    assert decision["g4"] is True
    assert decision["timing"]["mean"] == 13.0  # type: ignore[index]


@pytest.mark.parametrize("seconds", [13.01, 34.83])
def test_g4_rejects_latency_regression(seconds: float) -> None:
    sample_manifest = manifest()
    decision = decide_g4(sample_manifest, artifact(sample_manifest, seconds=seconds))
    assert decision["g4"] is False


def test_g4_rejects_output_mismatch() -> None:
    sample_manifest = manifest()
    run = artifact(sample_manifest)
    run["samples"][0]["output_sha256"] = digest("changed")  # type: ignore[index]
    decision = decide_g4(sample_manifest, run)
    assert decision["checks"]["output_equivalent"] is False  # type: ignore[index]


def test_g4_rejects_sample_reordering() -> None:
    sample_manifest = manifest()
    run = artifact(sample_manifest)
    run["samples"][0], run["samples"][1] = run["samples"][1], run["samples"][0]  # type: ignore[index]
    with pytest.raises(ValueError, match="frozen manifest"):
        decide_g4(sample_manifest, run)


def test_receipt_hashes_exact_evidence_set(tmp_path: Path) -> None:
    paths = {}
    for name in ("sample_manifest", "run_artifact", "decision"):
        path = tmp_path / f"{name}.json"
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    receipt = build_receipt(paths)
    assert set(receipt["files"]) == set(paths)  # type: ignore[arg-type]
