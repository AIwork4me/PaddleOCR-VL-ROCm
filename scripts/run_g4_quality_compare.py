"""Score only G4 pages whose candidate Markdown differs from the G3 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.artifact_utils import sha256_file
from eval.g4_performance import build_receipt, validate_sample_manifest
from eval.g4_quality import decide_g4_quality
from eval.task5_comparison import normalize_scorer_markdown


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_tree_sha256(root: Path) -> str:
    excluded = {".git", ".venv", "result", "__pycache__", ".pytest_cache", ".mypy_cache"}
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in excluded for part in relative.parts):
            continue
        rows.append(f"{relative.as_posix()}\t{sha256_file(path)}")
    if not rows:
        raise ValueError(f"Scorer source tree is empty: {root}")
    return _text_sha256("\n".join(rows))


def _git_commit(path: Path = Path(".")) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _prediction_path(directory: Path, image: str) -> Path:
    return directory / f"{Path(image).stem}.md"


def _run_scorer(
    *,
    scorer_python: Path,
    scorer_dir: Path,
    config_path: Path,
    output_root: Path,
    label: str,
) -> None:
    env = dict(os.environ)
    env.update(
        {
            "OMNIDOCBENCH_MATCH_WORKERS": "1",
            "OMNIDOCBENCH_TEDS_WORKERS": "1",
            "OMNIDOCBENCH_CDM_WORKERS": "1",
            "PYTHONUTF8": "1",
        }
    )
    result = subprocess.run(
        [
            str(scorer_python),
            str(scorer_dir / "pdf_validation.py"),
            "--config",
            str(config_path),
        ],
        cwd=output_root,
        env=env,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output_root / f"{label}-scorer.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{label} scorer failed with exit code {result.returncode}")


def _load_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Scorer output must be an object: {path}")
    return value


def _sample_average(values: dict[str, Any], image: str, field: str | None = None) -> float | None:
    prefix = f"{image}_["
    selected = []
    for key, raw in values.items():
        if not key.startswith(prefix):
            continue
        value = raw[field] if field is not None else raw
        selected.append(float(value))
    return fmean(selected) if selected else None


def _page_metrics(result_dir: Path, prefix: str, images: list[str]) -> dict[str, dict[str, float]]:
    per_page_files = {
        "text_edit": "text_block_per_page_edit.json",
        "formula_edit": "display_formula_per_page_edit.json",
        "table_edit": "table_per_page_edit.json",
        "reading_order_edit": "reading_order_per_page_edit.json",
    }
    page_sources = {
        metric: _load_dict(result_dir / f"{prefix}_{suffix}")
        for metric, suffix in per_page_files.items()
    }
    teds = _load_dict(result_dir / f"{prefix}_table_per_table_TEDS.json")
    cdm = _load_dict(result_dir / f"{prefix}_display_formula_per_sample_CDM.json")
    result: dict[str, dict[str, float]] = {}
    for image in images:
        metrics = {
            metric: float(source[image])
            for metric, source in page_sources.items()
            if image in source
        }
        for metric, field in (("teds", "TEDS"), ("teds_structure_only", "TEDS_structure_only")):
            score = _sample_average(teds, image, field)
            if score is not None:
                metrics[metric] = score
        score = _sample_average(cdm, image)
        if score is not None:
            metrics["cdm"] = score
        result[image] = metrics
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("eval/g4-v1.6-samples.json"))
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--performance-artifact", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--scorer-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite G4 quality evidence: {root}")
    root.mkdir(parents=True)
    manifest = _read_object(args.manifest)
    samples = validate_sample_manifest(manifest)
    reference_dir = args.reference_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    rows: list[dict[str, object]] = []
    scored_images: list[str] = []
    for sample in samples:
        image = str(sample["image"])
        reference_path = _prediction_path(reference_dir, image)
        candidate_path = _prediction_path(candidate_dir, image)
        if not reference_path.is_file() or not candidate_path.is_file():
            raise FileNotFoundError(f"Missing paired G4 prediction: {image}")
        reference_text = reference_path.read_text(encoding="utf-8")
        candidate_text = candidate_path.read_text(encoding="utf-8")
        reference_normalized = normalize_scorer_markdown(reference_text)
        candidate_normalized = normalize_scorer_markdown(candidate_text)
        reference_sha = sha256_file(reference_path)
        candidate_sha = sha256_file(candidate_path)
        if reference_sha == candidate_sha:
            relation = "exact"
        elif reference_normalized == candidate_normalized:
            relation = "normalized"
        else:
            relation = "scored"
            scored_images.append(image)
        rows.append(
            {
                "category": sample["category"],
                "image": image,
                "relation": relation,
                "reference_sha256": reference_sha,
                "candidate_sha256": candidate_sha,
                "reference_normalized_sha256": _text_sha256(reference_normalized),
                "candidate_normalized_sha256": _text_sha256(candidate_normalized),
                "metrics": {},
            }
        )
    if not scored_images:
        raise ValueError("Targeted G4 quality comparison requires at least one differing page")

    dataset = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise ValueError("OmniDocBench dataset must be a list")
    wanted = set(scored_images)
    subset = [
        page
        for page in dataset
        if isinstance(page, dict)
        and isinstance(page.get("page_info"), dict)
        and page["page_info"].get("image_path") in wanted
    ]
    found = {page["page_info"]["image_path"] for page in subset}
    if found != wanted or len(subset) != len(wanted):
        raise ValueError("The targeted GT subset does not match the differing G4 pages")
    subset_path = root / "g4-quality-subset-gt.json"
    _write(subset_path, subset)

    prediction_dirs = {
        "reference": root / "g4_quality_reference",
        "candidate": root / "g4_quality_candidate",
    }
    for label, directory in prediction_dirs.items():
        directory.mkdir()
        source = reference_dir if label == "reference" else candidate_dir
        for image in scored_images:
            shutil.copyfile(_prediction_path(source, image), _prediction_path(directory, image))

    contract_config = {
        "metrics": {
            "text_block": {"metric": ["Edit_dist"]},
            "display_formula": {"metric": ["Edit_dist", "CDM"], "cdm_workers": 1},
            "table": {"metric": ["TEDS", "Edit_dist"], "teds_workers": 1},
            "reading_order": {"metric": ["Edit_dist"]},
        },
        "dataset": {
            "dataset_name": "end2end_dataset",
            "match_method": "quick_match",
            "match_workers": 1,
            "quick_match_truncated_timeout_sec": 300,
            "match_timeout_sec": 420,
            "timeout_fallback_max_chunk_span": 10,
            "timeout_fallback_order_penalty": 0.10,
        },
    }
    contract_path = root / "g4-quality-scorer-contract.json"
    _write(contract_path, contract_config)
    for label, prediction_dir in prediction_dirs.items():
        config = {
            "end2end_eval": {
                "metrics": contract_config["metrics"],
                "dataset": {
                    **contract_config["dataset"],
                    "ground_truth": {"data_path": str(subset_path)},
                    "prediction": {"data_path": str(prediction_dir)},
                },
            }
        }
        config_path = root / f"{label}-config.json"
        _write(config_path, config)
        _run_scorer(
            scorer_python=args.scorer_python.resolve(),
            scorer_dir=args.scorer_dir.resolve(),
            config_path=config_path,
            output_root=root,
            label=label,
        )

    result_dir = root / "result"
    reference_metrics = _page_metrics(result_dir, "g4_quality_reference_quick_match", scored_images)
    candidate_metrics = _page_metrics(result_dir, "g4_quality_candidate_quick_match", scored_images)
    for row in rows:
        if row["relation"] != "scored":
            continue
        image = str(row["image"])
        reference = reference_metrics[image]
        candidate = candidate_metrics[image]
        if set(reference) != set(candidate) or not reference:
            raise ValueError(f"Scorer metric applicability mismatch for {image}")
        row["metrics"] = {
            metric: {"reference": reference[metric], "candidate": candidate[metric]}
            for metric in sorted(reference)
        }

    artifact = {
        "schema": 1,
        "benchmark": "OmniDocBench-v1.6",
        "project_commit": _git_commit(),
        "sample_manifest_sha256": sha256_file(args.manifest),
        "dataset_sha256": sha256_file(args.dataset_json),
        "scorer_commit": _git_commit(args.scorer_dir.resolve()),
        "scorer_tree_sha256": _source_tree_sha256(args.scorer_dir.resolve()),
        "scorer_config_sha256": sha256_file(contract_path),
        "performance_artifact_sha256": sha256_file(args.performance_artifact),
        "normalization": "task5-scorer-markdown-v1",
        "samples": rows,
    }
    artifact_path = root / "g4-quality-artifact.json"
    decision_path = root / "g4-quality-decision.json"
    receipt_path = root / "g4-quality-receipt.json"
    _write(artifact_path, artifact)
    decision = decide_g4_quality(manifest, artifact)
    _write(decision_path, decision)
    _write(
        receipt_path,
        build_receipt(
            {
                "sample_manifest": args.manifest,
                "performance_artifact": args.performance_artifact,
                "quality_artifact": artifact_path,
                "quality_decision": decision_path,
                "scorer_contract": contract_path,
                "subset_gt": subset_path,
            }
        ),
    )
    print(json.dumps(decision, indent=2))
    return 0 if decision["g4_quality"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
