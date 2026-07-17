#!/usr/bin/env python3
"""CDM-only computation with ThreadPoolExecutor (avoids Windows spawn issues)."""

import copy
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval" / ".omnidocbench"))
os.chdir(REPO_ROOT)

from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

from src.metrics.cal_metric import _strip_cdm_math_wrappers  # noqa: E402
from src.metrics.cdm_metric import CDM  # noqa: E402
from tqdm import tqdm  # noqa: E402

INPUT = "result/paddleocrvl_rocm_cdm_quick_match_display_formula_result.json"
OUTPUT_RESULT = "result/paddleocrvl_rocm_cdm_quick_match_display_formula_result.json"
OUTPUT_CDM = "result/paddleocrvl_rocm_cdm_quick_match_display_formula_per_sample_CDM.json"
WORKERS = int(os.getenv("OMNIDOCBENCH_CDM_WORKERS", "4"))
OUTPUT_ROOT = os.path.abspath("result/paddleocrvl_rocm_cdm_quick_match_display_formula/CDM")

print(f"[cdm-only] workers={WORKERS} output_root={OUTPUT_ROOT}")

samples = json.load(open(INPUT, encoding="utf-8"))
print(f"[cdm-only] loaded {len(samples)} samples")

if os.path.isdir(OUTPUT_ROOT):
    shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def process_one(idx_sample):
    idx, sample = idx_sample
    sample = copy.deepcopy(sample)
    sample["img_id_cdm"] = str(idx)

    gt_cdm = sample.get("gt_cdm", sample["gt"])
    pred_cdm = sample.get("pred_cdm", sample["pred"])
    gt_cdm = _strip_cdm_math_wrappers(gt_cdm)
    pred_cdm = pred_cdm.split("```latex")[-1].split("```")[0]
    pred_cdm = _strip_cdm_math_wrappers(pred_cdm)

    pred_cdm_alt = sample.get("pred_cdm_alt", "")
    if pred_cdm_alt:
        pred_cdm_alt = pred_cdm_alt.split("```latex")[-1].split("```")[0]
        pred_cdm_alt = _strip_cdm_math_wrappers(pred_cdm_alt)

    sample["gt_cdm"] = gt_cdm
    sample["pred_cdm"] = pred_cdm
    if pred_cdm_alt:
        sample["pred_cdm_alt"] = pred_cdm_alt

    try:
        cal = CDM(output_root=OUTPUT_ROOT)
        metrics = cal.evaluate(gt_cdm, pred_cdm, str(idx))
        cdm_score = metrics["F1_score"]
        cdm_error = metrics.get("cdm_eval_error")

        if pred_cdm_alt:
            metrics_alt = cal.evaluate(gt_cdm, pred_cdm_alt, str(idx) + "_alt")
            if metrics_alt["F1_score"] > cdm_score:
                cdm_score = metrics_alt["F1_score"]
                sample["pred_cdm"] = pred_cdm_alt
                cdm_error = metrics_alt.get("cdm_eval_error")

        if "metric" not in sample:
            sample["metric"] = {}
        sample["metric"]["CDM"] = cdm_score

        return {
            "idx": idx,
            "sample": sample,
            "cdm_score": cdm_score,
            "sample_key": sample["img_id"] + "_" + str(sample["gt_idx"]),
            "error": cdm_error,
        }
    except Exception as exc:
        if "metric" not in sample:
            sample["metric"] = {}
        sample["metric"]["CDM"] = 0.0
        return {
            "idx": idx,
            "sample": sample,
            "cdm_score": 0.0,
            "sample_key": sample["img_id"] + "_" + str(sample["gt_idx"]),
            "error": f"{type(exc).__name__}: {exc}",
        }


if __name__ == "__main__":
    worker_args = list(enumerate(samples))
    per_sample_score = {}
    cdm_samples = [None] * len(samples)
    exception_count = 0

    if WORKERS <= 1:
        for args in tqdm(worker_args, ascii=True, ncols=140, desc="CDM"):
            r = process_one(args)
            cdm_samples[r["idx"]] = r["sample"]
            per_sample_score[r["sample_key"]] = r["cdm_score"]
            if r["error"]:
                exception_count += 1
    else:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(process_one, a): a[0] for a in worker_args}
            for future in tqdm(
                as_completed(futures), total=len(worker_args), ascii=True, ncols=140, desc="CDM"
            ):
                r = future.result()
                cdm_samples[r["idx"]] = r["sample"]
                per_sample_score[r["sample_key"]] = r["cdm_score"]
                if r["error"]:
                    exception_count += 1

    with open(OUTPUT_RESULT, "w", encoding="utf-8") as f:
        json.dump(cdm_samples, f, indent=4, ensure_ascii=False)
    with open(OUTPUT_CDM, "w", encoding="utf-8") as f:
        json.dump(per_sample_score, f, indent=4, ensure_ascii=False)

    scores = [v for v in per_sample_score.values() if isinstance(v, (int, float))]
    nonzero = [s for s in scores if s > 0]
    mean_cdm = sum(nonzero) / len(nonzero) if nonzero else 0
    print(
        f"\n[cdm-only] DONE! total={len(scores)} nonzero={len(nonzero)} "
        f"zero={len(scores) - len(nonzero)} exceptions={exception_count} mean={mean_cdm:.4f}"
    )
