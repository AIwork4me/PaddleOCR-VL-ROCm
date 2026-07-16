#!/usr/bin/env python3
"""Formula CDM root cause analysis."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

CATEGORIES = {
    "EMPTY_PRED": "Empty/missing prediction from model",
    "ZERO_TOKENS": "Prediction produces zero renderable tokens",
    "FULL_STRUCTURE_MISMATCH": "Matrix/array structure differs",
    "NOTATION_STYLE": "Equivalent math, different notation",
    "CONTENT_DIFF": "Actual math content differs",
    "EXTRA_TEXT": "Prediction includes non-formula text artifacts",
    "MISSING_CONTENT": "Prediction missing significant content",
    "RENDER_DIFF": "Same LaTeX, different visual rendering",
    "OTHER": "Other / uncategorized",
}

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def build_key(entry):
    img = entry.get("img_id", entry.get("image_name", ""))
    gt_pos = str(entry.get("gt_position", ""))
    return f"{img}_{gt_pos}"

def categorize_pair(pair):
    roc_pred = pair["roc_pred"]
    gt = pair["gt"]
    if not roc_pred.strip() or roc_pred.strip() in ("$$$$", "\[\]"):
        return "EMPTY_PRED"
    if len(roc_pred) < 0.3 * len(gt):
        return "MISSING_CONTENT"
    for kw in ["Ignore", "ignore", "The", "This"]:
        if kw in roc_pred and kw not in gt:
            return "EXTRA_TEXT"
    if pair["roc_cdm"] == 0.0 and pair["gguf_cdm"] > 0:
        if pair.get("roc_gt_tokens", 0) == 0:
            return "ZERO_TOKENS"
        return "OTHER"
    if "\\\\" in gt or "\\\\" in roc_pred:
        gt_rows = gt.count("\\\\")
        roc_rows = roc_pred.count("\\\\")
        if abs(gt_rows - roc_rows) > 2:
            return "FULL_STRUCTURE_MISMATCH"
    if pair["roc_edit"] > 0.1:
        return "CONTENT_DIFF"
    return "NOTATION_STYLE"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roc-result", required=True, type=Path)
    parser.add_argument("--gguf-result", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--out-worst", type=Path, default=None)
    args = parser.parse_args()

    roc_data = load_json(args.roc_result)
    gguf_data = load_json(args.gguf_result)

    roc_by_key = {}
    for e in roc_data:
        roc_by_key[build_key(e)] = e

    gguf_by_key = {}
    for e in gguf_data:
        gguf_by_key[build_key(e)] = e

    common = set(roc_by_key) & set(gguf_by_key)
    print(f"Common formulas: {len(common)}")
    print(f"ROCm-only: {len(roc_by_key) - len(common)}")
    print(f"GGUF-only: {len(gguf_by_key) - len(common)}")

    pairs = []
    for key in sorted(common):
        r = roc_by_key[key]
        g = gguf_by_key[key]
        r_cdm = r.get("metric", {}).get("CDM", 0.0)
        g_cdm = g.get("metric", {}).get("CDM", 0.0)
        pairs.append({
            "img_id": r.get("img_id", ""),
            "gt_position": r.get("gt_position", ""),
            "gt": r.get("gt", ""),
            "roc_pred": r.get("pred", ""),
            "gguf_pred": g.get("pred", ""),
            "roc_cdm": r_cdm,
            "gguf_cdm": g_cdm,
            "cdm_diff": g_cdm - r_cdm,
            "roc_edit": r.get("edit", 0.0),
            "gguf_edit": g.get("edit", 0.0),
            "roc_gt_tokens": r.get("metric", {}).get("gt_tokens", 0),
            "roc_pred_tokens": r.get("metric", {}).get("pred_tokens", 0),
            "gguf_gt_tokens": g.get("metric", {}).get("gt_tokens", 0),
            "gguf_pred_tokens": g.get("metric", {}).get("pred_tokens", 0),
        })

    # Categorize
    cats = defaultdict(list)
    cat_loss = defaultdict(float)
    total_loss = 0.0
    for p in pairs:
        cat = categorize_pair(p)
        cats[cat].append(p)
        loss = p["cdm_diff"]
        cat_loss[cat] += loss
        total_loss += loss

    gguf_cdms = [p["gguf_cdm"] for p in pairs if p["gguf_cdm"] > 0]
    roc_cdms_nonzero = [p["roc_cdm"] for p in pairs if p["roc_cdm"] > 0]

    summary = {
        "overall": {
            "gguf_cdm_mean": round(statistics.mean(gguf_cdms), 6) if gguf_cdms else 0,
            "roc_nonzero_cdm_mean": round(statistics.mean(roc_cdms_nonzero), 6) if roc_cdms_nonzero else 0,
            "roc_zero_cdm_count": sum(1 for p in pairs if p["roc_cdm"] == 0.0),
            "cdm_gap": round(statistics.mean(gguf_cdms), 6) if gguf_cdms else 0,
            "total_common_pairs": len(pairs),
            "total_cdm_loss_sum": round(total_loss, 4),
        },
        "by_category": {},
        "worst_cases": sorted(pairs, key=lambda p: p["cdm_diff"], reverse=True)[:50],
    }

    for cat in sorted(cats):
        cp = cats[cat]
        loss = cat_loss[cat]
        pct = (loss / total_loss * 100) if total_loss > 0 else 0
        summary["by_category"][cat] = {
            "count": len(cp),
            "description": CATEGORIES.get(cat, cat),
            "cdm_loss_sum": round(loss, 4),
            "cdm_loss_pct": round(pct, 2),
            "avg_cdm_loss": round(loss / len(cp), 4) if cp else 0,
            "examples": [
                {
                    "img_id": p["img_id"],
                    "gt_position": p["gt_position"],
                    "roc_cdm": p["roc_cdm"],
                    "gguf_cdm": p["gguf_cdm"],
                    "roc_edit": p["roc_edit"],
                    "gt_preview": p["gt"][:150],
                    "roc_pred_preview": p["roc_pred"][:150],
                }
                for p in sorted(cp, key=lambda x: x["cdm_diff"], reverse=True)[:5]
            ],
        }

    output = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(output + "\n", encoding="utf-8")
        print(f"Written to {args.out}")

    if args.out_worst:
        with open(args.out_worst, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "img_id", "gt_position", "roc_cdm", "gguf_cdm", "cdm_diff",
                "roc_edit", "gguf_edit", "gt", "roc_pred",
            ])
            writer.writeheader()
            for p in summary["worst_cases"]:
                writer.writerow({
                    "img_id": p["img_id"], "gt_position": str(p["gt_position"]),
                    "roc_cdm": p["roc_cdm"], "gguf_cdm": p["gguf_cdm"],
                    "cdm_diff": p["cdm_diff"], "roc_edit": p["roc_edit"],
                    "gguf_edit": p["gguf_edit"], "gt": p["gt"],
                    "roc_pred": p["roc_pred"],
                })
        print(f"Worst CSV written to {args.out_worst}")

    print(output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
