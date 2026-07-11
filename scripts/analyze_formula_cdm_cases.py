from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _coerce_case(page: str, item: dict[str, Any]) -> dict[str, object] | None:
    raw_cdm = item.get("CDM", item.get("cdm"))
    if raw_cdm is None:
        return None
    cdm = float(raw_cdm)
    return {
        "page": page,
        "sample_id": str(item.get("sample_id", item.get("id", ""))),
        "cdm": cdm,
        "gt": item.get("gt", item.get("ground_truth", "")),
        "pred": item.get("pred", item.get("prediction", "")),
    }


def _coerce_scalar_case(sample_key: str, value: int | float) -> dict[str, object]:
    page, separator, sample_id = sample_key.rpartition("_[")
    if not separator or not sample_id.endswith("]"):
        page = sample_key
        sample_id = ""
    else:
        sample_id = sample_id[:-1]
    return {
        "page": page,
        "sample_id": sample_id,
        "cdm": float(value),
        "gt": "",
        "pred": "",
    }


def load_formula_scores(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: list[dict[str, object]] = []
    if isinstance(data, dict):
        for page, value in data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        case = _coerce_case(str(page), item)
                        if case is not None:
                            cases.append(case)
            elif isinstance(value, dict):
                case = _coerce_case(str(page), value)
                if case is not None:
                    cases.append(case)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                cases.append(_coerce_scalar_case(str(page), value))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                page = str(item.get("page", item.get("image", "")))
                case = _coerce_case(page, item)
                if case is not None:
                    cases.append(case)
    return cases


def summarize_cases(cases: list[dict[str, object]], threshold: float) -> dict[str, object]:
    ranked = sorted(cases, key=lambda case: float(case["cdm"]))
    return {
        "count": len(cases),
        "below_threshold_count": sum(1 for case in cases if float(case["cdm"]) < threshold),
        "zero_count": sum(1 for case in cases if float(case["cdm"]) == 0.0),
        "lowest_cases": ranked[:50],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Formula CDM per-sample cases.")
    parser.add_argument("--per-sample-cdm", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cases = load_formula_scores(args.per_sample_cdm)
    summary = summarize_cases(cases, threshold=args.threshold)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
