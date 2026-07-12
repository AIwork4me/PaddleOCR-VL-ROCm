from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

COMPONENTS = (
    ("Formula CDM", "display_formula", "CDM", "*_display_formula_result.json"),
    ("Table TEDS", "table", "TEDS", "*_table_result.json"),
)
DELTA_DEFINITION = "official_score - lightweight_score"


def _gt_idx(value: object) -> object:
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _sort_gt_idx(value: object) -> tuple[int, object]:
    if isinstance(value, bool):
        return (2, str(value))
    if isinstance(value, (int, float)):
        return (0, value)
    return (1, str(value))


def _case_key(component: str, img_id: object, gt_idx: object) -> tuple[str, str, object]:
    return component, str(img_id), _gt_idx(gt_idx)


def _load_error_metadata(
    result_dir: Path,
) -> dict[tuple[str, str, object], list[dict[str, object]]]:
    errors: dict[tuple[str, str, object], list[dict[str, object]]] = defaultdict(list)
    for path in sorted(result_dir.glob("*_metric_result*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for component, element, metric_name, _ in COMPONENTS:
            element_payload = payload.get(element, {})
            if not isinstance(element_payload, dict):
                continue
            metric_debug = element_payload.get("metric_debug", {})
            debug = metric_debug.get(metric_name, {}) if isinstance(metric_debug, dict) else {}
            if not isinstance(debug, dict):
                continue
            for field, kind in (
                ("timeout_cases", "timeout"),
                ("error_cases", "error"),
                ("exception_cases", "exception"),
            ):
                cases = debug.get(field, [])
                if not isinstance(cases, list):
                    continue
                for case in cases:
                    if not isinstance(case, dict) or not case.get("img_id"):
                        continue
                    metadata = {"kind": kind, **case}
                    key = _case_key(component, case["img_id"], case.get("gt_idx", ""))
                    errors[key].append(metadata)
    return errors


def load_component_samples(result_dir: Path) -> list[dict[str, object]]:
    """Load v1.6 Formula CDM and Table TEDS samples from a result directory."""
    error_metadata = _load_error_metadata(result_dir)
    samples: dict[tuple[str, str, object], dict[str, object]] = {}
    for component, _, metric_name, pattern in COMPONENTS:
        for path in sorted(result_dir.glob(pattern)):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"Expected a list of samples in {path}")
            for row in payload:
                if not isinstance(row, dict):
                    continue
                metric = row.get("metric")
                if not isinstance(metric, dict) or metric_name not in metric:
                    continue
                key = _case_key(component, row.get("img_id", ""), row.get("gt_idx", ""))
                if key in samples:
                    raise ValueError(f"Duplicate sample key {key!r} in {result_dir}")
                samples[key] = {
                    "component": component,
                    "img_id": key[1],
                    "gt_idx": key[2],
                    "score": float(metric[metric_name]),
                    "gt": row.get("gt", ""),
                    "pred": row.get("pred", ""),
                    "error_metadata": error_metadata.get(key, []),
                }
    return sorted(
        samples.values(),
        key=lambda row: (
            str(row["component"]),
            str(row["img_id"]),
            _sort_gt_idx(row["gt_idx"]),
        ),
    )


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return _stable_float(sum(items) / len(items))


def _stable_float(value: float) -> float:
    return round(value, 15)


def _loss_sort_key(row: dict[str, object]) -> tuple[float, str, tuple[int, object], str]:
    return (
        -float(row["delta"]),
        str(row["page"]),
        _sort_gt_idx(row.get("gt_idx", "")),
        str(row["component"]),
    )


def rank_deltas(
    reference: list[dict[str, object]], candidate: list[dict[str, object]]
) -> dict[str, object]:
    """Rank positive lightweight losses, with equal sample means within each page."""
    reference_by_key = {
        _case_key(row["component"], row["img_id"], row["gt_idx"]): row for row in reference
    }
    candidate_by_key = {
        _case_key(row["component"], row["img_id"], row["gt_idx"]): row for row in candidate
    }
    reference_only = sorted(set(reference_by_key) - set(candidate_by_key), key=_key_sort)
    candidate_only = sorted(set(candidate_by_key) - set(reference_by_key), key=_key_sort)

    ranked_samples: list[dict[str, object]] = []
    for key in sorted(set(reference_by_key) & set(candidate_by_key), key=_key_sort):
        official = reference_by_key[key]
        lightweight = candidate_by_key[key]
        ranked_samples.append(
            {
                "component": key[0],
                "page": key[1],
                "gt_idx": key[2],
                "official_score": float(official["score"]),
                "lightweight_score": float(lightweight["score"]),
                "delta": _stable_float(float(official["score"]) - float(lightweight["score"])),
                "gt": official.get("gt", lightweight.get("gt", "")),
                "official_prediction": official.get("pred", ""),
                "lightweight_prediction": lightweight.get("pred", ""),
                "official_error_metadata": official.get("error_metadata", []),
                "lightweight_error_metadata": lightweight.get("error_metadata", []),
            }
        )
    ranked_samples.sort(key=_loss_sort_key)

    samples_by_page: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in ranked_samples:
        samples_by_page[(str(row["component"]), str(row["page"]))].append(row)
    ranked_pages = [
        {
            "component": component,
            "page": page,
            "delta": _mean(float(row["delta"]) for row in rows),
            "sample_count": len(rows),
        }
        for (component, page), rows in samples_by_page.items()
    ]
    ranked_pages.sort(key=_loss_sort_key)

    pages_by_component: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in ranked_pages:
        pages_by_component[str(row["component"])].append(row)
    components = [
        {
            "component": component,
            "delta": _mean(float(row["delta"]) for row in pages),
            "page_count": len(pages),
            "sample_count": sum(int(row["sample_count"]) for row in pages),
        }
        for component, pages in sorted(pages_by_component.items())
    ]
    return {
        "delta_definition": DELTA_DEFINITION,
        "matched_sample_count": len(ranked_samples),
        "reference_only_keys": [_key_record(key) for key in reference_only],
        "lightweight_only_keys": [_key_record(key) for key in candidate_only],
        "components": components,
        "ranked_pages": ranked_pages,
        "ranked_samples": ranked_samples,
    }


def _key_sort(key: tuple[str, str, object]) -> tuple[str, str, tuple[int, object]]:
    return key[0], key[1], _sort_gt_idx(key[2])


def _key_record(key: tuple[str, str, object]) -> dict[str, object]:
    return {"component": key[0], "page": key[1], "gt_idx": key[2]}


def _limit_report(report: dict[str, object], top: int) -> dict[str, object]:
    limited = dict(report)
    limited["ranked_pages"] = list(report["ranked_pages"])[:top]
    limited["ranked_samples"] = list(report["ranked_samples"])[:top]
    limited["top"] = top
    return limited


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rank OmniDocBench v1.6 losses where delta is "
            "official_score - lightweight_score (positive means lightweight loss)."
        )
    )
    parser.add_argument("--official-result-dir", required=True, type=Path)
    parser.add_argument("--lightweight-result-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be at least 1")

    report = rank_deltas(
        load_component_samples(args.official_result_dir),
        load_component_samples(args.lightweight_result_dir),
    )
    output = _limit_report(report, args.top)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
