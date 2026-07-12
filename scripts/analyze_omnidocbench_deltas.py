from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

COMPONENTS = (
    ("Formula CDM", "display_formula", "CDM", "*_display_formula_result.json"),
    ("Table TEDS", "table", "TEDS", "*_table_result.json"),
)
DELTA_DEFINITION = "official_score - lightweight_score"
V16_COVERAGE = {
    "Formula CDM": {"sample_count": 2352, "page_count": 313},
    "Table TEDS": {"sample_count": 665, "page_count": 458},
}


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


def _canonical_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _gt_fingerprint(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric metric, got {value!r}")
    return float(value)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer count, got {value!r}")
    return value


def _case_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row["component"]),
        str(row["img_id"]),
        _canonical_value(row.get("gt_position", "")),
        str(row.get("gt_fingerprint") or _gt_fingerprint(row.get("gt", ""))),
    )


def _error_key(component: str, img_id: object, gt_idx: object) -> tuple[str, str, object]:
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
                    key = _error_key(component, case["img_id"], case.get("gt_idx", ""))
                    errors[key].append(metadata)
    return errors


def load_component_samples(result_dir: Path) -> list[dict[str, object]]:
    """Load v1.6 Formula CDM and Table TEDS samples from a result directory."""
    error_metadata = _load_error_metadata(result_dir)
    samples: dict[tuple[str, str, str, str], dict[str, object]] = {}
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
                normalized = {
                    "component": component,
                    "img_id": str(row.get("img_id", "")),
                    "gt_idx": _gt_idx(row.get("gt_idx", "")),
                    "gt_position": row.get("gt_position", ""),
                    "gt_fingerprint": _gt_fingerprint(row.get("gt", "")),
                    "score": float(metric[metric_name]),
                    "gt": row.get("gt", ""),
                    "pred": row.get("pred", ""),
                    "error_metadata": error_metadata.get(
                        _error_key(component, row.get("img_id", ""), row.get("gt_idx", "")), []
                    ),
                }
                key = _case_key(normalized)
                if key in samples:
                    raise ValueError(f"Duplicate sample key {key!r} in {result_dir}")
                samples[key] = normalized
    return sorted(
        samples.values(),
        key=lambda row: (
            str(row["component"]),
            str(row["img_id"]),
            str(row["gt_position"]),
            str(row["gt_fingerprint"]),
        ),
    )


def _metric_value(payload: dict[str, object], component: str) -> float:
    if component == "Formula CDM":
        path = ("display_formula", "page", "CDM", "ALL")
    else:
        path = ("table", "page", "TEDS", "ALL")
    value: object = payload
    for field in path:
        if not isinstance(value, dict) or field not in value:
            raise ValueError(f"Missing {'/'.join(path)} in companion metric")
        value = value[field]
    return _number(value)


def validate_v16_component_coverage(
    result_dir: Path, samples: list[dict[str, object]]
) -> dict[str, object]:
    """Fail closed unless samples cover the fixed OmniDocBench v1.6 corpus."""
    by_component: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample in samples:
        by_component[str(sample["component"])].append(sample)

    reconstructed: dict[str, float] = {}
    for component, expected in V16_COVERAGE.items():
        component_samples = by_component.get(component, [])
        pages = {str(row["img_id"]) for row in component_samples}
        if (
            len(component_samples) != expected["sample_count"]
            or len(pages) != expected["page_count"]
        ):
            raise ValueError(
                f"{component} coverage mismatch: samples={len(component_samples)}, "
                f"expected {expected['sample_count']}; pages={len(pages)}, "
                f"expected {expected['page_count']}"
            )
        scores_by_page: dict[str, list[float]] = defaultdict(list)
        for row in component_samples:
            scores_by_page[str(row["img_id"])].append(_number(row["score"]))
        reconstructed[component] = _mean(_mean(scores) for scores in scores_by_page.values())

    metric_paths = sorted(result_dir.glob("*_metric_result*.json"))
    metric_payloads = []
    for path in metric_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            metric_payloads.append((path, payload))
    if len(metric_payloads) != 1:
        raise ValueError(
            f"Expected exactly one companion metric result in {result_dir}, "
            f"found {len(metric_payloads)}"
        )
    metric_path, metric_payload = metric_payloads[0]
    for component, actual in reconstructed.items():
        recorded = _metric_value(metric_payload, component)
        if abs(actual - recorded) > 1e-12:
            raise ValueError(
                f"{component} reconstruction mismatch in {metric_path}: "
                f"reconstructed={actual}, recorded={recorded}"
            )
    return {
        component: {
            "sample_count": V16_COVERAGE[component]["sample_count"],
            "page_count": V16_COVERAGE[component]["page_count"],
            "reconstructed": reconstructed[component],
        }
        for component in V16_COVERAGE
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return _stable_float(sum(items) / len(items))


def _stable_float(value: float) -> float:
    return round(value, 15)


def _loss_sort_key(row: dict[str, object]) -> tuple[float, str, tuple[int, object], str]:
    return (
        -_number(row["delta"]),
        str(row["page"]),
        _sort_gt_idx(row.get("gt_idx", "")),
        str(row["component"]),
    )


def rank_deltas(
    reference: list[dict[str, object]], candidate: list[dict[str, object]]
) -> dict[str, object]:
    """Rank positive lightweight losses, with equal sample means within each page."""
    reference_by_key = {_case_key(row): row for row in reference}
    candidate_by_key = {_case_key(row): row for row in candidate}
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
                "gt_idx": official.get("gt_idx", ""),
                "lightweight_gt_idx": lightweight.get("gt_idx", ""),
                "gt_position": official.get("gt_position", ""),
                "gt_fingerprint": key[3],
                "official_score": _number(official["score"]),
                "lightweight_score": _number(lightweight["score"]),
                "delta": _stable_float(_number(official["score"]) - _number(lightweight["score"])),
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
            "delta": _mean(_number(row["delta"]) for row in rows),
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
            "delta": _mean(_number(row["delta"]) for row in pages),
            "page_count": len(pages),
            "sample_count": sum(_integer(row["sample_count"]) for row in pages),
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


def _key_sort(key: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
    return key


def _key_record(key: tuple[str, str, str, str]) -> dict[str, object]:
    return {
        "component": key[0],
        "page": key[1],
        "gt_position": json.loads(key[2]),
        "gt_fingerprint": key[3],
    }


def _limit_report(report: dict[str, object], top: int) -> dict[str, object]:
    limited = dict(report)
    ranked_pages = report["ranked_pages"]
    ranked_samples = report["ranked_samples"]
    if not isinstance(ranked_pages, list) or not isinstance(ranked_samples, list):
        raise ValueError("Ranked report must contain page and sample lists")
    limited["ranked_pages"] = ranked_pages[:top]
    limited["ranked_samples"] = ranked_samples[:top]
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

    official_samples = load_component_samples(args.official_result_dir)
    lightweight_samples = load_component_samples(args.lightweight_result_dir)
    validate_v16_component_coverage(args.official_result_dir, official_samples)
    validate_v16_component_coverage(args.lightweight_result_dir, lightweight_samples)
    report = rank_deltas(official_samples, lightweight_samples)
    output = _limit_report(report, args.top)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
