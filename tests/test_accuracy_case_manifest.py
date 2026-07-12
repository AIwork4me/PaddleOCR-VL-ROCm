from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent / "fixtures" / "accuracy" / "v16-root-cause-cases.json"
REQUIRED_BOUNDARIES = ["layout", "crop", "payload", "raw_vlm", "final_output"]
EXPECTED_DELTAS = {
    "formula_cdm": [
        0.904,
        0.2,
        0.1635,
        0.07966666666666655,
        0.013930555555555557,
    ],
    "table_teds": [
        0.3409420289855073,
        0.15384615384615397,
        0.14681114064407297,
        0.03286526420581071,
        0.030612244897959218,
    ],
    "text_edit": [
        0.2251210600925081,
        0.12666666666666668,
        0.07355387174274626,
        0.06564501150780636,
        0.06352941176470588,
    ],
    "reading_order": [
        0.2857142857142857,
        0.18518518518518517,
        0.16666666666666666,
        0.14285714285714285,
        0.13333333333333336,
    ],
}
EXPECTED_FORMULA_IDENTITIES = {
    "page-7dfc88d8-6d95-446c-b910-2410e8552f76.png": (
        [1],
        "472997a99cd7471e82aa3781aca9f04ba48e9ed4f1514a4742884eaa0a03cce6",
    ),
    "page-dad0f4e5-290f-496f-bbdd-099ad75c6ff0.png": (
        [15],
        "3127eff1948cabfc4ca288b0e2e02771987311ff57878c7f1ebc29462b578a03",
    ),
    "page-05746fc5-2045-4dea-94e7-4bbab648d702.png": (
        [12],
        "5c43e0e70103cb48cb999db06912b97b8ed3d186ae7ba4c903e620185134e983",
    ),
    "book_en_\u56fd\u5916\u6570\u5b66\u6559\u6750-\u6570\u8bba-Melvyn B. Nathanson\u2014Elementary Methods in Number Theory_0451.png": (
        [6],
        "dbe2694121056e76d1dd1d4b4dddf5348a8641dc0dbb68764558bd79ba3a9113",
    ),
    "yanbaopptmerge_9081a70ff98b3e7d640660a9412c447d.pdf_1287.jpg": (
        [52],
        "895c8f562a8cb3ced41db058e51e16b95d03d9205a8efb62b35dda635ed8c130",
    ),
}


def _load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _case_id(case: dict[str, object]) -> str:
    identity = case["source_identity"]
    assert isinstance(identity, dict)
    canonical_position = json.dumps(
        identity["gt_position"], ensure_ascii=False, separators=(",", ":")
    )
    value = "\0".join(
        (
            str(case["component"]),
            str(case["page"]),
            canonical_position,
            str(identity["gt_sha256"]),
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_manifest_locks_five_distinct_scalar_cases_per_component() -> None:
    manifest = _load_manifest()

    assert manifest["schema"] == 1
    cases = manifest["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 20

    for component, expected_deltas in EXPECTED_DELTAS.items():
        component_cases = [case for case in cases if case["component"] == component]
        assert len(component_cases) == 5
        assert len({case["page"] for case in component_cases}) == 5
        assert [case["page_delta"] for case in component_cases] == expected_deltas

    for case in cases:
        assert case["case_id"] == _case_id(case)
        assert case["required_boundaries"] == REQUIRED_BOUNDARIES
        assert isinstance(case["official_score"], (int, float))
        assert isinstance(case["lightweight_score"], (int, float))
        assert isinstance(case["page_delta"], (int, float))
        assert set(case["metadata"]) == {"gt_idx"}
        assert set(case["metadata"]["gt_idx"]) == {"official", "lightweight"}


def test_manifest_uses_canonical_formula_source_identity() -> None:
    cases = _load_manifest()["cases"]
    formula_cases = {
        case["page"]: (
            case["source_identity"]["gt_position"],
            case["source_identity"]["gt_sha256"],
        )
        for case in cases
        if case["component"] == "formula_cdm"
    }

    assert formula_cases == EXPECTED_FORMULA_IDENTITIES


def test_manifest_contains_no_raw_evidence() -> None:
    manifest = _load_manifest()
    forbidden_keys = {"gt", "prediction", "pred", "response", "trace", "raw"}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(manifest)


def test_manifest_requires_directml_first_without_cpu_fallback() -> None:
    contract = _load_manifest()["trace_contract"]

    assert contract["layout_providers_active_first"] == "DmlExecutionProvider"
    assert contract["allow_cpu_fallback"] is False
