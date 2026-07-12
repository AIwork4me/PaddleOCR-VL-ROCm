from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path

import pytest

MANIFEST_PATH = Path(__file__).parent / "fixtures" / "accuracy" / "v16-root-cause-cases.json"
TRACE_SUMMARY_PATH = (
    Path(__file__).parent / "fixtures" / "accuracy" / "v16-trace-capture-summary.json"
)
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
EXPECTED_CASE_IDENTITIES = {
    **{("formula_cdm", page): identity for page, identity in EXPECTED_FORMULA_IDENTITIES.items()},
    ("table_teds", "yanbaor2_yanbaoPPT_2098.jpg"): (
        [1],
        "23ffc5b34c9ee1ff3309a40548949a11e3dbef5081e2beffef6a2cf39693ed69",
    ),
    (
        "table_teds",
        "yanbaor2_3e1be78252e2fdfe1adf12bba38ec2a7b30699e152d61269aa6e5827f5adcc35.pdf_13.jpg",
    ): ([4], "8d4aa75b0bc094c65dc5b74e2e5c9d63c475778f2a4abd7fd91d819a9eb1abfd"),
    ("table_teds", "docstructbench_llm-raw-the-eye-o.O-TraneGuide.pdf_7.jpg"): (
        [1],
        "d78b2d89582d518b731f43d3a6cde495a1cddeceda1a0139af710b01c66f4a7c",
    ),
    (
        "table_teds",
        "eastmoney_ea59610b9b1a8f0df46f7a89da1116cbf256c772e1148f26017991e28c8bca21.pdf_18.jpg",
    ): ([3], "88865964d0b7d99d756c5f6190121cb60fbaed70a7e69943543677c101bdc871"),
    ("table_teds", "docstructbench_enbook-zlib-o.O-17761417.pdf_894.jpg"): (
        [12],
        "9645f2a786a6461a1b154424ab9508b38a74ee1a627123f6f40ef103a4066f7e",
    ),
    ("text_edit", "jiaocaineedrop_jiaocai_needrop_en_349.jpg"): (
        [57],
        "15fb723bc8818e037bab6ec80be4c171efa39dbe1c8e0ec96300f14c293bb069",
    ),
    ("text_edit", "PPT_lecture1_page_005.png"): (
        [4],
        "c8c025666a0cbc5cbe62cb27b4b677841001c0d66c8fcd83dbafa6e844d19832",
    ),
    ("text_edit", "magazine_TheEconomist.2023.12.09_page_048.png"): (
        [20],
        "60e839d400f5cb7eca0b31920a9b0cef815e5e8f755d6e1a16a917662ec6df33",
    ),
    ("text_edit", "page-2329f04a-41b3-435b-993a-a0652294b07d.png"): (
        [4],
        "80fab0c28f3d24ed95ca3267477e215af5f5fe1ef96a23c95963b26c25fd36d5",
    ),
    (
        "text_edit",
        "docstructbench_llm-raw-the-eye-o.O-Player%27s%20Options%20-%20Halflings.pdf_11.jpg",
    ): ([10], "f70f7923ce19f43703c1e2ace719c20c720ff7bd7a15088a4d79837b842ede80"),
    ("reading_order", "page-2329f04a-41b3-435b-993a-a0652294b07d.png"): (
        [],
        "c7136c8c22b6359af65a758ee1690117bc8a81bbb288e26bd57969157a5c45f4",
    ),
    ("reading_order", "page-21967f5d-667d-488e-a5b3-76b9d6f53656.png"): (
        [],
        "819a96a922c4cbdbcdb6bd97912b650c0c3512309ef476fee8ef579817a5d1a8",
    ),
    ("reading_order", "page-268266af-56c0-4b3b-9d07-73c6e50feb58.png"): (
        [],
        "177058ec0d2c019bac7aec5cf242db94bd714cddaef9695c871580fe35890749",
    ),
    ("reading_order", "page-4319d401-c9e8-4326-9869-7572cf2e0e96.png"): (
        [],
        "823afc90ba02222b0999d713e476b47b96e650fc69c8e3502148dc51f7f609d7",
    ),
    (
        "reading_order",
        "color_textbook_\u6559\u6750\u5168\u89e31+1\u4e8c\u5e74\u7ea7\u4e0b\u518c\u82f1\u8bed\u4e0a\u6d77\u725b\u6d25\u7248_page_006.png",
    ): ([], "27bed167feebda34670629d51900ab3d188ea591d93893360ef5203342374ef2"),
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _fingerprint(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


MANIFEST_KEYS = {"schema", "trace_contract", "cases"}
TRACE_CONTRACT_KEYS = {"layout_providers_active_first", "allow_cpu_fallback"}
CASE_KEYS = {
    "case_id",
    "component",
    "page",
    "source_identity",
    "metadata",
    "official_score",
    "lightweight_score",
    "page_delta",
    "required_boundaries",
}
SOURCE_IDENTITY_KEYS = {"gt_position", "gt_sha256"}
METADATA_KEYS = {"gt_idx"}
GT_IDX_KEYS = {"official", "lightweight"}


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


def _assert_index(value: object, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    assert isinstance(value, list)
    assert all(type(item) is int and item >= 0 for item in value)


def _assert_manifest_schema(manifest: object) -> None:
    assert isinstance(manifest, dict)
    assert set(manifest) == MANIFEST_KEYS
    assert type(manifest["schema"]) is int and manifest["schema"] == 1

    contract = manifest["trace_contract"]
    assert isinstance(contract, dict)
    assert set(contract) == TRACE_CONTRACT_KEYS
    assert contract["layout_providers_active_first"] == "DmlExecutionProvider"
    assert type(contract["allow_cpu_fallback"]) is bool

    cases = manifest["cases"]
    assert isinstance(cases, list)
    for case in cases:
        assert isinstance(case, dict)
        assert set(case) == CASE_KEYS
        assert isinstance(case["case_id"], str) and SHA256_PATTERN.fullmatch(case["case_id"])
        assert case["component"] in EXPECTED_DELTAS
        assert isinstance(case["page"], str) and case["page"]

        identity = case["source_identity"]
        assert isinstance(identity, dict)
        assert set(identity) == SOURCE_IDENTITY_KEYS
        _assert_index(identity["gt_position"])
        assert isinstance(identity["gt_sha256"], str)
        assert SHA256_PATTERN.fullmatch(identity["gt_sha256"])

        metadata = case["metadata"]
        assert isinstance(metadata, dict) and set(metadata) == METADATA_KEYS
        gt_idx = metadata["gt_idx"]
        assert isinstance(gt_idx, dict) and set(gt_idx) == GT_IDX_KEYS
        _assert_index(gt_idx["official"], nullable=True)
        _assert_index(gt_idx["lightweight"], nullable=True)

        for key in ("official_score", "lightweight_score", "page_delta"):
            value = case[key]
            assert type(value) in (int, float)
            assert math.isfinite(value) and 0 <= value <= 1
        boundaries = case["required_boundaries"]
        assert isinstance(boundaries, list)
        assert boundaries == REQUIRED_BOUNDARIES
        assert all(isinstance(boundary, str) for boundary in boundaries)


def _assert_boundary_fingerprints(boundaries: object, *, allow_unobservable: bool) -> None:
    assert isinstance(boundaries, dict)
    assert set(boundaries) == set(REQUIRED_BOUNDARIES)
    for boundary in REQUIRED_BOUNDARIES:
        observation = boundaries[boundary]
        assert isinstance(observation, dict)
        status = observation.get("status")
        if allow_unobservable and status == "unobservable":
            assert set(observation) == {"status"}
            continue
        assert set(observation) == {"status", "fingerprint"}
        assert status == "observable"
        fingerprint_value = observation["fingerprint"]
        assert isinstance(fingerprint_value, str) and fingerprint_value
        assert SHA256_PATTERN.fullmatch(fingerprint_value)


def _assert_trace_summary_schema(summary: object, manifest: dict[str, object]) -> None:
    assert isinstance(summary, dict)
    assert set(summary) == {"schema", "benchmark_version", "evidence_sources", "cases"}
    assert type(summary["schema"]) is int and summary["schema"] == 1
    assert summary["benchmark_version"] == "1.6"

    sources = summary["evidence_sources"]
    assert isinstance(sources, dict)
    assert set(sources) == {"manifest", "lightweight_capture_set", "official_scorer_artifacts"}
    assert sources["manifest"] == {
        "path": "v16-root-cause-cases.json",
        "sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
    }
    capture_set = sources["lightweight_capture_set"]
    assert isinstance(capture_set, dict)
    assert set(capture_set) == {"artifact_count", "sha256"}
    assert type(capture_set["artifact_count"]) is int and capture_set["artifact_count"] == 20
    assert isinstance(capture_set["sha256"], str)
    assert SHA256_PATTERN.fullmatch(capture_set["sha256"])
    artifacts = sources["official_scorer_artifacts"]
    assert isinstance(artifacts, list) and len(artifacts) == 4
    expected_roles = {"formula_cdm", "table_teds", "text_edit", "reading_order"}
    for artifact in artifacts:
        assert isinstance(artifact, dict)
        assert set(artifact) == {"role", "filename", "sha256"}
        assert artifact["role"] in expected_roles
        assert (
            artifact["filename"]
            == {
                "formula_cdm": "paddleocr_official_local_llamacpp_gguf_v16_quick_match_display_formula_result.json",
                "table_teds": "paddleocr_official_local_llamacpp_gguf_v16_quick_match_table_result.json",
                "text_edit": "paddleocr_official_local_llamacpp_gguf_v16_quick_match_text_block_result.json",
                "reading_order": "paddleocr_official_local_llamacpp_gguf_v16_quick_match_reading_order_result.json",
            }[artifact["role"]]
        )
        assert isinstance(artifact["sha256"], str)
        assert SHA256_PATTERN.fullmatch(artifact["sha256"])
    assert {artifact["role"] for artifact in artifacts} == expected_roles

    cases = summary["cases"]
    assert isinstance(cases, list)
    expected_case_ids = {case["case_id"] for case in manifest["cases"]}
    assert len(cases) == len(expected_case_ids)
    assert {case["case_id"] for case in cases} == expected_case_ids
    assert [case["case_id"] for case in cases] == [case["case_id"] for case in manifest["cases"]]

    for case in cases:
        assert isinstance(case, dict)
        assert set(case) == {"case_id", "component", "lightweight", "official"}
        manifest_case = next(
            item for item in manifest["cases"] if item["case_id"] == case["case_id"]
        )
        assert case["component"] == manifest_case["component"]
        lightweight = case["lightweight"]
        assert isinstance(lightweight, dict)
        assert set(lightweight) == {
            "layout_provider_requested",
            "layout_providers_active",
            "layout_fallback_disabled",
            "trace_artifact_sha256",
            "boundaries",
        }
        assert lightweight["layout_provider_requested"] == "auto"
        assert lightweight["layout_providers_active"] == [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert lightweight["layout_fallback_disabled"] is True
        assert isinstance(lightweight["trace_artifact_sha256"], str)
        assert SHA256_PATTERN.fullmatch(lightweight["trace_artifact_sha256"])
        _assert_boundary_fingerprints(lightweight["boundaries"], allow_unobservable=False)

        official = case["official"]
        assert isinstance(official, dict) and set(official) == {"boundaries"}
        _assert_boundary_fingerprints(official["boundaries"], allow_unobservable=True)
        assert official["boundaries"]["final_output"]["status"] == "observable"

    capture_contract = [
        {
            "case_id": case["case_id"],
            "trace_artifact_sha256": case["lightweight"]["trace_artifact_sha256"],
        }
        for case in cases
    ]
    assert _fingerprint(capture_contract) == capture_set["sha256"]


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


def test_manifest_locks_every_canonical_source_identity() -> None:
    cases = _load_manifest()["cases"]
    identities = {
        (case["component"], case["page"]): (
            case["source_identity"]["gt_position"],
            case["source_identity"]["gt_sha256"],
        )
        for case in cases
    }

    assert identities == EXPECTED_CASE_IDENTITIES


def test_manifest_is_scalar_only_strict_schema() -> None:
    _assert_manifest_schema(_load_manifest())


def test_manifest_requires_directml_first_without_cpu_fallback() -> None:
    contract = _load_manifest()["trace_contract"]

    assert contract["layout_providers_active_first"] == "DmlExecutionProvider"
    assert contract["allow_cpu_fallback"] is False


def _synthetic_trace_summary() -> dict[str, object]:
    manifest = _load_manifest()
    cases = []
    for case in manifest["cases"]:
        boundaries = {
            boundary: {"status": "observable", "fingerprint": "a" * 64}
            for boundary in REQUIRED_BOUNDARIES
        }
        cases.append(
            {
                "case_id": case["case_id"],
                "component": case["component"],
                "lightweight": {
                    "layout_provider_requested": "auto",
                    "layout_providers_active": [
                        "DmlExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                    "layout_fallback_disabled": True,
                    "trace_artifact_sha256": "a" * 64,
                    "boundaries": deepcopy(boundaries),
                },
                "official": {"boundaries": deepcopy(boundaries)},
            }
        )
    return {
        "schema": 1,
        "benchmark_version": "1.6",
        "evidence_sources": {
            "manifest": {
                "path": "v16-root-cause-cases.json",
                "sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
            },
            "lightweight_capture_set": {
                "artifact_count": 20,
                "sha256": _fingerprint(
                    [
                        {"case_id": case["case_id"], "trace_artifact_sha256": "a" * 64}
                        for case in cases
                    ]
                ),
            },
            "official_scorer_artifacts": [
                {"role": role, "filename": filename, "sha256": "a" * 64}
                for role, filename in {
                    "formula_cdm": "paddleocr_official_local_llamacpp_gguf_v16_quick_match_display_formula_result.json",
                    "table_teds": "paddleocr_official_local_llamacpp_gguf_v16_quick_match_table_result.json",
                    "text_edit": "paddleocr_official_local_llamacpp_gguf_v16_quick_match_text_block_result.json",
                    "reading_order": "paddleocr_official_local_llamacpp_gguf_v16_quick_match_reading_order_result.json",
                }.items()
            ],
        },
        "cases": cases,
    }


def test_committed_trace_summary_authenticates_all_canonical_captures() -> None:
    summary = json.loads(TRACE_SUMMARY_PATH.read_text(encoding="utf-8"))
    _assert_trace_summary_schema(summary, _load_manifest())


def test_trace_summary_accepts_all_canonical_cases_and_boundaries() -> None:
    _assert_trace_summary_schema(_synthetic_trace_summary(), _load_manifest())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda summary: summary["cases"].pop(),
        lambda summary: summary["cases"][0]["lightweight"]["boundaries"]["crop"].update(
            fingerprint=""
        ),
        lambda summary: summary["cases"][0]["lightweight"].update(
            api_key="credential-must-not-be-published"
        ),
        lambda summary: summary["cases"][0]["lightweight"].update(
            layout_providers_active=["CPUExecutionProvider", "DmlExecutionProvider"]
        ),
        lambda summary: summary["cases"][0]["official"]["boundaries"]["final_output"].clear(),
        lambda summary: summary["cases"][0]["official"]["boundaries"]["final_output"].update(
            status="unobservable"
        ),
        lambda summary: summary["cases"][0].update(raw_response="must-not-publish"),
        lambda summary: summary["cases"][0]["lightweight"]["boundaries"]["crop"].update(
            fingerprint="A" * 64
        ),
        lambda summary: summary.update(schema=True),
    ],
    ids=[
        "missing-case",
        "empty-fingerprint",
        "credential-key",
        "cpu-first",
        "missing-official-final",
        "unobservable-official-final",
        "raw-content-key",
        "uppercase-hash",
        "boolean-schema",
    ],
)
def test_trace_summary_rejects_incomplete_or_unqualified_evidence(mutation) -> None:
    summary = _synthetic_trace_summary()
    mutation(summary)

    with pytest.raises(AssertionError):
        _assert_trace_summary_schema(summary, _load_manifest())
