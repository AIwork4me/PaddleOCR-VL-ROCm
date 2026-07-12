from __future__ import annotations

import json

from scripts.compare_inference_traces import compare_traces
from scripts.record_trace import build_parser as build_record_parser
from scripts.record_trace import write_trace_jsonl


def _event(**changes):
    event = {
        "request_order": 0,
        "label": "formula",
        "bbox": [1, 2, 3, 4],
        "image_sha256": "crop",
        "prompt": "Formula Recognition:",
        "payload_fingerprint": "payload",
        "raw_result_sha256": "raw",
        "final_result_sha256": "final",
    }
    event.update(changes)
    return event


def test_compare_traces_classifies_crop_before_payload():
    reference = [_event(image_sha256="a", payload_fingerprint="x")]
    candidate = [_event(image_sha256="b", payload_fingerprint="y")]

    report = compare_traces(reference, candidate)

    assert report["differences"][0]["first_divergence"] == "crop_pixels"


def test_compare_traces_classifies_postprocess_difference():
    reference = [{"raw_result_sha256": "a", "final_result_sha256": "b"}]
    candidate = [{"raw_result_sha256": "a", "final_result_sha256": "c"}]

    report = compare_traces(reference, candidate)

    assert report["differences"][0]["first_divergence"] == "postprocess"


def test_compare_traces_uses_the_exact_difference_precedence():
    categories = [
        ("request_order", {"request_order": 1}),
        ("label", {"label": "text"}),
        ("bbox", {"bbox": [4, 3, 2, 1]}),
        ("crop_pixels", {"image_sha256": "other"}),
        ("prompt", {"prompt": "OCR:"}),
        ("payload", {"payload_fingerprint": "other"}),
        ("raw_result", {"raw_result_sha256": "other"}),
        ("postprocess", {"final_result_sha256": "other"}),
    ]

    for index, (expected, change) in enumerate(categories):
        candidate = _event(**change)
        for _, later_change in categories[index + 1 :]:
            candidate.update(later_change)
        report = compare_traces([_event()], [candidate])
        assert report["differences"][0]["first_divergence"] == expected


def test_compare_traces_reports_request_count_before_shared_event_differences():
    report = compare_traces([_event(), _event(request_order=1)], [_event(label="text")])

    assert report["differences"][0]["first_divergence"] == "request_count"
    assert report["summary"]["request_count"] == 1
    assert report["summary"]["label"] == 1
    assert report["reference_count"] == 2
    assert report["candidate_count"] == 1


def test_compare_traces_accepts_pipeline_block_field_names_and_payloads():
    reference = [_event()]
    candidate = [
        {
            "request_order": 0,
            "block_label": "formula",
            "block_bbox": [1, 2, 3, 4],
            "image_sha256": "crop",
            "prompt": "Formula Recognition:",
            "payload": {"model": "model.gguf", "temperature": 0.0},
            "raw_result_sha256": "raw",
            "final_result_sha256": "final",
        }
    ]
    reference[0].pop("payload_fingerprint")
    reference[0]["payload"] = {"temperature": 0.0, "model": "model.gguf"}

    report = compare_traces(reference, candidate)

    assert report["difference_count"] == 0
    assert report["differences"] == []


def test_compare_traces_limits_details_but_counts_every_difference():
    reference = [_event(request_order=index) for index in range(105)]
    candidate = [_event(request_order=index, label="text") for index in range(105)]

    report = compare_traces(reference, candidate)

    assert report["difference_count"] == 105
    assert report["summary"]["label"] == 105
    assert len(report["differences"]) == 100
    assert report["details_truncated"] is True


def test_record_trace_defaults_to_v16_llama_cpp():
    args = build_record_parser().parse_args([])

    assert args.api_model_name == "PaddleOCR-VL-1.6-GGUF.gguf"
    assert args.vlm_backend == "llama-cpp-server"
    assert args.trace_jsonl is None


def test_write_trace_jsonl_is_deterministic_redacted_and_preserves_fields(tmp_path):
    trace_path = tmp_path / "nested" / "trace.jsonl"
    events = [
        {
            "request_order": 0,
            "block_label": "formula",
            "payload": {
                "temperature": 0.0,
                "Authorization": "Bearer secret",
            },
            "url": "https://host/v1?token=secret",
            "raw_result_sha256": "raw",
            "final_result_sha256": "final",
        }
    ]

    write_trace_jsonl(trace_path, events)

    raw_line = trace_path.read_text(encoding="utf-8").splitlines()[0]
    written = json.loads(raw_line)
    assert raw_line == json.dumps(written, ensure_ascii=False, sort_keys=True)
    assert written["request_order"] == 0
    assert written["block_label"] == "formula"
    assert written["raw_result_sha256"] == "raw"
    assert written["final_result_sha256"] == "final"
    assert written["payload"]["Authorization"] == "<redacted>"
    assert written["url"].endswith("token=%3Credacted%3E")
    assert written["payload_fingerprint"]
