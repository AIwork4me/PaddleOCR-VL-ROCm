from __future__ import annotations

import json
from types import SimpleNamespace

from paddleocr_vl_rocm.contracts import fingerprint
from scripts import record_trace
from eval.task5_comparison import observation, unobservable
from scripts.compare_inference_traces import compare_boundary_documents, compare_traces
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


def test_canonical_comparison_prioritizes_fail_over_unknown():
    boundaries = {
        name: observation(name)
        for name in (
            "request_order",
            "label",
            "bbox",
            "crop_pixels",
            "prompt",
            "payload",
            "raw_result",
            "postprocess",
        )
    }
    reference = {"page": "page", "block_index": 0, "boundaries": dict(boundaries)}
    candidate = {"page": "page", "block_index": 0, "boundaries": dict(boundaries)}
    reference["boundaries"]["raw_result"] = unobservable()
    candidate["boundaries"]["postprocess"] = observation("different")

    report = compare_boundary_documents([reference], [candidate])

    assert report["verdict"] == "FAIL"
    assert report["first_divergence_counts"]["postprocess"] == 1


def test_record_trace_defaults_to_v16_llama_cpp():
    args = build_record_parser().parse_args([])

    assert args.api_model_name == "PaddleOCR-VL-1.6-GGUF.gguf"
    assert args.vlm_backend == "llama-cpp-server"
    assert args.layout_provider == "auto"
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
            "payload_fingerprint": "stale-before-redaction",
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
    assert written["payload_fingerprint"] == fingerprint(written["payload"])


def test_record_trace_resolves_and_propagates_actual_layout_providers(tmp_path, monkeypatch):
    captured = {}
    trace_path = tmp_path / "trace.jsonl"
    image_path = tmp_path / "input.png"
    image_path.touch()
    args = SimpleNamespace(
        server_url="http://127.0.0.1:8000/v1",
        api_model_name="model.gguf",
        vlm_backend="llama-cpp-server",
        layout_model=str(tmp_path / "layout"),
        layout_provider="auto",
        trace_jsonl=trace_path,
    )

    class FakeParser:
        def parse_args(self):
            return args

    class FakeLayout:
        def __init__(self, model_dir, providers, requested_provider):
            captured["model_dir"] = model_dir
            captured["providers"] = providers
            captured["requested_provider"] = requested_provider
            self.layout_provider_requested = requested_provider
            self.layout_providers_active = list(providers)

    def fake_run_light_parser(**kwargs):
        captured["run_kwargs"] = kwargs
        kwargs["vlm_trace_events"].append(
            {
                "request_order": 0,
                "payload": {"temperature": 0.0},
                "layout_provider_requested": kwargs["layout_provider_requested"],
                "layout_providers_active": kwargs["layout_providers_active"],
            }
        )
        result_path = kwargs["output_dir"] / "result.json"
        result_path.write_text("{}", encoding="utf-8")
        return result_path

    monkeypatch.setattr(record_trace, "build_parser", FakeParser)
    monkeypatch.setattr(record_trace, "IMAGES", [image_path])
    monkeypatch.setattr(record_trace, "FIXTURES", tmp_path / "fixtures")
    monkeypatch.setattr(record_trace, "GOLDEN", tmp_path / "fixtures" / "golden")
    monkeypatch.setattr(
        record_trace,
        "ort",
        SimpleNamespace(
            get_available_providers=lambda: [
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        record_trace,
        "platform",
        SimpleNamespace(system=lambda: "Windows"),
        raising=False,
    )
    monkeypatch.setattr(record_trace, "PPDocLayoutV3Onnx", FakeLayout, raising=False)
    monkeypatch.setattr(record_trace, "run_light_parser", fake_run_light_parser)

    record_trace.main()

    assert captured["providers"] == ["DmlExecutionProvider"]
    assert captured["requested_provider"] == "auto"
    assert captured["run_kwargs"]["layout_model"].layout_providers_active == [
        "DmlExecutionProvider"
    ]
    assert captured["run_kwargs"]["layout_provider_requested"] == "auto"
    assert captured["run_kwargs"]["layout_providers_active"] == ["DmlExecutionProvider"]
    written_events = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert written_events
    assert all(
        event["layout_provider_requested"] == "auto"
        and event["layout_providers_active"] == ["DmlExecutionProvider"]
        for event in written_events
    )
    record_meta = json.loads(
        (record_trace.FIXTURES / "record_meta.json").read_text(encoding="utf-8")
    )
    assert record_meta["layout_provider_requested"] == "auto"
    assert record_meta["layout_providers_active"] == ["DmlExecutionProvider"]
