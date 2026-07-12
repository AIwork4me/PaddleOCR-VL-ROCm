from __future__ import annotations

from threading import Event
from unittest.mock import Mock, patch

from PIL import Image

from paddleocr_vl_rocm.contracts import VLMRequestContract
from paddleocr_vl_rocm.encoding import _jpeg_bytes, _png_bytes, _sha256_hex
from paddleocr_vl_rocm.layout import LayoutBox
from paddleocr_vl_rocm.pipeline_core import run_light_parser
from paddleocr_vl_rocm.vlm.client import (
    OpenAICompatibleVLMClient,
    _completion_payload,
    _vlm_cache_key,
)


def test_complete_image_observer_receives_exact_llama_cpp_payload():
    observed = []
    response = Mock(status_code=200)
    response.json.return_value = {"choices": [{"message": {"content": "result"}}]}

    client = OpenAICompatibleVLMClient(
        "http://localhost:8080",
        "PaddleOCR-VL-1.6-GGUF.gguf",
        timeout=30.0,
        request_observer=observed.append,
    )
    with patch("paddleocr_vl_rocm.vlm.client.requests.post", return_value=response) as post:
        result = client.complete_image(
            "OCR:",
            image=Image.new("RGB", (3, 2), "white"),
            max_new_tokens=4096,
        )

    assert result == "result"
    assert len(observed) == 1
    request = observed[0]
    assert request.backend == "llama-cpp-server"
    assert request.model == "PaddleOCR-VL-1.6-GGUF.gguf"
    assert request.prompt == "OCR:"
    assert request.image_format == "PNG"
    assert request.image_size == (3, 2)
    assert request.payload == post.call_args.kwargs["json"]
    assert request.payload["temperature"] == 0.0
    assert request.payload["seed"] == 1
    assert request.payload["top_k"] == 1
    assert request.payload["top_p"] == 1.0
    assert request.payload["min_p"] == 0.0
    assert request.payload["repeat_penalty"] == 1.0
    assert request.payload["cache_prompt"] is False
    assert request.payload["max_tokens"] == 4096


def test_complete_image_observes_logical_request_on_memory_cache_hit():
    observed = []
    response = Mock(status_code=200)
    response.json.return_value = {"choices": [{"message": {"content": "cached"}}]}
    image = Image.new("RGB", (3, 2), "white")
    client = OpenAICompatibleVLMClient(
        "http://localhost:8080",
        "model.gguf",
        timeout=30.0,
        request_observer=observed.append,
    )

    with patch("paddleocr_vl_rocm.vlm.client.requests.post", return_value=response) as post:
        assert client.complete_image("OCR:", image=image, max_new_tokens=64) == "cached"
        assert client.complete_image("OCR:", image=image, max_new_tokens=64) == "cached"

    assert post.call_count == 1
    assert len(observed) == 2
    assert observed[1] == observed[0]


def test_complete_image_observes_logical_request_on_compat_cache_hit():
    observed = []
    image = Image.new("RGB", (3, 2), "white")
    image_sha256 = _sha256_hex(_png_bytes(image))
    cache_key = _vlm_cache_key("model.gguf", "OCR:", image_sha256, 64, 1)
    client = OpenAICompatibleVLMClient(
        "http://localhost:8080",
        "model.gguf",
        timeout=30.0,
        compat_cache={cache_key: "compat result"},
        request_observer=observed.append,
    )

    with patch("paddleocr_vl_rocm.vlm.client.requests.post") as post:
        result = client.complete_image("OCR:", image=image, max_new_tokens=64)

    assert result == "compat result"
    post.assert_not_called()
    assert len(observed) == 1
    assert observed[0].image_sha256 == image_sha256
    assert observed[0].payload["max_tokens"] == 64


def test_observer_redaction_copy_cannot_mutate_outgoing_payload():
    def mutate_observed(request):
        request.payload["model"] = "mutated"
        request.payload["messages"][0]["content"][1]["text"] = "mutated"

    response = Mock(status_code=200)
    response.json.return_value = {"choices": [{"message": {"content": "result"}}]}
    client = OpenAICompatibleVLMClient(
        "http://localhost:8080",
        "model.gguf",
        timeout=30.0,
        request_observer=mutate_observed,
    )

    with patch("paddleocr_vl_rocm.vlm.client.requests.post", return_value=response) as post:
        client.complete_image("OCR:", image=Image.new("RGB", (3, 2), "white"))

    outgoing = post.call_args.kwargs["json"]
    assert outgoing["model"] == "model.gguf"
    assert outgoing["messages"][0]["content"][1]["text"] == "OCR:"


def test_vllm_observer_receives_jpeg_metadata_and_exact_payload():
    observed = []
    response = Mock(status_code=200)
    response.json.return_value = {"choices": [{"message": {"content": "result"}}]}
    image = Image.new("RGB", (3, 2), "white")
    client = OpenAICompatibleVLMClient(
        "http://localhost:8080",
        "PaddleOCR-VL-1.5-0.9B",
        timeout=30.0,
        backend="vllm-server",
        request_observer=observed.append,
    )

    with patch("paddleocr_vl_rocm.vlm.client.requests.post", return_value=response) as post:
        client.complete_image(
            "OCR:",
            image=image,
            max_new_tokens=4096,
            min_pixels=112896,
            max_pixels=1003520,
        )

    request = observed[0]
    assert request.backend == "vllm-server"
    assert request.image_format == "JPEG"
    assert request.image_sha256 == _sha256_hex(_jpeg_bytes(image))
    assert request.image_size == (3, 2)
    assert request.payload == post.call_args.kwargs["json"]
    assert request.payload["max_completion_tokens"] == 4096
    assert request.payload["mm_processor_kwargs"] == {
        "min_pixels": 112896,
        "max_pixels": 1003520,
    }


def test_parallel_pipeline_observers_populate_their_own_trace_events(
    tmp_path, monkeypatch
):
    formula_observed = Event()

    class FakeLayout:
        def predict(self, *args, **kwargs):
            return (
                [
                    LayoutBox(0, "text", 1.0, [0, 0, 10, 10], 0),
                    LayoutBox(1, "formula", 1.0, [10, 0, 20, 10], 1),
                ],
                None,
            )

    class FakeClient:
        def __init__(self, *args, request_observer=None, **kwargs):
            self.request_observer = request_observer

        def complete_image(self, prompt, image, max_new_tokens, **kwargs):
            if prompt == "OCR:":
                assert formula_observed.wait(timeout=1.0)
            self.request_observer(
                VLMRequestContract(
                    backend="llama-cpp-server",
                    model="model.gguf",
                    prompt=prompt,
                    image_format="PNG",
                    image_sha256=f"sha-{prompt}",
                    image_size=image.size,
                    payload={
                        "max_tokens": max_new_tokens,
                        "skip_special_tokens": True,
                    },
                )
            )
            if prompt == "Formula Recognition:":
                formula_observed.set()
            return prompt

    monkeypatch.setattr("paddleocr_vl_rocm.pipeline_core.LlamaCppClient", FakeClient)
    image_path = tmp_path / "input.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    trace_events = []

    run_light_parser(
        input_path=image_path,
        output_dir=tmp_path / "output",
        model_dir=tmp_path,
        server_url="http://localhost:8080",
        vlm_backend="llama-cpp-server",
        api_model_name="model.gguf",
        max_new_tokens=64,
        timeout=30.0,
        prompt_label=None,
        use_layout_detection=True,
        use_chart_recognition=False,
        use_seal_recognition=False,
        seed=1,
        threshold=0.3,
        layout_model=FakeLayout(),
        skip_server_check=True,
        vlm_max_workers=2,
        vlm_trace_events=trace_events,
    )

    assert [event["request_order"] for event in trace_events] == [0, 1]
    assert trace_events[0]["block_label"] == "text"
    assert trace_events[0]["prompt"] == "OCR:"
    assert trace_events[0]["image_sha256"] == "sha-OCR:"
    assert trace_events[1]["block_label"] == "formula"
    assert trace_events[1]["prompt"] == "Formula Recognition:"
    assert trace_events[1]["image_sha256"] == "sha-Formula Recognition:"


def test_vllm_payload_matches_openai_compatible_shape():
    payload = _completion_payload(
        backend="vllm-server",
        model="PaddleOCR-VL-1.5-0.9B",
        prompt="OCR:",
        image_url="data:image/jpeg;base64,abc",
        max_new_tokens=4096,
        seed=1,
        min_pixels=112896,
        max_pixels=1003520,
    )

    assert payload["model"] == "PaddleOCR-VL-1.5-0.9B"
    assert payload["temperature"] == 0.0
    assert payload["skip_special_tokens"] is True
    assert payload["max_completion_tokens"] == 4096
    assert payload["mm_processor_kwargs"] == {"min_pixels": 112896, "max_pixels": 1003520}
    assert payload["messages"][0]["content"][0]["type"] == "image_url"
    assert payload["messages"][0]["content"][1] == {"type": "text", "text": "OCR:"}


def test_llama_cpp_payload_uses_local_deterministic_sampling_controls():
    payload = _completion_payload(
        backend="llama-cpp-server",
        model="PaddleOCR-VL-1.6-GGUF.gguf",
        prompt="Formula Recognition:",
        image_url="data:image/png;base64,abc",
        max_new_tokens=4096,
        seed=1,
        min_pixels=112896,
        max_pixels=1003520,
    )

    assert payload["model"] == "PaddleOCR-VL-1.6-GGUF.gguf"
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 1
    assert payload["top_p"] == 1.0
    assert payload["top_k"] == 1
    assert payload["min_p"] == 0.0
    assert payload["repeat_penalty"] == 1.0
    assert payload["cache_prompt"] is False
    assert payload["skip_special_tokens"] is True
    assert payload["max_tokens"] == 4096
    assert "mm_processor_kwargs" not in payload
