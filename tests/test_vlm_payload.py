from __future__ import annotations

from unittest.mock import Mock, patch

from PIL import Image

from paddleocr_vl_rocm.vlm.client import OpenAICompatibleVLMClient, _completion_payload


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
