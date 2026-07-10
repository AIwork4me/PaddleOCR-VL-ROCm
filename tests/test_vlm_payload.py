from __future__ import annotations

from paddleocr_vl_rocm.vlm.client import _completion_payload


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
