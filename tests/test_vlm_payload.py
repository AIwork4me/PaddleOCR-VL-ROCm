from __future__ import annotations

from paddleocr_vl_rocm.pipeline_core import _completion_payload


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
