from __future__ import annotations


def test_recorder_monkeypatch_target_exists():
    # scripts/record_trace.py patches this class.method; if the client ever moves,
    # this test fails loudly instead of the recorder silently breaking.
    from paddleocr_vl_rocm.vlm import client

    assert hasattr(client, "OpenAICompatibleVLMClient")
    assert hasattr(client.OpenAICompatibleVLMClient, "complete_image")
