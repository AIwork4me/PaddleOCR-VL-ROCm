import inspect
from pathlib import Path

import pytest
from PIL import Image

from paddleocr_vl_rocm import pipeline_core
from paddleocr_vl_rocm.pipeline import PaddleOCRVLROCm
from paddleocr_vl_rocm.pipeline_core import run_light_parser
from paddleocr_vl_rocm.timing import _covered_seconds, summarize_seconds


def test_summarize_seconds_uses_nearest_rank_percentiles():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]

    assert summarize_seconds(values) == {
        "count": 5,
        "mean": 22.0,
        "p50": 3.0,
        "p95": 100.0,
        "p99": 100.0,
        "max": 100.0,
    }


def test_summarize_seconds_empty():
    assert summarize_seconds([]) == {"count": 0}


def test_covered_seconds_merges_overlapping_intervals():
    assert _covered_seconds([(1.0, 3.0), (2.0, 4.0), (5.0, 6.0)]) == 4.0


def test_parser_does_not_read_clock_without_timing_observer(tmp_path, monkeypatch):
    image = Image.new("RGB", (2, 2), "white")
    monkeypatch.setattr(pipeline_core, "perf_counter", pytest.fail)
    monkeypatch.setattr(pipeline_core, "_open_crop_source", lambda _path: image)
    monkeypatch.setattr(pipeline_core, "_open_crop_source_bgr", lambda _path: None)
    monkeypatch.setattr(pipeline_core, "_make_blocks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(pipeline_core, "_result_payload", lambda *_args, **_kwargs: {})

    run_light_parser(
        input_path=tmp_path / "input.png",
        output_dir=tmp_path / "output",
        model_dir=Path("layout"),
        server_url="http://127.0.0.1:8111/v1",
        vlm_backend="llama-cpp-server",
        api_model_name="model.gguf",
        max_new_tokens=64,
        timeout=30.0,
        prompt_label="ocr",
        use_layout_detection=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        seed=1,
        threshold=0.3,
        skip_server_check=True,
    )


def test_timing_events_is_appended_after_existing_optional_parameters():
    parameters = list(inspect.signature(run_light_parser).parameters)

    assert parameters[-3:] == [
        "layout_provider_requested",
        "layout_providers_active",
        "timing_events",
    ]


def test_predict_clears_stale_timing_before_layout_initialization(tmp_path, monkeypatch):
    pipeline = PaddleOCRVLROCm()
    pipeline.last_timing = {"total_seconds": 1.0}
    monkeypatch.setattr(pipeline, "_layout", lambda: (_ for _ in ()).throw(RuntimeError("layout")))

    with pytest.raises(RuntimeError, match="layout"):
        pipeline.predict(tmp_path / "input.png")

    assert pipeline.last_timing is None
