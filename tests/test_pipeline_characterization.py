from __future__ import annotations

import json
from pathlib import Path

import pytest

from paddleocr_vl_rocm.pipeline_core import run_light_parser

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
GOLDEN = FIXTURES / "golden"
COMPAT = FIXTURES / "compat_cache.json"
LAYOUT_MODEL = REPO / "models" / "PP-DocLayoutV3-onnx"
IMAGES = sorted((REPO / "examples" / "input").glob("*.png"))


@pytest.fixture(autouse=True)
def _require_fixtures():
    if not COMPAT.exists() or not LAYOUT_MODEL.exists():
        pytest.skip("compat cache or layout model not present; run scripts/record_trace.py")


@pytest.mark.parametrize("image", IMAGES, ids=[p.stem for p in IMAGES])
def test_pipeline_matches_golden(tmp_path, image):
    json_path = run_light_parser(
        input_path=image,
        output_dir=tmp_path,
        model_dir=LAYOUT_MODEL,
        server_url="http://127.0.0.1:8000/v1",
        vlm_backend="vllm-server",
        api_model_name="PaddleOCR-VL-1.5-0.9B",
        max_new_tokens=4096,
        timeout=300.0,
        prompt_label=None,
        use_layout_detection=True,
        use_chart_recognition=False,
        use_seal_recognition=False,
        seed=1,
        threshold=0.3,
        compat_cache_path=COMPAT,
        display_input_path=str(image),
        skip_server_check=True,
    )
    actual = json.loads(json_path.read_text(encoding="utf-8"))
    expected = json.loads((GOLDEN / f"{image.stem}.json").read_text(encoding="utf-8"))
    assert actual == expected
