from __future__ import annotations

import json
from pathlib import Path

import pytest

from paddleocr_vl_rocm.pipeline_core import run_light_parser

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
GOLDEN = FIXTURES / "golden"
COMPAT = FIXTURES / "compat_cache.json"
META = FIXTURES / "record_meta.json"
IMAGES = sorted((REPO / "examples" / "input").glob("*.png"))
FLOAT_TOLERANCE = 1e-4


def _load_meta() -> dict | None:
    if not META.exists():
        return None
    return json.loads(META.read_text(encoding="utf-8"))


def _assert_json_close(actual, expected, path="$"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict"
        assert actual.keys() == expected.keys(), f"{path}: keys differ"
        for key in expected:
            _assert_json_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list"
        assert len(actual) == len(expected), f"{path}: list length differs"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_json_close(actual_item, expected_item, f"{path}[{index}]")
        return
    if isinstance(expected, float) or isinstance(actual, float):
        assert actual == pytest.approx(expected, abs=FLOAT_TOLERANCE), path
        return
    assert actual == expected, path


@pytest.fixture(autouse=True)
def _require_fixtures():
    meta = _load_meta()
    if not COMPAT.exists() or meta is None:
        pytest.skip("characterization fixtures not recorded; run scripts/record_trace.py")
    layout = Path(meta["layout_model"])
    if not layout.exists():
        pytest.skip(f"recorded layout model not found: {layout}")


@pytest.mark.parametrize("image", IMAGES, ids=[p.stem for p in IMAGES])
def test_pipeline_matches_golden(tmp_path, image):
    meta = _load_meta()
    json_path = run_light_parser(
        input_path=image,
        output_dir=tmp_path,
        model_dir=Path(meta["layout_model"]),
        server_url=meta["server_url"],
        vlm_backend=meta["vlm_backend"],
        api_model_name=meta["api_model_name"],
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
    _assert_json_close(actual, expected)
