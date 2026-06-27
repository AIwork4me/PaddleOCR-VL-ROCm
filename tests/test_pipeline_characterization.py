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


def _load_meta() -> dict | None:
    if not META.exists():
        return None
    return json.loads(META.read_text(encoding="utf-8"))


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
    assert actual == expected
