from __future__ import annotations

import json
import platform
from pathlib import Path

import onnxruntime as ort
import pytest

from paddleocr_vl_rocm.layout import PPDocLayoutV3Onnx, resolve_layout_providers
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


def _layout_replay_kwargs(meta: dict) -> dict:
    provider_keys = {"layout_provider_requested", "layout_providers_active"}
    present_keys = provider_keys.intersection(meta)
    if not present_keys:
        return {}
    if present_keys != provider_keys:
        raise RuntimeError("recorded layout provider metadata is incomplete")

    requested = meta["layout_provider_requested"]
    recorded_active = list(meta["layout_providers_active"])
    if not recorded_active:
        raise RuntimeError("recorded layout provider metadata has no active provider")
    providers = resolve_layout_providers(
        ort.get_available_providers(), requested, platform.system()
    )
    if providers[0] != recorded_active[0]:
        raise RuntimeError(
            "resolved layout provider does not match recorded layout provider: "
            f"{providers[0]} != {recorded_active[0]}"
        )
    layout_model = PPDocLayoutV3Onnx(
        Path(meta["layout_model"]),
        providers=providers,
        requested_provider=requested,
    )
    active = list(layout_model.layout_providers_active)
    if not active or active[0] != recorded_active[0]:
        actual = active[0] if active else "<none>"
        raise RuntimeError(
            "active layout provider does not match recorded layout provider: "
            f"{actual} != {recorded_active[0]}"
        )
    return {
        "layout_model": layout_model,
        "layout_provider_requested": requested,
        "layout_providers_active": active,
    }


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


@pytest.fixture
def _require_fixtures():
    meta = _load_meta()
    if not COMPAT.exists() or meta is None:
        pytest.skip("characterization fixtures not recorded; run scripts/record_trace.py")
    layout = Path(meta["layout_model"])
    if not layout.exists():
        pytest.skip(f"recorded layout model not found: {layout}")


def test_layout_replay_uses_recorded_provider_contract(monkeypatch):
    captured = {}

    class FakeLayout:
        def __init__(self, model_dir, providers, requested_provider):
            captured["model_dir"] = model_dir
            captured["providers"] = list(providers)
            captured["requested_provider"] = requested_provider
            self.layout_provider_requested = requested_provider
            self.layout_providers_active = ["DmlExecutionProvider"]

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setitem(globals(), "PPDocLayoutV3Onnx", FakeLayout)
    meta = {
        "layout_model": "models/PP-DocLayoutV3-onnx",
        "layout_provider_requested": "auto",
        "layout_providers_active": ["DmlExecutionProvider", "CPUExecutionProvider"],
    }

    kwargs = _layout_replay_kwargs(meta)

    assert captured["providers"] == ["DmlExecutionProvider"]
    assert captured["requested_provider"] == "auto"
    assert kwargs["layout_model"].layout_providers_active == ["DmlExecutionProvider"]
    assert kwargs["layout_provider_requested"] == "auto"
    assert kwargs["layout_providers_active"] == ["DmlExecutionProvider"]


def test_layout_replay_preserves_legacy_metadata_fallback():
    assert _layout_replay_kwargs({"layout_model": "models/PP-DocLayoutV3-onnx"}) == {}


def test_layout_replay_rejects_active_provider_contract_mismatch(monkeypatch):
    class FakeLayout:
        def __init__(self, model_dir, providers, requested_provider):
            self.layout_provider_requested = requested_provider
            self.layout_providers_active = ["CPUExecutionProvider"]

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setitem(globals(), "PPDocLayoutV3Onnx", FakeLayout)
    meta = {
        "layout_model": "models/PP-DocLayoutV3-onnx",
        "layout_provider_requested": "auto",
        "layout_providers_active": ["DmlExecutionProvider"],
    }

    with pytest.raises(RuntimeError, match="recorded layout provider"):
        _layout_replay_kwargs(meta)


@pytest.mark.parametrize("image", IMAGES, ids=[p.stem for p in IMAGES])
def test_pipeline_matches_golden(tmp_path, image, _require_fixtures):
    meta = _load_meta()
    layout_kwargs = _layout_replay_kwargs(meta)
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
        **layout_kwargs,
    )
    actual = json.loads(json_path.read_text(encoding="utf-8"))
    expected = json.loads((GOLDEN / f"{image.stem}.json").read_text(encoding="utf-8"))
    _assert_json_close(actual, expected)
