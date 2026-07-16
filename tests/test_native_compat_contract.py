from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from paddleocr_vl_rocm import cli
from paddleocr_vl_rocm.cli import build_parser
from paddleocr_vl_rocm.pipeline import PaddleOCRVLROCm
from paddleocr_vl_rocm.pipeline_core import run_light_parser
from paddleocr_vl_rocm.result import PaddleOCRVLROCmResult

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
CONTRACT = json.loads(
    (FIXTURES / "contracts" / "v16-native-output.json").read_text(encoding="utf-8")
)
GOLDEN_STEM = "handwrite_ch_demo"
GOLDEN_JSON = FIXTURES / "golden" / f"{GOLDEN_STEM}.json"
GOLDEN_MARKDOWN = FIXTURES / "golden" / f"{GOLDEN_STEM}.md"
COMPAT_CACHE = FIXTURES / "compat_cache.json"
RECORD_META = FIXTURES / "record_meta.json"

JSON_TYPES = {
    "string": str,
    "integer": int,
    "object": dict,
    "array": list,
}


def test_public_defaults_match_v16_contract():
    defaults = CONTRACT["defaults"]
    args = build_parser().parse_args(["--input", "input.png"])
    parameters = inspect.signature(PaddleOCRVLROCm.__init__).parameters

    for name, expected in defaults.items():
        assert getattr(args, name) == expected
        assert parameters[name].default == expected


def test_cli_keeps_explicit_vllm_backend_supported():
    args = build_parser().parse_args(
        [
            "--input",
            "input.png",
            "--api-model-name",
            "PaddleOCR-VL-1.5-0.9B",
            "--vlm-backend",
            "vllm-server",
        ]
    )

    assert args.api_model_name == "PaddleOCR-VL-1.5-0.9B"
    assert args.vlm_backend == "vllm-server"


def test_cli_forwards_contract_seed_to_pipeline(monkeypatch, tmp_path):
    captured = {}

    class FakeResult:
        def print(self):
            pass

        def save_to_json(self, output):
            return Path(output) / "input_res.json"

        def save_to_markdown(self, output, pretty=False):
            return Path(output) / "input.md"

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def predict(self, input_path):
            return FakeResult()

    monkeypatch.setattr(cli, "PaddleOCRVLROCm", FakePipeline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paddleocr-vl-rocm",
            "--input",
            "input.png",
            "--output",
            str(tmp_path),
            "--skip-server-check",
        ],
    )

    cli.main()

    assert captured["seed"] == CONTRACT["defaults"]["seed"]


def test_static_contract_check_cannot_skip_without_external_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "exists", lambda _path: False)
    skipped = False

    try:
        test_committed_golden_matches_static_v16_contract(tmp_path)
    except pytest.skip.Exception:
        skipped = True

    assert skipped is False, "checkout-independent contract assertions must not skip"


def test_committed_golden_matches_static_v16_contract(tmp_path):
    golden = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))

    for key in CONTRACT["json_keys"]:
        assert key in golden
        assert type(golden[key]) is JSON_TYPES[CONTRACT["json_types"][key]]
    for key, expected in CONTRACT["model_settings"].items():
        actual = golden["model_settings"][key]
        if expected == "array":
            assert isinstance(actual, list)
        else:
            assert actual is expected
    assert hashlib.sha256(GOLDEN_MARKDOWN.read_bytes()).hexdigest() == CONTRACT["markdown_sha256"]

    result = PaddleOCRVLROCmResult(golden, GOLDEN_MARKDOWN.read_text(encoding="utf-8"))
    saved_json = result.save_to_json(tmp_path / "saved")
    saved_markdown = result.save_to_markdown(tmp_path / "saved", pretty=False)
    assert saved_json.name.endswith(CONTRACT["json_suffix"])
    assert saved_markdown.name.endswith(CONTRACT["markdown_suffix"])
    assert json.loads(saved_json.read_text(encoding="utf-8")) == golden
    assert saved_markdown.read_bytes() == GOLDEN_MARKDOWN.read_bytes()


def _assert_json_close(actual, expected, path="$"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected object"
        assert actual.keys() == expected.keys(), f"{path}: keys differ"
        for key in expected:
            _assert_json_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected array"
        assert len(actual) == len(expected), f"{path}: length differs"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_json_close(actual_item, expected_item, f"{path}[{index}]")
        return
    if isinstance(expected, float) or isinstance(actual, float):
        assert actual == pytest.approx(expected, abs=1e-4), path
        return
    assert actual == expected, path


def test_native_replay_optionally_matches_committed_golden(tmp_path):
    golden = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
    input_path = Path(golden["input_path"])
    recorded_root = input_path.parents[2]
    meta = json.loads(RECORD_META.read_text(encoding="utf-8"))
    layout_model = recorded_root / meta["layout_model"]
    if not input_path.exists() or not layout_model.exists():
        pytest.skip("recorded input or layout model is unavailable")

    replay_dir = tmp_path / "replay"
    json_path = run_light_parser(
        input_path=input_path,
        output_dir=replay_dir,
        model_dir=layout_model,
        server_url=meta["server_url"],
        vlm_backend=meta["vlm_backend"],
        api_model_name=meta["api_model_name"],
        max_new_tokens=CONTRACT["defaults"]["max_new_tokens"],
        timeout=300.0,
        prompt_label=None,
        use_layout_detection=True,
        use_chart_recognition=False,
        use_seal_recognition=False,
        seed=CONTRACT["defaults"]["seed"],
        threshold=CONTRACT["defaults"]["threshold"],
        compat_cache_path=COMPAT_CACHE,
        display_input_path=str(input_path),
        skip_server_check=True,
        vlm_max_workers=CONTRACT["defaults"]["vlm_max_workers"],
    )

    _assert_json_close(json.loads(json_path.read_text(encoding="utf-8")), golden)
    assert (replay_dir / "result.md").read_bytes() == GOLDEN_MARKDOWN.read_bytes()
