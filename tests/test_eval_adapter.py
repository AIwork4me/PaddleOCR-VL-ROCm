from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from eval.task5_comparison import BOUNDARIES
from paddleocr_vl_rocm.pipeline import PaddleOCRVLROCm


def _load_adapter():
    script = Path("eval/PaddleOCRVLROCm_img2md.py")
    spec = importlib.util.spec_from_file_location("paddleocrvl_rocm_img2md", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_md_name_strips_extension():
    mod = _load_adapter()
    # OmniDocBench matcher looks up <img_name[:-4]>.md first
    assert mod.expected_md_name("page_001.png") == "page_001.md"
    assert mod.expected_md_name("doc.jpeg") == "doc.md"


def test_image_extensions_lowercase():
    mod = _load_adapter()
    exts = {e.lower() for e in mod.IMAGE_EXTENSIONS}
    assert {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"} <= exts


def test_iter_images_applies_deterministic_limit(tmp_path):
    mod = _load_adapter()
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ["b.png", "a.jpg", "notes.txt", "c.jpeg"]:
        (img_dir / name).write_bytes(b"x")

    selected = mod.iter_images(img_dir, limit_pages=2)

    assert [p.name for p in selected] == ["a.jpg", "b.png"]


def test_run_adapter_passes_limit_pages_to_official_engine(tmp_path, monkeypatch):
    mod = _load_adapter()
    captured = {}

    def fake_official_folder(**kwargs):
        captured.update(kwargs)
        return {
            "count": 1,
            "ok": 1,
            "fail": 0,
            "fallback": 0,
            "engine": "official",
            "limit_pages": 1,
            "stats": [],
        }

    monkeypatch.setattr(mod, "run_official_folder", fake_official_folder, raising=False)

    summary = mod.run_adapter(
        tmp_path / "images",
        tmp_path / "predictions",
        "http://127.0.0.1:8111/v1",
        engine="official",
        limit_pages=1,
    )

    assert summary["engine"] == "official"
    assert captured["limit_pages"] == 1


def test_official_result_prefers_plain_markdown_export():
    mod = _load_adapter()

    class FakeOfficialResult:
        def __init__(self):
            self.calls = []
            self.markdown = "<div>pretty markdown</div>"

        def _to_markdown(self, pretty=True):
            self.calls.append(pretty)
            return {"markdown_texts": "plain markdown"}

    result = FakeOfficialResult()

    assert mod._official_result_to_markdown(result) == "plain markdown"
    assert result.calls == [False]


def test_official_markdown_html_wrappers_become_scorer_friendly_markdown():
    mod = _load_adapter()
    markdown = (
        '<div style="text-align: center;"><img src="imgs/table_1.jpg"></div>\n\n'
        '<div style="text-align: center;"><b>Figure 1 &amp; caption</b></div>'
    )

    assert mod._normalize_official_markdown_for_omnidocbench(markdown) == (
        "![](imgs/table_1.jpg)\n\nFigure 1 & caption"
    )


def test_official_markdown_html_wrapper_preserves_formula_text():
    mod = _load_adapter()
    markdown = (
        '<div style="text-align: center;">'
        '<span class="formula">$\\frac{1}{2} &amp; x^2$</span>'
        "</div>"
    )

    assert mod._normalize_official_markdown_for_omnidocbench(markdown) == ("$\\frac{1}{2} & x^2$")


def test_run_adapter_resolves_defaults_from_env_local(tmp_path, monkeypatch):
    mod = _load_adapter()
    adapter_dir = tmp_path / "adapter"
    repo_root = tmp_path / "repo"
    adapter_dir.mkdir()
    repo_root.mkdir()
    (adapter_dir / ".env.local").write_text(
        "\n".join(
            [
                "PP_DOCLAYOUTV3_ONNX_DIR='C:/models/layout'",
                "LLAMA_HOST='127.0.0.1'",
                "LLAMA_PORT='8111'",
                "VL_REC_API_MODEL_NAME='PaddleOCR-VL-1.6-GGUF.gguf'",
            ]
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_lightweight_folder(**kwargs):
        captured.update(kwargs)
        return {"count": 0, "ok": 0, "fail": 0, "engine": "lightweight", "stats": []}

    monkeypatch.setattr(mod, "ADAPTER_DIR", adapter_dir, raising=False)
    monkeypatch.setattr(mod, "REPO_ROOT", repo_root, raising=False)
    monkeypatch.setattr(mod, "run_lightweight_folder", fake_lightweight_folder, raising=False)

    summary = mod.run_adapter(tmp_path / "images", tmp_path / "predictions", "")

    assert summary["engine"] == "lightweight"
    assert captured["layout_model"] == "C:/models/layout"
    assert captured["server_url"] == "http://127.0.0.1:8111/v1"
    assert captured["api_model_name"] == "PaddleOCR-VL-1.6-GGUF.gguf"
    assert captured["vlm_backend"] == "llama-cpp-server"


def test_run_adapter_dispatches_official_engine(tmp_path, monkeypatch):
    mod = _load_adapter()
    captured = {}

    def fake_official_folder(**kwargs):
        captured.update(kwargs)
        return {"count": 1, "ok": 1, "fail": 0, "fallback": 0, "engine": "official", "stats": []}

    monkeypatch.setattr(mod, "run_official_folder", fake_official_folder, raising=False)

    summary = mod.run_adapter(
        tmp_path / "images",
        tmp_path / "predictions",
        "http://127.0.0.1:8111/v1",
        engine="official",
        page_retries=2,
        fallback_pred_dir=tmp_path / "fallback",
    )

    assert summary["engine"] == "official"
    assert captured["server_url"] == "http://127.0.0.1:8111/v1"
    assert captured["page_retries"] == 2
    assert captured["fallback_pred_dir"] == tmp_path / "fallback"


def test_lightweight_folder_writes_run_stats_and_error_log(tmp_path, monkeypatch):
    mod = _load_adapter()
    img_dir = tmp_path / "images"
    out_dir = tmp_path / "predictions"
    img_dir.mkdir()
    (img_dir / "ok.png").write_bytes(b"fake image")
    (img_dir / "bad.png").write_bytes(b"fake image")
    out_dir.mkdir()
    (out_dir / "bad.md").write_text("stale output", encoding="utf-8")

    class FakeResult:
        markdown_text = "recognized"

    class FakePipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.initialized = False
            self.layout_provider_requested = "auto"
            self.layout_providers_active = []
            self.last_timing = None

        def _layout(self):
            self.initialized = True
            self.layout_providers_active = [
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ]

        def predict(self, image_path):
            assert self.initialized
            if image_path.name == "bad.png":
                raise RuntimeError("controlled failure")
            self.last_timing = {
                "decode_seconds": 0.1,
                "layout_seconds": 0.2,
                "crop_encode_seconds": 0.3,
                "vlm_seconds": 0.4,
                "finalize_seconds": 0.5,
                "total_seconds": 1.5,
            }
            return FakeResult()

    monkeypatch.setattr(mod, "PaddleOCRVLROCm", FakePipeline, raising=False)

    summary = mod.run_lightweight_folder(
        img_dir=img_dir,
        out_dir=out_dir,
        layout_model="layout",
        server_url="http://127.0.0.1:8111/v1",
        api_model_name="PaddleOCR-VL-1.6-GGUF.gguf",
        vlm_backend="llama-cpp-server",
    )

    assert summary["count"] == 2
    assert summary["ok"] == 1
    assert summary["fail"] == 1
    assert summary["fallback"] == 0
    assert summary["layout_provider_requested"] == "auto"
    assert summary["layout_providers_active"] == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert (out_dir / "ok.md").read_text(encoding="utf-8") == "recognized"
    assert not (out_dir / "bad.md").exists()
    assert "controlled failure" in (out_dir / "_errors.log").read_text(encoding="utf-8")
    stats = json.loads((out_dir / "_run_stats.json").read_text(encoding="utf-8"))
    assert stats["engine"] == "lightweight"
    assert stats["fallback"] == 0
    assert stats["layout_provider_requested"] == "auto"
    assert stats["layout_providers_active"][0] == "DmlExecutionProvider"
    assert stats["timing"]["count"] == 1
    assert stats["stage_timing"] == {
        "decode_seconds": {"count": 1, "mean": 0.1, "p50": 0.1, "p95": 0.1, "p99": 0.1, "max": 0.1},
        "layout_seconds": {"count": 1, "mean": 0.2, "p50": 0.2, "p95": 0.2, "p99": 0.2, "max": 0.2},
        "crop_encode_seconds": {
            "count": 1,
            "mean": 0.3,
            "p50": 0.3,
            "p95": 0.3,
            "p99": 0.3,
            "max": 0.3,
        },
        "vlm_seconds": {"count": 1, "mean": 0.4, "p50": 0.4, "p95": 0.4, "p99": 0.4, "max": 0.4},
        "finalize_seconds": {
            "count": 1,
            "mean": 0.5,
            "p50": 0.5,
            "p95": 0.5,
            "p99": 0.5,
            "max": 0.5,
        },
        "total_seconds": {"count": 1, "mean": 1.5, "p50": 1.5, "p95": 1.5, "p99": 1.5, "max": 1.5},
    }


def test_lightweight_trace_capture_is_opt_in_and_fingerprint_only(tmp_path, monkeypatch):
    mod = _load_adapter()
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "page.png").write_bytes(b"image")

    class FakeResult:
        markdown_text = "markdown  \n"

    class FakePipeline:
        def __init__(self, **kwargs):
            self.layout_provider_requested = "auto"
            self.layout_providers_active = []
            self.last_timing = None

        def _layout(self):
            pass

        def predict(self, image_path, *, vlm_trace_events=None):
            if vlm_trace_events is not None:
                vlm_trace_events.append(
                    {
                        "request_order": 0,
                        "block_label": "text",
                        "block_bbox": [1, 2, 3, 4],
                        "image_sha256": "crop-secret",
                        "prompt": "prompt-secret",
                        "payload": {"token": "payload-secret", "model": "model"},
                        "raw_result_sha256": "raw-secret",
                        "final_result_sha256": "final-secret",
                    }
                )
            return FakeResult()

    monkeypatch.setattr(mod, "PaddleOCRVLROCm", FakePipeline, raising=False)
    baseline = tmp_path / "baseline"
    observed = tmp_path / "observed"
    trace_dir = tmp_path / "traces"

    mod.run_lightweight_folder(img_dir=img_dir, out_dir=baseline)
    mod.run_lightweight_folder(img_dir=img_dir, out_dir=observed, trace_dir=trace_dir)

    assert (baseline / "page.md").read_bytes() == (observed / "page.md").read_bytes()
    assert (baseline / "_run_stats.json").read_bytes() == (
        observed / "_run_stats.json"
    ).read_bytes()
    assert not (tmp_path / "disabled-traces").exists()
    raw = (trace_dir / "page.jsonl").read_text(encoding="utf-8")
    event = json.loads(raw)
    assert event["page"] == "page"
    assert event["block_index"] == 0
    assert set(event["boundaries"]) == set(BOUNDARIES)
    assert all(
        set(value) <= {"status", "fingerprint"}
        for value in event["boundaries"].values()
    )
    for secret in ("crop-secret", "prompt-secret", "payload-secret", "raw-secret", "final-secret"):
        assert secret not in raw


def test_lightweight_zero_vlm_events_write_explicit_page_record(tmp_path, monkeypatch):
    mod = _load_adapter()
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "page.png").write_bytes(b"image")

    class FakeResult:
        markdown_text = "markdown"

    class FakePipeline:
        def __init__(self, **kwargs):
            self.layout_provider_requested = "auto"
            self.layout_providers_active = []
            self.last_timing = None

        def _layout(self):
            pass

        def predict(self, image_path, *, vlm_trace_events=None):
            return FakeResult()

    monkeypatch.setattr(mod, "PaddleOCRVLROCm", FakePipeline, raising=False)
    traces = tmp_path / "traces"

    mod.run_lightweight_folder(img_dir=img_dir, out_dir=tmp_path / "out", trace_dir=traces)

    event = json.loads((traces / "page.jsonl").read_text(encoding="utf-8"))
    assert event["block_index"] is None
    assert event["block_structure"] == {"status": "unobservable"}


def test_predict_optionally_forwards_vlm_trace_events(monkeypatch, tmp_path):
    captured = {}
    image = tmp_path / "input.png"
    image.write_bytes(b"image")

    class FakeLayout:
        layout_provider_requested = "auto"
        layout_providers_active = []

    pipeline = PaddleOCRVLROCm(skip_server_check=True)
    monkeypatch.setattr(pipeline, "_layout", lambda: FakeLayout())

    def fake_run_light_parser(**kwargs):
        captured.update(kwargs)
        (kwargs["output_dir"] / "result.json").write_text("{}", encoding="utf-8")
        (kwargs["output_dir"] / "result.md").write_text("markdown", encoding="utf-8")
        return kwargs["output_dir"] / "result.json"

    monkeypatch.setattr("paddleocr_vl_rocm.pipeline.run_light_parser", fake_run_light_parser)
    events = []

    result = pipeline.predict(image, vlm_trace_events=events)

    assert captured["vlm_trace_events"] is events
    assert result.markdown_text == "markdown"


def test_official_page_trace_observes_only_direct_block_fields():
    mod = _load_adapter()
    result = {
        "res": {
            "parsing_res_list": [
                {
                    "block_label": "text",
                    "block_bbox": [1, 2, 3, 4],
                    "prompt": "prompt",
                    "payload": {"model": "model"},
                    "raw_result": "raw",
                    "block_content": "postprocessed",
                }
            ]
        }
    }

    events = mod._official_page_trace_events("page", result, "markdown")

    assert len(events) == 1
    event = events[0]
    assert event["page"] == "page"
    assert event["block_index"] == 0
    assert event["boundaries"]["label"]["status"] == "observable"
    assert event["boundaries"]["bbox"]["status"] == "observable"
    assert event["boundaries"]["prompt"]["status"] == "observable"
    assert event["boundaries"]["payload"]["status"] == "observable"
    assert event["boundaries"]["raw_result"]["status"] == "observable"
    assert event["boundaries"]["postprocess"]["status"] == "observable"
    assert event["boundaries"]["crop_pixels"] == {"status": "unobservable"}


def test_official_does_not_infer_request_order_prompt_or_none_fields():
    mod = _load_adapter()
    events = mod._official_page_trace_events(
        "page",
        {
            "res": {
                "parsing_res_list": [
                    {
                        "block_label": None,
                        "request": {"payload": {"model": "model"}},
                        "raw_result": None,
                    }
                ]
            }
        },
        "markdown",
    )

    boundaries = events[0]["boundaries"]
    assert boundaries["request_order"] == {"status": "unobservable"}
    assert boundaries["label"] == {"status": "unobservable"}
    assert boundaries["prompt"] == {"status": "unobservable"}
    assert boundaries["raw_result"] == {"status": "unobservable"}
    assert boundaries["payload"]["status"] == "observable"


def test_official_empty_authenticated_block_list_is_not_zero_evidence():
    mod = _load_adapter()

    events = mod._official_page_trace_events(
        "page", {"res": {"parsing_res_list": []}}, "markdown"
    )

    assert len(events) == 1
    assert events[0]["block_index"] is None
    assert events[0]["block_structure"] == {"status": "unobservable"}


def test_official_and_lightweight_content_hash_boundaries_are_comparable():
    mod = _load_adapter()
    raw = "same raw result"
    final = "same postprocess"

    def digest(value):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    lightweight = mod._lightweight_page_trace_events(
        "page",
        [
            {
                "request_order": 0,
                "block_label": "text",
                "block_bbox": [1, 2, 3, 4],
                "raw_result_sha256": digest(raw),
                "final_result_sha256": digest(final),
            }
        ],
        "markdown",
    )[0]
    official = mod._official_page_trace_events(
        "page",
        {
            "res": {
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_bbox": [1, 2, 3, 4],
                        "raw_result": raw,
                        "block_content": final,
                    }
                ]
            }
        },
        "markdown",
    )[0]

    assert official["boundaries"]["raw_result"] == lightweight["boundaries"]["raw_result"]
    assert official["boundaries"]["postprocess"] == lightweight["boundaries"]["postprocess"]


def test_crop_pixels_and_prehashed_image_sha256_use_one_representation():
    mod = _load_adapter()
    pixels = b"same crop pixels"
    digest = hashlib.sha256(pixels).hexdigest()
    lightweight = mod._lightweight_page_trace_events(
        "page", [{"image_sha256": digest}], "markdown"
    )[0]
    same_official = mod._official_page_trace_events(
        "page", {"res": {"parsing_res_list": [{"crop_pixels": pixels}]}}, "markdown"
    )[0]
    different_official = mod._official_page_trace_events(
        "page",
        {"res": {"parsing_res_list": [{"crop_pixels": b"different pixels"}]}},
        "markdown",
    )[0]

    assert same_official["boundaries"]["crop_pixels"] == lightweight["boundaries"]["crop_pixels"]
    assert (
        different_official["boundaries"]["crop_pixels"]
        != lightweight["boundaries"]["crop_pixels"]
    )


def test_trace_write_failure_removes_stale_and_partial_trace(tmp_path, monkeypatch):
    mod = _load_adapter()
    img_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    trace_dir = tmp_path / "traces"
    img_dir.mkdir()
    trace_dir.mkdir()
    (img_dir / "page.png").write_bytes(b"image")
    target = trace_dir / "page.jsonl"
    target.write_text("stale", encoding="utf-8")

    class FakeResult:
        markdown_text = "markdown"

    class FakePipeline:
        def __init__(self, **kwargs):
            self.layout_provider_requested = "auto"
            self.layout_providers_active = []
            self.last_timing = None

        def _layout(self):
            pass

        def predict(self, image_path, *, vlm_trace_events=None):
            vlm_trace_events.append({"request_order": 0})
            return FakeResult()

    monkeypatch.setattr(mod, "PaddleOCRVLROCm", FakePipeline, raising=False)
    monkeypatch.setattr(mod.os, "replace", lambda source, destination: (_ for _ in ()).throw(OSError("replace failed")))

    with pytest.raises(SystemExit):
        mod.run_lightweight_folder(img_dir=img_dir, out_dir=out_dir, trace_dir=trace_dir)

    assert not target.exists()
    assert list(trace_dir.iterdir()) == []


def test_official_fallback_removes_stale_trace(tmp_path, monkeypatch):
    mod = _load_adapter()
    img_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    fallback_dir = tmp_path / "fallback"
    trace_dir = tmp_path / "traces"
    img_dir.mkdir()
    fallback_dir.mkdir()
    trace_dir.mkdir()
    (img_dir / "page.png").write_bytes(b"image")
    (fallback_dir / "page.md").write_text("fallback", encoding="utf-8")
    stale = trace_dir / "page.jsonl"
    stale.write_text("stale", encoding="utf-8")

    class FailingOfficial:
        def __init__(self, **kwargs):
            pass

        def predict(self, image_path):
            raise RuntimeError("controlled")

    monkeypatch.setitem(
        sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCRVL=FailingOfficial)
    )

    summary = mod.run_official_folder(
        img_dir=img_dir,
        out_dir=out_dir,
        server_url="http://server/v1",
        api_model_name="model",
        page_retries=0,
        fallback_pred_dir=fallback_dir,
        trace_dir=trace_dir,
    )

    assert summary["fallback"] == 1
    assert not stale.exists()


@pytest.mark.parametrize("engine", ["lightweight", "official"])
def test_duplicate_image_stems_fail_before_inference(tmp_path, monkeypatch, engine):
    mod = _load_adapter()
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "page.png").write_bytes(b"image")
    (img_dir / "page.jpg").write_bytes(b"image")

    if engine == "lightweight":
        class ForbiddenPipeline:
            def __init__(self, **kwargs):
                raise AssertionError("inference initialized before stem validation")

        monkeypatch.setattr(mod, "PaddleOCRVLROCm", ForbiddenPipeline, raising=False)

        def call():
            return mod.run_lightweight_folder(img_dir=img_dir, out_dir=tmp_path / "out")
    else:
        class ForbiddenOfficial:
            def __init__(self, **kwargs):
                raise AssertionError("inference initialized before stem validation")

        monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCRVL=ForbiddenOfficial))

        def call():
            return mod.run_official_folder(
                img_dir=img_dir,
                out_dir=tmp_path / "out",
                server_url="http://server/v1",
                api_model_name="model",
            )

    with pytest.raises(ValueError, match="Duplicate output stem"):
        call()


def test_official_page_trace_without_authenticated_blocks_is_page_level_unknown():
    mod = _load_adapter()

    events = mod._official_page_trace_events("page", {"markdown": "body"}, "body\r\n")

    assert events == [mod.official_page_trace("page", {"markdown": "body"}, "body\r\n")]
    event = events[0]
    assert event["block_index"] is None
    assert event["block_structure"] == {"status": "unobservable"}
    assert all(value == {"status": "unobservable"} for value in event["boundaries"].values())
    assert event["page_postprocess"]["status"] == "observable"
    assert "body" not in json.dumps(event)


def test_official_folder_materializes_generator_results(tmp_path, monkeypatch):
    mod = _load_adapter()
    img_dir = tmp_path / "images"
    out_dir = tmp_path / "predictions"
    img_dir.mkdir()
    (img_dir / "page.png").write_bytes(b"fake image")

    class FakeOfficialResult:
        def __init__(self, markdown):
            self.markdown = markdown

    class FakeOfficialPipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, image_path):
            return (FakeOfficialResult(markdown) for markdown in ("first page", "second page"))

    monkeypatch.setitem(
        sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCRVL=FakeOfficialPipeline)
    )

    summary = mod.run_official_folder(
        img_dir=img_dir,
        out_dir=out_dir,
        server_url="http://127.0.0.1:8111/v1",
        api_model_name="PaddleOCR-VL-1.6-GGUF.gguf",
    )

    assert summary["ok"] == 1
    assert (out_dir / "page.md").read_text(encoding="utf-8") == "first page\n\nsecond page"


def test_official_folder_preserves_same_directory_fallback_after_failure(tmp_path, monkeypatch):
    mod = _load_adapter()
    img_dir = tmp_path / "images"
    out_dir = tmp_path / "predictions"
    img_dir.mkdir()
    out_dir.mkdir()
    (img_dir / "page.png").write_bytes(b"fake image")
    fallback_markdown = "existing fallback prediction"
    (out_dir / "page.md").write_text(fallback_markdown, encoding="utf-8")

    class FailingOfficialPipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, image_path):
            raise RuntimeError("controlled official failure")

    monkeypatch.setitem(
        sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCRVL=FailingOfficialPipeline)
    )

    summary = mod.run_official_folder(
        img_dir=img_dir,
        out_dir=out_dir,
        server_url="http://127.0.0.1:8111/v1",
        api_model_name="PaddleOCR-VL-1.6-GGUF.gguf",
        page_retries=0,
        fallback_pred_dir=out_dir,
    )

    assert (out_dir / "page.md").read_text(encoding="utf-8") == fallback_markdown
    assert summary["ok"] == 1
    assert summary["fail"] == 0
    assert summary["fallback"] == 1
    assert summary["stats"][0]["status"].startswith("fallback:")


def test_official_folder_empty_generator_uses_same_directory_fallback(tmp_path, monkeypatch):
    mod = _load_adapter()
    img_dir = tmp_path / "images"
    out_dir = tmp_path / "predictions"
    img_dir.mkdir()
    out_dir.mkdir()
    (img_dir / "page.png").write_bytes(b"fake image")
    fallback_markdown = "existing fallback prediction"
    (out_dir / "page.md").write_text(fallback_markdown, encoding="utf-8")

    class EmptyGeneratorOfficialPipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, image_path):
            return (result for result in ())

    monkeypatch.setitem(
        sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCRVL=EmptyGeneratorOfficialPipeline)
    )

    summary = mod.run_official_folder(
        img_dir=img_dir,
        out_dir=out_dir,
        server_url="http://127.0.0.1:8111/v1",
        api_model_name="PaddleOCR-VL-1.6-GGUF.gguf",
        page_retries=0,
        fallback_pred_dir=out_dir,
    )

    assert (out_dir / "page.md").read_text(encoding="utf-8") == fallback_markdown
    assert summary["ok"] == 1
    assert summary["fallback"] == 1
    assert summary["stats"][0]["status"].startswith("fallback:")


def test_official_folder_empty_generator_without_fallback_fails_page(tmp_path, monkeypatch):
    mod = _load_adapter()
    img_dir = tmp_path / "images"
    out_dir = tmp_path / "predictions"
    img_dir.mkdir()
    (img_dir / "page.png").write_bytes(b"fake image")

    class EmptyGeneratorOfficialPipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, image_path):
            return (result for result in ())

    monkeypatch.setitem(
        sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCRVL=EmptyGeneratorOfficialPipeline)
    )

    try:
        mod.run_official_folder(
            img_dir=img_dir,
            out_dir=out_dir,
            server_url="http://127.0.0.1:8111/v1",
            api_model_name="PaddleOCR-VL-1.6-GGUF.gguf",
            page_retries=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected empty official iterable to fail the page")

    assert not (out_dir / "page.md").exists()
    stats = (out_dir / "_run_stats.json").read_text(encoding="utf-8")
    assert '"fail": 1' in stats
    assert "Official PaddleOCRVL predict() returned no page results." in stats
