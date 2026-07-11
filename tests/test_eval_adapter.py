from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


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

        def predict(self, image_path):
            if image_path.name == "bad.png":
                raise RuntimeError("controlled failure")
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
    assert (out_dir / "ok.md").read_text(encoding="utf-8") == "recognized"
    assert not (out_dir / "bad.md").exists()
    assert "controlled failure" in (out_dir / "_errors.log").read_text(encoding="utf-8")
    stats = (out_dir / "_run_stats.json").read_text(encoding="utf-8")
    assert '"engine": "lightweight"' in stats
    assert '"fallback": 0' in stats


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
