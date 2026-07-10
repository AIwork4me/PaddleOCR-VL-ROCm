"""Server-free unit test for eval/run_eval._resolve_report_path.

Locks in that the OmniDocBench metric report path is resolved under the
checkout directory (where pdf_validation.py runs and writes its result/ dir),
not against the orchestrator's own CWD.
"""

import importlib.util
from pathlib import Path


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("run_eval", Path("eval/run_eval.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_path_under_checkout(tmp_path):
    mod = _load_run_eval()
    pred = tmp_path / "predictions" / "paddleocrvl_rocm"  # basename -> paddleocrvl_rocm
    out = mod._resolve_report_path(tmp_path, pred, "quick_match")
    assert out == tmp_path / "result" / "paddleocrvl_rocm_quick_match_metric_result.json"


def test_report_path_uses_match_method(tmp_path):
    mod = _load_run_eval()
    pred = tmp_path / "predictions" / "paddleocrvl_rocm"
    out = mod._resolve_report_path(tmp_path, pred, "naive_match")
    assert out == tmp_path / "result" / "paddleocrvl_rocm_naive_match_metric_result.json"


def test_report_path_distinct_checkouts(tmp_path):
    mod = _load_run_eval()
    pred = tmp_path / "predictions" / "paddleocrvl_rocm"
    checkout_a = tmp_path / "checkout_a"
    checkout_b = tmp_path / "checkout_b"
    out_a = mod._resolve_report_path(checkout_a, pred, "quick_match")
    out_b = mod._resolve_report_path(checkout_b, pred, "quick_match")
    assert out_a == checkout_a / "result" / "paddleocrvl_rocm_quick_match_metric_result.json"
    assert out_b == checkout_b / "result" / "paddleocrvl_rocm_quick_match_metric_result.json"
    assert out_a != out_b


def test_artifact_profile_sets_official_predictions_dir():
    mod = _load_run_eval()
    args = type(
        "Args",
        (),
        {
            "artifact_profile": "official-local",
            "version": "v16",
            "predictions_dir": str(mod.DEFAULT_PREDICTIONS_DIR),
            "engine": "official",
            "cdm": False,
        },
    )()

    mod.apply_artifact_profile_defaults(args)

    assert args.predictions_dir == "predictions/paddleocr_official_local_llamacpp_gguf_v16"


def test_stage_eval_copies_official_metric_report(tmp_path, monkeypatch):
    mod = _load_run_eval()
    checkout = tmp_path / "checkout"
    report = checkout / "result" / "paddleocr_official_local_llamacpp_gguf_v16_quick_match_metric_result.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"text_block": {"page": {"Edit_dist": {"ALL": 0.1}}}}', encoding="utf-8")
    predictions = tmp_path / "predictions" / "paddleocr_official_local_llamacpp_gguf_v16"
    predictions.mkdir(parents=True)
    (predictions / "_run_stats.json").write_text(
        '{"count": 1, "ok": 1, "fail": 0, "fallback": 0, "engine": "official", "stats": []}',
        encoding="utf-8",
    )
    copied = tmp_path / "results" / "metric.json"
    summary = tmp_path / "results" / "summary.json"

    monkeypatch.setattr(mod, "_ensure_omnidocbench_checkout", lambda: checkout)
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(mod, "_resolve_report_path", lambda checkout, predictions_dir, match_method: report)

    args = type(
        "Args",
        (),
        {
            "config": "eval/configs/omnidocbench_v16.yaml",
            "version": "v16",
            "predictions_dir": str(predictions),
            "match_method": "quick_match",
            "copy_report": str(copied),
            "run_summary": str(summary),
            "cdm": False,
        },
    )()

    mod.stage_eval(args)

    assert copied.read_text(encoding="utf-8").startswith('{"text_block"')
    assert summary.is_file()


def test_stage_infer_dispatches_to_run_adapter(tmp_path, monkeypatch, capsys):
    mod = _load_run_eval()
    dataset = tmp_path / "data"
    images = dataset / "images"
    images.mkdir(parents=True)
    predictions = tmp_path / "predictions"
    captured = {}

    class FakeAdapter:
        @staticmethod
        def run_adapter(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {"count": 1, "ok": 1, "fail": 0, "engine": kwargs["engine"], "stats": []}

    monkeypatch.setattr(mod, "_server_reachable", lambda server_url: True)
    monkeypatch.setattr(mod, "_load_script_module", lambda name, path: FakeAdapter)

    args = type(
        "Args",
        (),
        {
            "server_url": "http://127.0.0.1:8111/v1",
            "dataset_dir": str(dataset),
            "version": "v16",
            "predictions_dir": str(predictions),
            "layout_model": "layout",
            "api_model_name": "PaddleOCR-VL-1.6-GGUF.gguf",
            "vlm_backend": "llama-cpp-server",
            "engine": "official",
            "page_retries": 2,
            "fallback_pred_dir": str(tmp_path / "fallback"),
            "limit_pages": 3,
        },
    )()

    mod.stage_infer(args)

    assert captured["args"] == (images, predictions, "http://127.0.0.1:8111/v1")
    assert captured["kwargs"] == {
        "engine": "official",
        "layout_model": "layout",
        "api_model_name": "PaddleOCR-VL-1.6-GGUF.gguf",
        "vlm_backend": "llama-cpp-server",
        "page_retries": 2,
        "fallback_pred_dir": str(tmp_path / "fallback"),
        "limit_pages": 3,
    }
    assert "[infer] 1/1 pages succeeded" in capsys.readouterr().out
