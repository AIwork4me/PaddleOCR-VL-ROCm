"""Server-free unit test for eval/run_eval._resolve_report_path.

Locks in that the OmniDocBench metric report path is resolved under the
checkout directory (where pdf_validation.py runs and writes its result/ dir),
not against the orchestrator's own CWD.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("run_eval", Path("eval/run_eval.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _allow_test_checkout(mod, monkeypatch):
    monkeypatch.setattr(mod, "_validate_scorer_checkout", lambda checkout: None)


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fail", 1),
        ("fallback", 1),
        ("ok", 1650),
        ("limit_pages", 16),
    ],
)
def test_release_prediction_stats_reject_incomplete_runs(tmp_path, field, value):
    mod = _load_run_eval()
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    predictions = tmp_path / "predictions"
    images.mkdir(parents=True)
    predictions.mkdir()
    for index in range(1651):
        (images / f"{index}.png").touch()
    stats = {
        "count": 1651,
        "ok": 1651,
        "fail": 0,
        "fallback": 0,
        "limit_pages": None,
        "engine": "official",
        "stats": [],
    }
    stats[field] = value
    (predictions / "_run_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "version": "v16",
            "dataset_dir": str(dataset),
            "copy_report": "metric.json",
            "run_summary": None,
            "cdm": False,
        },
    )()

    with pytest.raises(SystemExit):
        mod._validate_release_prediction_stats(args, predictions)


def test_release_prediction_stats_accept_complete_clean_run(tmp_path):
    mod = _load_run_eval()
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    predictions = tmp_path / "predictions"
    images.mkdir(parents=True)
    predictions.mkdir()
    for index in range(1651):
        (images / f"{index}.png").touch()
    stats = {
        "count": 1651,
        "ok": 1651,
        "fail": 0,
        "fallback": 0,
        "limit_pages": None,
        "engine": "official",
        "stats": [],
    }
    (predictions / "_run_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "version": "v16",
            "dataset_dir": str(dataset),
            "copy_report": "metric.json",
            "run_summary": None,
            "cdm": False,
        },
    )()

    mod._validate_release_prediction_stats(args, predictions)


def test_stage_eval_validates_checkout_before_rendering_config(tmp_path, monkeypatch):
    mod = _load_run_eval()
    checkout = tmp_path / "checkout"
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    calls = []

    class BenchmarkContract:
        @staticmethod
        def validate_checkout(candidate):
            calls.append(("validate", candidate))

    monkeypatch.setattr(mod, "_ensure_omnidocbench_checkout", lambda: checkout)
    monkeypatch.setattr(mod, "_load_script_module", lambda name, path: BenchmarkContract)

    def stop_at_render(*args, **kwargs):
        assert calls == [("validate", checkout)]
        raise RuntimeError("stop after checkout validation")

    monkeypatch.setattr(mod, "_render_eval_config", stop_at_render)
    args = type(
        "Args",
        (),
        {
            "config": "eval/configs/omnidocbench_v16.yaml",
            "version": "v16",
            "dataset_dir": None,
            "predictions_dir": str(predictions),
            "match_method": "quick_match",
            "copy_report": None,
            "run_summary": None,
            "cdm": False,
        },
    )()

    with pytest.raises(RuntimeError, match="stop after checkout validation"):
        mod.stage_eval(args)


def test_stage_eval_copies_official_metric_report(tmp_path, monkeypatch):
    mod = _load_run_eval()
    checkout = tmp_path / "checkout"
    report = (
        checkout
        / "result"
        / "paddleocr_official_local_llamacpp_gguf_v16_quick_match_metric_result.json"
    )
    report.parent.mkdir(parents=True)
    report.write_text('{"text_block": {"page": {"Edit_dist": {"ALL": 0.1}}}}', encoding="utf-8")
    predictions = tmp_path / "predictions" / "paddleocr_official_local_llamacpp_gguf_v16"
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    (images / "page.png").write_bytes(b"image")
    predictions.mkdir(parents=True)
    (predictions / "_run_stats.json").write_text(
        '{"count": 1, "ok": 1, "fail": 0, "fallback": 0, "engine": "official", "stats": []}',
        encoding="utf-8",
    )
    copied = tmp_path / "results" / "metric.json"
    summary = tmp_path / "results" / "summary.json"

    monkeypatch.setattr(mod, "_ensure_omnidocbench_checkout", lambda: checkout)
    _allow_test_checkout(mod, monkeypatch)
    monkeypatch.setitem(mod.VERSION_DATASET_DIRS, "v16", dataset)

    def fake_run(*args, **kwargs):
        report.write_text('{"text_block": {"page": {"Edit_dist": {"ALL": 0.1}}}}', encoding="utf-8")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(
        mod, "_resolve_report_path", lambda checkout, predictions_dir, match_method: report
    )

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


def test_stage_eval_refuses_to_publish_limited_predictions(tmp_path, monkeypatch):
    mod = _load_run_eval()
    checkout = tmp_path / "checkout"
    predictions = tmp_path / "predictions" / "paddleocr_official_local_llamacpp_gguf_v16"
    predictions.mkdir(parents=True)
    (predictions / "_run_stats.json").write_text(
        json.dumps(
            {
                "count": 16,
                "ok": 16,
                "fail": 0,
                "fallback": 0,
                "engine": "official",
                "limit_pages": 16,
                "stats": [],
            }
        ),
        encoding="utf-8",
    )
    copied = tmp_path / "results" / "metric.json"

    monkeypatch.setattr(mod, "_ensure_omnidocbench_checkout", lambda: checkout)
    _allow_test_checkout(mod, monkeypatch)

    args = type(
        "Args",
        (),
        {
            "config": "eval/configs/omnidocbench_v16.yaml",
            "version": "v16",
            "dataset_dir": None,
            "predictions_dir": str(predictions),
            "match_method": "quick_match",
            "copy_report": str(copied),
            "run_summary": None,
            "cdm": True,
        },
    )()

    try:
        mod.stage_eval(args)
    except SystemExit as exc:
        assert "full unbounded inference" in str(exc)
    else:
        raise AssertionError("Expected limited predictions to be rejected before scoring")


def test_stage_eval_refuses_to_publish_when_dataset_count_mismatches(tmp_path, monkeypatch):
    mod = _load_run_eval()
    checkout = tmp_path / "checkout"
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    predictions = tmp_path / "predictions" / "paddleocr_official_local_llamacpp_gguf_v16"
    images.mkdir(parents=True)
    predictions.mkdir(parents=True)
    for name in ("a.png", "b.jpg", "c.jpeg"):
        (images / name).write_bytes(b"image")
    (predictions / "_run_stats.json").write_text(
        json.dumps(
            {
                "count": 2,
                "ok": 2,
                "fail": 0,
                "fallback": 0,
                "engine": "official",
                "limit_pages": None,
                "stats": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_ensure_omnidocbench_checkout", lambda: checkout)
    _allow_test_checkout(mod, monkeypatch)

    args = type(
        "Args",
        (),
        {
            "config": "eval/configs/omnidocbench_v16.yaml",
            "version": "v16",
            "dataset_dir": str(dataset),
            "predictions_dir": str(predictions),
            "match_method": "quick_match",
            "copy_report": str(tmp_path / "results" / "metric.json"),
            "run_summary": None,
            "cdm": False,
        },
    )()

    try:
        mod.stage_eval(args)
    except SystemExit as exc:
        assert "does not match dataset image count" in str(exc)
    else:
        raise AssertionError("Expected mismatched prediction count to be rejected")


def test_stage_eval_uses_version_dataset_count_without_dataset_override(tmp_path, monkeypatch):
    mod = _load_run_eval()
    checkout = tmp_path / "checkout"
    dataset = tmp_path / "data" / "omnidocbench" / "v16"
    images = dataset / "images"
    predictions = tmp_path / "predictions" / "paddleocr_official_local_llamacpp_gguf_v16"
    images.mkdir(parents=True)
    predictions.mkdir(parents=True)
    for name in ("a.png", "b.jpg"):
        (images / name).write_bytes(b"image")
    (predictions / "_run_stats.json").write_text(
        json.dumps(
            {
                "count": 1,
                "ok": 1,
                "fail": 0,
                "fallback": 0,
                "engine": "official",
                "limit_pages": None,
                "stats": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_ensure_omnidocbench_checkout", lambda: checkout)
    _allow_test_checkout(mod, monkeypatch)
    monkeypatch.setitem(mod.VERSION_DATASET_DIRS, "v16", dataset)

    args = type(
        "Args",
        (),
        {
            "config": "eval/configs/omnidocbench_v16.yaml",
            "version": "v16",
            "dataset_dir": None,
            "predictions_dir": str(predictions),
            "match_method": "quick_match",
            "copy_report": str(tmp_path / "results" / "metric.json"),
            "run_summary": None,
            "cdm": False,
        },
    )()

    try:
        mod.stage_eval(args)
    except SystemExit as exc:
        assert "does not match dataset image count" in str(exc)
    else:
        raise AssertionError("Expected default dataset count mismatch to be rejected")


def test_stage_eval_fails_when_expected_report_is_missing(tmp_path, monkeypatch):
    mod = _load_run_eval()
    checkout = tmp_path / "checkout"
    predictions = tmp_path / "predictions" / "paddleocrvl_rocm"
    predictions.mkdir(parents=True)
    (predictions / "_run_stats.json").write_text(
        '{"count": 1, "ok": 1, "fail": 0, "fallback": 0, "limit_pages": null, "stats": []}',
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_ensure_omnidocbench_checkout", lambda: checkout)
    _allow_test_checkout(mod, monkeypatch)
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *args, **kwargs: type("R", (), {"returncode": 0})()
    )

    args = type(
        "Args",
        (),
        {
            "config": "eval/configs/omnidocbench_v16.yaml",
            "version": "v16",
            "dataset_dir": None,
            "predictions_dir": str(predictions),
            "match_method": "quick_match",
            "copy_report": None,
            "run_summary": None,
            "cdm": False,
        },
    )()

    try:
        mod.stage_eval(args)
    except SystemExit as exc:
        assert "expected report not found" in str(exc)
    else:
        raise AssertionError("Expected missing report to fail eval")


def test_stage_eval_removes_stale_report_before_scoring(tmp_path, monkeypatch):
    mod = _load_run_eval()
    checkout = tmp_path / "checkout"
    predictions = tmp_path / "predictions" / "paddleocrvl_rocm"
    report = checkout / "result" / f"{predictions.name}_quick_match_metric_result.json"
    predictions.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    report.write_text('{"stale": true}', encoding="utf-8")
    (predictions / "_run_stats.json").write_text(
        '{"count": 1, "ok": 1, "fail": 0, "fallback": 0, "limit_pages": null, "stats": []}',
        encoding="utf-8",
    )

    def fake_run(*args, **kwargs):
        assert not report.exists()
        report.write_text('{"fresh": true}', encoding="utf-8")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(mod, "_ensure_omnidocbench_checkout", lambda: checkout)
    _allow_test_checkout(mod, monkeypatch)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    args = type(
        "Args",
        (),
        {
            "config": "eval/configs/omnidocbench_v16.yaml",
            "version": "v16",
            "dataset_dir": None,
            "predictions_dir": str(predictions),
            "match_method": "quick_match",
            "copy_report": None,
            "run_summary": None,
            "cdm": False,
        },
    )()

    mod.stage_eval(args)

    assert json.loads(report.read_text(encoding="utf-8")) == {"fresh": True}


def test_stage_eval_passes_rendered_config_for_selected_predictions_without_cdm(
    tmp_path, monkeypatch
):
    mod = _load_run_eval()
    checkout = tmp_path / "checkout"
    predictions = tmp_path / "predictions" / "paddleocr_official_local_llamacpp_gguf_v16"
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    report = checkout / "result" / f"{predictions.name}_quick_match_metric_result.json"
    images.mkdir(parents=True)
    (images / "page.png").write_bytes(b"image")
    predictions.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    report.write_text('{"text_block": {"page": {"Edit_dist": {"ALL": 0.1}}}}', encoding="utf-8")
    captured = {}

    def fake_run(cmd, **kwargs):
        config_path = Path(cmd[cmd.index("--config") + 1])
        captured["config_path"] = config_path
        captured["config"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        report.write_text('{"text_block": {"page": {"Edit_dist": {"ALL": 0.1}}}}', encoding="utf-8")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(mod, "_ensure_omnidocbench_checkout", lambda: checkout)
    _allow_test_checkout(mod, monkeypatch)
    monkeypatch.setitem(mod.VERSION_DATASET_DIRS, "v16", dataset)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    args = type(
        "Args",
        (),
        {
            "config": "eval/configs/omnidocbench_v16.yaml",
            "version": "v16",
            "predictions_dir": str(predictions),
            "match_method": "quick_match",
            "copy_report": None,
            "run_summary": None,
            "cdm": False,
        },
    )()

    mod.stage_eval(args)

    eval_config = captured["config"]["end2end_eval"]
    assert captured["config_path"] != Path(args.config).resolve()
    assert eval_config["dataset"]["prediction"]["data_path"] == str(predictions.resolve())
    assert eval_config["dataset"]["ground_truth"]["data_path"] == str(
        Path("data/omnidocbench/v16/OmniDocBench.json").resolve()
    )
    assert eval_config["metrics"]["display_formula"]["metric"] == ["Edit_dist"]


def test_stage_eval_passes_rendered_config_with_cdm_when_requested(tmp_path, monkeypatch):
    mod = _load_run_eval()
    checkout = tmp_path / "checkout"
    predictions = tmp_path / "predictions" / "paddleocr_official_local_llamacpp_gguf_v16"
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    report = checkout / "result" / f"{predictions.name}_quick_match_metric_result.json"
    images.mkdir(parents=True)
    (images / "page.png").write_bytes(b"image")
    predictions.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    report.write_text('{"text_block": {"page": {"Edit_dist": {"ALL": 0.1}}}}', encoding="utf-8")
    (predictions / "_run_stats.json").write_text(
        '{"count": 1, "ok": 1, "fail": 0, "fallback": 0, "limit_pages": null, "stats": []}',
        encoding="utf-8",
    )
    captured = {}

    def fake_run(cmd, **kwargs):
        config_path = Path(cmd[cmd.index("--config") + 1])
        captured["config"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        report.write_text('{"text_block": {"page": {"Edit_dist": {"ALL": 0.1}}}}', encoding="utf-8")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(mod, "_ensure_omnidocbench_checkout", lambda: checkout)
    _allow_test_checkout(mod, monkeypatch)
    monkeypatch.setitem(mod.VERSION_DATASET_DIRS, "v16", dataset)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    args = type(
        "Args",
        (),
        {
            "config": "eval/configs/omnidocbench_v16.yaml",
            "version": "v16",
            "predictions_dir": str(predictions),
            "match_method": "quick_match",
            "copy_report": None,
            "run_summary": None,
            "cdm": True,
        },
    )()

    mod.stage_eval(args)

    assert captured["config"]["end2end_eval"]["metrics"]["display_formula"]["metric"] == [
        "Edit_dist",
        "CDM",
    ]


def test_stage_eval_uses_checkout_venv_python_when_available(tmp_path, monkeypatch):
    mod = _load_run_eval()
    checkout = tmp_path / "checkout"
    venv_python = checkout / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    predictions = tmp_path / "predictions" / "paddleocr_official_local_llamacpp_gguf_v16"
    report = checkout / "result" / f"{predictions.name}_quick_match_metric_result.json"
    predictions.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    report.write_text('{"text_block": {"page": {"Edit_dist": {"ALL": 0.1}}}}', encoding="utf-8")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        report.write_text('{"text_block": {"page": {"Edit_dist": {"ALL": 0.1}}}}', encoding="utf-8")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(mod, "_ensure_omnidocbench_checkout", lambda: checkout)
    _allow_test_checkout(mod, monkeypatch)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    args = type(
        "Args",
        (),
        {
            "config": "eval/configs/omnidocbench_v16.yaml",
            "version": "v16",
            "predictions_dir": str(predictions),
            "match_method": "quick_match",
            "copy_report": None,
            "run_summary": None,
            "cdm": False,
        },
    )()

    mod.stage_eval(args)

    assert captured["cmd"][0] == str(venv_python)


def test_stage_eval_sets_pythonutf8_for_windows_omnidocbench_subprocess(tmp_path, monkeypatch):
    mod = _load_run_eval()
    checkout = tmp_path / "checkout"
    predictions = tmp_path / "predictions" / "paddleocr_official_local_llamacpp_gguf_v16"
    report = checkout / "result" / f"{predictions.name}_quick_match_metric_result.json"
    predictions.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    report.write_text('{"text_block": {"page": {"Edit_dist": {"ALL": 0.1}}}}', encoding="utf-8")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})
        report.write_text('{"text_block": {"page": {"Edit_dist": {"ALL": 0.1}}}}', encoding="utf-8")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(mod, "_ensure_omnidocbench_checkout", lambda: checkout)
    _allow_test_checkout(mod, monkeypatch)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    args = type(
        "Args",
        (),
        {
            "config": "eval/configs/omnidocbench_v16.yaml",
            "version": "v16",
            "predictions_dir": str(predictions),
            "match_method": "quick_match",
            "copy_report": None,
            "run_summary": None,
            "cdm": False,
        },
    )()

    mod.stage_eval(args)

    assert captured["env"]["PYTHONUTF8"] == "1"


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
