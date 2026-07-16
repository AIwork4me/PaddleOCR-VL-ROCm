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
    monkeypatch.setattr(
        mod,
        "_validate_scorer_checkout",
        lambda checkout: {"commit": "pinned", "blobs": {"metrics.py": "blob"}},
    )


def _allow_test_release_stats(mod, monkeypatch):
    monkeypatch.setattr(mod, "_validate_release_prediction_stats", lambda args, path: None)


def _use_test_dataset_manifest(mod, monkeypatch, tmp_path):
    dataset = tmp_path / "authenticated dataset"
    dataset.mkdir(exist_ok=True)
    (dataset / "OmniDocBench.json").write_text("[]", encoding="utf-8")
    monkeypatch.setitem(mod.VERSION_DATASET_DIRS, "v16", dataset)
    return dataset


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
            "provenance": None,
        },
    )()

    mod.apply_artifact_profile_defaults(args)

    assert args.predictions_dir == "predictions/paddleocr_official_local_llamacpp_gguf_v16"
    assert args.provenance == (
        "results/omnidocbench/v16/paddleocr_official_local_llamacpp_gguf_provenance.json"
    )


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
        "stats": [{"image": f"page-{index:04d}.png", "status": "ok"} for index in range(1651)],
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
        "stats": [{"image": f"page-{index:04d}.png", "status": "ok"} for index in range(1651)],
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


def test_release_prediction_stats_accept_exact_known_official_failure(tmp_path):
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
        "ok": 1650,
        "fail": 1,
        "fallback": 0,
        "limit_pages": None,
        "engine": "official",
        "stats": [{"image": f"page-{index:04d}.png", "status": "ok"} for index in range(1650)]
        + [
            {
                "image": "newspaper_The Times UK_0801@magazinesclubnew_page_031.png",
                "status": "failed: output does not match the expected peg-native format",
            }
        ],
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
            "engine": "official",
        },
    )()

    mod._validate_release_prediction_stats(args, predictions)


def test_release_prediction_stats_reject_known_failure_with_residual_markdown(tmp_path):
    mod = _load_run_eval()
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    predictions = tmp_path / "predictions"
    images.mkdir(parents=True)
    predictions.mkdir()
    for index in range(1651):
        (images / f"{index}.png").touch()
    failed_image = "newspaper_The Times UK_0801@magazinesclubnew_page_031.png"
    stats = {
        "count": 1651,
        "ok": 1650,
        "fail": 1,
        "fallback": 0,
        "limit_pages": None,
        "engine": "official",
        "stats": [{"image": f"page-{index:04d}.png", "status": "ok"} for index in range(1650)]
        + [{"image": failed_image, "status": "failed: peg-native"}],
    }
    (predictions / "_run_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    (predictions / f"{Path(failed_image).stem}.md").write_text(
        "synthetic fallback", encoding="utf-8"
    )
    args = type(
        "Args",
        (),
        {
            "version": "v16",
            "dataset_dir": str(dataset),
            "copy_report": "metric.json",
            "run_summary": None,
            "cdm": False,
            "engine": "official",
        },
    )()

    with pytest.raises(SystemExit, match="must not exist"):
        mod._validate_release_prediction_stats(args, predictions)


def _release_stats_args(dataset_dir):
    return type(
        "Args",
        (),
        {
            "version": "v16",
            "dataset_dir": str(dataset_dir),
            "copy_report": "metric.json",
            "run_summary": None,
            "cdm": False,
        },
    )()


def _write_release_stats(predictions, **overrides):
    stats = {
        "count": 1651,
        "ok": 1651,
        "fail": 0,
        "fallback": 0,
        "limit_pages": None,
        "engine": "official",
        "stats": [],
    }
    stats.update(overrides)
    predictions.mkdir()
    (predictions / "_run_stats.json").write_text(json.dumps(stats), encoding="utf-8")


def test_release_prediction_stats_reject_missing_dataset_images(tmp_path):
    mod = _load_run_eval()
    predictions = tmp_path / "predictions"
    _write_release_stats(predictions)

    with pytest.raises(SystemExit, match="dataset image count"):
        mod._validate_release_prediction_stats(
            _release_stats_args(tmp_path / "missing-dataset"), predictions
        )


@pytest.mark.parametrize("missing_fields", [("count",), ("ok",), ("count", "ok"), ("limit_pages",)])
def test_release_prediction_stats_reject_missing_required_fields(
    tmp_path, monkeypatch, missing_fields
):
    mod = _load_run_eval()
    predictions = tmp_path / "predictions"
    _write_release_stats(predictions)
    stats_path = predictions / "_run_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for field in missing_fields:
        stats.pop(field)
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    monkeypatch.setattr(mod, "_dataset_image_count", lambda args: 1651)

    with pytest.raises(SystemExit):
        mod._validate_release_prediction_stats(_release_stats_args(tmp_path), predictions)


def test_release_prediction_stats_reject_wrong_v16_count_even_when_dataset_matches(
    tmp_path, monkeypatch
):
    mod = _load_run_eval()
    predictions = tmp_path / "predictions"
    _write_release_stats(predictions, count=1650, ok=1650)
    monkeypatch.setattr(mod, "_dataset_image_count", lambda args: 1650)

    with pytest.raises(SystemExit, match="1651"):
        mod._validate_release_prediction_stats(_release_stats_args(tmp_path), predictions)


def test_non_release_eval_preserves_missing_dataset_behavior(tmp_path):
    mod = _load_run_eval()
    args = type(
        "Args",
        (),
        {
            "version": "v16",
            "dataset_dir": str(tmp_path / "missing-dataset"),
            "copy_report": None,
            "run_summary": None,
            "cdm": False,
        },
    )()

    mod._validate_release_prediction_stats(args, tmp_path / "missing-predictions")


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
    dataset_manifest = dataset / "OmniDocBench.json"
    dataset_manifest.write_text('{"pages": ["page.png"]}', encoding="utf-8")
    predictions.mkdir(parents=True)
    (predictions / "page.md").write_text("prediction", encoding="utf-8")
    (predictions / "_run_stats.json").write_text(
        '{"count": 1, "ok": 1, "fail": 0, "fallback": 0, "engine": "official", "stats": []}',
        encoding="utf-8",
    )
    copied = tmp_path / "results" / "metric.json"
    summary = tmp_path / "results" / "summary.json"
    provenance = tmp_path / "results" / "provenance.json"

    monkeypatch.setattr(mod, "_ensure_omnidocbench_checkout", lambda: checkout)
    _allow_test_checkout(mod, monkeypatch)
    _allow_test_release_stats(mod, monkeypatch)
    monkeypatch.setitem(mod.VERSION_DATASET_DIRS, "v16", dataset)

    def fake_run(*args, **kwargs):
        report.write_text('{"text_block": {"page": {"Edit_dist": {"ALL": 0.1}}}}', encoding="utf-8")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod.subprocess, "check_output", lambda *args, **kwargs: "repo-head\n")
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
            "dataset_dir": str(dataset),
            "provenance": str(provenance),
            "server_url": "http://127.0.0.1:8111/v1",
            "api_model_name": "PaddleOCR-VL-1.6-GGUF.gguf",
            "engine": "official",
        },
    )()

    mod.stage_eval(args)

    assert copied.read_text(encoding="utf-8").startswith('{"text_block"')
    assert summary.is_file()
    written_provenance = json.loads(provenance.read_text(encoding="utf-8"))
    assert written_provenance["omnidocbench"]["commit"] == "pinned"
    assert len(written_provenance["dataset_sha256"]) == 64
    assert len(written_provenance["config_sha256"]) == 64
    assert len(written_provenance["prediction_manifest_sha256"]) == 64


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
    _use_test_dataset_manifest(mod, monkeypatch, tmp_path)
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
    _use_test_dataset_manifest(mod, monkeypatch, tmp_path)
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
    dataset_manifest = dataset / "OmniDocBench.json"
    dataset_manifest.write_text("[]", encoding="utf-8")
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
        dataset_manifest.resolve()
    )
    assert eval_config["metrics"]["display_formula"]["metric"] == ["Edit_dist"]


def test_render_eval_config_uses_authenticated_unicode_dataset_manifest(tmp_path):
    mod = _load_run_eval()
    config = tmp_path / "base.yaml"
    config.write_text(
        """end2end_eval:
  dataset:
    ground_truth:
      data_path: unsafe/default.json
    prediction:
      data_path: ignored
  metrics:
    display_formula:
      metric: [Edit_dist]
""",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    dataset_manifest = tmp_path / "dataset with spaces 数据" / "OmniDocBench.json"
    dataset_manifest.parent.mkdir()
    dataset_manifest.write_text("[]", encoding="utf-8")

    rendered = mod._render_eval_config(
        config,
        predictions,
        ground_truth_manifest=dataset_manifest,
        cdm=False,
        destination_dir=tmp_path / "rendered",
    )

    eval_config = yaml.safe_load(rendered.read_text(encoding="utf-8"))["end2end_eval"]
    assert eval_config["dataset"]["ground_truth"]["data_path"] == str(
        dataset_manifest.resolve()
    )


def test_stage_eval_passes_rendered_config_with_cdm_when_requested(tmp_path, monkeypatch):
    mod = _load_run_eval()
    checkout = tmp_path / "checkout"
    predictions = tmp_path / "predictions" / "paddleocr_official_local_llamacpp_gguf_v16"
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    report = checkout / "result" / f"{predictions.name}_quick_match_metric_result.json"
    images.mkdir(parents=True)
    (images / "page.png").write_bytes(b"image")
    (dataset / "OmniDocBench.json").write_text("[]", encoding="utf-8")
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
    _allow_test_release_stats(mod, monkeypatch)
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
    _use_test_dataset_manifest(mod, monkeypatch, tmp_path)
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


def test_explicit_scorer_python_overrides_checkout_venv(tmp_path):
    mod = _load_run_eval()
    checkout = tmp_path / "checkout"
    checkout_python = checkout / ".venv" / "Scripts" / "python.exe"
    checkout_python.parent.mkdir(parents=True)
    checkout_python.write_text("", encoding="utf-8")
    authenticated_python = tmp_path / "authenticated" / "python.exe"
    authenticated_python.parent.mkdir()
    authenticated_python.write_text("", encoding="utf-8")

    resolved = mod._resolve_eval_python(checkout, str(authenticated_python.resolve()))

    assert resolved == str(authenticated_python.resolve())


def test_stage_eval_sets_pythonutf8_for_windows_omnidocbench_subprocess(tmp_path, monkeypatch):
    mod = _load_run_eval()
    _use_test_dataset_manifest(mod, monkeypatch, tmp_path)
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
            "trace_dir": str(tmp_path / "traces"),
            "layout_profile_prefix": str(tmp_path / "layout-profile"),
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
        "trace_dir": str(tmp_path / "traces"),
        "layout_profile_prefix": str(tmp_path / "layout-profile"),
    }
    assert "[infer] 1/1 pages succeeded" in capsys.readouterr().out
