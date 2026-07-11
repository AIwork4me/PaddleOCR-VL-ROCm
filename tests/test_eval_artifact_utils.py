from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_artifacts():
    script = Path("eval/artifact_utils.py")
    spec = importlib.util.spec_from_file_location("artifact_utils", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_local_paths_are_engine_identified():
    mod = _load_artifacts()

    paths = mod.official_local_paths("v16", cdm=False)
    cdm_paths = mod.official_local_paths("v16", cdm=True)

    assert paths.predictions_dir == Path("predictions/paddleocr_official_local_llamacpp_gguf_v16")
    assert paths.metric_result == Path(
        "results/omnidocbench/v16/paddleocr_official_local_llamacpp_gguf_quick_match_metric_result.json"
    )
    assert cdm_paths.metric_result == Path(
        "results/omnidocbench/v16/paddleocr_official_local_llamacpp_gguf_quick_match_metric_result_cdm.json"
    )
    assert paths.provenance == Path(
        "results/omnidocbench/v16/paddleocr_official_local_llamacpp_gguf_provenance.json"
    )


def test_write_run_summary_and_provenance(tmp_path):
    mod = _load_artifacts()
    predictions = tmp_path / "predictions"
    results_dir = tmp_path / "results"
    predictions.mkdir()
    stats_path = predictions / "_run_stats.json"
    metric_path = results_dir / "metric.json"
    stats_path.write_text(
        json.dumps(
            {"count": 3, "ok": 2, "fail": 1, "fallback": 1, "engine": "official", "stats": []}
        ),
        encoding="utf-8",
    )
    metric_path.parent.mkdir(parents=True)
    metric_path.write_text(
        json.dumps({"text_block": {"page": {"Edit_dist": {"ALL": 0.1}}}}), encoding="utf-8"
    )

    summary_path = mod.write_run_summary(
        save_name="paddleocr_official_local_llamacpp_gguf_quick_match",
        run_stats_path=stats_path,
        metric_result_path=metric_path,
        destination=results_dir / "summary.json",
        cdm=False,
    )
    provenance_path = mod.write_provenance(
        destination=results_dir / "provenance.json",
        git_commit="abc123",
        engine="official",
        server_url="http://127.0.0.1:8111/v1",
        api_model_name="PaddleOCR-VL-1.6-GGUF.gguf",
        adapter_command="python eval/run_eval.py --stage infer --engine official",
        scoring_config_path=Path("eval/configs/omnidocbench_v16.yaml"),
        dataset_manifest_path=Path("data/omnidocbench/v16/OmniDocBench.json"),
        predictions_dir=predictions,
        metric_result_paths=[metric_path],
        run_summary_paths=[summary_path],
        run_stats_path=stats_path,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert summary["run_stats"]["ok"] == 2
    assert summary["metric_result_path"] == str(metric_path)
    assert provenance["git_commit"] == "abc123"
    assert provenance["ok_pages"] == 2
    assert provenance["fallback_pages"] == 1


def test_cdm_all_exception_metric_is_marked_invalid(tmp_path):
    mod = _load_artifacts()
    predictions = tmp_path / "predictions"
    results_dir = tmp_path / "results"
    predictions.mkdir()
    stats_path = predictions / "_run_stats.json"
    metric_path = results_dir / "metric_cdm.json"
    stats_path.write_text(
        json.dumps(
            {"count": 2, "ok": 2, "fail": 0, "fallback": 0, "engine": "official", "stats": []}
        ),
        encoding="utf-8",
    )
    metric_path.parent.mkdir(parents=True)
    metric_path.write_text(
        json.dumps(
            {
                "display_formula": {
                    "page": {"CDM": {"ALL": 0.0}},
                    "metric_debug": {
                        "CDM": {
                            "sample_count": 2,
                            "exception_case_count": 2,
                            "exception_cases": [
                                {
                                    "reason": (
                                        "FileNotFoundError: [Errno 2] No such file or directory: "
                                        "'result/.../temp_gt/.../gt_sample_0.tex'"
                                    )
                                }
                            ],
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    summary_path = mod.write_run_summary(
        save_name="paddleocr_official_local_llamacpp_gguf_v16_quick_match",
        run_stats_path=stats_path,
        metric_result_path=metric_path,
        destination=results_dir / "summary_cdm.json",
        cdm=True,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["readme_metrics"]["formula_cdm_percent"] is None
    assert summary["metric_quality"]["formula_cdm"]["valid"] is False
    assert "all CDM samples raised exceptions" in summary["metric_quality"]["formula_cdm"]["reason"]
