from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from eval import benchmark_contract


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


def test_extract_notebook_metrics_uses_official_page_fields_and_rounding():
    mod = _load_artifacts()
    metric = {
        "text_block": {"all": {"Edit_dist": {"ALL_page_avg": 0.0344448}}},
        "display_formula": {
            "all": {"CDM": {"all": 0.9617079}},
            "page": {"CDM": {"ALL": 0.96502201}},
        },
        "table": {
            "all": {"TEDS": {"all": 0.9304263}},
            "page": {
                "TEDS": {"ALL": 0.94239317},
                "TEDS_structure_only": {"ALL": 0.955},
            },
        },
        "reading_order": {"all": {"Edit_dist": {"ALL_page_avg": 0.1294874}}},
    }

    values = mod.extract_notebook_metrics(metric)

    assert values == {
        "text_edit_dist": 0.034,
        "formula_cdm_percent": 96.502,
        "table_teds_percent": 94.239,
        "table_teds_structure_only_percent": 95.5,
        "reading_order_edit_dist": 0.129,
        "overall": 95.78033333333333,
    }


def test_extract_notebook_metrics_matches_tracked_official_local_artifact():
    mod = _load_artifacts()
    metric_path = Path(
        "results/omnidocbench/v16/"
        "paddleocr_official_local_llamacpp_gguf_quick_match_metric_result_cdm.json"
    )

    values = mod.extract_notebook_metrics(json.loads(metric_path.read_text(encoding="utf-8")))

    assert values["overall"] == 95.78033333333333


def test_metric_quality_requires_clean_formula_and_table_debug_counts():
    mod = _load_artifacts()
    metric = {
        "display_formula": {
            "metric_debug": {
                "CDM": {
                    "sample_count": 2,
                    "timeout_case_count": 0,
                    "exception_case_count": 0,
                }
            }
        },
        "table": {
            "metric_debug": {
                "TEDS": {
                    "sample_count": 1,
                    "timeout_case_count": 0,
                    "error_case_count": 0,
                }
            }
        },
    }

    quality = mod.analyze_metric_quality(metric)

    assert quality["formula_cdm"]["valid"] is True
    assert quality["formula_cdm"]["timeout_case_count"] == 0
    assert quality["formula_cdm"]["exception_case_count"] == 0
    assert quality["table_teds"]["valid"] is True
    assert quality["table_teds"]["timeout_case_count"] == 0
    assert quality["table_teds"]["error_case_count"] == 0


def test_extract_readme_metrics_suppresses_failed_quality_metrics_and_overall():
    mod = _load_artifacts()
    metric = {
        "text_block": {"all": {"Edit_dist": {"ALL_page_avg": 0.0344}}},
        "display_formula": {
            "page": {"CDM": {"ALL": 0.965}},
            "metric_debug": {
                "CDM": {
                    "sample_count": 2,
                    "timeout_case_count": 1,
                    "exception_case_count": 0,
                }
            },
        },
        "table": {
            "page": {
                "TEDS": {"ALL": 0.9424},
                "TEDS_structure_only": {"ALL": 0.955},
            },
            "metric_debug": {
                "TEDS": {
                    "sample_count": 1,
                    "timeout_case_count": 0,
                    "error_case_count": 1,
                }
            },
        },
        "reading_order": {"all": {"Edit_dist": {"ALL_page_avg": 0.1295}}},
    }

    values = mod.extract_readme_metrics(metric)

    assert values["formula_cdm_percent"] is None
    assert values["table_teds_percent"] is None
    assert values["overall"] is None


def test_write_run_summary_and_provenance(tmp_path):
    mod = _load_artifacts()
    predictions = tmp_path / "predictions"
    results_dir = tmp_path / "results"
    predictions.mkdir()
    (predictions / "page-2.md").write_text("second", encoding="utf-8")
    (predictions / "page-1.md").write_text("first", encoding="utf-8")
    stats_path = predictions / "_run_stats.json"
    metric_path = results_dir / "metric.json"
    config_path = tmp_path / "config.yaml"
    dataset_path = tmp_path / "OmniDocBench.json"
    config_path.write_text("metrics: [TEDS, CDM]\n", encoding="utf-8")
    dataset_path.write_text('{"pages": [1, 2, 3]}', encoding="utf-8")
    stats_path.write_text(
        json.dumps(
            {
                "count": 3,
                "ok": 2,
                "fail": 1,
                "fallback": 1,
                "engine": "official",
                "layout_provider_requested": "auto",
                "layout_providers_active": [
                    "DmlExecutionProvider",
                    "CPUExecutionProvider",
                ],
                "stats": [
                    {
                        "image": "bad.png",
                        "status": "fail: controlled",
                        "error": "x" * 500,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    metric_path.parent.mkdir(parents=True)
    metric_path.write_text(
        json.dumps(
            {
                "text_block": {"all": {"Edit_dist": {"ALL_page_avg": 0.0344448}}},
                "display_formula": {
                    "page": {"CDM": {"ALL": 0.96502201}},
                    "metric_debug": {
                        "CDM": {
                            "sample_count": 2,
                            "timeout_case_count": 0,
                            "exception_case_count": 0,
                        }
                    },
                },
                "table": {
                    "page": {
                        "TEDS": {"ALL": 0.94239317},
                        "TEDS_structure_only": {"ALL": 0.955},
                    },
                    "metric_debug": {
                        "TEDS": {
                            "sample_count": 1,
                            "timeout_case_count": 0,
                            "error_case_count": 0,
                        }
                    },
                },
                "reading_order": {"all": {"Edit_dist": {"ALL_page_avg": 0.1294874}}},
            }
        ),
        encoding="utf-8",
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
        scoring_config_path=config_path,
        dataset_manifest_path=dataset_path,
        predictions_dir=predictions,
        metric_result_paths=[metric_path],
        run_summary_paths=[summary_path],
        run_stats_path=stats_path,
        omnidocbench={
            "commit": benchmark_contract.OMNIDOCBENCH_V16_COMMIT,
            "blobs": benchmark_contract.SCORING_BLOBS,
        },
        dataset_sha256=mod.sha256_file(dataset_path),
        config_sha256=mod.sha256_file(config_path),
        prediction_manifest_sha256=mod.prediction_manifest_sha256(predictions),
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert "run_stats" not in summary
    assert summary["run_stats_summary"] == {
        "count": 3,
        "ok": 2,
        "fail": 1,
        "fallback": 1,
        "limit_pages": None,
        "failure_samples": [{"image": "bad.png", "status": "fail: controlled", "error": "x" * 200}],
    }
    assert summary["metric_result_path"] == str(metric_path)
    assert summary["notebook_metrics"]["overall"] == 95.78033333333333
    assert summary["layout_provider_requested"] == "auto"
    assert summary["layout_providers_active"][0] == "DmlExecutionProvider"
    assert provenance["git_commit"] == "abc123"
    assert provenance["ok_pages"] == 2
    assert provenance["fallback_pages"] == 1
    assert provenance["layout_provider_requested"] == "auto"
    assert provenance["layout_providers_active"] == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert provenance["omnidocbench"]["commit"] == benchmark_contract.OMNIDOCBENCH_V16_COMMIT
    assert len(provenance["dataset_sha256"]) == 64
    assert len(provenance["config_sha256"]) == 64
    assert len(provenance["prediction_manifest_sha256"]) == 64


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
    assert summary["metric_quality"]["formula_cdm"]["reason"] == (
        "CDM requires samples>0, timeouts=0, errors=0; "
        "samples=2, timeouts=0, errors=2"
    )
