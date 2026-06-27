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
