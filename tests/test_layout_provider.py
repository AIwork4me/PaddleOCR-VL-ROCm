from __future__ import annotations

import platform
import sys
from types import SimpleNamespace

import pytest

from paddleocr_vl_rocm.layout import PPDocLayoutV3Onnx, resolve_layout_providers
from paddleocr_vl_rocm.pipeline import PaddleOCRVLROCm


def test_windows_auto_requires_directml():
    assert resolve_layout_providers(
        ["DmlExecutionProvider", "CPUExecutionProvider"], "auto", "Windows"
    ) == ["DmlExecutionProvider"]


def test_windows_auto_never_silently_falls_back_to_cpu():
    with pytest.raises(RuntimeError, match="onnxruntime-directml"):
        resolve_layout_providers(["CPUExecutionProvider"], "auto", "Windows")


def test_explicit_cpu_is_available_for_troubleshooting():
    assert resolve_layout_providers(["CPUExecutionProvider"], "cpu", "Windows") == [
        "CPUExecutionProvider"
    ]


def test_linux_auto_uses_cpu():
    assert resolve_layout_providers(
        ["DmlExecutionProvider", "CPUExecutionProvider"], "auto", "Linux"
    ) == ["CPUExecutionProvider"]


def test_invalid_provider_choice_is_rejected():
    with pytest.raises(ValueError, match="Unsupported layout provider"):
        resolve_layout_providers(["CPUExecutionProvider"], "cuda", "Windows")


def test_explicit_directml_must_be_available():
    with pytest.raises(RuntimeError, match="onnxruntime-directml"):
        resolve_layout_providers(["CPUExecutionProvider"], "directml", "Windows")


def test_layout_session_disables_fallback_and_records_active_provider(tmp_path, monkeypatch):
    (tmp_path / "inference.onnx").touch()

    class FakeSession:
        def __init__(self, _path, sess_options, providers):
            self.requested_providers = providers
            self.fallback_disabled = False

        def disable_fallback(self):
            self.fallback_disabled = True

        def get_providers(self):
            return list(self.requested_providers)

    fake_ort = SimpleNamespace(
        SessionOptions=type("SessionOptions", (), {}),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    model = PPDocLayoutV3Onnx(tmp_path, providers=["DmlExecutionProvider"])

    assert model.session.fallback_disabled is True
    assert model.active_providers == ["DmlExecutionProvider"]


def test_layout_session_rejects_directml_activation_mismatch(tmp_path, monkeypatch):
    (tmp_path / "inference.onnx").touch()

    class FakeSession:
        def __init__(self, _path, sess_options, providers):
            pass

        def disable_fallback(self):
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

    fake_ort = SimpleNamespace(
        SessionOptions=type("SessionOptions", (), {}),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(RuntimeError, match="failed to activate"):
        PPDocLayoutV3Onnx(tmp_path, providers=["DmlExecutionProvider"])


def test_pipeline_stores_requested_and_active_layout_providers(monkeypatch):
    class FakeLayout:
        def __init__(self, _model_dir, providers):
            self.active_providers = list(providers)

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        "onnxruntime.get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr("paddleocr_vl_rocm.pipeline.PPDocLayoutV3Onnx", FakeLayout)
    pipeline = PaddleOCRVLROCm(layout_provider="auto")

    pipeline._layout()

    assert pipeline.layout_provider == "auto"
    assert pipeline.active_layout_providers == ["DmlExecutionProvider"]
