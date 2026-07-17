from __future__ import annotations

import platform
import sys
from pathlib import Path
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
    assert model.layout_provider_requested == "directml"
    assert model.layout_providers_active == ["DmlExecutionProvider"]
    assert model.layout_fallback_disabled is True


def test_non_evidence_session_records_missing_disable_fallback(tmp_path, monkeypatch):
    (tmp_path / "inference.onnx").touch()

    class FakeSession:
        def __init__(self, _path, sess_options, providers):
            self.requested_providers = providers

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

    assert model.layout_fallback_disabled is False


def test_evidence_session_requires_disable_fallback_api(tmp_path, monkeypatch):
    (tmp_path / "inference.onnx").touch()

    class FakeSession:
        def __init__(self, _path, sess_options, providers):
            self.requested_providers = providers

        def get_providers(self):
            return list(self.requested_providers)

    fake_ort = SimpleNamespace(
        SessionOptions=type("SessionOptions", (), {}),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(RuntimeError, match="disable_fallback"):
        PPDocLayoutV3Onnx(
            tmp_path,
            providers=["DmlExecutionProvider"],
            profiling_prefix=tmp_path / "profiles" / "layout",
        )


def test_finish_profiling_is_idempotent_and_uses_resolved_prefix(tmp_path, monkeypatch):
    (tmp_path / "inference.onnx").touch()
    profile = tmp_path / "profiles" / "layout_2026.json"

    class FakeSession:
        end_calls = 0

        def __init__(self, _path, sess_options, providers):
            self.options = sess_options
            self.requested_providers = providers

        def disable_fallback(self):
            pass

        def get_providers(self):
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

        def end_profiling(self):
            self.end_calls += 1
            profile.write_text("[]", encoding="utf-8")
            return str(profile)

    fake_ort = SimpleNamespace(
        SessionOptions=type("SessionOptions", (), {}),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    prefix = tmp_path / "profiles" / "layout"

    model = PPDocLayoutV3Onnx(
        tmp_path,
        providers=["DmlExecutionProvider"],
        requested_provider="auto",
        profiling_prefix=prefix,
    )

    assert model.session.options.enable_profiling is True
    assert model.session.options.profile_file_prefix == str(prefix.resolve())
    assert model.finish_profiling() == profile.resolve()
    assert model.finish_profiling() == profile.resolve()
    assert model.session.end_calls == 1


def test_finish_profiling_does_not_end_session_twice_after_missing_path(tmp_path, monkeypatch):
    (tmp_path / "inference.onnx").touch()
    missing_profile = tmp_path / "profiles" / "missing.json"

    class FakeSession:
        end_calls = 0

        def __init__(self, _path, sess_options, providers):
            self.requested_providers = providers

        def disable_fallback(self):
            pass

        def get_providers(self):
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

        def end_profiling(self):
            self.end_calls += 1
            return str(missing_profile)

    fake_ort = SimpleNamespace(
        SessionOptions=type("SessionOptions", (), {}),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    model = PPDocLayoutV3Onnx(
        tmp_path,
        providers=["DmlExecutionProvider"],
        profiling_prefix=tmp_path / "profiles" / "layout",
    )

    with pytest.raises(FileNotFoundError):
        model.finish_profiling()
    with pytest.raises(FileNotFoundError):
        model.finish_profiling()

    assert model.session.end_calls == 1


def test_finish_profiling_is_disabled_by_default(tmp_path, monkeypatch):
    (tmp_path / "inference.onnx").touch()

    class FakeSession:
        def __init__(self, _path, sess_options, providers):
            self.requested_providers = providers

        def disable_fallback(self):
            pass

        def get_providers(self):
            return list(self.requested_providers)

    fake_ort = SimpleNamespace(
        SessionOptions=type("SessionOptions", (), {}),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    assert PPDocLayoutV3Onnx(tmp_path).finish_profiling() is None


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
        def __init__(self, _model_dir, providers, requested_provider, profiling_prefix=None):
            self.active_providers = list(providers)
            self.layout_provider_requested = requested_provider
            self.layout_providers_active = list(providers)
            self.layout_fallback_disabled = True
            self.profiling_prefix = profiling_prefix

        def finish_profiling(self):
            return Path("profile.json") if self.profiling_prefix else None

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        "onnxruntime.get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr("paddleocr_vl_rocm.pipeline.PPDocLayoutV3Onnx", FakeLayout)
    prefix = Path("profiles/layout")
    pipeline = PaddleOCRVLROCm(layout_provider="auto", layout_profile_prefix=prefix)

    pipeline._layout()

    assert pipeline.layout_provider == "auto"
    assert pipeline.active_layout_providers == ["DmlExecutionProvider"]
    assert pipeline.layout_provider_requested == "auto"
    assert pipeline.layout_providers_active == ["DmlExecutionProvider"]
    assert pipeline.layout_fallback_disabled is True
    assert pipeline._layout_model.profiling_prefix == prefix
    assert pipeline.finish_layout_profiling() == Path("profile.json")


def test_pipeline_passes_layout_provider_metadata_to_trace(tmp_path, monkeypatch):
    captured = {}

    class FakeLayout:
        layout_provider_requested = "auto"
        layout_providers_active = ["DmlExecutionProvider", "CPUExecutionProvider"]

    def fake_run_light_parser(**kwargs):
        captured.update(kwargs)
        kwargs["timing_events"].append(
            {
                "decode_seconds": 0.1,
                "layout_seconds": 0.2,
                "crop_encode_seconds": 0.3,
                "vlm_seconds": 0.4,
                "finalize_seconds": 0.5,
                "total_seconds": 1.5,
            }
        )
        path = kwargs["output_dir"] / "result.json"
        path.write_text('{"input_path": "input.png"}', encoding="utf-8")
        return path

    pipeline = PaddleOCRVLROCm(layout_provider="auto")
    monkeypatch.setattr(pipeline, "_layout", lambda: FakeLayout())
    monkeypatch.setattr("paddleocr_vl_rocm.pipeline.run_light_parser", fake_run_light_parser)

    pipeline.predict(tmp_path / "input.png")

    assert captured["layout_provider_requested"] == "auto"
    assert captured["layout_providers_active"] == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert pipeline.last_timing == {
        "decode_seconds": 0.1,
        "layout_seconds": 0.2,
        "crop_encode_seconds": 0.3,
        "vlm_seconds": 0.4,
        "finalize_seconds": 0.5,
        "total_seconds": 1.5,
    }
