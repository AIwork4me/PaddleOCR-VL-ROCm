from __future__ import annotations

import platform
import tempfile
from pathlib import Path

from .layout import PPDocLayoutV3Onnx, resolve_layout_providers
from .pipeline_core import run_light_parser
from .result import PaddleOCRVLROCmResult


class PaddleOCRVLROCm:
    """Lightweight PaddleOCR-VL pipeline using ONNXRuntime layout and a ROCm VLM endpoint."""

    def __init__(
        self,
        layout_model_dir: str | Path = "models/PP-DocLayoutV3-onnx",
        vlm_server_url: str = "http://127.0.0.1:8000/v1",
        api_model_name: str = "PaddleOCR-VL-1.6-GGUF.gguf",
        vlm_backend: str = "llama-cpp-server",
        max_new_tokens: int = 4096,
        timeout: float = 300.0,
        seed: int = 1,
        threshold: float = 0.3,
        vlm_max_workers: int = 1,
        layout_provider: str = "auto",
    ) -> None:
        self.layout_model_dir = Path(layout_model_dir)
        self.vlm_server_url = vlm_server_url
        self.api_model_name = api_model_name
        self.vlm_backend = vlm_backend
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.seed = seed
        self.threshold = threshold
        self.vlm_max_workers = vlm_max_workers
        self.layout_provider = layout_provider
        self.layout_provider_requested = layout_provider
        self.layout_providers_active: list[str] = []
        self.active_layout_providers = self.layout_providers_active
        self._layout_model: PPDocLayoutV3Onnx | None = None

    def _layout(self) -> PPDocLayoutV3Onnx:
        if self._layout_model is None:
            import onnxruntime

            available = onnxruntime.get_available_providers()
            providers = resolve_layout_providers(available, self.layout_provider, platform.system())
            self._layout_model = PPDocLayoutV3Onnx(
                self.layout_model_dir,
                providers=providers,
                requested_provider=self.layout_provider_requested,
            )
            self.layout_providers_active = list(self._layout_model.layout_providers_active)
            self.active_layout_providers = self.layout_providers_active
        return self._layout_model

    def predict(self, image_path: str | Path) -> PaddleOCRVLROCmResult:
        image = Path(image_path)
        layout = self._layout()
        with tempfile.TemporaryDirectory(prefix="paddleocr_vl_rocm_") as tmp:
            tmp_dir = Path(tmp)
            json_path = run_light_parser(
                input_path=image,
                output_dir=tmp_dir,
                model_dir=self.layout_model_dir,
                server_url=self.vlm_server_url,
                vlm_backend=self.vlm_backend,
                api_model_name=self.api_model_name,
                max_new_tokens=self.max_new_tokens,
                timeout=self.timeout,
                prompt_label=None,
                use_layout_detection=True,
                use_chart_recognition=False,
                use_seal_recognition=False,
                seed=self.seed,
                threshold=self.threshold,
                display_input_path=str(image),
                vlm_repeats=1,
                vlm_max_workers=self.vlm_max_workers,
                layout_model=layout,
                layout_provider_requested=layout.layout_provider_requested,
                layout_providers_active=layout.layout_providers_active,
            )
            import json

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = (
                (tmp_dir / "result.md").read_text(encoding="utf-8")
                if (tmp_dir / "result.md").exists()
                else ""
            )
        return PaddleOCRVLROCmResult(payload, markdown)
