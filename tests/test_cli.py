from __future__ import annotations

from paddleocr_vl_rocm.cli import build_parser


def test_cli_parser_accepts_documented_smoke_command():
    args = build_parser().parse_args(
        [
            "--input",
            "examples/input/handwrite_ch_demo.png",
            "--output",
            "outputs/smoke",
            "--layout-model",
            "models/PP-DocLayoutV3-onnx",
            "--server-url",
            "http://127.0.0.1:8000/v1",
            "--api-model-name",
            "PaddleOCR-VL-1.5-0.9B",
            "--vlm-backend",
            "vllm-server",
            "--layout-provider",
            "directml",
        ]
    )

    assert args.input.endswith("handwrite_ch_demo.png")
    assert args.vlm_backend == "vllm-server"
    assert args.api_model_name == "PaddleOCR-VL-1.5-0.9B"
    assert args.layout_provider == "directml"


def test_cli_layout_provider_defaults_to_auto():
    args = build_parser().parse_args(["--input", "input.png"])

    assert args.layout_provider == "auto"
