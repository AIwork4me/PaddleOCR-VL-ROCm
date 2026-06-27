from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PaddleOCRVLROCm
from .server import check_openai_compatible_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL-ROCm lightweight ONNXRuntime + ROCm VLM inference."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", default="outputs", help="Output directory.")
    parser.add_argument(
        "--layout-model",
        default="models/PP-DocLayoutV3-onnx",
        help="PP-DocLayoutV3 ONNX model directory.",
    )
    parser.add_argument(
        "--server-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible VLM server URL."
    )
    parser.add_argument("--api-model-name", default="PaddleOCR-VL-1.5-0.9B", help="VLM model id.")
    parser.add_argument(
        "--vlm-backend", choices=["vllm-server", "llama-cpp-server"], default="vllm-server"
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--vlm-max-workers", type=int, default=1)
    parser.add_argument("--skip-server-check", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.skip_server_check:
        check_openai_compatible_server(args.server_url)
    pipeline = PaddleOCRVLROCm(
        layout_model_dir=args.layout_model,
        vlm_server_url=args.server_url,
        api_model_name=args.api_model_name,
        vlm_backend=args.vlm_backend,
        max_new_tokens=args.max_new_tokens,
        timeout=args.timeout,
        threshold=args.threshold,
        vlm_max_workers=args.vlm_max_workers,
    )
    result = pipeline.predict(args.input)
    output = Path(args.output)
    result.print()
    json_path = result.save_to_json(output)
    md_path = result.save_to_markdown(output, pretty=False)
    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
