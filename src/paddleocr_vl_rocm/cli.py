from __future__ import annotations

import argparse
import sys
from enum import IntEnum
from pathlib import Path

from .doctor import _redacted_url, checks_to_json, doctor_exit_code, render_checks, run_doctor
from .pipeline import PaddleOCRVLROCm
from .server import check_openai_compatible_server
from .setup import SetupDownloadError, SetupOptions, setup_managed_runtime, start_managed_server


class ExitCode(IntEnum):
    OK = 0
    USAGE = 2
    ENVIRONMENT = 10
    DOWNLOAD = 11
    SERVER = 12
    INFERENCE = 13
    PARTIAL = 14


class _ServerError(RuntimeError):
    pass


def _powershell_command(args: list[str]) -> str:
    quoted = [f"'{value.replace(chr(39), chr(39) * 2)}'" for value in args]
    return f"& {' '.join(quoted)}"


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", default="outputs", help="Output directory.")
    parser.add_argument(
        "--layout-model",
        default="models/PP-DocLayoutV3-onnx",
        help="PP-DocLayoutV3 ONNX model directory.",
    )
    parser.add_argument(
        "--layout-provider",
        choices=["auto", "directml", "cpu"],
        default="auto",
        help="Layout execution provider (Windows auto requires DirectML).",
    )
    parser.add_argument(
        "--server-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible VLM server URL."
    )
    parser.add_argument(
        "--api-model-name", default="PaddleOCR-VL-1.6-GGUF.gguf", help="VLM model id."
    )
    parser.add_argument(
        "--vlm-backend",
        choices=["vllm-server", "llama-cpp-server"],
        default="llama-cpp-server",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--vlm-max-workers", type=int, default=1)
    parser.add_argument("--skip-server-check", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL-ROCm lightweight ONNXRuntime + ROCm VLM inference."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    _add_inference_arguments(parser)
    return parser


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paddleocr-vl-rocm run",
        description="Run lightweight PaddleOCR-VL inference.",
    )
    parser.add_argument("input", help="Input image path.")
    _add_inference_arguments(parser)
    return parser


def _setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paddleocr-vl-rocm setup")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--auto", action="store_true", help="Install and start llama-server.")
    start.add_argument("--no-start", action="store_true", help="Install without starting.")
    parser.add_argument("--root", default=None, help="Managed installation root.")
    parser.add_argument("--force", action="store_true", help="Reinstall the pinned runtime.")
    return parser


def _doctor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paddleocr-vl-rocm doctor")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--config", default=None, help="Managed config.json path.")
    parser.add_argument("--server-url", default=None, help="Existing OpenAI-compatible endpoint.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "setup":
        args = _setup_parser().parse_args(values[1:])
        args.command = "setup"
        return args
    if values and values[0] == "doctor":
        args = _doctor_parser().parse_args(values[1:])
        args.command = "doctor"
        return args
    if values and values[0] == "run":
        args = _run_parser().parse_args(values[1:])
        args.command = "run"
        return args
    args = build_parser().parse_args(values)
    args.command = "legacy"
    return args


def _run_inference(args: argparse.Namespace) -> None:
    if not args.skip_server_check:
        try:
            check_openai_compatible_server(args.server_url)
        except RuntimeError as exc:
            raise _ServerError(str(exc)) from exc
    pipeline = PaddleOCRVLROCm(
        layout_model_dir=args.layout_model,
        vlm_server_url=args.server_url,
        api_model_name=args.api_model_name,
        vlm_backend=args.vlm_backend,
        max_new_tokens=args.max_new_tokens,
        timeout=args.timeout,
        seed=args.seed,
        threshold=args.threshold,
        vlm_max_workers=args.vlm_max_workers,
        layout_provider=args.layout_provider,
        skip_server_check=True,
    )
    result = pipeline.predict(args.input)
    output = Path(args.output)
    result.print()
    json_path = result.save_to_json(output)
    md_path = result.save_to_markdown(output, pretty=False)
    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


def _run_setup(args: argparse.Namespace) -> ExitCode:
    root = Path(args.root).expanduser() if args.root else None
    try:
        result = setup_managed_runtime(SetupOptions(root=root, force=args.force))
    except SetupDownloadError as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return ExitCode.DOWNLOAD
    except Exception as exc:
        print(f"Setup environment failed: {exc}", file=sys.stderr)
        return ExitCode.ENVIRONMENT
    if args.auto and not args.no_start:
        try:
            start_managed_server(result)
        except Exception as exc:
            print(f"Server start failed: {exc}", file=sys.stderr)
            return ExitCode.SERVER
    print(f"Managed installation ready: {result.root}")
    run_command = _powershell_command(
        [
            "paddleocr-vl-rocm",
            "run",
            r"C:\path\to\image.png",
            "--layout-model",
            str(result.layout_model_dir),
            "--server-url",
            "http://127.0.0.1:8111/v1",
        ]
    )
    if args.no_start or not args.auto:
        start_command = _powershell_command(
            [
                str(result.llama_server),
                "-m",
                str(result.main_gguf),
                "--mmproj",
                str(result.mmproj),
                "--host",
                "127.0.0.1",
                "--port",
                "8111",
                "-ngl",
                "99",
            ]
        )
        print(f"Start server: {start_command}")
    print(f"Next: {run_command}")
    return ExitCode.OK


def _run_doctor(args: argparse.Namespace) -> ExitCode:
    checks = run_doctor(args.config, server_url=args.server_url)
    if args.json:
        print(checks_to_json(checks))
    else:
        render_checks(checks)
    return ExitCode.ENVIRONMENT if doctor_exit_code(checks) else ExitCode.OK


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "setup":
        return _run_setup(args)
    if args.command == "doctor":
        return _run_doctor(args)
    try:
        _run_inference(args)
    except _ServerError:
        print(
            f"Server error: unable to use {_redacted_url(args.server_url)}; "
            "run 'paddleocr-vl-rocm doctor --server-url <url>'.",
            file=sys.stderr,
        )
        return ExitCode.SERVER
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return ExitCode.INFERENCE
    return ExitCode.OK


if __name__ == "__main__":
    raise SystemExit(main())
