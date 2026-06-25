"""OmniDocBench staged evaluation orchestrator.

Thin coordinator that runs the OmniDocBench evaluation in independent stages so
the heavy / environment-sensitive steps (VLM inference, eval harness) can each be
gated and run on their own:

  download -> infer -> eval        (or ``all``)

Each stage checks its prerequisites and fails with a clear message rather than
crashing:

  * ``download`` -> ``eval.download_omnidocbench.main`` for the chosen version.
  * ``infer``    -> the PaddleOCR-VL-ROCm adapter's ``process_folder`` against
    the dataset images dir. Requires the VLM server to be reachable.
  * ``eval``     -> ``python pdf_validation.py --config <yaml>`` inside the
    OmniDocBench checkout at ``eval/.omnidocbench/``. The checkout is NOT
    cloned here (this task is structural only); if absent we print setup
    instructions and exit non-zero. After eval, locate and print the
    ``result/<save>_metric_result.json`` report path.

Live end-to-end runs are PENDING: there is no VLM server and no OmniDocBench env
in this task's environment. This orchestrator is verified structurally only.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("run_eval")

# Managed OmniDocBench checkout location (gitignored). Not cloned in this task.
OMNIDOCBENCH_CHECKOUT = Path("eval/.omnidocbench")
# Eval entry script relative to the checkout root.
PDF_VALIDATION = "pdf_validation.py"
# OmniDocBench writes results under ./result/; the score table is named
# <save>_metric_result.json where <save> = basename(prediction.data_path)_<match_method>.
RESULT_DIR = Path("result")

# --- version-specific defaults -------------------------------------------------

VERSION_CONFIGS = {
    "v15": "eval/configs/omnidocbench_v15.yaml",
    "v16": "eval/configs/omnidocbench_v16.yaml",
}
VERSION_DATASET_DIRS = {
    "v15": Path("data/omnidocbench/v15"),
    "v16": Path("data/omnidocbench/v16"),
}
DEFAULT_PREDICTIONS_DIR = Path("predictions/paddleocrvl_rocm")
DEFAULT_SERVER_URL = "http://127.0.0.1:8000/v1"
DEFAULT_LAYOUT_MODEL = "models/PP-DocLayoutV3-onnx"
DEFAULT_API_MODEL_NAME = "PaddleOCR-VL-1.5-0.9B"


def _load_script_module(name: str, path: Path):
    """Import a sibling script (eval/ is not a package) via importlib."""
    if not path.exists():
        raise SystemExit(f"Required script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- stages --------------------------------------------------------------------


def stage_download(args: argparse.Namespace) -> None:
    dl = _load_script_module("download_omnidocbench", Path("eval/download_omnidocbench.py"))
    target_dir = args.target_dir or str(VERSION_DATASET_DIRS[args.version])
    # Call main() programmatically by reconstructing the same effect as the CLI.
    revision = args.revision
    if revision is None:
        revision = dl.VERSIONS[args.version]
    resolved = dl.download_dataset(
        args.repo_id,
        Path(target_dir).expanduser().resolve(),
        revision=revision,
    )
    print(f"[download] OmniDocBench {args.version} ready: {resolved}")


def _server_reachable(server_url: str) -> bool:
    try:
        from paddleocr_vl_rocm.server import check_openai_compatible_server
    except ImportError:
        return False
    try:
        check_openai_compatible_server(server_url)
        return True
    except RuntimeError as exc:
        log.error("VLM server check failed: %s", exc)
        return False


def stage_infer(args: argparse.Namespace) -> None:
    server_url = args.server_url
    if not _server_reachable(server_url):
        raise SystemExit(
            f"VLM server not reachable at {server_url}. Start the OpenAI-compatible "
            "server before running the infer stage. (Live run PENDING.)"
        )

    adapter = _load_script_module("paddleocrvl_adapter", Path("eval/PaddleOCRVLROCm_img2md.py"))
    dataset_dir = Path(args.dataset_dir or VERSION_DATASET_DIRS[args.version])
    images_dir = dataset_dir / "images"
    if not images_dir.is_dir():
        raise SystemExit(
            f"Dataset images dir not found: {images_dir}. Run the 'download' stage first."
        )
    out_dir = Path(args.predictions_dir)
    summary = adapter.process_folder(
        images_dir,
        out_dir,
        layout_model=args.layout_model,
        server_url=server_url,
        api_model_name=args.api_model_name,
        vlm_backend="vllm-server",
    )
    print(f"[infer] {summary['ok']}/{summary['count']} pages succeeded -> {out_dir}")


def _ensure_omnidocbench_checkout() -> Path:
    """Return the OmniDocBench checkout root, or raise SystemExit with instructions."""
    # A couple of sentinel files that indicate a real checkout.
    markers = [PDF_VALIDATION, "src/cli.py"]
    checkout_ok = OMNIDOCBENCH_CHECKOUT.is_dir() and all(
        (OMNIDOCBENCH_CHECKOUT / m).exists() for m in markers
    )
    if checkout_ok:
        return OMNIDOCBENCH_CHECKOUT.resolve()
    raise SystemExit(
        "OmniDocBench checkout not found at "
        f"{OMNIDOCBENCH_CHECKOUT.resolve()}.\n"
        "This task does NOT clone it. Set it up manually, e.g.:\n"
        "  git clone https://github.com/opendatalab/OmniDocBench.git "
        f"{OMNIDOCBENCH_CHECKOUT}\n"
        f"  cd {OMNIDOCBENCH_CHECKOUT} && pip install -e .\n"
        "(Pin a commit for reproducibility.) Live run PENDING."
    )


def _resolve_report_path(config_path: Path, predictions_dir: Path, match_method: str) -> Path:
    save = f"{predictions_dir.name}_{match_method}"
    return RESULT_DIR / f"{save}_metric_result.json"


def stage_eval(args: argparse.Namespace) -> None:
    checkout = _ensure_omnidocbench_checkout()
    config = Path(args.config or VERSION_CONFIGS[args.version])
    if not config.is_file():
        raise SystemExit(f"OmniDocBench config not found: {config}")

    predictions_dir = Path(args.predictions_dir)
    if not predictions_dir.is_dir():
        raise SystemExit(
            f"Predictions dir not found: {predictions_dir}. Run the 'infer' stage first."
        )

    # pdf_validation.py is run from the checkout cwd (it writes ./result/ there).
    cmd = [sys.executable, PDF_VALIDATION, "--config", str(config.resolve())]
    print(f"[eval] Running in {checkout}: {' '.join(cmd)}")
    # NOTE: not actually executed during this structural task. When run for real:
    result = subprocess.run(cmd, cwd=str(checkout), check=False)
    if result.returncode != 0:
        raise SystemExit(f"pdf_validation.py exited {result.returncode}")

    # Match the report path. match_method defaults to quick_match per config templates.
    match_method = args.match_method
    report = _resolve_report_path(config, predictions_dir, match_method)
    if report.exists():
        print(f"[eval] Report ready: {report}")
    else:
        print(
            f"[eval] pdf_validation completed but expected report not found at {report}; "
            f"check {RESULT_DIR.resolve()} for the *_metric_result.json file."
        )


# --- entrypoint ----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OmniDocBench staged evaluation orchestrator (download/infer/eval/all)."
    )
    parser.add_argument(
        "--stage",
        choices=["download", "infer", "eval", "all"],
        default="all",
        help="Stage to run. Default: all.",
    )
    parser.add_argument("--version", choices=sorted(VERSION_CONFIGS), default="v16")
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Override dataset target dir (download stage). Default: data/omnidocbench/<version>.",
    )
    parser.add_argument("--dataset-dir", default=None, help="Dataset root dir (infer stage).")
    parser.add_argument(
        "--predictions-dir",
        default=str(DEFAULT_PREDICTIONS_DIR),
        help=f"Predictions output dir (infer/eval stages). Default: {DEFAULT_PREDICTIONS_DIR}.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Override OmniDocBench eval config yaml (eval stage). "
            "Default: eval/configs/omnidocbench_<version>.yaml."
        ),
    )
    parser.add_argument(
        "--match-method",
        default="quick_match",
        help="match_method used to locate the report (<save>_<match_method>_metric_result.json).",
    )
    parser.add_argument("--repo-id", default="opendatalab/OmniDocBench")
    parser.add_argument(
        "--revision", default=None, help="Override HF dataset revision (download stage)."
    )
    parser.add_argument(
        "--server-url", default=DEFAULT_SERVER_URL, help="VLM OpenAI-compatible base URL."
    )
    parser.add_argument("--layout-model", default=DEFAULT_LAYOUT_MODEL)
    parser.add_argument("--api-model-name", default=DEFAULT_API_MODEL_NAME)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.stage in ("download", "all"):
        stage_download(args)
    if args.stage in ("infer", "all"):
        stage_infer(args)
    if args.stage in ("eval", "all"):
        stage_eval(args)


if __name__ == "__main__":
    main()
