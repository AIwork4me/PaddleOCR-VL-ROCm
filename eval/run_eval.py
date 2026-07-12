"""OmniDocBench staged evaluation orchestrator.

Thin coordinator that runs the OmniDocBench evaluation in independent stages so
the heavy / environment-sensitive steps (VLM inference, eval harness) can each be
gated and run on their own:

  download -> infer -> eval        (or ``all``)

Each stage checks its prerequisites and fails with a clear message rather than
crashing:

  * ``download`` -> ``eval.download_omnidocbench.main`` for the chosen version.
  * ``infer``    -> the PaddleOCR-VL-ROCm adapter's ``run_adapter`` against
    the dataset images dir. Requires the VLM server to be reachable.
  * ``eval``     -> ``python pdf_validation.py --config <yaml>`` inside the
    OmniDocBench checkout at ``eval/.omnidocbench/``. The checkout is NOT
    cloned here (this task is structural only); if absent we print setup
    instructions and exit non-zero. After eval, locate and print the
    ``result/<save>_metric_result.json`` report path.

Live runs require a reachable VLM server and a prepared OmniDocBench checkout;
each stage validates those prerequisites before it runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
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
DEFAULT_SERVER_URL = "http://127.0.0.1:8111/v1"
DEFAULT_LAYOUT_MODEL = "models/PP-DocLayoutV3-onnx"
DEFAULT_API_MODEL_NAME = "PaddleOCR-VL-1.6-GGUF.gguf"
DEFAULT_VLM_BACKEND = "llama-cpp-server"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}


def _load_script_module(name: str, path: Path):
    """Import a sibling script (eval/ is not a package) via importlib."""
    if not path.exists():
        raise SystemExit(f"Required script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_artifact_utils():
    return _load_script_module("eval_artifact_utils", Path("eval/artifact_utils.py"))


def _load_release_contract():
    return _load_script_module("eval_release_contract", Path("eval/release_contract.py"))


def apply_artifact_profile_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "artifact_profile", "default") != "official-local":
        return
    artifacts = _load_artifact_utils()
    paths = artifacts.official_local_paths(args.version, cdm=getattr(args, "cdm", False))
    if args.predictions_dir == str(DEFAULT_PREDICTIONS_DIR):
        args.predictions_dir = paths.predictions_dir.as_posix()
    if getattr(args, "copy_report", None) is None:
        args.copy_report = paths.metric_result.as_posix()
    if getattr(args, "run_summary", None) is None:
        args.run_summary = paths.run_summary.as_posix()
    if getattr(args, "provenance", None) is None:
        args.provenance = paths.provenance.as_posix()


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
    summary = adapter.run_adapter(
        images_dir,
        out_dir,
        server_url,
        engine=args.engine,
        layout_model=args.layout_model,
        api_model_name=args.api_model_name,
        vlm_backend=args.vlm_backend,
        page_retries=args.page_retries,
        fallback_pred_dir=args.fallback_pred_dir,
        limit_pages=args.limit_pages,
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


def _resolve_report_path(checkout: Path, predictions_dir: Path, match_method: str) -> Path:
    """Resolve the OmniDocBench metric report path under the checkout.

    ``pdf_validation.py`` runs with ``cwd=checkout`` and writes its ``result/``
    directory there, so the report lives at
    ``<checkout>/result/<save>_metric_result.json`` where
    ``<save> = basename(prediction.data_path) + "_" + match_method``.
    """
    save = f"{predictions_dir.name}_{match_method}"
    return checkout / RESULT_DIR / f"{save}_metric_result.json"


def _requires_full_prediction_stats(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "cdm", False)
        or getattr(args, "copy_report", None)
        or getattr(args, "run_summary", None)
    )


def _dataset_image_count(args: argparse.Namespace) -> int | None:
    dataset_dir_arg = getattr(args, "dataset_dir", None)
    dataset_dir = Path(dataset_dir_arg) if dataset_dir_arg else VERSION_DATASET_DIRS[args.version]
    images_dir = dataset_dir / "images"
    if not images_dir.is_dir():
        return None
    return sum(1 for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def _validate_release_prediction_stats(args: argparse.Namespace, predictions_dir: Path) -> None:
    if not _requires_full_prediction_stats(args):
        return
    stats_path = predictions_dir / "_run_stats.json"
    if not stats_path.is_file():
        raise SystemExit(
            f"Prediction run stats not found: {stats_path}. Run full unbounded inference first."
        )
    run_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if "limit_pages" not in run_stats or run_stats["limit_pages"] is not None:
        raise SystemExit(
            "Refusing to publish/evaluate official evidence from limited predictions: "
            f"{stats_path}. "
            "Run full unbounded inference first so _run_stats.json has limit_pages=null."
        )
    expected_count = _dataset_image_count(args)
    actual_count = run_stats.get("count")
    if expected_count is None:
        raise SystemExit(
            f"Release evidence requires an available dataset image count: {stats_path}"
        )
    if actual_count != expected_count:
        raise SystemExit(
            f"Prediction count {actual_count} does not match dataset image count "
            f"{expected_count}. Run full unbounded inference before scoring."
        )
    release_contract = _load_release_contract()
    try:
        release_contract.validate_release_run_stats(
            run_stats,
            version=args.version,
            engine=getattr(args, "engine", run_stats.get("engine", "")),
        )
    except ValueError as exc:
        raise SystemExit(f"Release prediction contract failed for {stats_path}: {exc}") from exc


def _render_eval_config(
    base_config: Path, predictions_dir: Path, *, cdm: bool, destination_dir: Path
) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to render the OmniDocBench eval config.") from exc

    config_data = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    eval_config = config_data["end2end_eval"]
    ground_truth_path = Path(eval_config["dataset"]["ground_truth"]["data_path"])
    if not ground_truth_path.is_absolute():
        ground_truth_path = ground_truth_path.expanduser().resolve()
    eval_config["dataset"]["ground_truth"]["data_path"] = str(ground_truth_path)
    eval_config["dataset"]["prediction"]["data_path"] = str(predictions_dir.expanduser().resolve())

    formula_metrics = list(eval_config["metrics"]["display_formula"].get("metric", []))
    if cdm:
        if "CDM" not in formula_metrics:
            formula_metrics.append("CDM")
    else:
        formula_metrics = [metric for metric in formula_metrics if metric != "CDM"]
    eval_config["metrics"]["display_formula"]["metric"] = formula_metrics

    destination_dir.mkdir(parents=True, exist_ok=True)
    rendered = destination_dir / base_config.name
    rendered.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return rendered


def _resolve_eval_python(checkout: Path) -> str:
    for candidate in (
        checkout / ".venv" / "Scripts" / "python.exe",
        checkout / ".venv" / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _validate_scorer_checkout(checkout: Path) -> dict[str, object]:
    benchmark_contract = _load_script_module(
        "benchmark_contract", Path("eval/benchmark_contract.py")
    )
    return benchmark_contract.validate_checkout(checkout)


def stage_eval(args: argparse.Namespace) -> None:
    checkout = _ensure_omnidocbench_checkout()
    checkout_contract = _validate_scorer_checkout(checkout)
    config = Path(args.config or VERSION_CONFIGS[args.version])
    if not config.is_file():
        raise SystemExit(f"OmniDocBench config not found: {config}")

    predictions_dir = Path(args.predictions_dir)
    if not predictions_dir.is_dir():
        raise SystemExit(
            f"Predictions dir not found: {predictions_dir}. Run the 'infer' stage first."
        )
    _validate_release_prediction_stats(args, predictions_dir)
    match_method = args.match_method
    report = _resolve_report_path(checkout, predictions_dir, match_method)
    if report.exists():
        report.unlink()

    # pdf_validation.py is run from the checkout cwd (it writes ./result/ there).
    # Render a runtime config so the subprocess sees the selected prediction
    # directory and explicit CDM/non-CDM formula metrics.
    with tempfile.TemporaryDirectory(prefix="paddleocr_eval_config_") as config_dir:
        rendered_config = _render_eval_config(
            config,
            predictions_dir,
            cdm=bool(getattr(args, "cdm", False)),
            destination_dir=Path(config_dir),
        )
        cmd = [
            _resolve_eval_python(checkout),
            PDF_VALIDATION,
            "--config",
            str(rendered_config.resolve()),
        ]
        print(f"[eval] Running in {checkout}: {' '.join(cmd)}")
        eval_env = {**os.environ, "PYTHONUTF8": "1"}
        result = subprocess.run(cmd, cwd=str(checkout), check=False, env=eval_env)
    if result.returncode != 0:
        raise SystemExit(f"pdf_validation.py exited {result.returncode}")

    if report.exists():
        print(f"[eval] Report ready: {report}")
        copied = None
        summary = None
        if getattr(args, "copy_report", None):
            artifacts = _load_artifact_utils()
            copied = artifacts.copy_metric_report(report, Path(args.copy_report))
            print(f"[eval] Copied report: {copied}")
            if getattr(args, "run_summary", None):
                save_name = f"{predictions_dir.name}_{match_method}"
                summary = artifacts.write_run_summary(
                    save_name=save_name,
                    run_stats_path=predictions_dir / "_run_stats.json",
                    metric_result_path=copied,
                    destination=Path(args.run_summary),
                    cdm=bool(getattr(args, "cdm", False)),
                )
                print(f"[eval] Run summary ready: {summary}")
        if getattr(args, "provenance", None):
            if copied is None:
                raise SystemExit("Writing provenance requires --copy-report.")
            artifacts = _load_artifact_utils()
            dataset_root = Path(
                getattr(args, "dataset_dir", None) or VERSION_DATASET_DIRS[args.version]
            )
            dataset_manifest = dataset_root / "OmniDocBench.json"
            if not dataset_manifest.is_file():
                raise SystemExit(f"Dataset manifest not found: {dataset_manifest}")
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            provenance = artifacts.write_provenance(
                destination=Path(args.provenance),
                git_commit=git_commit,
                engine=getattr(args, "engine", ""),
                server_url=getattr(args, "server_url", DEFAULT_SERVER_URL),
                api_model_name=getattr(args, "api_model_name", DEFAULT_API_MODEL_NAME),
                adapter_command=(
                    "python eval/run_eval.py --stage infer "
                    f"--version {args.version} --engine {getattr(args, 'engine', '')} "
                    "--artifact-profile official-local "
                    f"--server-url {getattr(args, 'server_url', DEFAULT_SERVER_URL)} "
                    f"--api-model-name {getattr(args, 'api_model_name', DEFAULT_API_MODEL_NAME)}"
                ),
                scoring_config_path=config,
                dataset_manifest_path=dataset_manifest,
                predictions_dir=predictions_dir,
                metric_result_paths=[copied],
                run_summary_paths=[] if summary is None else [summary],
                run_stats_path=predictions_dir / "_run_stats.json",
                omnidocbench=checkout_contract,
                dataset_sha256=artifacts.sha256_file(dataset_manifest),
                config_sha256=artifacts.sha256_file(config),
                prediction_manifest_sha256=artifacts.prediction_manifest_sha256(predictions_dir),
            )
            print(f"[eval] Provenance ready: {provenance}")
    else:
        raise SystemExit(
            f"pdf_validation completed but expected report not found at {report}; "
            f"check {(checkout / RESULT_DIR).resolve()} for the *_metric_result.json file."
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
    parser.add_argument("--engine", choices=["lightweight", "official"], default="lightweight")
    parser.add_argument(
        "--vlm-backend",
        default=DEFAULT_VLM_BACKEND,
        help=(
            "VLM backend for the lightweight engine only; ignored by the official engine. "
            f"Default: {DEFAULT_VLM_BACKEND}."
        ),
    )
    parser.add_argument("--page-retries", type=int, default=1)
    parser.add_argument("--fallback-pred-dir", default=None)
    parser.add_argument(
        "--artifact-profile", choices=["default", "official-local"], default="default"
    )
    parser.add_argument("--limit-pages", type=int, default=None)
    parser.add_argument("--copy-report", default=None)
    parser.add_argument("--run-summary", default=None)
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--cdm", action="store_true")
    args = parser.parse_args()
    apply_artifact_profile_defaults(args)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.stage in ("download", "all"):
        stage_download(args)
    if args.stage in ("infer", "all"):
        stage_infer(args)
    if args.stage in ("eval", "all"):
        stage_eval(args)


if __name__ == "__main__":
    main()
