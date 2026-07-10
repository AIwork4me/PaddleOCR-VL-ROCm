"""PaddleOCR-VL-ROCm adapter for OmniDocBench."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import time
import traceback
from collections.abc import Iterable
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
DEFAULT_ENGINE = "lightweight"
DEFAULT_LOCAL_API_MODEL_NAME = "PaddleOCR-VL-1.6-GGUF.gguf"
DEFAULT_VLM_BACKEND = "llama-cpp-server"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")


def expected_md_name(image_name: str) -> str:
    """Return the Markdown filename OmniDocBench's matcher looks up."""
    return Path(image_name).stem + ".md"


def _read_env_local(repo_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    env_file = repo_root / ".env.local"
    if not env_file.is_file():
        return values
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _read_adapter_env() -> dict[str, str]:
    root_values = _read_env_local(REPO_ROOT)
    adapter_values = _read_env_local(ADAPTER_DIR)
    return {**root_values, **adapter_values}


def run_adapter(
    img_dir,
    out_dir,
    server_url: str = "",
    *,
    engine: str = DEFAULT_ENGINE,
    layout_model: str | None = None,
    api_model_name: str | None = None,
    vlm_backend: str = DEFAULT_VLM_BACKEND,
    page_retries: int = 1,
    fallback_pred_dir: str | Path | None = None,
) -> dict:
    env = _read_adapter_env()
    default_layout = (
        layout_model
        or os.environ.get("ADAPTER_LAYOUT_MODEL")
        or env.get("PP_DOCLAYOUTV3_ONNX_DIR")
        or "models/PP-DocLayoutV3-onnx"
    )
    llama_host = env.get("LLAMA_HOST") or "127.0.0.1"
    llama_port = env.get("LLAMA_PORT") or "8111"
    resolved_server = (
        server_url
        or os.environ.get("ADAPTER_SERVER_URL")
        or f"http://{llama_host}:{llama_port}/v1"
    )
    default_api_model = (
        api_model_name
        or os.environ.get("ADAPTER_API_MODEL_NAME")
        or env.get("VL_REC_API_MODEL_NAME")
        or DEFAULT_LOCAL_API_MODEL_NAME
    )

    selected_engine = (engine or DEFAULT_ENGINE).strip().lower()
    if selected_engine == "lightweight":
        return run_lightweight_folder(
            img_dir=Path(img_dir),
            out_dir=Path(out_dir),
            layout_model=default_layout,
            server_url=resolved_server,
            api_model_name=default_api_model,
            vlm_backend=vlm_backend,
        )
    if selected_engine == "official":
        return run_official_folder(
            img_dir=Path(img_dir),
            out_dir=Path(out_dir),
            server_url=resolved_server,
            api_model_name=default_api_model,
            page_retries=page_retries,
            fallback_pred_dir=Path(fallback_pred_dir) if fallback_pred_dir else None,
        )
    raise ValueError(f"Unsupported engine '{engine}'. Use lightweight or official.")


def run_lightweight_folder(
    *,
    img_dir: Path,
    out_dir: Path,
    layout_model: str = "models/PP-DocLayoutV3-onnx",
    server_url: str = "http://127.0.0.1:8000/v1",
    api_model_name: str = DEFAULT_LOCAL_API_MODEL_NAME,
    vlm_backend: str = DEFAULT_VLM_BACKEND,
) -> dict:
    """Run the local lightweight pipeline over every image in ``img_dir``."""
    if not img_dir.is_dir():
        raise SystemExit(f"Image directory not found: {img_dir}")
    try:
        PipelineClass = PaddleOCRVLROCm  # type: ignore[name-defined]
    except NameError:
        from paddleocr_vl_rocm import PaddleOCRVLROCm as PipelineClass
    pipeline = PipelineClass(
        layout_model_dir=layout_model,
        vlm_server_url=server_url,
        api_model_name=api_model_name,
        vlm_backend=vlm_backend,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_path = out_dir / "_errors.log"
    errors_path.unlink(missing_ok=True)
    stats: list[dict] = []
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    for img in images:
        start = time.time()
        destination = out_dir / expected_md_name(img.name)
        destination.unlink(missing_ok=True)
        try:
            result = pipeline.predict(img)
            destination.write_text(result.markdown_text, encoding="utf-8")
            stats.append(
                {"image": img.name, "status": "ok", "seconds": round(time.time() - start, 2)}
            )
        except Exception as exc:  # noqa: BLE001 - record failure, continue
            tb = traceback.format_exc()
            with open(out_dir / "_errors.log", "a", encoding="utf-8") as fh:
                fh.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {img.name}: {exc}\n{tb}\n")
            stats.append(
                {
                    "image": img.name,
                    "status": f"failed: {exc}",
                    "seconds": round(time.time() - start, 2),
                    "traceback": tb,
                }
            )

    ok_count = sum(1 for s in stats if s["status"] == "ok")
    summary = {
        "count": len(images),
        "ok": ok_count,
        "fail": len(images) - ok_count,
        "fallback": 0,
        "engine": "lightweight",
        "stats": stats,
    }
    (out_dir / "_run_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if len(images) > 0 and ok_count < 0.5 * len(images):
        raise SystemExit(2)
    return summary


def process_folder(
    img_dir: Path,
    out_dir: Path,
    *,
    layout_model: str = "models/PP-DocLayoutV3-onnx",
    server_url: str = "http://127.0.0.1:8000/v1",
    api_model_name: str = DEFAULT_LOCAL_API_MODEL_NAME,
    vlm_backend: str = DEFAULT_VLM_BACKEND,
) -> dict:
    return run_lightweight_folder(
        img_dir=img_dir,
        out_dir=out_dir,
        layout_model=layout_model,
        server_url=server_url,
        api_model_name=api_model_name,
        vlm_backend=vlm_backend,
    )


def _official_result_to_markdown(result: object) -> str:
    def markdown_from_mapping(value: dict) -> str | None:
        for key in ("markdown_texts", "markdown", "md", "content", "markdown_text", "text"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        return None

    if isinstance(result, str):
        return result
    official_export = getattr(result, "_to_markdown", None)
    if callable(official_export):
        try:
            exported = official_export(pretty=False)
        except TypeError:
            exported = None
        if isinstance(exported, dict):
            mapped = markdown_from_mapping(exported)
            if mapped is not None:
                return mapped
        if isinstance(exported, str):
            return exported
    markdown = getattr(result, "markdown", None)
    if isinstance(markdown, str):
        return markdown
    if isinstance(markdown, dict):
        mapped = markdown_from_mapping(markdown)
        if mapped is not None:
            return mapped
    if isinstance(result, dict):
        mapped = markdown_from_mapping(result)
        if mapped is not None:
            return mapped
    json_value = getattr(result, "json", None)
    if isinstance(json_value, dict):
        mapped = markdown_from_mapping(json_value)
        if mapped is not None:
            return mapped
        res = json_value.get("res")
        if isinstance(res, dict):
            mapped = markdown_from_mapping(res)
            if mapped is not None:
                return mapped
    for method_name in ("to_markdown", "export_markdown"):
        method = getattr(result, method_name, None)
        if callable(method):
            value = method()
            if isinstance(value, str):
                return value
    raise TypeError("Official PaddleOCRVL result did not expose Markdown text.")


_CENTERED_IMAGE_DIV_RE = re.compile(
    r"<div[^>]*style=[\"'][^\"']*text-align:\s*center;?[^\"']*[\"'][^>]*>\s*"
    r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"'][^>]*>\s*</div>",
    re.IGNORECASE | re.DOTALL,
)
_CENTERED_TEXT_DIV_RE = re.compile(
    r"<div[^>]*style=[\"'][^\"']*text-align:\s*center;?[^\"']*[\"'][^>]*>\s*(.*?)\s*</div>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _normalize_official_markdown_for_omnidocbench(markdown: str) -> str:
    def replace_image(match: re.Match[str]) -> str:
        return f"![]({html.unescape(match.group(1))})"

    def replace_text(match: re.Match[str]) -> str:
        inner = _HTML_TAG_RE.sub("", match.group(1))
        return html.unescape(inner.strip())

    markdown = _CENTERED_IMAGE_DIV_RE.sub(replace_image, markdown)
    markdown = _CENTERED_TEXT_DIV_RE.sub(replace_text, markdown)
    return re.sub(r"\n{3,}", "\n\n", markdown)


def run_official_folder(
    *,
    img_dir: Path,
    out_dir: Path,
    server_url: str,
    api_model_name: str,
    page_retries: int = 1,
    fallback_pred_dir: Path | None = None,
) -> dict:
    if not img_dir.is_dir():
        raise SystemExit(f"Image directory not found: {img_dir}")
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise RuntimeError(
            "Official engine requires PaddleOCR. Install the local PaddleOCR dependency first."
        ) from exc

    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_backend="llama-cpp-server",
        vl_rec_server_url=server_url,
        vl_rec_api_model_name=api_model_name,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_path = out_dir / "_errors.log"
    stats_path = out_dir / "_run_stats.json"
    errors_path.unlink(missing_ok=True)
    stats_path.unlink(missing_ok=True)

    stats: list[dict] = []
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    page_retries = max(0, int(page_retries))

    for img in images:
        start = time.time()
        destination = out_dir / expected_md_name(img.name)
        fallback_path = (
            fallback_pred_dir / expected_md_name(img.name)
            if fallback_pred_dir is not None
            else None
        )
        fallback_is_destination = (
            fallback_path is not None and fallback_path.resolve() == destination.resolve()
        )
        if not fallback_is_destination:
            destination.unlink(missing_ok=True)
        last_exc: Exception | None = None
        last_tb = ""
        attempts = 0
        for attempt in range(page_retries + 1):
            attempts = attempt + 1
            try:
                result = pipeline.predict(str(img))
                if isinstance(result, Iterable) and not isinstance(result, (str, bytes, dict)):
                    result = list(result)
                if isinstance(result, list):
                    if not result:
                        raise RuntimeError("Official PaddleOCRVL predict() returned no page results.")
                    markdown = "\n\n".join(_official_result_to_markdown(item) for item in result)
                else:
                    markdown = _official_result_to_markdown(result)
                markdown = _normalize_official_markdown_for_omnidocbench(markdown)
                destination.write_text(markdown, encoding="utf-8")
                stats.append(
                    {
                        "image": img.name,
                        "status": "ok",
                        "seconds": round(time.time() - start, 2),
                        "attempts": attempts,
                    }
                )
                break
            except Exception as exc:
                last_exc = exc
                last_tb = traceback.format_exc()
                if attempt < page_retries:
                    time.sleep(min(2.0, 0.25 * attempts))
                    continue
        else:
            if fallback_path is not None and fallback_path.is_file():
                if not fallback_is_destination:
                    shutil.copyfile(fallback_path, destination)
                status = f"fallback: {last_exc}"
            else:
                status = f"failed: {last_exc}"
            with open(errors_path, "a", encoding="utf-8") as fh:
                fh.write(
                    f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {img.name}: {last_exc} "
                    f"(attempts={attempts})\n{last_tb}\n"
                )
                if fallback_path is not None and fallback_path.is_file():
                    action = "retained at" if fallback_is_destination else "copied from"
                    fh.write(f"FALLBACK prediction {action}: {fallback_path}\n")
            stats.append(
                {
                    "image": img.name,
                    "status": status,
                    "seconds": round(time.time() - start, 2),
                    "attempts": attempts,
                    "traceback": last_tb,
                }
            )

    ok_count = sum(1 for s in stats if s["status"] == "ok" or s["status"].startswith("fallback:"))
    fallback_count = sum(1 for s in stats if s["status"].startswith("fallback:"))
    summary = {
        "count": len(images),
        "ok": ok_count,
        "fail": len(images) - ok_count,
        "fallback": fallback_count,
        "engine": "official",
        "stats": stats,
    }
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(images) > 0 and ok_count < 0.5 * len(images):
        raise SystemExit(2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL-ROCm adapter for OmniDocBench: write per-page .md"
    )
    parser.add_argument("--img-dir", required=True, help="Dataset images directory.")
    parser.add_argument(
        "--out-dir", required=True, help="Output flat dir of <basename>.md predictions."
    )
    parser.add_argument("--layout-model", default="models/PP-DocLayoutV3-onnx")
    parser.add_argument("--server-url", default="")
    parser.add_argument("--api-model-name", default=None)
    parser.add_argument(
        "--vlm-backend",
        default=DEFAULT_VLM_BACKEND,
        help=(
            "VLM backend for the lightweight engine only; ignored by the official engine. "
            f"Default: {DEFAULT_VLM_BACKEND}."
        ),
    )
    parser.add_argument("--engine", choices=["lightweight", "official"], default=DEFAULT_ENGINE)
    parser.add_argument(
        "--page-retries", type=int, default=int(os.environ.get("PADDLEOCR_VL_PAGE_RETRIES", "1"))
    )
    parser.add_argument("--fallback-pred-dir", default=os.environ.get("PADDLEOCR_VL_FALLBACK_PRED_DIR"))
    args = parser.parse_args()
    summary = run_adapter(
        Path(args.img_dir),
        Path(args.out_dir),
        args.server_url,
        engine=args.engine,
        layout_model=args.layout_model,
        api_model_name=args.api_model_name,
        vlm_backend=args.vlm_backend,
        page_retries=args.page_retries,
        fallback_pred_dir=args.fallback_pred_dir,
    )
    print(summary)


if __name__ == "__main__":
    main()
