"""PaddleOCR-VL-ROCm adapter for OmniDocBench."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import tempfile
import time
import traceback
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from eval.task5_comparison import BOUNDARIES, normalize_scorer_markdown, observation, unobservable
from paddleocr_vl_rocm.timing import summarize_seconds

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
DEFAULT_ENGINE = "lightweight"
DEFAULT_LOCAL_API_MODEL_NAME = "PaddleOCR-VL-1.6-GGUF.gguf"
DEFAULT_VLM_BACKEND = "llama-cpp-server"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")


def expected_md_name(image_name: str) -> str:
    """Return the Markdown filename OmniDocBench's matcher looks up."""
    return Path(image_name).stem + ".md"


def iter_images(img_dir: Path, limit_pages: int | None = None) -> list[Path]:
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if limit_pages is None:
        return images
    limit = max(0, int(limit_pages))
    return images[:limit]


def _validated_images(img_dir: Path, limit_pages: int | None) -> list[Path]:
    all_images = iter_images(img_dir)
    by_stem: dict[str, list[str]] = {}
    for image in all_images:
        by_stem.setdefault(image.stem, []).append(image.name)
    duplicates = {stem: names for stem, names in by_stem.items() if len(names) > 1}
    if duplicates:
        raise ValueError(f"Duplicate output stem(s): {duplicates}")
    if limit_pages is None:
        return all_images
    return all_images[: max(0, int(limit_pages))]


def _prepare_trace_dir(trace_dir: Path | None) -> None:
    if trace_dir is None:
        return
    if trace_dir.exists():
        if not trace_dir.is_dir():
            raise ValueError(f"Trace path must be a directory: {trace_dir}")
        if next(trace_dir.iterdir(), None) is not None:
            raise ValueError(f"Trace directory must be empty before inference: {trace_dir}")
        return
    trace_dir.mkdir(parents=True)


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
    limit_pages: int | None = None,
    trace_dir: str | Path | None = None,
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
        server_url or os.environ.get("ADAPTER_SERVER_URL") or f"http://{llama_host}:{llama_port}/v1"
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
            limit_pages=limit_pages,
            trace_dir=Path(trace_dir) if trace_dir is not None else None,
        )
    if selected_engine == "official":
        return run_official_folder(
            img_dir=Path(img_dir),
            out_dir=Path(out_dir),
            server_url=resolved_server,
            api_model_name=default_api_model,
            page_retries=page_retries,
            fallback_pred_dir=Path(fallback_pred_dir) if fallback_pred_dir else None,
            limit_pages=limit_pages,
            trace_dir=Path(trace_dir) if trace_dir is not None else None,
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
    limit_pages: int | None = None,
    trace_dir: Path | None = None,
) -> dict:
    """Run the local lightweight pipeline over every image in ``img_dir``."""
    if not img_dir.is_dir():
        raise SystemExit(f"Image directory not found: {img_dir}")
    images = _validated_images(img_dir, limit_pages)
    _prepare_trace_dir(trace_dir)
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
    pipeline._layout()
    layout_provider_requested = pipeline.layout_provider_requested
    layout_providers_active = list(pipeline.layout_providers_active)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_path = out_dir / "_errors.log"
    errors_path.unlink(missing_ok=True)
    stats: list[dict] = []
    for img in images:
        start = time.time()
        destination = out_dir / expected_md_name(img.name)
        trace_path = trace_dir / f"{img.stem}.jsonl" if trace_dir is not None else None
        destination.unlink(missing_ok=True)
        if trace_path is not None:
            trace_path.unlink(missing_ok=True)
        try:
            trace_events: list[dict[str, Any]] | None = [] if trace_dir is not None else None
            if trace_events is None:
                result = pipeline.predict(img)
            else:
                result = pipeline.predict(img, vlm_trace_events=trace_events)
            if trace_events is not None:
                canonical_events = _lightweight_page_trace_events(
                    img.stem, trace_events, result.markdown_text
                )
                _write_trace_jsonl(trace_path, canonical_events)
            destination.write_text(result.markdown_text, encoding="utf-8")
            page_stats = {
                "image": img.name,
                "status": "ok",
                "seconds": round(time.time() - start, 2),
            }
            if pipeline.last_timing is not None:
                page_stats["timing"] = dict(pipeline.last_timing)
            stats.append(page_stats)
        except Exception as exc:  # noqa: BLE001 - record failure, continue
            if trace_path is not None:
                trace_path.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)
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
    successful_stats = [item for item in stats if item["status"] == "ok"]
    stage_names = (
        "decode_seconds",
        "layout_seconds",
        "crop_encode_seconds",
        "vlm_seconds",
        "finalize_seconds",
        "total_seconds",
    )
    summary = {
        "count": len(images),
        "ok": ok_count,
        "fail": len(images) - ok_count,
        "fallback": 0,
        "engine": "lightweight",
        "layout_provider_requested": layout_provider_requested,
        "layout_providers_active": layout_providers_active,
        "limit_pages": limit_pages,
        "timing": summarize_seconds([float(item["seconds"]) for item in successful_stats]),
        "stats": stats,
    }
    stage_events = [item["timing"] for item in successful_stats if "timing" in item]
    if stage_events:
        summary["stage_timing"] = {
            stage: summarize_seconds([float(event[stage]) for event in stage_events])
            for stage in stage_names
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
    limit_pages: int | None = None,
    trace_dir: Path | None = None,
) -> dict:
    return run_lightweight_folder(
        img_dir=img_dir,
        out_dir=out_dir,
        layout_model=layout_model,
        server_url=server_url,
        api_model_name=api_model_name,
        vlm_backend=vlm_backend,
        limit_pages=limit_pages,
        trace_dir=trace_dir,
    )


def _canonical_boundary(value: object, *, available: bool) -> dict[str, str]:
    return observation(value) if available else unobservable()


def _prehashed_observation(value: object) -> dict[str, str]:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        return unobservable()
    return {"status": "observable", "fingerprint": value}


def _lightweight_page_trace_events(
    page: str, events: list[dict[str, Any]], markdown: str
) -> list[dict[str, object]]:
    fields = {
        "request_order": ("request_order",),
        "label": ("block_label", "label"),
        "bbox": ("block_bbox", "bbox"),
        "crop_pixels": ("image_sha256",),
        "prompt": ("prompt",),
        "payload": ("payload",),
        "raw_result": ("raw_result_sha256",),
        "postprocess": ("final_result_sha256",),
    }
    page_postprocess = observation(normalize_scorer_markdown(markdown))
    if not events:
        return [unobservable_page_trace(page, markdown)]
    canonical: list[dict[str, object]] = []
    for block_index, event in enumerate(events):
        boundaries: dict[str, dict[str, str]] = {}
        for boundary, names in fields.items():
            found, value = _direct_field(event, *names)
            boundaries[boundary] = (
                _prehashed_observation(value)
                if boundary == "crop_pixels" and found
                else _canonical_boundary(value, available=found)
            )
        canonical.append(
            {
                "page": page,
                "block_index": block_index,
                "boundaries": boundaries,
                "page_postprocess": page_postprocess,
            }
        )
    return canonical


def _write_trace_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(
        json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for event in events
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _direct_field(value: Mapping[str, object], *names: str) -> tuple[bool, object]:
    for name in names:
        if name in value and value[name] is not None:
            return True, value[name]
    return False, None


def _official_mapping(result: object) -> Mapping[str, object] | None:
    if isinstance(result, Mapping):
        value = result
    else:
        json_value = getattr(result, "json", None)
        if not isinstance(json_value, Mapping):
            return None
        value = json_value
    nested = value.get("res")
    return nested if isinstance(nested, Mapping) else value


def _extract_authenticated_official_blocks(
    result: object,
) -> list[Mapping[str, object]] | None:
    if isinstance(result, list):
        combined: list[Mapping[str, object]] = []
        for item in result:
            blocks = _extract_authenticated_official_blocks(item)
            if blocks is None:
                return None
            combined.extend(blocks)
        return combined
    value = _official_mapping(result)
    if value is None:
        return None
    blocks = value.get("parsing_res_list")
    if not isinstance(blocks, list) or not all(isinstance(item, Mapping) for item in blocks):
        return None
    return blocks


def unobservable_page_trace(page: str, markdown: str) -> dict[str, object]:
    """Return a page record when no authenticated block collection is available."""
    return {
        "page": page,
        "block_index": None,
        "block_structure": unobservable(),
        "boundaries": {name: unobservable() for name in BOUNDARIES},
        "page_postprocess": observation(normalize_scorer_markdown(markdown)),
    }


def official_page_trace(page: str, result: object, markdown: str) -> dict[str, object]:
    """Compatibility wrapper for conservative Official page traces."""
    return unobservable_page_trace(page, markdown)


def _official_page_trace_events(
    page: str, result: object, markdown: str
) -> list[dict[str, object]]:
    blocks = _extract_authenticated_official_blocks(result)
    if not blocks:
        return [unobservable_page_trace(page, markdown)]
    page_postprocess = observation(normalize_scorer_markdown(markdown))
    events: list[dict[str, object]] = []
    for block_index, block in enumerate(blocks):
        request_found, request = _direct_field(block, "request")
        request_mapping = request if request_found and isinstance(request, Mapping) else None

        label_found, label = _direct_field(block, "block_label", "label")
        bbox_found, bbox = _direct_field(block, "block_bbox", "bbox", "coordinate")
        crop_sha_found, crop_sha = _direct_field(block, "image_sha256", "crop_sha256")
        crop_pixels_found, crop_pixels = _direct_field(block, "crop_pixels")
        prompt_found, prompt = _direct_field(block, "prompt")
        if not prompt_found and request_mapping is not None:
            prompt_found, prompt = _direct_field(request_mapping, "prompt")
        payload_found, payload = _direct_field(block, "payload")
        if not payload_found and request_mapping is not None:
            payload_found, payload = _direct_field(request_mapping, "payload")
        raw_found, raw = _direct_field(block, "raw_result", "raw_text")
        post_found, post = _direct_field(block, "block_content", "postprocess", "content")
        if raw_found and isinstance(raw, str):
            raw = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if post_found and isinstance(post, str):
            post = hashlib.sha256(post.encode("utf-8")).hexdigest()
        if crop_pixels_found and isinstance(crop_pixels, bytes):
            crop_boundary = _prehashed_observation(hashlib.sha256(crop_pixels).hexdigest())
        elif crop_sha_found:
            crop_boundary = _prehashed_observation(crop_sha)
        else:
            crop_boundary = unobservable()
        values = {
            "request_order": _direct_field(block, "request_order"),
            "label": (label_found, label),
            "bbox": (bbox_found, bbox),
            "prompt": (prompt_found, prompt),
            "payload": (payload_found, payload),
            "raw_result": (raw_found, raw),
            "postprocess": (post_found, post),
        }
        events.append(
            {
                "page": page,
                "block_index": block_index,
                "boundaries": {
                    name: _canonical_boundary(value, available=available)
                    for name, (available, value) in values.items()
                }
                | {"crop_pixels": crop_boundary},
                "page_postprocess": page_postprocess,
            }
        )
    return events


_CENTERED_IMAGE_DIV_RE = re.compile(
    r"<div[^>]*style=[\"'][^\"']*text-align:\s*center;?[^\"']*[\"'][^>]*>\s*"
    r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"'][^>]*>\s*</div>",
    re.IGNORECASE | re.DOTALL,
)
_CENTERED_TEXT_DIV_RE = re.compile(
    r"<div[^>]*style=[\"'][^\"']*text-align:\s*center;?[^\"']*[\"'][^>]*>\s*(.*?)\s*</div>",
    re.IGNORECASE | re.DOTALL,
)
_INLINE_FORMATTING_TAG_RE = re.compile(r"</?(?:b|strong|i|em|span)\b[^>]*>", re.IGNORECASE)


def _normalize_official_markdown_for_omnidocbench(markdown: str) -> str:
    def replace_image(match: re.Match[str]) -> str:
        return f"![]({html.unescape(match.group(1))})"

    def replace_text(match: re.Match[str]) -> str:
        inner = _INLINE_FORMATTING_TAG_RE.sub("", match.group(1))
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
    limit_pages: int | None = None,
    trace_dir: Path | None = None,
) -> dict:
    if not img_dir.is_dir():
        raise SystemExit(f"Image directory not found: {img_dir}")
    images = _validated_images(img_dir, limit_pages)
    _prepare_trace_dir(trace_dir)
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
    page_retries = max(0, int(page_retries))

    for img in images:
        start = time.time()
        destination = out_dir / expected_md_name(img.name)
        trace_path = trace_dir / f"{img.stem}.jsonl" if trace_dir is not None else None
        if trace_path is not None:
            trace_path.unlink(missing_ok=True)
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
                        raise RuntimeError(
                            "Official PaddleOCRVL predict() returned no page results."
                        )
                    markdown = "\n\n".join(_official_result_to_markdown(item) for item in result)
                else:
                    markdown = _official_result_to_markdown(result)
                markdown = _normalize_official_markdown_for_omnidocbench(markdown)
                if trace_dir is not None:
                    _write_trace_jsonl(
                        trace_path, _official_page_trace_events(img.stem, result, markdown)
                    )
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
                if trace_path is not None:
                    trace_path.unlink(missing_ok=True)
                if attempt < page_retries:
                    time.sleep(min(2.0, 0.25 * attempts))
                    continue
        else:
            if trace_path is not None:
                trace_path.unlink(missing_ok=True)
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
        "limit_pages": limit_pages,
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
    parser.add_argument(
        "--fallback-pred-dir", default=os.environ.get("PADDLEOCR_VL_FALLBACK_PRED_DIR")
    )
    parser.add_argument("--limit-pages", type=int, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
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
        limit_pages=args.limit_pages,
        trace_dir=args.trace_dir,
    )
    print(summary)


if __name__ == "__main__":
    main()
