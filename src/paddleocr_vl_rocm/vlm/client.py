from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from ..constants import IMAGE_LABELS, PROMPTS
from ..contracts import VLMRequestContract, redact
from ..encoding import _data_url_from_bytes, _image_data_url, _jpeg_bytes, _png_bytes, _sha256_hex
from ..server import normalize_server_url
from ..utils import get_logger

_logger = get_logger(__name__)


def _vlm_cache_key(
    model: str,
    prompt: str,
    image_sha256: str,
    max_new_tokens: int | None,
    seed: int,
) -> str:
    return json.dumps(
        {
            "schema": "paddleocr-vl-local-vlm-cache-v1",
            "model": model,
            "prompt": prompt,
            "image_sha256": image_sha256,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _load_vlm_compat_cache(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("entries"), dict):
        data = data["entries"]
    if not isinstance(data, dict):
        raise ValueError(f"Invalid VLM compatibility cache: {path}")
    cache: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"Invalid VLM compatibility cache entry in {path}")
        cache[key] = value
    return cache


def _content_from_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _completion_payload(
    backend: str,
    model: str,
    prompt: str,
    image_url: str,
    max_new_tokens: int | None,
    seed: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    if backend == "llama-cpp-server":
        payload.update(
            {
                "seed": seed,
                "top_p": 1.0,
                "skip_special_tokens": True,
                "top_k": 1,
                "min_p": 0.0,
                "repeat_penalty": 1.0,
                "cache_prompt": False,
            }
        )
        if max_new_tokens is not None:
            payload["max_tokens"] = max_new_tokens
    elif backend == "vllm-server":
        payload["skip_special_tokens"] = True
        if min_pixels is not None or max_pixels is not None:
            payload["mm_processor_kwargs"] = {}
            if min_pixels is not None:
                payload["mm_processor_kwargs"]["min_pixels"] = min_pixels
            if max_pixels is not None:
                payload["mm_processor_kwargs"]["max_pixels"] = max_pixels
        if max_new_tokens is not None:
            payload["max_completion_tokens"] = max_new_tokens
    elif max_new_tokens is not None:
        payload["max_tokens"] = max_new_tokens
    return payload


class OpenAICompatibleVLMClient:
    def __init__(
        self,
        server_url: str,
        model: str,
        timeout: float,
        backend: str = "llama-cpp-server",
        seed: int = 1,
        compat_cache: dict[str, str] | None = None,
        request_observer: Callable[[VLMRequestContract], None] | None = None,
    ) -> None:
        if backend not in {"llama-cpp-server", "vllm-server"}:
            raise ValueError(
                "Unsupported VLM backend for the lightweight parser: "
                f"{backend}. Expected 'llama-cpp-server' or 'vllm-server'."
            )
        self.backend = backend
        self.base_url = normalize_server_url(server_url)
        self.model = model
        self.timeout = timeout
        self.seed = seed
        self._cache: dict[str, str] = {}
        self._compat_cache = compat_cache or {}
        self._request_observer = request_observer

    def complete_image(
        self,
        prompt: str,
        image: Image.Image | None = None,
        image_path: Path | None = None,
        max_new_tokens: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        use_client_cache: bool = True,
    ) -> str:
        if image is None and image_path is None:
            raise ValueError("Either image or image_path is required.")
        if image is not None:
            if self.backend == "vllm-server":
                image_bytes = _jpeg_bytes(image)
                image_url = _data_url_from_bytes(image_bytes, "image/jpeg")
            else:
                image_bytes = _png_bytes(image)
                image_url = _data_url_from_bytes(image_bytes, "image/png")
        else:
            image_path = image_path  # type: ignore[assignment]
            image_bytes = image_path.read_bytes()  # type: ignore[union-attr]
            image_url = _image_data_url(image_path)  # type: ignore[arg-type]
        image_sha256 = _sha256_hex(image_bytes)
        cache_key = _vlm_cache_key(
            self.model,
            prompt=prompt,
            image_sha256=image_sha256,
            max_new_tokens=max_new_tokens,
            seed=self.seed,
        )
        if use_client_cache and cache_key in self._cache:
            return self._cache[cache_key]
        if use_client_cache and cache_key in self._compat_cache:
            text = self._compat_cache[cache_key]
            self._cache[cache_key] = text
            return text
        payload = _completion_payload(
            self.backend,
            self.model,
            prompt=prompt,
            image_url=image_url,
            max_new_tokens=max_new_tokens,
            seed=self.seed,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if self._request_observer is not None:
            self._request_observer(
                VLMRequestContract(
                    backend=self.backend,
                    model=self.model,
                    prompt=prompt,
                    image_format="JPEG" if self.backend == "vllm-server" else "PNG",
                    image_sha256=image_sha256,
                    image_size=image.size if image is not None else (0, 0),
                    payload=redact(payload),
                )
            )
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                    _logger.warning(
                        "VLM request %s (attempt %d), retrying in %.1fs",
                        response.status_code,
                        attempt + 1,
                        1.5 * (attempt + 1),
                    )
                    time.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = exc
                _logger.warning("VLM request error (attempt %d): %s", attempt + 1, exc)
                if attempt >= 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError("VLM request failed without a response.")
        text = _content_from_response(response.json())
        if use_client_cache:
            self._cache[cache_key] = text
        return text


LlamaCppClient = OpenAICompatibleVLMClient


def _prompt_for_label(
    label: str, use_chart_recognition: bool, use_seal_recognition: bool
) -> str | None:
    if label in IMAGE_LABELS:
        return None
    if label == "chart" and not use_chart_recognition:
        return None
    if label == "seal" and not use_seal_recognition:
        return None
    if "formula" in label and label != "formula_number":
        return "Formula Recognition:"
    return PROMPTS.get(label, "OCR:")
