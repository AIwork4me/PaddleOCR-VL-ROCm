from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SECRET_KEYS = {"authorization", "api_key", "apikey", "token", "access_token"}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if key.lower() in SECRET_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str) and "://" in value:
        parts = urlsplit(value)
        query = [
            (key, "<redacted>" if key.lower() in SECRET_KEYS else item)
            for key, item in parse_qsl(parts.query)
        ]
        return urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
        )
    return value


@dataclass(frozen=True)
class VLMRequestContract:
    backend: str
    model: str
    prompt: str
    image_format: str
    image_sha256: str
    image_size: tuple[int, int]
    payload: dict[str, Any]

    def fingerprint(self) -> str:
        return fingerprint(redact(asdict(self)))


@dataclass(frozen=True)
class BlockTrace:
    request_order: int
    label: str
    bbox: tuple[float, float, float, float]
    request: VLMRequestContract
    raw_result_sha256: str
    final_result_sha256: str
