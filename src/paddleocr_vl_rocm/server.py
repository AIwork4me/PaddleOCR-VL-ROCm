from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from .utils import get_console


def normalize_server_url(server_url: str) -> str:
    return server_url.rstrip("/")


def _models_url(server_url: str) -> str:
    parsed = urlsplit(normalize_server_url(server_url))
    path = parsed.path.rstrip("/")
    if not path.endswith("/models"):
        path = f"{path}/models"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


def _redacted_url(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or "<host>"
    if ":" in host:
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme or "http", host, parsed.path, "", ""))


def check_openai_compatible_server(server_url: str, timeout: float = 10.0) -> dict[str, Any]:
    models_url = _models_url(server_url)
    display_url = _redacted_url(models_url)
    console = get_console()

    try:
        response = requests.get(models_url, timeout=timeout)
    except requests.RequestException:
        raise RuntimeError(
            f"Failed to connect to OpenAI-compatible server at {display_url}"
        ) from None

    try:
        if response.status_code >= 400:
            raise RuntimeError(
                f"Server health check failed: GET {display_url} -> HTTP {response.status_code}"
            )

        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError(f"Server returned non-JSON response from {display_url}") from None

        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise RuntimeError(f"Server returned an invalid /v1/models payload from {display_url}")
        console.print(f"[green]/v1/models: PASS[/green] ({len(models)} model(s))")
    finally:
        response.close()

    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check an OpenAI-compatible llama.cpp server.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8111/v1")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    check_openai_compatible_server(args.server_url, timeout=args.timeout)


if __name__ == "__main__":
    main()
