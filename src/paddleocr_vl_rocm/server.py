from __future__ import annotations

from typing import Any

import requests

from .utils import get_console


def normalize_server_url(server_url: str) -> str:
    return server_url.rstrip("/")


def check_openai_compatible_server(server_url: str, timeout: float = 10.0) -> dict[str, Any]:
    base = normalize_server_url(server_url)
    models_url = f"{base}/models"
    console = get_console()

    try:
        response = requests.get(models_url, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to connect to OpenAI-compatible server at {models_url}: {exc}"
        ) from exc

    if response.status_code >= 400:
        body = response.text[:2000]
        raise RuntimeError(
            f"Server health check failed: GET {models_url} -> HTTP {response.status_code}\n{body}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Server returned non-JSON response from {models_url}: {response.text[:1000]}"
        ) from exc

    models = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(models, list):
        console.print("[green]/v1/models: PASS[/green]")
        for item in models:
            model_id = item.get("id") if isinstance(item, dict) else str(item)
            console.print(f"  - {model_id}")
    else:
        console.print("[green]/v1/models: PASS[/green]")
        console.print(payload)

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
