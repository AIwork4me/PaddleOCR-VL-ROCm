from __future__ import annotations

import argparse
import json
import traceback
from typing import Any


def check_official_dependency(
    *,
    construct: bool,
    server_url: str,
    api_model_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "paddleocr_found": False,
        "PaddleOCRVL_found": False,
        "constructed": False,
        "server_url": server_url,
        "api_model_name": api_model_name,
        "error": "",
    }
    try:
        from paddleocr import PaddleOCRVL

        result["paddleocr_found"] = True
        result["PaddleOCRVL_found"] = True
        if construct:
            PaddleOCRVL(
                pipeline_version="v1.6",
                vl_rec_backend="llama-cpp-server",
                vl_rec_server_url=server_url,
                vl_rec_api_model_name=api_model_name,
            )
            result["constructed"] = True
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic gate
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Check official PaddleOCRVL availability.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8111/v1")
    parser.add_argument("--api-model-name", default="PaddleOCR-VL-1.6-GGUF.gguf")
    parser.add_argument("--construct", action="store_true")
    args = parser.parse_args()

    result = check_official_dependency(
        construct=args.construct,
        server_url=args.server_url,
        api_model_name=args.api_model_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
