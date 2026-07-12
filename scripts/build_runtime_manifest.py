from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RUNTIME_VERSION = "b9884"
RUNTIME_COMMIT = "86961efd5"
RUNTIME_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/b9884/"
    "llama-b9884-bin-win-hip-radeon-x64.zip"
)
RUNTIME_SHA256 = "3f48fd5fa9cfa26c0f537a2c1e9ebe5931bc46540b224e6fcede2fc3bb8cb07f"
RUNTIME_SIZE = 323202730
MODEL_REPO = "PaddlePaddle/PaddleOCR-VL-1.6-GGUF"
MODEL_REVISION = "511b09642bb324401f15f97cc23bc67e8f0a291d"
LAYOUT_REPO = "PaddlePaddle/PP-DocLayoutV3_onnx"
LAYOUT_REVISION = "46bbdf188bb0a772c08aed74882ce7e51a8f1ea6"
MAIN_GGUF_SHA256 = "f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8"
MAIN_GGUF_SIZE = 935769056
MMPROJ_SHA256 = "204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a"
MMPROJ_SIZE = 881770560
LAYOUT_ONNX_SHA256 = "45bf71750b00739a41fc209f132eb104a4d6b5bb29483c9078164d8b87cf28ba"
LAYOUT_ONNX_SIZE = 130502049
LAYOUT_CONFIG_SHA256 = "506fcfac13b3b546ae40d7886b44126420f392adb694e3f8bb6a6286a1f90fdc"
LAYOUT_CONFIG_SIZE = 1482


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(
    path: Path,
    *,
    name: str,
    url: str,
    destination: str,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    size = path.stat().st_size
    digest = sha256_file(path)
    if expected_size is not None and size != expected_size:
        raise ValueError(f"Unexpected size for {name}: {size} != {expected_size}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Unexpected SHA-256 for {name}: {digest} != {expected_sha256}")
    return {
        "name": name,
        "url": url,
        "destination": destination,
        "size": size,
        "sha256": digest,
    }


def describe_runtime(path: Path) -> dict[str, object]:
    return describe(
        path,
        name="llama-cpp-hip-runtime",
        url=RUNTIME_URL,
        destination="runtime/llama-b9884-bin-win-hip-radeon-x64.zip",
        expected_size=RUNTIME_SIZE,
        expected_sha256=RUNTIME_SHA256,
    )


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    model_base = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}"
    layout_base = f"https://huggingface.co/{LAYOUT_REPO}/resolve/{LAYOUT_REVISION}"
    resources = [
        describe_runtime(args.runtime_archive),
        describe(
            args.main_gguf,
            name="paddleocr-vl-main-gguf",
            url=f"{model_base}/PaddleOCR-VL-1.6-GGUF.gguf",
            destination="models/PaddleOCR-VL-1.6-GGUF.gguf",
            expected_size=MAIN_GGUF_SIZE,
            expected_sha256=MAIN_GGUF_SHA256,
        ),
        describe(
            args.mmproj,
            name="paddleocr-vl-mmproj",
            url=f"{model_base}/PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
            destination="models/PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
            expected_size=MMPROJ_SIZE,
            expected_sha256=MMPROJ_SHA256,
        ),
        describe(
            args.layout_onnx,
            name="pp-doclayout-v3-onnx",
            url=f"{layout_base}/inference.onnx",
            destination="models/PP-DocLayoutV3-onnx/inference.onnx",
            expected_size=LAYOUT_ONNX_SIZE,
            expected_sha256=LAYOUT_ONNX_SHA256,
        ),
        describe(
            args.layout_config,
            name="pp-doclayout-v3-config",
            url=f"{layout_base}/inference.yml",
            destination="models/PP-DocLayoutV3-onnx/inference.yml",
            expected_size=LAYOUT_CONFIG_SIZE,
            expected_sha256=LAYOUT_CONFIG_SHA256,
        ),
    ]
    return {
        "schema": 1,
        "runtime": {"version": RUNTIME_VERSION, "commit": RUNTIME_COMMIT},
        "manual_download_pages": {
            "en": "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx",
            "zh-CN": "https://modelscope.cn/models/PaddlePaddle/PP-DocLayoutV3_onnx",
        },
        "resources": resources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the pinned Windows AMD resource manifest.")
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--main-gguf", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--layout-onnx", type=Path, required=True)
    parser.add_argument("--layout-config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/paddleocr_vl_rocm/assets/runtime-manifest.json"),
    )
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
