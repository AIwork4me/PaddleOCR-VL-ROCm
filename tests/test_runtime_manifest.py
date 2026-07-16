from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest

from scripts.build_runtime_manifest import describe_runtime, sha256_file

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "src" / "paddleocr_vl_rocm" / "assets" / "runtime-manifest.json"
EXPECTED_RESOURCES = {
    "llama-cpp-hip-runtime": {
        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b9884/llama-b9884-bin-win-hip-radeon-x64.zip",
        "destination": "runtime/llama-b9884-bin-win-hip-radeon-x64.zip",
        "size": 323202730,
        "sha256": "3f48fd5fa9cfa26c0f537a2c1e9ebe5931bc46540b224e6fcede2fc3bb8cb07f",
    },
    "paddleocr-vl-main-gguf": {
        "url": "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/resolve/511b09642bb324401f15f97cc23bc67e8f0a291d/PaddleOCR-VL-1.6-GGUF.gguf",
        "destination": "models/PaddleOCR-VL-1.6-GGUF.gguf",
        "size": 935769056,
        "sha256": "f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8",
    },
    "paddleocr-vl-mmproj": {
        "url": "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/resolve/511b09642bb324401f15f97cc23bc67e8f0a291d/PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
        "destination": "models/PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
        "size": 881770560,
        "sha256": "204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a",
    },
    "pp-doclayout-v3-onnx": {
        "url": "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx/resolve/46bbdf188bb0a772c08aed74882ce7e51a8f1ea6/inference.onnx",
        "destination": "models/PP-DocLayoutV3-onnx/inference.onnx",
        "size": 130502049,
        "sha256": "45bf71750b00739a41fc209f132eb104a4d6b5bb29483c9078164d8b87cf28ba",
    },
    "pp-doclayout-v3-config": {
        "url": "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx/resolve/46bbdf188bb0a772c08aed74882ce7e51a8f1ea6/inference.yml",
        "destination": "models/PP-DocLayoutV3-onnx/inference.yml",
        "size": 1482,
        "sha256": "506fcfac13b3b546ae40d7886b44126420f392adb694e3f8bb6a6286a1f90fdc",
    },
}


def test_sha256_file_hashes_without_read_bytes(tmp_path, monkeypatch):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"runtime asset")
    monkeypatch.setattr(Path, "read_bytes", lambda _self: (_ for _ in ()).throw(AssertionError()))

    assert sha256_file(path) == hashlib.sha256(b"runtime asset").hexdigest()


def test_describe_runtime_rejects_unverified_archive(tmp_path):
    archive = tmp_path / "runtime.zip"
    archive.write_bytes(b"not the official runtime")

    with pytest.raises(ValueError, match="Unexpected size for llama-cpp-hip-runtime"):
        describe_runtime(archive)


def test_runtime_manifest_is_fully_pinned():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["schema"] == 1
    assert manifest["runtime"] == {"version": "b9884", "commit": "86961efd5"}
    assert manifest["manual_download_pages"] == {
        "en": "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx",
        "zh-CN": "https://modelscope.cn/models/PaddlePaddle/PP-DocLayoutV3_onnx",
    }
    resources = {item["name"]: item for item in manifest["resources"]}
    assert len(resources) == len(manifest["resources"])
    assert resources == {
        name: {"name": name, **metadata} for name, metadata in EXPECTED_RESOURCES.items()
    }
    for item in resources.values():
        assert item["url"].startswith("https://")
        assert len(item["sha256"]) == 64
        assert item["sha256"] == item["sha256"].lower()
        assert item["size"] > 0
        destination = PurePosixPath(item["destination"])
        assert not destination.is_absolute()
        assert ".." not in destination.parts
    layout_resources = [item for item in resources.values() if item["name"].startswith("pp-")]
    assert all(
        "/resolve/46bbdf188bb0a772c08aed74882ce7e51a8f1ea6/" in item["url"]
        for item in layout_resources
    )
