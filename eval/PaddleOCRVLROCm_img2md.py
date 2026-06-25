"""PaddleOCR-VL-ROCm adapter for OmniDocBench.

Mirrors OmniDocBench's ``tools/model_infer/PaddleOCR_img2md.py``: a standalone
offline script that, for each dataset image, runs our pipeline and writes one
``<image_basename_no_ext>.md`` file into a flat output directory. OmniDocBench's
matcher consumes those pre-generated Markdown files directly (it never imports
this adapter), so no JSON is emitted for the harness.

Per-page failures are caught and recorded so a single bad page does not abort
the run (a missing page scores zero in the harness).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from paddleocr_vl_rocm import PaddleOCRVLROCm

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")


def expected_md_name(image_name: str) -> str:
    """Return the Markdown filename OmniDocBench's matcher looks up.

    The matcher's first lookup is ``<img_name[:-4]>.md`` (basename minus
    extension). ``Path.stem`` strips a single extension regardless of length.
    """
    return Path(image_name).stem + ".md"


def process_folder(
    img_dir: Path,
    out_dir: Path,
    *,
    layout_model: str = "models/PP-DocLayoutV3-onnx",
    server_url: str = "http://127.0.0.1:8000/v1",
    api_model_name: str = "PaddleOCR-VL-1.5-0.9B",
    vlm_backend: str = "vllm-server",
) -> dict:
    """Run the pipeline over every image in ``img_dir`` and write per-page ``.md``.

    Returns a summary dict with ``count``, ``ok``, and per-image ``stats``.
    """
    pipeline = PaddleOCRVLROCm(
        layout_model_dir=layout_model,
        vlm_server_url=server_url,
        api_model_name=api_model_name,
        vlm_backend=vlm_backend,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: list[dict] = []
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    for img in images:
        start = time.time()
        try:
            result = pipeline.predict(img)
            md_path = out_dir / expected_md_name(img.name)
            md_path.write_text(result.markdown_text, encoding="utf-8")
            stats.append(
                {"image": img.name, "status": "ok", "seconds": round(time.time() - start, 2)}
            )
        except Exception as exc:  # noqa: BLE001 - record failure, continue (page scored as empty otherwise)
            stats.append(
                {
                    "image": img.name,
                    "status": f"failed: {exc}",
                    "seconds": round(time.time() - start, 2),
                }
            )
    return {
        "count": len(images),
        "ok": sum(1 for s in stats if s["status"] == "ok"),
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL-ROCm adapter for OmniDocBench: write per-page .md"
    )
    parser.add_argument("--img-dir", required=True, help="Dataset images directory.")
    parser.add_argument(
        "--out-dir", required=True, help="Output flat dir of <basename>.md predictions."
    )
    parser.add_argument("--layout-model", default="models/PP-DocLayoutV3-onnx")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-model-name", default="PaddleOCR-VL-1.5-0.9B")
    parser.add_argument("--vlm-backend", default="vllm-server")
    args = parser.parse_args()
    summary = process_folder(
        Path(args.img_dir),
        Path(args.out_dir),
        layout_model=args.layout_model,
        server_url=args.server_url,
        api_model_name=args.api_model_name,
        vlm_backend=args.vlm_backend,
    )
    print(summary)


if __name__ == "__main__":
    main()
