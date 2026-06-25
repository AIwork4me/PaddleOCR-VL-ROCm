"""Record a VLM compat cache + golden outputs for characterization tests.

Run once against a live OpenAI-compatible VLM server. Produces:
  tests/fixtures/compat_cache.json
  tests/fixtures/golden/<stem>.json
  tests/fixtures/golden/<stem>.md

After recording, tests/test_pipeline_characterization.py replays without a server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paddleocr_vl_rocm.encoding import _jpeg_bytes, _png_bytes, _sha256_hex
from paddleocr_vl_rocm.pipeline_core import run_light_parser
from paddleocr_vl_rocm.vlm import client
from paddleocr_vl_rocm.vlm.client import _vlm_cache_key

REPO = Path(__file__).resolve().parent.parent
IMAGES = sorted((REPO / "examples" / "input").glob("*.png"))
FIXTURES = REPO / "tests" / "fixtures"
GOLDEN = FIXTURES / "golden"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-model-name", default="PaddleOCR-VL-1.5-0.9B")
    parser.add_argument("--layout-model", default="models/PP-DocLayoutV3-onnx")
    args = parser.parse_args()

    FIXTURES.mkdir(parents=True, exist_ok=True)
    GOLDEN.mkdir(parents=True, exist_ok=True)

    recorded: dict[str, str] = {}
    original = client.OpenAICompatibleVLMClient.complete_image

    def recording(
        self,
        prompt,
        image=None,
        image_path=None,
        max_new_tokens=None,
        min_pixels=None,
        max_pixels=None,
        use_client_cache=True,
    ):
        text = original(
            self,
            prompt,
            image=image,
            image_path=image_path,
            max_new_tokens=max_new_tokens,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_client_cache=use_client_cache,
        )
        if image is not None:
            raw = _jpeg_bytes(image) if self.backend == "vllm-server" else _png_bytes(image)
        else:
            raw = image_path.read_bytes()
        key = _vlm_cache_key(
            self.model,
            prompt=prompt,
            image_sha256=_sha256_hex(raw),
            max_new_tokens=max_new_tokens,
            seed=self.seed,
        )
        recorded[key] = text
        return text

    client.OpenAICompatibleVLMClient.complete_image = recording  # type: ignore[assignment]
    try:
        for img in IMAGES:
            out_dir = FIXTURES / "_tmp_record"
            out_dir.mkdir(parents=True, exist_ok=True)
            json_path = run_light_parser(
                input_path=img,
                output_dir=out_dir,
                model_dir=Path(args.layout_model),
                server_url=args.server_url,
                vlm_backend="vllm-server",
                api_model_name=args.api_model_name,
                max_new_tokens=4096,
                timeout=300.0,
                prompt_label=None,
                use_layout_detection=True,
                use_chart_recognition=False,
                use_seal_recognition=False,
                seed=1,
                threshold=0.3,
                display_input_path=str(img),
                skip_server_check=False,
            )
            stem = img.stem
            (GOLDEN / f"{stem}.json").write_text(
                json_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            md = out_dir / "result.md"
            if md.exists():
                (GOLDEN / f"{stem}.md").write_text(md.read_text(encoding="utf-8"), encoding="utf-8")
    finally:
        client.OpenAICompatibleVLMClient.complete_image = original  # type: ignore[assignment]
        (FIXTURES / "_tmp_record").mkdir(parents=True, exist_ok=True)

    (FIXTURES / "compat_cache.json").write_text(
        json.dumps({"entries": recorded}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Recorded {len(recorded)} VLM responses and {len(IMAGES)} golden outputs into {FIXTURES}"
    )


if __name__ == "__main__":
    main()
