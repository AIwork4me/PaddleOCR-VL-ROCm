# PaddleOCR-VL-ROCm

Lightweight PaddleOCR-VL inference for Windows and Linux systems that use an
ONNXRuntime layout model plus a ROCm-backed OpenAI-compatible VLM server.

This repository keeps the inference path small:

- No PaddlePaddle runtime is required for inference.
- PP-DocLayoutV3 runs through ONNXRuntime.
- Visual language recognition is served by your ROCm vLLM or llama.cpp endpoint.
- Outputs are saved as PaddleOCR-VL-style JSON and Markdown files.

## Validation Result

The lightweight ONNXRuntime path has been validated against the Paddle native
pipeline on 1355 images.

| Item | Result |
|---|---:|
| Full-run success | 1355 / 1355 |
| Payload alignment | 1355 / 1355 |
| Layout, crop, request order, request payload | Strictly aligned |

## Install

```powershell
git clone <your-repo-url> PaddleOCR-VL-ROCm
cd PaddleOCR-VL-ROCm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Linux/macOS:

```bash
git clone <your-repo-url> PaddleOCR-VL-ROCm
cd PaddleOCR-VL-ROCm
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Prepare Models

Place the PP-DocLayoutV3 ONNX files here:

```text
models/PP-DocLayoutV3-onnx/
  inference.onnx
  inference.yml
```

Download the verified ONNX model directly from Hugging Face:

```powershell
pip install -e .[download]
python scripts/download_ppdoclayoutv3_onnx.py
```

Model link:

```text
https://huggingface.co/AlexTransformer/PP-DocLayoutV3-onnx
```

If you already have the verified ONNX directory locally, copy it with:

```powershell
python scripts/download_ppdoclayoutv3_onnx.py `
  --source-dir C:\path\to\PP-DocLayoutV3-onnx `
  --target-dir models/PP-DocLayoutV3-onnx
```

Start or provide an OpenAI-compatible VLM server. For vLLM, the server should
expose:

```text
http://127.0.0.1:8000/v1/models
http://127.0.0.1:8000/v1/chat/completions
```

Check the endpoint:

```powershell
paddleocr-vl-rocm-check-server --server-url http://127.0.0.1:8000/v1
```

## CLI Usage

```powershell
paddleocr-vl-rocm `
  --input examples/input/handwrite_ch_demo.png `
  --output outputs/smoke `
  --layout-model models/PP-DocLayoutV3-onnx `
  --server-url http://127.0.0.1:8000/v1 `
  --api-model-name PaddleOCR-VL-1.5-0.9B `
  --vlm-backend vllm-server
```

Expected outputs:

```text
outputs/smoke/handwrite_ch_demo_res.json
outputs/smoke/handwrite_ch_demo.md
```

## Python API

```python
from paddleocr_vl_rocm import PaddleOCRVLROCm

pipeline = PaddleOCRVLROCm(
    layout_model_dir="models/PP-DocLayoutV3-onnx",
    vlm_server_url="http://127.0.0.1:8000/v1",
    api_model_name="PaddleOCR-VL-1.5-0.9B",
)

result = pipeline.predict("examples/input/handwrite_ch_demo.png")
result.print()
result.save_to_json("outputs")
result.save_to_markdown("outputs", pretty=False)
```

## Example Images

The smoke images are copied from `ppocrv6_onnx/test_images`:

- `handwrite_ch_demo.png`
- `handwrite_en_demo.png`
- `ancient_demo.png`
- `japan_demo.png`
- `magazine.png`
- `magazine_vetical.png`
- `pinyin_demo.png`

## Output Format

JSON contains:

- `input_path`
- `width`, `height`
- `layout_det_res`
- `parsing_res_list`
- `model_settings`

Markdown contains the recognized document content in reading order.

## Tests

```powershell
python -m compileall -q src/paddleocr_vl_rocm
python -m pytest -q
paddleocr-vl-rocm --help
```

## Evaluation (OmniDocBench v1.6, local AMD Windows)

Scores in this repository are local measurements from the Windows + AMD Radeon
+ llama.cpp/GGUF + OmniDocBench/CDM environment. They are not claimed from a
Linux vLLM/BF16 reference path.

| Engine | Text Edit-dist ↓ | Reading-order Edit-dist ↓ | Table TEDS ↑ | Formula CDM ↑ | Notes |
|---|---:|---:|---:|---:|---|
| Lightweight local engine | 0.035 | 0.129 | 94.00 | 94.40 | Existing recorded local CDM artifact |
| Official local engine | pending | pending | pending | pending | Pending reproduction from tracked local artifacts; companion setup evidence is non-tracked context |
| [Public PaddleOCR-VL-1.6 target](https://arxiv.org/abs/2606.03264) | 0.035 | 0.129 | 94.64 | 97.49 | External reference, shown for context only |

The project goal is to align inputs, outputs, parameters, and local evaluation
evidence. Remaining gaps are reported by engine instead of hidden.

### Running the eval

End-to-end benchmark scoring lives under [`eval/`](eval/README.md). It runs in
three gated stages — `download` → `infer` → `eval`:

```powershell
python eval/run_eval.py --stage all --version v16
```

See [`eval/README.md`](eval/README.md) for prerequisites, the three stages, the
CDM/Docker note, v1.5 vs v1.6 differences, and where scores land.

> **Setting up OmniDocBench on a fresh machine?** See our companion repo
> [`omnidocbench-amd-windows`](https://github.com/AIwork4me/omnidocbench-amd-windows) —
> a one-command automated setup guide with CLAUDE.md for AI-agent orchestration,
> covering VLM server, CDM environment, and the full pitfalls knowledge base.

## Development

Install with dev tooling and run the full local check:

```powershell
pip install -e .[dev]
./scripts/check.ps1   # Linux/macOS: bash scripts/check.sh
```

The check runs `compileall`, `ruff check`, `ruff format --check`, `mypy src`, and `pytest`.

To establish the characterization fixtures (requires the VLM server once):

```powershell
python scripts/record_trace.py --server-url http://127.0.0.1:8000/v1
```

This records `tests/fixtures/compat_cache.json` and golden outputs so `tests/test_pipeline_characterization.py` can replay the pipeline byte-for-byte without a server. The test skips automatically if fixtures or the layout model are absent.

## Notes

ROCm acceleration is provided by the VLM server. This Python package handles
the ONNXRuntime layout stage, document crop routing, VLM requests, and result
serialization.
