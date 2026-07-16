# PaddleOCR-VL-ROCm

Document image to Markdown inference for Windows AMD GPUs. PP-DocLayoutV3 runs through ONNX Runtime DirectML, while PaddleOCR-VL 1.6 is served by the pinned llama.cpp HIP runtime. The legacy external OpenAI-compatible endpoint workflow remains supported on Windows and Linux.

[Chinese documentation](README.zh-CN.md)

## Evidence status

This historical evidence contains reconstructed OmniDocBench v1.6 results, not a fresh release acceptance run. It uses scorer commit [`147cd5ac9472002f5751221d390bf00abdbc0d2f`](docs/accuracy-root-cause-v16.md), rounds Text, Formula, and Table to three decimals, and then computes Overall.

| Historical path | Text Edit | Formula CDM | Table TEDS | Overall |
|---|---:|---:|---:|---:|
| Official local | 0.034 | 96.502 | 94.239 | 95.7803 |
| Lightweight ROCm | 0.034 | 96.922 | 94.322 | 95.9480 |

The provenance and reconstruction are recorded in [`docs/accuracy-root-cause-v16.md`](docs/accuracy-root-cause-v16.md). Official inference currently has 1,650 successful pages and one deterministic `peg-native` HTTP 500 for `newspaper_The Times UK_0801@magazinesclubnew_page_031.png`, tracked in [PaddleOCR issue #18248](https://github.com/PaddlePaddle/PaddleOCR/issues/18248). The project owner approved this single, immutable known failure. Scoring still includes all 1,651 ground-truth pages and treats that page as an empty prediction, so the exception cannot inflate the score. The issue remains open and is not described as a PaddlePaddle maintainer resolution. G3 accuracy and G4 performance have not passed; pre-G3 timing remains diagnostic.

## Compatibility demo

The tracked sample [`examples/input/magazine.png`](examples/input/magazine.png) and its [`Markdown`](tests/fixtures/golden/magazine.md) and [`structured JSON`](tests/fixtures/golden/magazine.json) golden outputs show the public output shape. This is a compatibility demo, not release evidence; the goldens do not establish current hardware speed or G3/G4 acceptance.

## Windows AMD managed setup

Requirements: Windows 10/11, an AMD GPU with a working HIP runtime, Python 3.10-3.13, and enough disk space. Managed setup pins llama.cpp HIP `b9884` (`86961efd5`) and verifies every downloaded file by size and SHA-256.

```powershell
pip install -e .[download]
paddleocr-vl-rocm setup --auto
paddleocr-vl-rocm doctor
paddleocr-vl-rocm run examples/input/magazine.png
```

`setup --auto` downloads, verifies, installs, and starts the local server. Use `setup --no-start` to install without starting it, or `--root` to select a managed data directory. No telemetry is sent.

English users can download PP-DocLayoutV3 ONNX directly from [Hugging Face](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx). Chinese users can use the [ModelScope mirror](https://modelscope.cn/models/PaddlePaddle/PP-DocLayoutV3_onnx).

## Existing server

Keep an existing llama.cpp, vLLM, or compatible endpoint and use the same pipeline without managed runtime assets:

```powershell
pip install -e .[download]
paddleocr-vl-rocm doctor --server-url http://127.0.0.1:8111/v1
paddleocr-vl-rocm run examples/input/magazine.png --server-url http://127.0.0.1:8111/v1
paddleocr-vl-rocm --input examples/input/magazine.png --server-url http://127.0.0.1:8111/v1
```

The last command is the backward-compatible legacy form. Use `--api-model-name` when the endpoint requires a specific model identifier.

## Python API

```python
from paddleocr_vl_rocm import PaddleOCRVLROCm

pipeline = PaddleOCRVLROCm(layout_model_dir="models/PP-DocLayoutV3-onnx", vlm_server_url="http://127.0.0.1:8111/v1")
result = pipeline.predict("examples/input/magazine.png")
print(result.markdown_text)
```

## Support matrix

| Path | Status | Notes |
|---|---|---|
| Windows 10/11 + AMD + managed llama.cpp HIP | Supported | [Environment doctor evidence](docs/windows-amd-doctor-evidence-2026-07-12.md) detects Windows 11, Radeon 8060S, and HIP; full release gates remain pending |
| Windows + existing OpenAI-compatible server | Supported | `doctor --server-url` validates the endpoint |
| Linux + existing OpenAI-compatible server | Supported | Server ownership remains external |
| macOS | Not supported | No managed runtime or tested layout provider |

## Benchmark reproduction

See [`eval/README.md`](eval/README.md) for the pinned OmniDocBench v1.6 checkout, inference stages, official metric definitions, and artifact gates. Do not publish a score from an incomplete run, fallback output, mismatched scorer, or unverified artifact.

## Troubleshooting

- Run `paddleocr-vl-rocm doctor` first. Every failed check includes remediation.
- Use `paddleocr-vl-rocm doctor --json` for a redacted hardware report.
- DirectML must be the first active layout provider; the pipeline fails closed rather than silently using CPU fallback.
- Downloads are resumable. A size or SHA-256 mismatch never replaces a verified installation.

## Contributing and security

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request. Report security issues privately as described in [`SECURITY.md`](SECURITY.md).
