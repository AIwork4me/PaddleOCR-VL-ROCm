---
model_id: paddleocr-vl-1.6
backend: llama-cpp-server
hardware:
  gpu: "AMD gfx1100"
  vram_min_gb: 48
environment:
  type: docker
  rocm: "7.2"
command: |
  # 1. Start llama.cpp server with GGUF:
  llama-server -m models/paddleocr-vl-bf16.gguf \
    --mmproj models/mmproj-bf16.gguf --port 8080 --n-gpu-layers 99

  # 2. Run adapter:
  python adapter/run_adapter.py \
    --platform linux-rocm --backend llama-cpp-server \
    --server-url http://127.0.0.1:8080/v1 \
    --img-dir /root/datasets/OmniDocBench_data/images \
    --out-dir /tmp/paddleocr-predictions

  # 3. Score:
  omnidocbench-rocm run --stage score --platform linux-rocm --cdm \
    --predictions-dir /tmp/paddleocr-predictions \
    --run-stats /tmp/paddleocr-predictions/_run_stats.json --version v16
expected_overall:
  value: 95.77
  tolerance: 0.5
---

# Reproduce PaddleOCR-VL 1.6 (95.77) on AMD ROCm

## Prerequisites

```bash
rocminfo | grep -E "Name:|VRAM"    # must show gfx1100 + ≥48 GB
ls -la /dev/kfd                     # must exist
```

## Quickstart

PaddleOCR-VL uses llama.cpp GGUF with HIP backend + ONNX Runtime for layout detection.

```bash
# 1. Start llama.cpp server
llama-server -m models/paddleocr-vl-bf16.gguf \
  --mmproj models/mmproj-bf16.gguf --port 8080 --n-gpu-layers 99

# 2. Run adapter
python adapter/run_adapter.py \
  --platform linux-rocm --backend llama-cpp-server \
  --server-url http://127.0.0.1:8080/v1 \
  --img-dir /root/datasets/OmniDocBench_data/images \
  --out-dir /tmp/paddleocr-predictions

# 3. Score
omnidocbench-rocm run --stage score --platform linux-rocm --cdm \
  --predictions-dir /tmp/paddleocr-predictions \
  --run-stats /tmp/paddleocr-predictions/_run_stats.json --version v16
```

## Expected output

Overall **95.77** (±0.5). Text 96.88, Table TEDS 93.44%, Formula CDM 93.94%.

## If it fails

See [OmniDocBench-ROCm pitfalls](https://github.com/AIwork4me/OmniDocBench-ROCm/docs/pitfalls.md).
