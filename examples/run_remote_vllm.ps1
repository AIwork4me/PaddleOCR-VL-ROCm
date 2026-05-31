$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

paddleocr-vl-rocm `
  --input examples/input/handwrite_ch_demo.png `
  --output outputs/smoke `
  --layout-model models/PP-DocLayoutV3-onnx `
  --server-url http://127.0.0.1:8000/v1 `
  --api-model-name PaddleOCR-VL-1.5-0.9B `
  --vlm-backend vllm-server

