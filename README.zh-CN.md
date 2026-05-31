# PaddleOCR-VL-ROCm

这是一个轻量版 PaddleOCR-VL 推理仓库，使用 ONNXRuntime 执行
PP-DocLayoutV3 layout 检测，并通过 ROCm 加速的 OpenAI-compatible VLM 服务
完成视觉语言识别。

这个仓库的目标是把推理链路做干净：

- 推理时不需要 PaddlePaddle runtime。
- PP-DocLayoutV3 通过 ONNXRuntime 执行。
- VLM 识别由你的 ROCm vLLM 或 llama.cpp 服务承载。
- 输出为 PaddleOCR-VL 风格的 JSON 和 Markdown。

## 验证结果

本仓库的 ONNXRuntime 轻量链路已在 1355 张图片上与 Paddle 原生链路完成验证。

| 项目 | 结果 |
|---|---:|
| 全量成功 | 1355 / 1355 |
| Payload 对齐 | 1355 / 1355 |
| Layout、crop、请求顺序、请求 payload | 严格对齐 |

## 安装

```powershell
git clone <your-repo-url> PaddleOCR-VL-ROCm
cd PaddleOCR-VL-ROCm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## 准备模型

把 PP-DocLayoutV3 ONNX 文件放到：

```text
models/PP-DocLayoutV3-onnx/
  inference.onnx
  inference.yml
```

直接从 Hugging Face 下载已验证的 ONNX 模型：

```powershell
pip install -e .[download]
python scripts/download_ppdoclayoutv3_onnx.py
```

模型链接：

```text
https://huggingface.co/AlexTransformer/PP-DocLayoutV3-onnx
```

如果你本地已经有已验证的 ONNX 目录，也可以复制：

```powershell
python scripts/download_ppdoclayoutv3_onnx.py `
  --source-dir C:\path\to\PP-DocLayoutV3-onnx `
  --target-dir models/PP-DocLayoutV3-onnx
```

然后准备一个 OpenAI-compatible VLM 服务，例如 vLLM：

```text
http://127.0.0.1:8000/v1/models
http://127.0.0.1:8000/v1/chat/completions
```

检查服务：

```powershell
paddleocr-vl-rocm-check-server --server-url http://127.0.0.1:8000/v1
```

## 命令行使用

```powershell
paddleocr-vl-rocm `
  --input examples/input/handwrite_ch_demo.png `
  --output outputs/smoke `
  --layout-model models/PP-DocLayoutV3-onnx `
  --server-url http://127.0.0.1:8000/v1 `
  --api-model-name PaddleOCR-VL-1.5-0.9B `
  --vlm-backend vllm-server
```

输出文件：

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

## 示例图片

示例图片来自 `ppocrv6_onnx/test_images`：

- `handwrite_ch_demo.png`
- `handwrite_en_demo.png`
- `ancient_demo.png`
- `japan_demo.png`
- `magazine.png`
- `magazine_vetical.png`
- `pinyin_demo.png`

## 输出格式

JSON 包含：

- `input_path`
- `width`、`height`
- `layout_det_res`
- `parsing_res_list`
- `model_settings`

Markdown 保存按阅读顺序组织后的识别结果。

## 测试

```powershell
python -m compileall -q src/paddleocr_vl_rocm
python -m pytest -q
paddleocr-vl-rocm --help
```

## 说明

ROCm 加速发生在 VLM 服务端。本仓库负责 ONNXRuntime layout、文档区域裁剪、
VLM 请求和结果序列化。
