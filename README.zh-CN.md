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

## 评测（OmniDocBench v1.6）

基于 OmniDocBench v1.6（1,651 页）的基准打分。使用同一个 PaddleOCR-VL-1.6
模型，通过轻量 ONNXRuntime + llama.cpp（HIP/ROCm）推理路径，与官方
PaddleOCR-VL-1.6 发表数据（[arXiv 2606.03264](https://arxiv.org/abs/2606.03264)）对比：

| 指标 | 本仓库 | 官方 1.6 | 说明 |
|---|---:|---:|---|
| 文本 Edit-dist ↓ | **0.035**（96.5%） | 0.033（96.7%） | 差 0.24pt |
| 阅读顺序 Edit-dist ↓ | **0.129**（87.1%） | 0.127（87.3%） | 差 0.25pt |
| 表格 TEDS ↑ | **0.940** | 0.948 | 差 0.76pt |
| 公式 Edit-dist ↓ | **0.094**（90.6%） | — | 有效指标 |
| 公式 CDM ↑ | **0.944** | 0.975 | 差 3.1pt |

**Hard 子集（296 页）：** 文本 0.058 · 公式 0.143 · 表格 TEDS 0.912 · 阅读顺序 0.182。

文本与阅读顺序两项与官方模型差 0.25pt 以内，表格 TEDS 差 0.76pt——验证了轻量
ONNX+llama.cpp 路线使用同一个 PaddleOCR-VL-1.6 模型能达到接近官方 Paddle 原生
管线的识别质量。

### 运行评测

针对 OmniDocBench（v1.5 与 v1.6）的端到端基准打分位于
[`eval/`](eval/README.md) 目录。它分三个带前置检查的阶段运行 ——
`download` → `infer` → `eval`：

```powershell
python eval/run_eval.py --stage all --version v16
```

前置条件、三个阶段、CDM/Docker 说明、v1.5 与 v1.6 的差异，以及分数落地位置，
请参见 [`eval/README.md`](eval/README.md)。

## 开发

安装开发工具并运行完整的本地检查：

```powershell
pip install -e .[dev]
./scripts/check.ps1   # Linux/macOS: bash scripts/check.sh
```

该检查会运行 `compileall`、`ruff check`、`ruff format --check`、`mypy src` 和 `pytest`。

要建立 characterization 固定数据（需要一次 VLM 服务）：

```powershell
python scripts/record_trace.py --server-url http://127.0.0.1:8000/v1
```

这会记录 `tests/fixtures/compat_cache.json` 和 golden 输出，使 `tests/test_pipeline_characterization.py` 可以在没有服务的情况下逐字节重放推理链路。如果固定数据或 layout 模型缺失，该测试会自动跳过。

## 说明

ROCm 加速发生在 VLM 服务端。本仓库负责 ONNXRuntime layout、文档区域裁剪、
VLM 请求和结果序列化。
