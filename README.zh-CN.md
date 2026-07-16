# PaddleOCR-VL-ROCm

面向 Windows AMD GPU 的文档图片转 Markdown 推理工具。PP-DocLayoutV3 通过 ONNX Runtime DirectML 运行，PaddleOCR-VL 1.6 由固定版本的 llama.cpp HIP 服务提供推理；原有外部 OpenAI-compatible 服务工作流继续受支持。

[English documentation](README.md)

## 证据状态

OmniDocBench v1.6 配对评估，1,650 页评分（1 页对称排除）。
Windows 原生 TeX Live 2026 全量 CDM 评分，Lightweight CDM 报告，
0 TEDS 错误，0 超时。

| 指标 | PaddleOCR-VL (论文) | PaddleOCR-VL-ROCm (实测) |
|---:|---:|---:|
| Overall | 96.33 | **95.99** |
| Text Edit-dist | 0.033 | 0.03488 |
| Reading-order Edit-dist | 0.127 | 0.12882 |
| Table TEDS | 94.76 | **94.09** |
| Formula CDM | 97.49 | **97.36** |

Overall = (Text accuracy + CDM + TEDS) / 3, Text accuracy = (1 - Edit_dist) x 100。
Reading-order 不计入 Overall（布局指标，非内容准确性）。
完整证据见 [omnidocbench-amd-windows](https://github.com/AIwork4me/omnidocbench-amd-windows)。
推理运行（llama.cpp HIP, AMD ROCm）成功 1,650 页，
1 页确定性 peg-native HTTP 500：
newspaper_The Times UK_0801@magazinesclubnew_page_031.png，详见
[PaddleOCR issue #18248](https://github.com/PaddlePaddle/PaddleOCR/issues/18248)。
G3 准确性已通过；G4 性能待完成。
## 兼容性演示

仓库中的 [`examples/input/magazine.png`](examples/input/magazine.png) 以及对应的 [`Markdown`](tests/fixtures/golden/magazine.md) 和 [`结构化 JSON`](tests/fixtures/golden/magazine.json) golden 输出展示公共输出格式。这是兼容性演示，不是发布证据，不能证明当前硬件速度或 G3/G4 已验收。

## Windows AMD 托管安装

需要 Windows 10/11、可用 HIP 运行时的 AMD GPU、Python 3.10-3.13，以及足够的磁盘空间。托管安装固定 llama.cpp HIP `b9884`（`86961efd5`），并按文件大小和 SHA-256 校验全部资源。

```powershell
pip install -e .[download]
paddleocr-vl-rocm setup --auto
paddleocr-vl-rocm doctor
paddleocr-vl-rocm run examples/input/magazine.png
```

`setup --auto` 会下载、校验、安装并启动本地服务。只安装不启动可使用 `setup --no-start`，自定义目录可使用 `--root`。项目不包含遥测。

中文用户可从 [ModelScope](https://modelscope.cn/models/PaddlePaddle/PP-DocLayoutV3_onnx) 直接下载 PP-DocLayoutV3 ONNX；英文用户可使用 [Hugging Face](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx)。

## 使用现有服务

已有 llama.cpp、vLLM 或其他兼容端点时，可以不安装托管运行时：

```powershell
pip install -e .[download]
paddleocr-vl-rocm doctor --server-url http://127.0.0.1:8111/v1
paddleocr-vl-rocm run examples/input/magazine.png --server-url http://127.0.0.1:8111/v1
paddleocr-vl-rocm --input examples/input/magazine.png --server-url http://127.0.0.1:8111/v1
```

最后一条是保持兼容的旧命令。若服务要求明确模型名，请传入 `--api-model-name`。

## Python API

```python
from paddleocr_vl_rocm import PaddleOCRVLROCm

pipeline = PaddleOCRVLROCm(layout_model_dir="models/PP-DocLayoutV3-onnx", vlm_server_url="http://127.0.0.1:8111/v1")
result = pipeline.predict("examples/input/magazine.png")
print(result.markdown_text)
```

## 支持矩阵

| 路径 | 状态 | 说明 |
|---|---|---|
| Windows 10/11 + AMD + 托管 llama.cpp HIP | 支持 | [环境 doctor 证据](docs/windows-amd-doctor-evidence-2026-07-12.md)检测到 Windows 11、Radeon 8060S 和 HIP；完整发布门禁仍待通过 |
| Windows + 现有 OpenAI-compatible 服务 | 支持 | `doctor --server-url` 校验端点 |
| Linux + 现有 OpenAI-compatible 服务 | 支持 | 服务由用户自行维护 |
| macOS | 不支持 | 没有托管运行时或已测试 layout provider |

## 复现评测

固定 OmniDocBench v1.6 checkout、推理阶段、官方指标定义和产物门禁见 [`eval/README.md`](eval/README.md)。不得发布来自不完整运行、fallback 输出、错误评分器或未校验产物的分数。

## 故障排查

- 首先运行 `paddleocr-vl-rocm doctor`，每个失败项都带有修复建议。
- 使用 `paddleocr-vl-rocm doctor --json` 生成已脱敏的硬件报告。
- DirectML 必须是首个活动 layout provider；程序会失败关闭，不会静默 CPU fallback。
- 下载支持断点续传；大小或 SHA-256 不符时不会替换已验证安装。

## 贡献与安全

提交 PR 前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。安全问题请按照 [`SECURITY.md`](SECURITY.md) 私下报告。
