# PaddleOCR-VL-ROCm

面向 Windows AMD GPU 的文档图片转 Markdown 推理工具。PP-DocLayoutV3 通过 ONNX Runtime DirectML 运行，PaddleOCR-VL 1.6 由固定版本的 llama.cpp HIP 服务提供推理；原有外部 OpenAI-compatible 服务工作流继续受支持。

[English documentation](README.md)

## 证据状态

下表是 OmniDocBench v1.6 的历史证据和重算结果，不是新鲜的发布验收结果。评分器固定在提交 [`147cd5ac9472002f5751221d390bf00abdbc0d2f`](docs/accuracy-root-cause-v16.md)，Text、Formula、Table 分别四舍五入到三位后再计算 Overall。

| 历史路径 | Text Edit | Formula CDM | Table TEDS | Overall |
|---|---:|---:|---:|---:|
| 官方本地路径 | 0.034 | 96.502 | 94.239 | 95.7803 |
| 轻量 ROCm 路径 | 0.034 | 96.922 | 94.322 | 95.9480 |

证据来源和重算过程见 [`docs/accuracy-root-cause-v16.md`](docs/accuracy-root-cause-v16.md)。新鲜官方运行因上游 PEG parser 的确定性 HTTP 500 停在 1650/1651。G3 精度与 G4 性能尚未通过，因此项目不宣称已经获得发布验收分数或延迟；G3 前计时仅用于诊断。

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
