# PaddleOCR-VL-ROCm

![PaddleOCR-VL-ROCm 在 Windows AMD GPU 上将文档转换为 Markdown 和 JSON](docs/assets/paddleocr-vl-rocm-readme-hero.png.jpg)

在 Windows AMD GPU 上本地运行 PaddleOCR-VL 1.6，将文档图片转换为 Markdown
和结构化 JSON。

本项目使用混合后端；整个流水线并非全部通过 ROCm 执行：

```text
图片
→ PP-DocLayoutV3 / ONNX Runtime DirectML
→ 区域裁剪
→ PaddleOCR-VL / llama.cpp HIP
→ Markdown + JSON
```

[English documentation](README.md)

> 发布状态：**v0.1.0 仍处于 BLOCKED**，但 **G3 精度已 PASS**，
> 验收 Overall 为 **95.99**。PaddleOCR 已在线下确认该结果，项目 Maintainer
> 于 2026-07-17 决定无需再次全量运行。G2、G4、G5 仍为 BLOCKED。详见
> [G3 Maintainer 验收记录](docs/releases/0.1.0-g3-attestation.md)和
> [OmniDocBench v1.6 事实表](docs/benchmarks/omnidocbench-v1.6.md)。

| 指标 | PaddleOCR-VL（论文） | PaddleOCR-VL-ROCm（已验收） |
|---:|---:|---:|
| Overall | 96.33 | **95.99** |
| Text Edit-dist | 0.033 | 0.03488 |
| Reading-order Edit-dist | 0.127 | 0.12882 |
| Table TEDS | 94.76 | **94.09** |
| Formula CDM | 97.49 | **97.36** |

## 输入与输出

仓库包含真实的兼容性样例：

![杂志输入样例](examples/input/magazine.png)

- 输入：[`examples/input/magazine.png`](examples/input/magazine.png)
- Golden Markdown：[`tests/fixtures/golden/magazine.md`](tests/fixtures/golden/magazine.md)
- Golden 结构化 JSON：[`tests/fixtures/golden/magazine.json`](tests/fixtures/golden/magazine.json)

这是兼容性演示，不是发布证据。它展示公共输出格式，但不能证明当前硬件性能，
也不能证明 G4 已验收。

## 为什么需要它

PaddleOCR-VL 通常依赖在 Windows AMD 环境中较难部署的 VLM 服务栈。本项目组合了：

- DirectML 文档布局推理；
- 固定版本的 Windows llama.cpp HIP runtime 和经过哈希校验的 GGUF 资源；
- 外部 OpenAI-compatible endpoint 路径；
- 稳定的 CLI 与 Python 输出契约；
- 可审计的 OmniDocBench 工具。

## 已验证范围

- 一台 Windows 11 / Radeon 8060S 机器完成了缓存资源校验安装、DirectML
  布局激活、托管 server 冒烟推理和外部 server 冒烟推理。
- 该记录没有保存精确的 AMD 驱动和 HIP runtime 版本，因此不能作为可复现的
  性能或发布门禁 benchmark。
- 托管下载 manifest 固定了 2.27 GB（2.12 GiB）资源的大小和 SHA-256。
- 正式评分分母是 1,651 个 GT 页面。已批准的 official-local 运行包含
  1,650 个成功预测和 1 个失败页；失败页按空预测计分。它不是“对称排除 1 页”
  后得到的 1,650 页成绩。

证据和限制详见[兼容性矩阵](docs/compatibility/windows-amd.md)和
[benchmark 事实表](docs/benchmarks/omnidocbench-v1.6.md)。

## 五分钟快速开始

推荐环境：Windows 11、Python 3.11、PowerShell、受当前 AMD HIP SDK 支持的
AMD GPU，以及至少 5 GiB 可用磁盘空间。

```powershell
git clone https://github.com/AIwork4me/PaddleOCR-VL-ROCm.git
cd PaddleOCR-VL-ROCm

py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[download]"

paddleocr-vl-rocm setup --auto
paddleocr-vl-rocm doctor
paddleocr-vl-rocm run examples/input/magazine.png
```

`setup --auto` 会下载、校验、安装资源，并在 8111 端口启动托管 server。
默认根目录为 `%LOCALAPPDATA%\PaddleOCR-VL-ROCm`；模型位于 `models\`，
runtime 位于 `runtime\`，下载缓存位于 `cache\`，活动路径记录在
`config.json` 中。

使用其他磁盘或目录：

```powershell
paddleocr-vl-rocm setup --auto --root D:\PaddleOCR-VL-ROCm
```

使用 `setup --no-start` 可以只安装、不启动 server。CLI 会打印稍后启动
`llama-server.exe` 所需的完整命令。

项目目前没有托管的 `stop` 或 `clean` 命令。删除托管根目录前，请先停止由你
启动的 `llama-server.exe` 进程；不要在未检查内容时删除共享的 `--root`。

## 使用已有 server

如需保留自己的 llama.cpp、vLLM 或其他 OpenAI-compatible endpoint：

```powershell
pip install -e ".[download]"
paddleocr-vl-rocm doctor --server-url http://127.0.0.1:8111/v1
paddleocr-vl-rocm run examples/input/magazine.png --server-url http://127.0.0.1:8111/v1
```

仍支持向后兼容的旧命令形式：

```powershell
paddleocr-vl-rocm --input examples/input/magazine.png --server-url http://127.0.0.1:8111/v1
```

当 endpoint 要求特定模型标识时使用 `--api-model-name`。外部 server 自己负责
GPU/runtime 兼容性；本仓库不会验证或安装该 server。

## CLI

```text
paddleocr-vl-rocm setup [--auto | --no-start] [--root PATH] [--force]
paddleocr-vl-rocm doctor [--json] [--config PATH] [--server-url URL]
paddleocr-vl-rocm run INPUT [--output DIR] [--layout-model DIR]
                         [--layout-provider auto|directml|cpu]
                         [--server-url URL] [--api-model-name NAME]
                         [--vlm-max-workers N]
```

CLI、Python API 和底层 parser 共享公共并发默认值
`vlm_max_workers=8`。仅在内存压力或 server 请求容量不足时调低。

中文用户可从
[ModelScope](https://modelscope.cn/models/PaddlePaddle/PP-DocLayoutV3_onnx)
获取 PP-DocLayoutV3 ONNX；英文用户可使用
[Hugging Face](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx)。
托管安装会自动下载固定版本。

## Python API

```python
from paddleocr_vl_rocm import PaddleOCRVLROCm

pipeline = PaddleOCRVLROCm(
    layout_model_dir="models/PP-DocLayoutV3-onnx",
    vlm_server_url="http://127.0.0.1:8111/v1",
)
result = pipeline.predict("examples/input/magazine.png")
print(result.markdown_text)
```

## 输出结构

CLI 会在 `--output`（默认 `outputs`）下写入 `result.md` 和 `result.json`。
JSON 包含来源路径、页面尺寸、有序 block、标签、边界框、识别内容和 provider
元数据。坐标和标签属于版本化兼容性契约；修改布局或序列化逻辑时，应与仓库内
golden fixtures 对比。

## 复现评测

下载数据或运行评分前，请先阅读 [`eval/README.md`](eval/README.md)。其中固定了
OmniDocBench checkout，说明推理/评分阶段，并拒绝不完整的发布 artifact。
公开数字和门禁状态只在
[OmniDocBench v1.6 事实表](docs/benchmarks/omnidocbench-v1.6.md)维护。

不要发布子集、scorer 不匹配、使用 fallback 或未经验证 artifact 的成绩。

## 常见问题

- **PowerShell 阻止激活：**执行
  `Set-ExecutionPolicy -Scope Process Bypass`，再激活虚拟环境。
- **8111 端口被占用：**停止预期进程，或在其他端口运行外部 server，并传入
  `--server-url`。
- **下载失败：**重新运行 setup；部分下载可以续传。代理、DNS 和 GitHub
  release assets 必须可访问。
- **DirectML 不可用：**更新 AMD 显卡驱动，再运行
  `paddleocr-vl-rocm doctor --json`。Windows 托管验证要求 DirectML 排在首位，
  且不会静默回退到 CPU。
- **HIP DLL 或 server 启动失败：**对照 AMD 当前 Windows HIP 支持表检查 GPU
  和系统，再查看
  `%LOCALAPPDATA%\PaddleOCR-VL-ROCm\logs\server.log`。
- **诊断信息敏感：**公开 Doctor JSON 或日志前，删除用户名、本地路径、token、
  私有文档和 endpoint 凭据。

## 已知限制

- v0.1.0 尚未达到发布条件；G2、G4、G5 为 BLOCKED，G3 已 PASS。
- 项目只记录了一台 Windows AMD 机器的冒烟验证。
- README 曾展示的 27 页性能结论没有仓库内原始 timing artifact，因此已撤回，
  不再作为公开性能结论。
- 托管安装仅支持 Windows，且没有 stop/cleanup 命令。
- 空缓存公网安装尚未通过发布验收；已通过的是预校验缓存安装。

## Roadmap

参见 [`ROADMAP.md`](ROADMAP.md)。近期重点是关闭 G2、完成由 artifact 支撑的
G4 benchmark、收集可复现硬件报告，以及完成全新网络安装验证。

## 贡献、安全与许可证

提交改动前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)，并选择合适的
[Issue Form](.github/ISSUE_TEMPLATE)。安全问题请按
[`SECURITY.md`](SECURITY.md) 私下报告。项目使用 [MIT License](LICENSE)。
