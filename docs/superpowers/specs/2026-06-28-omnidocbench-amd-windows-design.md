# OmniDocBench AMD Windows 评测系统搭建指南 — 设计

- 日期：2026-06-28
- 状态：草案，待 review
- 定位：独立一站式开源 repo，帮助 AMD Windows 用户从零搭建 OmniDocBench v1.6 全量评测系统

## 1. 背景与目标

在 Windows + AMD GPU 上搭建 OmniDocBench v1.6 全量评测系统极具挑战——涉及 llama.cpp HIP 构建、TeX Live CJK/CDM 环境、ImageMagick 7、ghostscript、WSL Linux 子系统、`\mathcolor` 渲染 bug 等 20+ 个已知坑。本 repo 将整个搭建过程固化成**自动化脚本 + AI-agent 可执行的 CLAUDE.md + 详细知识库**，让今后所有 AMD Windows 用户可以复刻。

**核心原则：**
- **PaddleOCR-VL-1.6 是验证范例**——证明搭建成功，不是 repo 的唯一用途
- **搭建好后可评测任何文档解析模型**——换模型只需写一个适配器
- **易懂**：每个步骤解释是什么 / 为什么 / 解决什么问题
- **易用**：脚本自动化 + CLAUDE.md 让 Claude Code / OpenCode 全自动搭建和验证

## 2. 架构

### 2.1 模型无关 vs 模型相关

```
eval-infra/        ← 模型无关，搭一次永久受益
  01-omnidocbench/    OmniDocBench 代码 + 数据集
  02-cdm-environment/ CDM 全套依赖（WSL + TL2026 + IM7 + fixes）
  03-scoring/         评分脚本 + 配置（接受任意 .md 预测）

adapters/          ← 模型相关，每个模型一个
  _template/          新模型适配器模板
  paddleocr-vl-1.6/   参考范例（完整验证过）
```

### 2.2 适配器接口规范

每个适配器只需做一件事：

```python
def run_adapter(img_dir: Path, out_dir: Path, server_url: str = ""):
    """输入图片目录 → 输出 <basename>.md 预测目录"""
```

评测基础设施读取这些 .md 文件，与模型完全解耦。

### 2.3 数据流

```
OmniDocBench 图片 (1651 张)
  │
  ▼ [adapter] 推理
  predictions/<model_name>/<basename>.md  (每页一个 Markdown)
  │
  ▼ [eval-infra/03-scoring]
  OmniDocBench 读 .md + GT → 匹配 → Edit_dist + TEDS（Windows 原生）
  │
  ▼ [eval-infra/03-scoring/cdm]
  CDM 渲染公式 → 颜色 bbox 匹配 → CDM 分数（WSL Linux 内）
```

## 3. Repo 结构

```
omnidocbench-amd-windows/
├── CLAUDE.md                       # AI-agent 编排（全自动搭建 + 验证）
├── README.md                       # 总览 + 快速开始（双语）
├── README.zh-CN.md
│
├── eval-infra/                     # ★ 模型无关的评测基础设施
│   ├── 01-omnidocbench/
│   │   ├── setup.ps1               # 下载 OmniDocBench 代码 + 数据集
│   │   ├── verify.ps1              # 验证：代码 + 数据集就绪
│   │   ├── config/                 # eval 配置模板（v1.5/v1.6/Hard 子集）
│   │   └── README.md               # 是什么/为什么/预期结果
│   ├── 02-cdm-environment/
│   │   ├── setup.sh                # WSL 内一键搭建 CDM 环境（9 步）
│   │   ├── verify.sh               # 验证：编译 CJK 公式→PDF→PNG→有颜色→CDM F1>0
│   │   └── README.md               # 每步是什么/为什么/对应哪个坑
│   └── 03-scoring/
│       ├── score.ps1               # Edit_dist + TEDS 评分（Windows 原生）
│       ├── score-cdm.sh            # CDM 评分（WSL 内，含 \mathcolor fix）
│       ├── verify.ps1              # 验证：四项指标非零
│       └── README.md
│
├── adapters/                       # ★ 模型适配器
│   ├── _template/
│   │   ├── run_adapter.py          # 接口模板 + 注释
│   │   ├── setup.ps1               # 模型环境安装模板
│   │   └── README.md               # 如何写适配器
│   └── paddleocr-vl-1.6/           # 参考范例
│       ├── 01-vlm-server/
│       │   ├── setup.ps1           # llama.cpp HIP + 模型下载
│       │   ├── verify.ps1          # curl /v1/models
│       │   └── README.md
│       ├── 02-layout-model/
│       │   ├── setup.ps1           # PP-DocLayoutV3 ONNX 下载
│       │   └── README.md
│       ├── run_adapter.py          # 推理适配器（图片→.md）
│       └── README.md
│
├── scripts/                        # 跨模块工具
│   ├── detect-mirrors.ps1          # 自动检测网络→选择镜像源→输出 env 文件
│   ├── wsl-ensure.ps1             # 确保 WSL 已安装（含 rootfs 导入）
│   └── full-verify.ps1            # 全链验证（从环境到出分）
│
└── docs/
    ├── pitfalls.md                 # 踩坑知识库（最有价值的部分）
    └── architecture.md             # 架构图 + 数据流
```

## 4. CLAUDE.md 设计

CLAUDE.md 是 AI-agent 编排层。核心原则：**高层编排（指向脚本）+ 异常处理（指向知识库）+ 人机协作点（显式标注）**。

### 结构

```markdown
# CLAUDE.md

## 这个 repo 做什么
在 AMD Windows 上从零搭建 OmniDocBench v1.6 全量评测系统。
以 PaddleOCR-VL-1.6 为范例验证。搭好后可评测任何文档解析模型。

## ⚠️ 人工介入点（agent 遇到时暂停）
1. WSL 安装 → wsl --install + 重启
2. VLM 服务器 → 确认 GPU 被占用
3. ImageMagick 安装 → 可能需要 UAC

## 执行流程
Step 0: detect-mirrors.ps1 → wsl-ensure.ps1
Step 1: eval-infra/01-omnidocbench/ (setup → verify)
Step 2: eval-infra/02-cdm-environment/ (setup → verify) ← 最难
Step 3: adapters/paddleocr-vl-1.6/ (setup server → run adapter)
Step 4: eval-infra/03-scoring/ (score → score-cdm → verify)

## 异常处理速查
| 症状 | 查 |
|---|---|
| 下载超时 | pitfalls.md#network |
| WSL 相关 | pitfalls.md#wsl |
| CDM = 0 | pitfalls.md#cdm-zero |
| \mathcolor 黑色 | pitfalls.md#mathcolor |

## 成功标准
四项指标全部非零：text < 0.1 · TEDS > 0.85 · CDM > 0.85
```

### 设计点

1. **verify 脚本是 agent 的"眼睛"**：每个 setup 后跟 verify，返回 0/1 + 诊断。agent 只需 check exit code。
2. **异常指向 pitfalls.md**：不把解决方案塞进 CLAUDE.md，而是"症状 → 查对应章节"。
3. **人工介入点 ⚠️ 显式标注**：agent 遇到时暂停 + 提示用户。

## 5. CDM 环境自动化（eval-infra/02-cdm-environment/）

这是 repo 最有价值的部分——把 20+ 个坑固化成幂等脚本。

### setup.sh 的 9 个步骤

| # | 做什么 | 为什么 | 对应的坑 |
|---|---|---|---|
| 1 | apt install texlive-lang-cjk/chinese/latex-extra imagemagick ghostscript | CDM 基础依赖 | Ubuntu texlive 缺 CJK.sty |
| 2 | 下载 + 安装 TL2026（USTC CTAN）| TL2026 有 \mathcolor + 完整 CJK | \mathcolor 在 2021 texlive 渲染黑色 |
| 3 | 复制 CJK.sty + gkai + arphic 字体 | TL2026 自带，Ubuntu 缺 | CJK.sty/c70gkai.fd 不存在 |
| 4 | 注入 gkaiu font map → pdftex.map | pdflatex 找不到字体位图 | "Font gkaiu5f not found" |
| 5 | 下载 IM7 AppImage + 系统级安装 | IM6 把彩色 PNG 渲染成灰度 | "grayscale PNG" → CDM F1=0 |
| 6 | IM6 policy.xml 允许 PDF | 安全策略禁止读 PDF | "security policy PDF" |
| 7 | 安装 IM7 依赖 libfribidi 等 | AppImage 缺系统库 | "libfribidi.so.0 not found" |
| 8 | OmniDocBench 代码 + \DeclareDocumentCommand fix | \mathcolor 在 TL2026 渲染黑色 | CDM 全部 F1=0 |
| 9 | Python venv + 依赖 | OmniDocBench 要 Python <3.12 | 版本冲突 |

### verify.sh

端到端验证全链路：pdflatex 编译 CJK 公式 → PDF → magick 转 PNG → PIL 检查颜色 > 2 →
CDM.evaluate 相同公式 F1 > 0 → echo OK / echo 诊断。

## 6. 镜像策略（detect-mirrors.ps1）

检测顺序：HuggingFace → ModelScope → GitHub → gitclone/ghproxy → pypi → 清华/USTC pypi →
CTAN(tug.org) → USTC/TUNA CTAN。

输出：`mirrors.env`（后续脚本读取）。

| 资源 | 国内主源 | 海外备源 |
|---|---|---|
| OmniDocBench 数据集 | ModelScope `OpenDataLab/OmniDocBench` | HF |
| PaddleOCR-VL-1.6-GGUF | ModelScope `PaddlePaddle/PaddleOCR-VL-1.6-GGUF` | HF |
| PP-DocLayoutV3 ONNX | ModelScope `AlexTransformer/PP-DocLayoutV3-onnx` | HF |
| llama.cpp HIP 构建 | ModelScope（参考脚本 02）| HF/GitHub |
| OmniDocBench 评测代码 | gitclone.com | GitHub |
| ImageMagick 7 AppImage | ghproxy.net / ghfast.top | GitHub |
| TeX Live 2026 | USTC/TUNA CTAN | tug.org |
| Python 包 | 清华 pypi | pypi.org |
| Ubuntu rootfs (WSL) | USTC ubuntu-cdimage | cloud-images.ubuntu.com |

## 7. pitfalls.md 知识库结构

按症状索引（用户/agent 遇到问题时按症状查找）：

```
# pitfalls.md

## #network — 网络问题
### GitHub 不可达 → 用 gitclone.com 或 ghproxy.net
### HuggingFace 不可达 → 用 ModelScope 或 hf-mirror.com
### tug.org 不可达 → 用 USTC/TUNA CTAN 镜像
### Microsoft Store 不可达（WSL 安装） → 手动 rootfs 导入

## #wsl — WSL 问题
### wsl --install 失败（raw.githubusercontent 超时） → rootfs 手动导入
### AppImage "Permission denied" → FUSE 未装 → --appimage-extract

## #cdm-zero — CDM 全部 F1=0
### 根因排查决策树
### #mathcolor — \mathcolor 渲染黑色
### #ghostscript — gs 找不到 gs_init.ps
### #grayscale — IM6 灰度问题 → 装 IM7
### #gkaiu-map — gkaiu 字体不在 pdftex.map
### #im-policy — IM6 安全策略禁止 PDF
### #posix — CDM 代码 POSIX-only → 必须在 WSL/Linux 跑

## #layout — 布局模型问题
### ONNX model not found → 下载脚本 + 路径

## #vlm — VLM 服务器问题
### llama-server 启动失败 → 检查 HIP SDK / NGL / 模型路径
### 500 Server Error → 图片太大 → 降低 max_tokens
```

每条：症状（一句话）→ 根因（为什么）→ 解决（具体命令）→ 验证（怎么确认修好了）。

## 8. 范围之外

- Docker 方案（作为备选，不在主线）
- v1.5 评测（配置模板提供，不自动化）
- 非 AMD GPU（NVIDIA/Intel）适配器（模板提供，社区贡献）
- CI/CD（本地脚本验证，不上 GitHub Actions）
