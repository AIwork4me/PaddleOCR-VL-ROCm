# 交接文档：WSL 安装+重启后继续（CDM 对齐 + 表格真因）

- 写于：2026-06-27
- 分支：`feat/engineering-quality` @ `0d7d5a1`（未合并到 main）
- 目标：**全面对齐** OmniDocBench v1.6（公式 CDM + Overall + 表格 TEDS 对齐官方 PaddleOCR-VL-1.6）

---

## 0. 一句话现状

Plan A（重构）+ Plan B 评测链 + #1 Hard 子集 + #3 表格真因排查 **全部完成**。
**唯一卡住的是 #2 的公式 CDM**：OmniDocBench 的 CDM 代码是 **POSIX-only**（用 shell-string 命令 + `/bin/sh`），在 Windows 上 `cmd.exe` 解析不了 → 跑不出分数。**依赖全部装好且单独验证可用**，卡在 CDM 代码层。**解决：装 WSL，在 Linux 里跑 CDM**（Linux 下 CDM 原生可用）。

---

## 1. 已完成（证据）

| 项 | 状态 | 关键产出 |
|---|---|---|
| Plan A 分层重构 | ✅ 完成 + 证明 | `pipeline_core.py` 1400→224 行；原始 vs 重构 **7/7 字节一致**（commit `77b5999`） |
| 工程脚手架 | ✅ | ruff/mypy/pytest、`scripts/check.sh`、特征化测试网（`tests/fixtures/` 已提交） |
| uv 环境 | ✅ | 本项目 `.venv`(py3.13) + 桌面 `OmniDocBench/.venv`(py3.11) |
| v1.6 预测 | ✅ 1650/1651 | `predictions/paddleocrvl_rocm/*.md`（1 张超大报纸页 VLM 硬 500，按空页算） |
| v1.6 全量分数（Edit_dist 口径 + 表格 TEDS） | ✅ | `results/omnidocbench/v16/paddleocrvl_rocm_quick_match_*.json` |
| v1.6 Hard 子集（296 页） | ✅ | `results/omnidocbench/v16/paddleocrvl_rocm_hard_quick_match_*.json` |
| #3 表格真因 | ✅ 排查完毕 | 见下文「#3 结论」 |

## 2. 当前分数 vs 官方 PaddleOCR-VL-1.6（v1.6）

| 维度 | 我们 | 官方 | 对齐 |
|---|---:|---:|---|
| 文本 Edit_dist ↓ | 0.0348（96.5%） | 0.033（96.7%） | ✅ ~0.2pt |
| 阅读顺序 Edit_dist ↓ | 0.1289（87.1%） | 0.127（87.3%） | ✅ ~0.2pt |
| 表格 TEDS ↑ | 0.9290 | 0.9476 | ⚠️ 结构差 1.7pt（见 #3） |
| 表格 TEDS-S ↑ | 0.9543 | 0.9711 | ⚠️ 1.7pt |
| 公式 **CDM** ↑ | **待跑** | 0.9749 | ⏳ 本文档目标 |
| Overall ↑ | 待算 | 0.9633 | ⏳ 依赖 CDM |

## 3. #3 表格真因结论（已排查，证据驱动）

逐项排除：❌量化（模型是 **BF16 未量化**，128 个 BF16 tensor，2.0 字节/参数）、❌后端数值（llama.cpp ≈ Paddle 原生，同图表格 char 相似度 0.9965）、❌匹配/解析失败（99% 表格已匹配）、❌内容识别（Edit_dist ≈ 94.9% ≈ 官方）、❌我们的 OTSL→HTML 丢 span（我们的 span 与参考项目 `pd_native` 177/178 一致）。

**结论：差距是结构性的（TEDS/TEDS-S 各低 ~1.7pt），且我们的管线 ≈ 参考项目 `paddleocr_vl_onnx` 的 Paddle-原生臂。** 要彻底定位+修掉那 1.7pt，需在同一批 v1.6 表格页上跑**官方 PP-StructureV3**（用参考项目的 `.venv-pd`，PaddlePaddle 原生），把它的表格 HTML 结构与我们逐字段 diff。**不是我们代码的 bug**。

---

## 4. 重启后恢复步骤（按顺序）

### Step 1 — 装 WSL（你来，一次性）
以管理员 PowerShell：
```powershell
wsl --install -d Ubuntu-22.04
```
> 选 **22.04**：自带 Python 3.10（在 OmniDocBench 要求的 `<3.12` 范围内，省去 deadsnakes PPA）。
装完会要求**重启**。重启后首次进 Ubuntu 设个用户名/密码即可。然后回来跟我说「继续」。

### Step 2 —（我来）在 WSL Linux 里搭 CDM 环境
WSL 里跑（我可以从宿主用 `wsl.exe bash -c "..."` 驱动）。要点：
1. apt 换国内源（USTC/TUNA），装 CDM 依赖：`texlive-lang-chinese texlive-lang-cjk texlive-latex-extra texlive-fonts-recommended texlive-science imagemagick ghostscript python3-venv git`
2. **在 WSL 里全新 clone OmniDocBench**（**不要用**桌面那个 `<omnidocbench-worktree>` —— 它被我打了 Windows 专用的补丁 `_win_q`/`>NUL`/路径-flatten，在 Linux 下会反向出问题）。用 gitclone 镜像：`git clone https://gitclone.com/github.com/opendatalab/OmniDocBench.git`
3. 建 Linux venv（python3.10），装 OmniDocBench 依赖（**不锁版本**，同 Windows 那次的清单：`apted beautifulsoup4 evaluate func-timeout Levenshtein loguru lxml numpy pandas Pillow pylatexenc PyYAML scipy tabulate tqdm nltk matplotlib`，pip 走清华源）

### Step 3 —（我来）在 WSL 里跑 CDM 评测
写一个 v1.6 CDM config（路径用 `/mnt/c/...`），指向**已存在的预测**和 GT：
```yaml
end2end_eval:
  metrics:
    text_block: { metric: [Edit_dist] }
    display_formula: { metric: [Edit_dist, CDM], cdm_workers: 8 }
    table: { metric: [TEDS, Edit_dist], teds_workers: 16 }
    reading_order: { metric: [Edit_dist] }
  dataset:
    dataset_name: end2end_dataset
    ground_truth: { data_path: /mnt/c/Users/rocm/Desktop/PaddleOCR-VL-ROCm/OmniDocBench_data/OmniDocBench.json }
    prediction:   { data_path: /mnt/c/Users/rocm/Desktop/PaddleOCR-VL-ROCm/predictions/paddleocrvl_rocm }
    match_method: quick_match
    match_workers: 24
```
> Linux 下 CDM 的 pdflatex+magick+gs 走 `/bin/sh`，**原生可用**，不用任何补丁/`GS_LIB`。
> 只算分（预测已存在）——**不需要 llama-server**。
> 为加速，可把 `predictions/` 和 `OmniDocBench_data/` 复制进 WSL 原生文件系统（`~/`）再跑（`/mnt/c` 跨文件系统 I/O 较慢）。
跑完读 `result/.../..._metric_result.json` 的 `display_formula.CDM` → **公式 CDM 分数**；`Overall` 见 run_summary。

### Step 4 —（可选）#3 官方 PP-StructureV3 表格对照
用参考项目 `<user-home>\Desktop\paddleocr_vl_onnx\.venv-pd`（PaddlePaddle 原生）跑官方管线，在同一批 v1.6 表格页产出表格 HTML，与我们 `predictions/paddleocrvl_rocm/*.md` 里的表格逐字段 diff，定位那 1.7pt 结构差。可在 WSL 或 Windows 做。

---

## 5. 关键环境备忘（避免重踩坑）

- **网络**：`github.com`/`huggingface.co`/`tug.org` 在这台机器被墙；可用的镜像：**modelscope**（数据集）、**gitclone.com / ghproxy.net / ghfast.top**（GitHub）、**USTC/TUNA CTAN**（TeX）、**USTC/TUNA pypi**。
- **数据**已下好在 `OmniDocBench_data/`（1651 图 + 42MB manifest，via modelscope）。
- **Windows 上已装好且单独验证可用的 CDM 依赖**（重启后仍在，但 **Windows 原生跑 CDM 走不通——代码是 POSIX-only**，所以走 WSL）：
  - TeX Live 2026：`C:/texlive/2026`（scheme-medium + collection-langcjk + collection-langchinese，含 CJK.sty + gkai + 自带 tlgs Ghostscript）
  - ImageMagick：`C:/Program Files/ImageMagick-7.1.2-Q16-HDRI/magick.exe`
  - Ghostscript：TeX Live 自带 `C:/texlive/2026/tlpkg/tlgs/bin/gswin64c.exe`（需 `GS_LIB` 指向 tlgs 全部子目录才能被 ImageMagick 调用）
  - **Windows 跑 pdf_validation 必须 `PYTHONUTF8=1`**（OmniDocBench 读 manifest 没指定 utf-8，中文 Windows 默认 GBK 会崩）
- **llama-server**（评测时已停，释放了 GPU）：重启后如需重测某些页，`powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<user-home>\Desktop\paddleocr_vl_onnx\scripts\05_start_llama_server.ps1" -Ngl 99`（端口 8111，模型 `PaddleOCR-VL-1.6-GGUF.gguf`，backend `llama-cpp-server`）。
- 桌面 `<omnidocbench-worktree>` 是 **被我打了 Windows 补丁**的副本（`src/metrics/cdm/modules/latex2bbox_color.py`：`_win_q` 双引号、`>NUL`、path-flatten）。WSL 里**另起干净 clone**。
- `eval/.omnidocbench` 是指向桌面 OmniDocBench 的 **junction**；`predictions/paddleocrvl_rocm_cdm`/`_hard` 是指向预测目录的 junction（用来生成不同 save_name）。

## 6. 文件地图
- 本项目：`<repo>`
- 设计/计划：`docs/superpowers/specs/2026-06-25-engineering-quality-upgrade-design.md`、`docs/superpowers/plans/2026-06-25-engineering-quality-refactor.md`、`…-omnidocbench-eval.md`
- 进度账本（最权威）：`.superpowers/sdd/progress.md`（gitignored scratch）
- 本地评测 config（gitignored）：`eval/configs/run_v16_local.yaml`（Edit_dist 口径）、`run_v16_cdm_local.yaml`（含 CDM，Windows 跑过但 CDM=0）、`run_v16_hard_local.yaml`（Hard 子集）
- 数据：`OmniDocBench_data/`、预测：`predictions/paddleocrvl_rocm/`、分数：`results/omnidocbench/v16/`

---

**重启后回来跟我说「继续」，我会：在 WSL 里搭 CDM 环境 → 跑公式 CDM + Overall → 报数；然后（如你要）做 #3 官方表格对照。**
