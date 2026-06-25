# 设计：工程质量与可维护性升级（Phase 1）

- 日期：2026-06-25
- 状态：草案，待 review
- 分支策略：新建特性分支 `feat/engineering-quality`（不在 `main` 上直接改动）

## 1. 背景与目标

`PaddleOCR-VL-ROCm` 是一个轻量级 PaddleOCR-VL 推理库：用 ONNXRuntime 跑
PP-DocLayoutV3 版面模型（绕开 PaddlePaddle 运行时），把视觉语言识别丢给外部的
OpenAI 兼容 VLM 服务器（vLLM / llama.cpp，跑在 ROCm GPU 上），输出 PaddleOCR-VL
风格的 JSON + Markdown。README 里的「1355 图对照验证」即 **OmniDocBench v1.5**
评测集（1,355 页）。

本阶段目标是把它升级到「顶级开源项目」应有的**工程质量与可维护性**，并把内部对照
验证升级为**标准、可对外公布的 OmniDocBench v1.5 / v1.6 基准分数**。

### 成功标准

1. 1400 行的 `pipeline_core.py` 拆分为单一职责模块，每个模块可独立测试。
2. **公开 API 保持兼容**（`PaddleOCRVLROCm` / `PaddleOCRVLROCmResult` / 现有 CLI
   参数），且贴合官方 PaddleOCR-VL 风格 —— 作为新内部实现的薄门面。
3. OmniDocBench v1.5（1,355 页）作为**特征化回归网**：重构前后 JSON / Markdown
   输出逐字段一致。
4. 落地本地工程脚手架：`ruff`（lint + format）、`mypy`、`pytest` 配置 + 一个本地
   检查脚本（CI 留到发布阶段）。
5. OmniDocBench **v1.5 & v1.6 可在本地一键评测**，产出按维度的标准分数并固化进仓库。

### 约束（已与用户确认）

- **回归网可用**：OmniDocBench v1.5 数据集 + 校验脚本本机可跑 → 可放心激进重构。
- **破坏性改动边界**：公开 API 保持兼容且贴合官方风格；内部实现 / 模块拆分可自由调整。
- **暂不上 CI**：本阶段只配本地 `ruff` / `mypy` / `pytest` 与检查脚本；GitHub Actions
  延后到发布阶段。
- **OmniDocBench 做完整评测链**（非仅 demo）。
- **新建特性分支** `feat/engineering-quality` 进行。

## 2. Workstream 1：分层重构

### 2.1 模块拆分

把 `src/paddleocr_vl_rocm/pipeline_core.py`（约 1400 行）按下表拆分。所有新模块都在
`src/paddleocr_vl_rocm/` 下。

| 新模块 | 职责（从 pipeline_core 迁出的符号） |
|---|---|
| `models.py` | 共享 dataclass：`LightBlock`、`TableCell` |
| `encoding.py` | image↔data-url / base64、`_sha256_hex` 等纯函数 |
| `geometry.py` | `_area`、`_overlap_ratio`、`_projection_overlap_ratio`、`_polygon_overlap_ratio`、`_filter_overlap_boxes` |
| `imageio.py` | 读图（`_open_crop_source` / `_open_crop_source_bgr`）、`_crop`、`_crop_from_bgr`、`_crop_margin`、`_merge_images` |
| `preprocess.py` | `_make_blocks`、`_merge_blocks`、表格内图片 token 化（`_gather_imgs_for_table_tokens`、`_paint_token`、`_tokenize_figure_of_table`、`_untokenize_figure_of_table`、`_construct_img_path`） |
| `vlm/__init__.py` + `vlm/client.py` | `OpenAICompatibleVLMClient`（= `LlamaCppClient`）、`_completion_payload`、`_content_from_response`、`_vlm_cache_key`、`_load_vlm_compat_cache`、`_prompt_for_label`、缓存 |
| `table.py` | OTSL→HTML：`_otsl_extract_tokens_and_text`、`_otsl_pad_to_square`、`_otsl_parse_texts`、`_convert_otsl_to_html` |
| `content.py` | `_normalize_vlm_result`、`_truncate_repetitive_content`、`_find_shortest_repeating_substring`、`_find_repeating_suffix`、`_format_block_content`、`_should_keep_text_newlines`、`_has_cjk` |
| `markdown.py` | `_markdown_from_blocks`、`_markdown_content_for_block`、`_format_title_text`、`_collapse_soft_newlines`、`_normalize_markdown_newlines` |
| `serialize.py` | `_result_payload`、`_layout_boxes_to_json`、`_blocks_with_orders`、`_block_to_json` |
| `pipeline_core.py`（瘦身） | 仅保留 `run_light_parser` 编排 |

不变 / 轻改：

- `pipeline.py`：公开类 `PaddleOCRVLROCm`，签名不变。
- `result.py`：公开类 `PaddleOCRVLROCmResult`，签名不变。
- `layout.py`：已较内聚，本阶段保留；如拆分，按 `layout/io`、`layout/postproc`、
  `layout/polygon`、`layout/order` 轻拆（可选，不阻塞主线）。
- `server.py` / `utils.py` / `cli.py`：基本不动；`cli.py` 内部 `print` 可换为 logging。

模块级常量（`IMAGE_LABELS`、`NON_MERGE_LABELS`、`SKIP_ORDER_LABELS`、
`MARKDOWN_IGNORE_LABELS`、`DEFAULT_MIN_PIXELS`、`DEFAULT_MAX_PIXELS`、`PROMPTS`、
`OTSL_*`）就近放到使用它们的模块；多处共用的进一个小的 `constants.py`。

> 注意：现有测试 `tests/test_vlm_payload.py` 从 `pipeline_core` 导入 `_completion_payload`。
拆分后该符号迁至 `vlm/client.py`，需同步更新测试导入路径（测试属于内部，可改）。

### 2.2 特征化测试（重构安全网）

代码中已存在 `compat_cache`（`_load_vlm_compat_cache`）与 `vlm_trace_events` 机制，
天然支持「录制-回放」式确定性测试：

- **录制**：用真实 VLM 服务器在 OmniDocBench v1.5 上跑一次，落盘一份 trace / compat
  cache（每页 prompt + 图 sha + VLM 原始输出）。
- **回放**：之后测试无需 VLM 服务器即可逐字节复现输出。
- **金标准**：把 v1.5 的 JSON / Markdown 输出（或其 sha256）作为 golden artifact 存入
  仓库；一个 pytest 用回放模式重跑，断言逐字段一致。

这样重构期间任何行为偏移都会被测试立即捕获。

### 2.3 工程脚手架（本地，不含 CI）

- `pyproject.toml`：增加 `[tool.ruff]`（lint + format，含行长度规则）、`[tool.mypy]`
  （对 `src/` 开启，逐步收紧）、`[tool.pytest.ini_options]`。
- 删除 `requirements.txt`（与 `pyproject.toml` 重复、有漂移风险）。`pyproject.toml`
  成为依赖的唯一来源。README 现用 `pip install -e .[dev]`，不受影响。
- `scripts/check.sh` + `scripts/check.ps1`：一键本地检查 = `python -m compileall` +
  `ruff check` + `ruff format --check` + `mypy src` + `pytest -q`。
- 内部代码的 `print` → `logging`；CLI 对用户可见的输出仍用 `rich`（`utils.get_console`）。

## 3. Workstream 2：OmniDocBench v1.5 / v1.6 本地评测

按 OmniDocBench（`opendatalab/OmniDocBench`，CVPR 2025）自身的评测流实现：

**评测流**：数据集（每页图 + ground-truth markdown/json）→ 适配器跑每页产出预测
markdown → `src/core/pipeline_eval.py` 比对算分（text_block 编辑距离 / table TEDS /
display_formula / reading_order / layout CDM）→ 按维度聚合报告。

适配器契约已核实：官方 `tools/model_infer/PaddleOCR_img2md.py` 即
`predict(img) → save_to_json() + save_to_markdown(pretty=False)`，与本库公开 API
完全一致。我们的适配器是其近一比一克隆（`PPStructureV3()` 换成指向 VLM 服务器的
`PaddleOCRVLROCm(...)`）。

### 交付物

1. **适配器** `PaddleOCRVLROCm_img2md.py`（镜像官方 PaddleOCR 适配器）。
2. **数据集获取脚本**：从 HuggingFace 拉取 v1.5（1,355 页）& v1.6，放入 OmniDocBench
   期望的目录布局。
3. **可复现 runner**（`paddleocr-vl-rocm-eval` 命令或脚本）：
   - pin 住某个 OmniDocBench commit（可复现，不 vendoring 进本包）。
   - 校验数据 → 跑适配器（打到 VLM 服务器）→ 跑指标 → 写分数报告。
4. **结果固化**：`results/omnidocbench/{v1.5,v1.6}/` + README 中一张按维度的分数表。
5. **子集 / 演示模式**：OmniDocBench 自带 `demo_data/`（19 页）做快速冒烟；外加可配
   页数上限做部分跑。
6. **回放模式**：预测结果缓存后可不打 VLM 服务器直接重算分数（复用 §2.2 的
   compat cache 思路）。

### 关键事实 / 假设

- 适配器近乎零成本，因公开 API 已对齐。
- OmniDocBench 作为 **pin 住 commit 的 git 依赖**（clone 到受管理目录，不 vendoring）。
- 生成预测时需要 ROCm VLM 服务器在跑；回放模式可免推理。
- v1.6 的确切页数 / 数据集位置需从 HF 确认并 pin revision（HF 上 ~1,651 页，应为
  v1.6 / 当前集）。
- OmniDocBench 的 CDM 指标（`metrics/cdm/`，内容距离模型）可能有额外安装步骤，
  实现时核实；必要时提供无 CDM 的精简评分作为兜底。

## 4. 排序与协同

**Workstream 1（重构）先行 → Workstream 2（OmniDocBench）在后。**

协同点：重构前跑一次基线分数、重构后再跑一次，**两次分数一致 = 重构零回归的最强
证据**，把「评测」变成「回归网」的一部分。

## 5. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 全量跑很重（GPU × ~3000 页 × 两版） | 子集 / 演示模式 + 回放缓存 |
| OmniDocBench 依赖版本可能与本库冲突 | 评测在隔离 venv 运行 |
| CDM 指标需额外模型 / 安装 | 实现时核实 `metrics/cdm`，必要时提供精简评分兜底 |
| v1.6 数据集确切形态 | 实现时从 HF 确认并 pin revision |
| 公开 API 被意外破坏 | 特征化测试 + 公开门面单独的契约测试 |

## 6. 范围之外（后续阶段）

- 发布到 GitHub（issue / PR 模板、CONTRIBUTING、CODE_OF_CONDUCT、badges）、PyPI 发布、
  Docker、CHANGELOG、release 流程。
- 文档站、API 参考自动生成。
- GitHub Actions CI（本阶段只配本地）。
- 功能扩展：批处理、PDF、更多 VLM 后端、layout 模型加速（DirectML/ROCm）、异步并发。
