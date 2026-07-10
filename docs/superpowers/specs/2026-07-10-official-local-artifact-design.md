# Design: Official Local Artifact and Alignment Loop

- Date: 2026-07-10
- Status: Approved for implementation planning
- Scope: Produce repository-backed official-engine OmniDocBench v1.6 artifacts on the local AMD Windows + llama.cpp/GGUF environment, then use them as the evidence baseline for lightweight accuracy and speed work.

## 1. Background

The previous local reference-quality upgrade added a production evaluation
adapter with two engines:

- `lightweight`: PaddleOCR-VL-ROCm's ONNXRuntime layout plus local
  OpenAI-compatible VLM server path.
- `official`: PaddleOCR's `PaddleOCRVL` doc_parser path, pointed at the same
  local llama.cpp/GGUF VLM server and exporting evaluation-oriented Markdown.

That work also updated the documentation to mark official-engine scores as
pending because this repository does not yet contain engine-identified official
prediction or score artifacts. The companion
`C:\Users\rocm\Desktop\omnidocbench-amd-windows` repository has strong official
evidence, including a full `1651` page official run with `1650` successful pages
and `1` failure, but those artifacts are not this repository's tracked evidence.

Current local discovery shows:

- `paddleocr_vl_rocm` imports from this repository.
- `paddleocr` is not installed in the current repository Python environment.
- Local lightweight v1.6 score artifacts exist under `results/omnidocbench/v16/`.
- Local official artifacts exist in the companion repository, but not under this
  repository's `results/` tree.

The user has approved installing/configuring the official `paddleocr` dependency
in this repository environment so the official engine can be reproduced here.

## 2. Goals

1. Install or configure the local official `paddleocr` dependency required by
   `--engine official`.
2. Verify the official engine with a tiny local smoke run before any full run.
3. Run an official-engine OmniDocBench v1.6 artifact path in this repository
   using the local Windows + AMD + llama.cpp/GGUF environment only.
4. Save predictions, `_run_stats.json`, metric results, run summaries, and
   provenance files with engine-identifying names.
5. Update README score tables only after repository-backed artifacts exist.
6. Use official-vs-lightweight local artifacts to drive the next accuracy and
   speed improvements.

## 3. Non-Goals

- Do not set up Linux vLLM, BF16, SGLang, FastDeploy, Docker inference, or any
  cross-machine reference path.
- Do not copy companion official scores into README as if they were produced by
  this repository.
- Do not rewrite OmniDocBench ground truth or tune benchmark scores.
- Do not optimize lightweight output before official-local evidence exists.
- Do not commit generated full prediction Markdown unless explicitly selected
  as a release artifact; prefer tracked summaries, metrics, provenance, and
  compact diagnostics.

## 4. Approach Options

### Recommended: Local Official Reproduction First

Install/configure official `paddleocr`, run a small official-engine smoke set,
then run full v1.6 official inference and scoring. Artifacts are saved with
names such as `paddleocr_official_local_llamacpp_gguf_*`, so README claims can
point to repository-backed evidence.

Trade-off: This is the longest path because full inference and CDM scoring are
heavy, but it gives the project the strongest evidence chain.

### Alternative: Import Companion Artifacts

Copy official predictions/results from `omnidocbench-amd-windows` and record
their provenance.

Trade-off: Fast, but weaker. It proves compatibility with the companion setup,
not reproduction through this repository's current commands.

### Alternative: Optimize Lightweight First

Skip official reproduction and directly tune the lightweight path.

Trade-off: Fastest to start coding, but risky. Without a same-repo official
baseline, low-score cases can be misclassified as adapter bugs when they are
actually local model/server-output differences.

## 5. Recommended Design

### 5.1 Dependency Setup

Add a clear, local-only official-engine setup path. The setup may be a script,
documentation section, or both, but it must verify:

- `import paddleocr` succeeds.
- `from paddleocr import PaddleOCRVL` succeeds.
- `PaddleOCRVL(pipeline_version="v1.6", vl_rec_backend="llama-cpp-server", ...)`
  can be constructed without importing PaddleOCR at adapter module import time.

If installation is manual or environment-sensitive, the workflow must fail with
an actionable message instead of silently skipping official reproduction.

### 5.2 Artifact Naming

Use engine-identifying directories and result names:

- Predictions:
  `predictions/paddleocr_official_local_llamacpp_gguf_v16`
- Non-CDM results:
  `results/omnidocbench/v16/paddleocr_official_local_llamacpp_gguf_quick_match_metric_result.json`
- CDM results:
  `results/omnidocbench/v16/paddleocr_official_local_llamacpp_gguf_quick_match_metric_result_cdm.json`
- Run summaries:
  matching `*_run_summary*.json`
- Provenance:
  `results/omnidocbench/v16/paddleocr_official_local_llamacpp_gguf_provenance.json`

The provenance file must include:

- date/time of run;
- git commit;
- engine;
- VLM server URL;
- requested model name;
- adapter command;
- scoring config path;
- dataset manifest path;
- prediction directory;
- number of pages, ok pages, failed pages, fallback pages;
- paths to metric and run-summary files.

### 5.3 Gated Execution

The official path must run in increasing cost gates:

1. **Import gate:** official dependency imports and `--help` still works without
   eager PaddleOCR imports.
2. **Smoke gate:** one to three local pages run with `--engine official`.
3. **Hard/subset gate:** a small representative subset runs and produces
   non-zero metrics.
4. **Full non-CDM gate:** full v1.6 official predictions score Text,
   Reading-order, Table, and Formula Edit-distance.
5. **Full CDM gate:** full v1.6 official predictions score CDM in the already
   configured local CDM environment.

A later gate must not start if an earlier gate fails.

### 5.4 README Update Rule

README and README.zh-CN may replace the official row's `pending` values only
after this repository contains matching official-local artifacts. The row must
name the environment as local Windows + AMD + llama.cpp/GGUF, not native Linux
vLLM or any cross-machine path.

If the official full run succeeds but CDM is not yet rerun, README may publish
only the non-CDM metrics and leave Formula CDM pending.

### 5.5 Alignment Diagnostics

After official-local artifacts exist, compare them with lightweight artifacts:

- Compare `_run_stats.json` counts, failure pages, fallback pages, and timings.
- Compare metric summaries by engine and metric.
- Build a hard-case list from formula CDM zero/low cases, empty predictions,
  malformed LaTeX, and text/table regressions.
- Classify each case as:
  - adapter/output-format issue;
  - malformed prediction LaTeX;
  - empty or missed formula;
  - matching/normalization issue;
  - true local model/server-output difference;
  - pending.

Only implement quality fixes when a case-level probe proves the root cause and
a test can lock the behavior.

### 5.6 Speed Work

Speed improvements come after artifact-backed accuracy comparison. Any speed
change must preserve output by default or prove the output delta is acceptable
with before/after subset evidence. Preferred first targets:

- better run timing summaries;
- retry/fallback controls;
- cache reuse for repeated diagnostics;
- conservative concurrency tuning with identical output checks.

## 6. Testing and Verification

Fast checks:

- `python -m py_compile` over tracked `src/` and `eval/*.py` files.
- `python -m pytest -q`.
- `python eval/PaddleOCRVLROCm_img2md.py --help`.
- `python eval/run_eval.py --help`.

Official dependency checks:

- `python -c "from paddleocr import PaddleOCRVL; print(PaddleOCRVL)"`.
- A one-page `--engine official` smoke run.

Artifact checks:

- Prediction directory contains expected page Markdown count.
- `_run_stats.json` has `engine == "official"`.
- Metric result and run summary files exist with engine-identifying names.
- Provenance JSON points to existing files and the current git commit.
- README official row remains pending until artifact checks pass.

## 7. Success Criteria

1. The current repository environment can run `--engine official`.
2. At least one official smoke prediction is produced by this repository.
3. Full or staged v1.6 official-local results are saved with engine-identifying
   artifact names.
4. README score claims match repository-backed artifacts exactly.
5. A lightweight-vs-official diagnostic report identifies the next concrete
   accuracy fixes.
6. All fast checks pass.
7. Changes are committed and pushed to
   `https://github.com/AIwork4me/PaddleOCR-VL-ROCm`.

## 8. Risks

- Installing official `paddleocr` may be slow or dependency-sensitive on this
  local Windows environment.
- Full official inference is long-running and depends on the local llama.cpp
  server staying healthy.
- Full CDM scoring is environment-sensitive even though the local CDM stack has
  already been validated.
- Generated prediction Markdown can be large; commit only the artifact classes
  chosen for repository evidence.
- The official-local score may still trail the public target because the local
  serving path is llama.cpp/GGUF, not Linux vLLM/BF16. That gap must be reported
  honestly.
