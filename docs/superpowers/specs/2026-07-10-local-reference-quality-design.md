# Design: Local Reference Quality Upgrade

- Date: 2026-07-10
- Status: Draft, pending review
- Scope: Improve PaddleOCR-VL-ROCm as a top-tier open-source project using only this machine's Windows + AMD + llama.cpp/GGUF + OmniDocBench environment for validation.

## 1. Background

OmniDocBench v1.6 is already configured and validated locally in
`<omnidocbench-worktree>`. That companion repository
contains the latest evidence for PaddleOCR-VL-1.6 on this machine:

- The lightweight PaddleOCR-VL-ROCm path is strong on text, reading order, and
  tables, but Formula CDM trails the official PaddleOCRVL doc_parser path.
- The official doc_parser path plus evaluation-oriented Markdown recovered a
  large part of the Formula CDM gap on the same AMD Windows + llama.cpp/GGUF
  serving stack.
- Known evaluator and adapter fixes already exist in the companion repository:
  determinant-style Formula CDM normalization, long-text timeout fallback,
  official adapter retry/fallback, run stats, and persistent error logs.
- The remaining gap is not treated as a request to add a Linux vLLM/BF16 path.
  All validation for this project must be based on the local machine only.

This repository currently lags behind the proven adapter and evaluation
workflow. Its OmniDocBench adapter is lightweight-only, defaults to a v1.5 model
name, lacks official/reference-engine mode, lacks run statistics, and does not
persist detailed per-page failures. That makes it harder to explain accuracy
differences, reproduce scores, and present the project as a polished open-source
system.

## 2. Goals

1. Make this repository's inference and evaluation entry points strictly align
   with the proven local OmniDocBench workflow.
2. Clearly separate three concerns: lightweight inference implementation,
   official/reference adapter comparison, and OmniDocBench scoring.
3. Improve local accuracy, stability, and speed within the AMD Windows
   llama.cpp/GGUF environment only.
4. Publish score claims that are backed by local OmniDocBench v1.6 artifacts and
   clearly labeled by engine and environment.
5. Push the completed work to `https://github.com/AIwork4me/PaddleOCR-VL-ROCm`.

## 3. Non-Goals

- Do not set up Linux vLLM, BF16, SGLang, FastDeploy, Docker inference, or any
  cross-machine reference path.
- Do not rewrite ground truth or tune benchmark scores.
- Do not make broad unrelated refactors outside the evaluation, adapter,
  inference-quality, and documentation surface needed for this task.
- Do not treat public PaddleOCR-VL numbers as proof unless reproduced or
  contrasted with local evidence.

## 4. Approach Options

### Recommended: Proven Adapter Migration First, Then Local Optimization

Migrate the already validated adapter and evaluation improvements from
`omnidocbench-amd-windows` into this repository, then run targeted local
diagnostics and optimizations. This gives the project a reliable baseline before
attempting speed or quality improvements.

Trade-off: Some remaining Formula CDM gap may still be a real limitation of the
local llama.cpp/GGUF path, but the project will report that honestly with local
evidence.

### Alternative: Optimize Lightweight Inference First

Tune the current lightweight path before adding official/reference mode.

Trade-off: Faster to touch the core pipeline, but riskier because current score
differences mix adapter gaps, output format differences, and model-output
issues.

### Alternative: Documentation-Only Release Polish

Only document the existing local scores and known limitations.

Trade-off: Low risk, but it does not fix the current adapter drift or improve
the project's engineering quality.

## 5. Recommended Design

### 5.1 Evaluation Adapter

Replace the current minimal `eval/PaddleOCRVLROCm_img2md.py` with a production
adapter that supports two local engines:

- `lightweight`: the existing ONNXRuntime layout + OpenAI-compatible VLM path.
- `official`: PaddleOCR's `PaddleOCRVL` doc_parser path, configured for the
  same local llama.cpp/GGUF server and exporting evaluation-oriented Markdown
  with `pretty=False`.

Both engines must write one `<image_stem>.md` per page, plus:

- `_run_stats.json` with page count, ok/fail count, engine, fallback count, and
  per-page timing.
- `_errors.log` with timestamped full tracebacks.
- A non-zero exit when fewer than 50% of pages succeed.
- Optional per-page retry for the official engine.
- Optional explicit fallback prediction directory for known transient official
  parser/server failures.

Defaults must resolve from CLI flags, `ADAPTER_*` environment variables, and
`.env.local` where available. The v1.6 local GGUF model name should be the
documented default when no override is present.

### 5.2 Evaluation Orchestrator

Extend `eval/run_eval.py` so the `infer` stage can select engine, backend,
retry, fallback directory, and model name. The orchestrator should call the
adapter's documented `run_adapter` entry point, not a private lightweight-only
function.

The eval stage remains dependent on the local `eval/.omnidocbench` checkout or
the companion setup. It should fail with clear instructions when prerequisites
are missing.

### 5.3 Inference Parameter Alignment

Make request parameters explicit and testable:

- `temperature=0.0`, deterministic seed, llama.cpp sampling controls, and
  max-token fields.
- vLLM-only `mm_processor_kwargs` stays supported, but local validation does not
  depend on vLLM.
- Trace events should capture backend, model, prompt, image format, crop size,
  request order, token limit, and min/max pixel intent.

The project should document that llama.cpp/GGUF cannot consume the same
`min_pixels` / `max_pixels` request path as vLLM, so local score claims are
environment-specific.

### 5.4 Local Quality Work

After the adapter migration, use local OmniDocBench evidence to improve quality:

- Compare lightweight and official local predictions on hard Formula CDM cases.
- Classify failures into malformed LaTeX, empty prediction, extraction/matching
  mismatch, and true model-output difference.
- Add narrow normalization or post-processing only when a local probe proves it
  preserves formula meaning and improves renderability.
- Add tests for any output normalization before changing production behavior.

No fix is accepted without a reproducible local failure case and a verification
run showing improvement or no regression.

### 5.5 Local Speed Work

Speed improvements must preserve output by default:

- Keep `vlm_max_workers` configurable and conservative by default.
- Add or improve timing summaries in `_run_stats.json`.
- Avoid changing decoding or image encoding defaults unless a characterization
  test proves output stability.
- Prefer caching, clearer batching boundaries, and failed-page retry controls
  over speculative prompt or crop changes.

### 5.6 Documentation and Reporting

README and eval docs should report local results by engine:

- Lightweight local engine.
- Official local engine, if reproduced in this repository.
- Hardware/software environment: Windows, AMD GPU, llama.cpp/GGUF server,
  OmniDocBench v1.6, local CDM environment.

The documentation must distinguish:

- Public PaddleOCR-VL paper targets.
- Local official-engine measurements.
- Local lightweight-engine measurements.

Claims like "native precision aligned" must be backed by local artifacts and
phrased with the measured gap.

## 6. Testing and Verification

Fast checks:

- `python -m compileall -q src/paddleocr_vl_rocm eval`
- `python -m pytest -q`
- `python eval/PaddleOCRVLROCm_img2md.py --help`
- `python eval/run_eval.py --help`

Adapter tests:

- `expected_md_name` preserves OmniDocBench naming.
- Adapter module imports without `paddleocr` installed.
- `_official_result_to_markdown` prefers `_to_markdown(pretty=False)`.
- Official Markdown HTML wrappers normalize to scorer-friendly Markdown.
- Run stats and error logs are written on controlled failures.
- `run_adapter` resolves `.env.local` and environment defaults correctly.

Local live verification:

- Use the already configured local VLM server and OmniDocBench environment.
- Run a small hard subset first.
- Run full v1.6 inference/scoring only after subset checks pass.
- Copy final score artifacts into `results/omnidocbench/v16/` with names that
  identify engine, match method, and CDM/non-CDM mode.

## 7. Success Criteria

1. The repository exposes a local, reproducible evaluation path for both
   lightweight and official engines.
2. Input/output parameters and Markdown export behavior are documented and
   covered by tests.
3. Local OmniDocBench v1.6 scores are updated from fresh artifacts or explicitly
   marked as previously recorded artifacts.
4. The README presents honest, engine-specific local scores and the remaining
   known gap.
5. The local check suite passes.
6. Changes are committed and pushed to
   `https://github.com/AIwork4me/PaddleOCR-VL-ROCm`.

## 8. Risks

- Full OmniDocBench inference is long-running and depends on the local VLM
  server staying healthy.
- The official PaddleOCR package may not be installed in this repository's
  environment, so tests must keep official-engine imports lazy.
- Some Formula CDM gaps may be true local server/model-output differences and
  should be documented rather than masked.
- Existing untracked local eval artifacts must not be accidentally committed.
