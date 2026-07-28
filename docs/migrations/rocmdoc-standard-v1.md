# Migration Record — PaddleOCR-VL-ROCm → ROCmDoc Model Repository Standard v1

Decision log, not a benchmark report. No unrun GPU test is claimed.

## Central lock
- central repository: AIwork4me/OmniDocBench-ROCm; commit `ccd466ef317fd6a710131db3a19ec9d55a65ce2e` (`.rocmdoc/spec-lock.json`)
- migrated 2026-07-27; mode `all-safe` (no GPU inference, no new scoring)
- baseline commit `f0cb4014be5f9f98593f6b08afbc2404f049df4d` (branch `main`; one untracked `.superpowers/sdd/` dir left untouched)

## Structural fix
`NOTICE` was missing (ADR-0006) — created. `model_card.json` (v1) retained; its
badge is already the central `community` enum.

## v2 result (single)
`paddleocr-vl-1-6__windows-hip__llama-cpp__bf16__v1-6__…` (PRIMARY), Overall
**95.77**, `submitted`. Legacy v1 `model_card.json` (95.77) maps to it.

## Score conflict (no auto-resolution, §15.4)
- **95.77** = canonical (raw-evidence-derived; model_card submetrics compute to 95.77, matches `metric_result_cdm` + REPRO.yaml). 
- **95.99** (README "G3 PASS, maintainer-accepted out-of-band"; also the official_reference) — different/official number.
- **96.33** (README table) — different number.
- The official PaddleOCR-VL 95.99 is an external reference (different engine) → `docs/benchmark-context.md`, never in `results[]`.

## ⚠️ Weak / inconsistent evidence (blocks assurance upgrade)
- **Platform ambiguity:** REPRO.yaml says `linux-rocm/gfx1100`, but the measured
  artifact (`paddleocr_official_local_llamacpp_gguf_*`) carries **Windows-native
  paths** + the model_card hardware is **Strix Halo (windows-hip)**. The v2 result
  is recorded as `windows-hip` (strongest signal); REPRO's linux-rocm claim is
  unverified and recorded as a `planned` implementation in `rocmdoc.yaml`.
- **Unpinned weights:** REPRO `weights.revision`/`sha256` = `not_recorded`; no
  `prediction_manifest_sha256` in provenance (hash unverifiable).
- The result is therefore capped at **`submitted`**. `evidence-complete` requires
  weights pinned + hash verifiable + platform confirmed — none established here.

## Mixed pipeline (§15.1)
The measured result is a MIXED pipeline on Strix Halo: layout = PP-DocLayoutV3 via
**ONNX Runtime DirectML** (NOT ROCm) + VLM = **llama.cpp HIP**. Modeled with
`implementation.components`; the DirectML layout component is never described as
ROCm. `rocmdoc.yaml` declares `windows-hip/llama-cpp` (supported) +
`windows-hip/onnx-directml` (supported, layout component) + `linux-rocm/llama-cpp`
(planned).

## License (§15.5 / §18)
Repo code is MIT, but the model is NOT blanket MIT: code = MIT (open-source-ai);
**weights = Apache-2.0** (upstream PaddleOCR-VL model + weights, per the
PaddlePaddle/PaddleOCR-VL HuggingFace card). At v1 migration this was recorded as
`unknown` (weight license not verified — never assumed open-source-ai, ADR-0010);
verified Apache-2.0 on 2026-07-28. The weight revision/sha256 remains unpinned
(REPRO weights.sha256 = not_recorded) — a reproducibility gap, not a license gap.
See `NOTICE`.

## Conformance (central `ccd466e`)
| profile | result | against |
|---|---|---|
| structural | CONFORMANT | repo (NOTICE added) |
| base | PASS | real `paddleocr-vl-rocm` CLI |
| runtime-core | PASS | real CLI (doctor `status` added) |
| benchmark-omnidocbench-v16 | PASS | fake-CLI fixture |
| reproducible-score | PASS | v2 primary (provenance complete; no sha to fail) |
| pytest | pre-existing: 4 scorer-preflight Windows-path failures + 2 onnxruntime collection errors (env: DirectML dep not installed) — NOT introduced by this migration |

## Rollback
Additive: `rocmdoc.yaml`, `model_card_v2.json`, `.rocmdoc/spec-lock.json`, `NOTICE`,
`docs/migrations/`, `docs/benchmark-context.md`, `src/paddleocr_vl_rocm/standard_cli.py`,
`tests/fixtures/`, `scripts/generate_results_block.py`. Edits: `cli.py` (standard
commands + doctor `status` + `import json`), README generated block. No historical
result value altered.
