# OmniDocBench v1.6 Local Evidence

All artifacts in this directory are local Windows + AMD + llama.cpp/GGUF
measurements. They are not Linux vLLM/BF16 reference-path measurements.

## Current ROCm Lightweight/Local Evidence

| Artifact | Source | Notes |
|---|---|---|
| `paddleocrvl_rocm_cdm_quick_match_metric_result_windows_native_2026-07-11.json` | `omnidocbench-amd-windows` Windows-native CDM run | Current local ROCm CDM evidence for `predictions/paddleocrvl_rocm_cdm` |
| `paddleocrvl_rocm_cdm_quick_match_run_summary_windows_native_2026-07-11.json` | same run | Records 1651 pages, 2352 CDM samples, 0 CDM errors/exceptions |

## Score Aggregation Conventions

The dated Windows-native 2026-07-11 artifacts contain two valid score views
with different aggregation conventions. Under the OmniDocBench official
leaderboard/notebook convention, each Overall component is rounded to three
decimals first. Historical lightweight values are Text Edit-distance **0.034**,
Table TEDS **94.322**, Formula CDM **96.922**, and Overall **95.9480**. The
historical official-local comparison is Text **0.034**, Table **94.239**,
Formula **96.502**, and Overall **95.7803**. Reading-order Edit-distance
**0.128238** is reported separately and does not enter Overall.

These are reconstructed historical values, not fresh release evidence. See
`docs/accuracy-root-cause-v16.md` for provenance. The official path has 1,650
successful pages plus one project-approved known `peg-native` failure linked to
PaddleOCR issue #18248. Scoring retains all 1,651 GT pages and treats that page
as an empty prediction.

## Fresh Official-Local r7 G0 Evidence

The independently reviewed score-only r7 recovery reports Text Edit-distance
**0.035**, Formula CDM **96.485%**, Table TEDS **94.244%**, and Overall
**95.743** under three-decimal component rounding, with reading order excluded.
It authenticates and reuses the immutable r5 inference source; r7 did not rerun
inference. See the [tracked G0 evidence receipt](../../../docs/releases/0.1.0-g0-evidence.md),
SHA-256 `d0b7fcbe389e03439b5ba65126008fa5ee828a59e358ae0347c5bb6a51648a04`, produced from commit
`fd91cb0a2d75b0a18d16b1bb34652a148cb59b9e`.

This is fresh official-local G0 integrity evidence. It is not a Lightweight/G3
score and is not a Linux reference score.

The lower-level raw all-values from `metric_result` are retained for audit:
Table TEDS **93.1345** and Formula CDM **96.7129**. These values are not
contradictory; always name the artifact and convention when comparing scores.

## Historical Artifacts

Existing `paddleocrvl_rocm_*` and `paddleocr_official_local_llamacpp_gguf_*`
artifacts are retained for comparison. Do not mix score rows unless prediction
directory, adapter version, config, and CDM environment are explicitly named.
