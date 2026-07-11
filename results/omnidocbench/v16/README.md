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
with different aggregation conventions. The project README defaults to the
OmniDocBench official leaderboard/notebook convention: page-level Text
Edit-distance **0.03402**, Reading-order Edit-distance **0.12824**, Table TEDS
**94.3222**, Formula CDM **96.9219**, and Overall **95.9475**. This matches the
aggregation style used by OmniDocBench's `tools/generate_result_tables.ipynb`.

The lower-level raw all-values from `metric_result` are retained for audit:
Table TEDS **93.1345** and Formula CDM **96.7129**. These values are not
contradictory; always name the artifact and convention when comparing scores.

## Historical Artifacts

Existing `paddleocrvl_rocm_*` and `paddleocr_official_local_llamacpp_gguf_*`
artifacts are retained for comparison. Do not mix score rows unless prediction
directory, adapter version, config, and CDM environment are explicitly named.
