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
with different aggregation conventions. The README score row uses the raw
all-values from `metric_result`: Table TEDS **93.1345** and Formula CDM
**96.7129**. The paired `run_summary` artifact's
`notebook_metric_summary` records the notebook/page convention instead:
Table TEDS **94.3222** and Formula CDM **96.9219**. The latter applies the
notebook's page denominators, so these values are not contradictory; always
name the artifact and convention when comparing scores.

## Historical Artifacts

Existing `paddleocrvl_rocm_*` and `paddleocr_official_local_llamacpp_gguf_*`
artifacts are retained for comparison. Do not mix score rows unless prediction
directory, adapter version, config, and CDM environment are explicitly named.
