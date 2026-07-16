## Formula CDM Gap Investigation -- Closeout

**Date**: 2026-07-16
**Trigger**: Paper 97.49 vs ROCm 96.15, reported gap 1.34 pp
**Status**: CLOSED -- gap does not exist

### Result

| Source | CDM | Notes |
|---|---|---|
| Paper (PaddleOCR-VL, vLLM NVIDIA) | 97.49 | Published baseline |
| GGUF llama.cpp (WSL) | 97.33 | Per-sample CDM, 2324/2352 non-zero |
| ROCm llama.cpp (Windows, **after fix**) | **97.36** | Per-sample CDM, 2282/2352 non-zero |

The true gap between ROCm and GGUF is **0.03 pp** (ROCm slightly higher).
The original 96.15 number was an incorrect aggregate from a CDM evaluation where
all 2352 Windows samples failed with FileNotFoundError.

### Hypotheses tested

| Hypothesis | Verdict | Evidence |
|---|---|---|
| GGUF quantization loses formula precision | **Rejected** | 98.2% of predictions byte-identical between GGUF and ROCm |
| Windows vs Linux TeX rendering causes CDM gap | **Rejected** | Windows TeX Live 2026 (97.36%) = Linux WSL (97.33%) |

### Root cause of the original (false) gap

Four bugs in eval/.omnidocbench/src/metrics/cdm/ prevented CDM from
computing on Windows:

1. _safe_temp_prefix used os.path.basename(os.path.normpath(...)) which
   produced incorrect path components in multiprocessing on Windows.
2. tmp_dir was passed as a relative path to tempfile.TemporaryDirectory.
3. output_root was a relative path in the CDM evaluator.
4. open(jsonl) defaulted to gbk encoding, crashing on CJK bbox tokens.

Fixed in PaddleOCR-VL-ROCm commit a707d89.

### Files

- Per-sample CDM: result/paddleocrvl_rocm_cdm_quick_match_display_formula_per_sample_CDM.json
- Formula-level comparison: docs/formula-cdm-final-comparison-v2.json
- CDM re-run script: scripts/run_cdm_only.py
- Gap analysis script: scripts/analyze_cdm_root_cause.py

### Open (deferred)

- 43 book_zh_* CJK formulas still produce zero renderable tokens on Windows.
  CJK font configuration issue, independent of model quality.
  Tracked separately.
