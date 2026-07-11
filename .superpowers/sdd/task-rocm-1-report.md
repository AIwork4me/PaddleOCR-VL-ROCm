# Task 1 Report

Status: DONE

## Files changed

- `README.md`
- `README.zh-CN.md`
- `results/omnidocbench/v16/README.md`
- `results/omnidocbench/v16/paddleocrvl_rocm_cdm_quick_match_metric_result_windows_native_2026-07-11.json`
- `results/omnidocbench/v16/paddleocrvl_rocm_cdm_quick_match_run_summary_windows_native_2026-07-11.json`

The score row is explicitly labeled as the latest Windows-native local ROCm CDM
evidence for `predictions/paddleocrvl_rocm_cdm`. The provenance README states
that these are Windows + AMD + llama.cpp/GGUF measurements, not Linux
vLLM/BF16 reference-path measurements, and retains historical artifacts for
comparison.

## Commit

`79f8482` (`docs: reconcile local rocm cdm evidence`)

## Commands and results

- `Get-Item C:\Users\rocm\Desktop\omnidocbench-amd-windows\eval-infra\01-omnidocbench\OmniDocBench\result\paddleocrvl_rocm_cdm_quick_match_metric_result.json | Format-List FullName,Length`: passed; source size `16380` bytes.
- `Get-Item C:\Users\rocm\Desktop\omnidocbench-amd-windows\eval-infra\01-omnidocbench\OmniDocBench\result\paddleocrvl_rocm_cdm_quick_match_run_summary.json | Format-List FullName,Length`: passed; source size `10393` bytes.
- `Copy-Item C:\Users\rocm\Desktop\omnidocbench-amd-windows\eval-infra\01-omnidocbench\OmniDocBench\result\paddleocrvl_rocm_cdm_quick_match_metric_result.json results\\omnidocbench\\v16\\paddleocrvl_rocm_cdm_quick_match_metric_result_windows_native_2026-07-11.json`: passed.
- `Copy-Item C:\Users\rocm\Desktop\omnidocbench-amd-windows\eval-infra\01-omnidocbench\OmniDocBench\result\paddleocrvl_rocm_cdm_quick_match_run_summary.json results\\omnidocbench\\v16\\paddleocrvl_rocm_cdm_quick_match_run_summary_windows_native_2026-07-11.json`: passed.
- `python -c "import json, pathlib; p=pathlib.Path('results/omnidocbench/v16'); m=json.loads((p/'paddleocrvl_rocm_cdm_quick_match_metric_result_windows_native_2026-07-11.json').read_text(encoding='utf-8')); s=json.loads((p/'paddleocrvl_rocm_cdm_quick_match_run_summary_windows_native_2026-07-11.json').read_text(encoding='utf-8')); print({'text': round(m['text_block']['all']['Edit_dist']['ALL_page_avg'], 5), 'reading_order': round(m['reading_order']['all']['Edit_dist']['ALL_page_avg'], 5), 'table': round(m['table']['all']['TEDS']['all'] * 100, 4), 'cdm': round(m['display_formula']['all']['CDM']['all'] * 100, 4), 'pages': m['match_debug']['page_count'], 'cdm_samples': s['stage_execution']['metrics']['display_formula']['CDM']['sample_count'], 'cdm_errors': s['stage_execution']['metrics']['display_formula']['CDM']['error_case_count'], 'cdm_exceptions': s['stage_execution']['metrics']['display_formula']['CDM']['exception_case_count']})"`: passed; values were text `0.03402`, reading order `0.12824`, table `93.1345`, CDM `96.7129`, 1651 pages, 2352 CDM samples, 0 CDM errors, and 0 CDM exceptions.
- `python -m pytest tests/test_eval_artifact_utils.py tests/test_eval_adapter.py -q`: passed, `................ [100%]`.
- `rg -n "96\\.7129|windows_native_2026-07-11|Linux vLLM/BF16" README.md README.zh-CN.md results\\omnidocbench\\v16\\README.md`: passed; required references found.
- `git diff --check`: passed; no whitespace errors.
- `git add -- README.md README.zh-CN.md results/omnidocbench/v16/README.md results/omnidocbench/v16/paddleocrvl_rocm_cdm_quick_match_metric_result_windows_native_2026-07-11.json results/omnidocbench/v16/paddleocrvl_rocm_cdm_quick_match_run_summary_windows_native_2026-07-11.json`: passed; only task files staged.
- `git diff --cached --check`: passed.
- `git commit -m "docs: reconcile local rocm cdm evidence"`: passed; commit `79f8482` created.

## Concerns

Git emitted normal LF-to-CRLF working-copy warnings for Markdown files. The
pre-existing untracked `data/`, `eval/.omnidocbench/`, and `logs/` directories
were left untouched and uncommitted.

## Follow-up Fix: Aggregation Convention Clarification

Status: DONE

The provenance documentation now distinguishes the two score conventions in
the Windows-native 2026-07-11 artifacts. The README score rows are explicitly
labeled as raw `metric_result` all-values: Table TEDS `93.1345` and Formula CDM
`96.7129`. The paired `run_summary.notebook_metric_summary` is documented as
the notebook/page convention: Table TEDS `94.3222` and Formula CDM `96.9219`.
The copied JSON artifact contents were not changed.

## Fix Commands and Results

- `rg -n "96\.7129|96\.9219|93\.1345|94\.3222" README.md README.zh-CN.md results\omnidocbench\v16\README.md`: passed; all four values were found across the requested files.
- `python -m pytest tests/test_eval_artifact_utils.py tests/test_eval_adapter.py -q`: passed, `................ [100%]`.
- `git diff --check`: passed; only normal LF-to-CRLF working-copy warnings were emitted.
- `git diff --stat`: passed; three README files changed, with no JSON artifact changes.

## Follow-up Concerns

The pre-existing untracked `data/`, `eval/.omnidocbench/`, and `logs/`
directories remain untouched and uncommitted.
