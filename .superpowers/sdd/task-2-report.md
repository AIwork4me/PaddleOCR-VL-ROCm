# Task 2: Production Adapter Migration Report

## Status

Completed and committed as `aa656f5 feat: add local official eval adapter`.

## Implementation

- Added `run_adapter` with `.env.local` parsing, environment overrides, and lightweight/official dispatch.
- Migrated the existing lightweight runner to `run_lightweight_folder` with lazy imports, error logging, run statistics, failure counts, and the requested majority-success exit threshold.
- Retained `process_folder` as a compatibility wrapper.
- Added official PaddleOCRVL execution with lazy imports, retry handling, fallback prediction copying, Markdown extraction, OmniDocBench normalization, error logging, and run statistics.
- Routed the command-line interface through `run_adapter` and added engine, retry, and fallback options.

## Files Changed

- Modified: `eval/PaddleOCRVLROCm_img2md.py`
- Exercised without modification: `tests/test_eval_adapter.py`
- Created after the implementation commit: `.superpowers/sdd/task-2-report.md`

## GREEN Evidence

Command:

```powershell
python -m pytest tests/test_eval_adapter.py -q
```

Result:

```text
.......                                                                  [100%]
```

Exit code: `0` (`7 passed`). `git diff --check` also exited `0`.

## Self-Review

- Verified all exact Task 2 interfaces are present and the CLI invokes `run_adapter`.
- Verified production imports for both pipeline implementations are lazy, allowing contract tests and unrelated adapter uses to load without the local runtime.
- Verified the requested defaults and exact official pipeline configuration values are used.
- Verified lightweight and official summaries include failure counts and persist `_run_stats.json`; per-page failures write `_errors.log`.
- Verified official Markdown conversion prioritizes `_to_markdown(pretty=False)` and normalization converts centered image/text wrappers for OmniDocBench.
- No Task 1 contract tests were changed, preserving their original intent.

## Residual Validation Boundary

The contract suite uses fakes and does not execute a live PaddleOCRVL server or the local official PaddleOCR installation. That dependency is intentionally imported only when the official engine is selected; live inference validation requires the local runtime and model server.

## Review Fixes

Fixed the official adapter's handling of iterable `predict()` results and stale output cleanup on reruns.

### RED Evidence

Command:

```powershell
python -m pytest tests/test_eval_adapter.py -q -k official_folder_materializes_generator_results
```

Result: failed with `SystemExit: 2` because a fake official pipeline returned a generator of two result objects, which the runner passed directly to `_official_result_to_markdown` instead of iterating.

### GREEN Evidence

Commands:

```powershell
python -m pytest tests/test_eval_adapter.py -q -k official_folder_materializes_generator_results
python -m pytest tests/test_eval_adapter.py -q
```

Results: targeted regression passed (`1 passed`) and the full adapter suite passed (`8 passed`), both with exit code `0`.

### Files Changed

- Modified: `eval/PaddleOCRVLROCm_img2md.py`
- Modified: `tests/test_eval_adapter.py`
- Modified: `.superpowers/sdd/task-2-report.md`

## Re-review Fix: Same-directory Fallback Preservation

When `fallback_pred_dir` is the same directory as `out_dir`, the official runner now retains the existing page Markdown while inference is attempted. On inference failure, it recognizes that file as the fallback without calling `shutil.copyfile` on a source and destination that are the same path.

### RED Evidence

Command:

```powershell
python -m pytest tests/test_eval_adapter.py -q -k same_directory_fallback
```

Result: failed with `SystemExit: 2`. The runner deleted the pre-created `page.md` before prediction failed, so no fallback remained.

### GREEN Evidence

Commands:

```powershell
python -m pytest tests/test_eval_adapter.py -q -k same_directory_fallback
python -m pytest tests/test_eval_adapter.py -q
```

Results: the targeted regression passed (`1 passed`), and the full adapter suite passed (`9 passed`), both with exit code `0`.

### Files Changed

- Modified: `eval/PaddleOCRVLROCm_img2md.py`
- Modified: `tests/test_eval_adapter.py`
- Modified: `.superpowers/sdd/task-2-report.md`
