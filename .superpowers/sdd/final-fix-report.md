# Final Local Eval Adapter Fix Report

Date: 2026-07-10

## Fixed Findings

- The adapter and lightweight entry points now default `--vlm-backend` to
  `llama-cpp-server`. CLI help states that this option is used only by the
  lightweight engine and ignored by the official engine.
- An official `predict()` iterable that materializes to no page results now
  raises a controlled page error before any empty Markdown is written. The
  normal retry and fallback path handles it; a same-directory fallback remains
  unchanged, and no-fallback pages are recorded as failed.
- Lightweight summaries and `_run_stats.json` now include `"fallback": 0` to
  match the official summary schema.
- The `run_eval.py` module docstring now describes prerequisite-gated live
  runs instead of a stale environment-specific pending status.

## TDD Evidence

- Added the empty official generator fallback regression and the no-fallback
  page-failure regression before implementation. The focused suite failed on
  the expected empty output behavior, then passed after the controlled failure
  check was added.
- Updated the lightweight stats test to assert `fallback` in both the returned
  summary and serialized stats; it failed with `KeyError` before the schema
  addition and passed afterward.

## Verification

- `python -m pytest tests/test_eval_adapter.py -q` -> 11 passed.
- `python eval/PaddleOCRVLROCm_img2md.py --help` -> confirms the local default
  and engine-scoped help text.
- `python eval/run_eval.py --help` -> confirms the matching engine-scoped help
  text.
