# Known Upstream Page Exception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept one immutable, publicly documented official-local PEG failure without deleting it from OmniDocBench's 1,651-page scoring denominator.

**Architecture:** Add one shared Python release contract consumed by both evaluation and PowerShell entry points. Persist the approved exception in run summaries and provenance, while leaving scorer inputs and notebook metric extraction unchanged.

**Tech Stack:** Python 3.10+, JSON, PowerShell, pytest, OmniDocBench v1.6.

## Global Constraints

- The only allowed filename is `newspaper_The Times UK_0801@magazinesclubnew_page_031.png`.
- The evidence URL is `https://github.com/PaddlePaddle/PaddleOCR/issues/18248`.
- Accepted stats are exactly `count=1651`, `ok=1650`, `fail=1`, `fallback=0`, `limit_pages=null`.
- The failure must contain `peg-native`; no prediction, fallback, or synthetic output may be added.
- Official scoring still includes all 1,651 GT pages and scores this page as empty.
- G3 still requires Overall >=96.13 and clean CDM/TEDS quality.

---

### Task 1: Implement the shared release exception contract

**Files:**
- Create: `eval/release_contract.py`
- Create: `tests/test_release_contract.py`
- Modify: `eval/run_eval.py`
- Modify: `tests/test_eval_report_path.py`
- Modify: `scripts/run_official_local_v16.ps1`
- Modify: `tests/test_run_official_local_v16_script.py`

**Interfaces:**
- Produces: `KNOWN_V16_OFFICIAL_FAILURE`.
- Produces: `validate_release_run_stats(run_stats, *, version, engine) -> list[dict[str, str]]`.
- Produces CLI: `python eval/release_contract.py --stats <path> --version v16 --engine official`.

- [ ] Write failing tests for the exact accepted failure and rejection of wrong filename, signature, counts, fallback, limited run, and non-official engine.
- [ ] Run focused tests and confirm failure because the module does not exist.
- [ ] Implement the immutable contract and CLI.
- [ ] Replace duplicated strict checks in `eval/run_eval.py` with the shared validator.
- [ ] Make PowerShell call the shared validator before CDM scoring.
- [ ] Run focused Python and PowerShell contract tests.

### Task 2: Persist exception provenance and update claims

**Files:**
- Modify: `eval/artifact_utils.py`
- Modify: `tests/test_eval_artifact_utils.py`
- Modify: `docs/accuracy-root-cause-v16.md`
- Modify: `eval/README.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `docs/releases/0.1.0-readiness.md`

**Interfaces:**
- Run summaries and provenance include `approved_known_failures`.

- [ ] Write failing artifact and documentation contract tests.
- [ ] Record the validated exception in summary/provenance writers.
- [ ] Update bilingual wording to distinguish success coverage from scoring denominator.
- [ ] Verify no document says 1,651 predictions succeeded or that PaddlePaddle maintainers resolved the issue.

### Task 3: Regenerate full evidence and reassess gates

**Files:**
- Regenerate: official prediction stats, metric result, run summary, and provenance under the existing ignored/generated paths.
- Update tracked evidence only after all gates pass.

- [ ] Run full unbounded official-local inference against all 1,651 pages.
- [ ] Validate exactly 1,650 successes and the one approved failure.
- [ ] Run full official v1.6 scoring with CDM/TEDS quality checks.
- [ ] Run the lightweight full scoring path on the same scorer contract.
- [ ] Recompute notebook metrics and compare them with G3 Overall >=96.13.
- [ ] Run compileall, Ruff, Ruff format, mypy, full pytest, build, and `git diff --check`.
- [ ] Request independent review and commit each completed task separately.
- [ ] Unlock performance Tasks 2-5 only if the newly generated G3 evidence passes.
