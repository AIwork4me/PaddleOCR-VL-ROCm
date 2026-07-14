# Task 5 Empty-Log Redaction Hotfix Design

## Context

The first authorized real Task 5 Preflight attempt,
`task5-20260715-paired-a1`, failed before inference on Windows PowerShell 5.1.
The native command runner passed an empty stdout or stderr string to
`Protect-LoggedText`. PowerShell 5.1 accepts empty input in `ConvertFrom-Json`
but returns `$null`; the redaction path then called `.Replace()` on that null
value. The attempt is durably `invalid` and must never be reused or modified.

The model server on port 8111 and the sealed G0 receipt/output hashes were
healthy at failure time. No page inference ran.

## Approved behavior

`Protect-LoggedText` treats null, empty, and whitespace-only command streams as
ordinary safe text and returns their exact string representation without JSON
parsing. A literal JSON `null` is emitted as the textual JSON value `null`, not
as a PowerShell null pipeline result. Non-empty JSON, JSONL, embedded JSON,
pretty multiline JSON, free text, and sensitive-value redaction retain their
existing behavior.

The fix is local to `scripts/run_task5_paired_v16.ps1` and its executable tests
in `tests/test_run_task5_paired_v16_script.py`. It must not weaken strict command
records, persisted-log hashing, lifecycle checks, or pre-persistence redaction.

## TDD and verification

The RED test executes the real PowerShell 5.1 `Invoke-LoggedNative` path with:

- empty stdout and non-empty stderr;
- non-empty stdout and empty stderr;
- both streams empty;
- whitespace-only output;
- textual JSON `null`.

Each case must create a valid redacted log and strict command record without a
null-method exception. Existing structured/multiline secret tests must remain
green. After the minimal fix, run the focused runner suite, Task 1 authority and
release suites, exact Ruff lint/format/diff gates, residual-process audit, and
one final full pytest on the final bytes. Require independent read-only review
with Critical and Important findings at zero.

## Real-run recovery

Commit the reviewed fix. Preserve `task5-20260715-paired-a1` byte-for-byte as an
invalid attempt. Re-run real Preflight using a new AttemptId,
`task5-20260715-paired-a2`, with the same explicit r7 root, dataset, layout,
runtime configuration, scorer interpreter, server identity, API model, and G0
receipt. Only after a2 Preflight and manifest validation succeed may Official,
Lightweight, Score, Compare, and Decide continue.

No manual stage-state edits, receipt fabrication, environment override, or
copying from a1 are permitted.
