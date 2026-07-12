# Evidence Task 3 Report

## Outcome

- Captured and immediately qualified all 20 manifest cases with the existing
  `scripts/record_trace.py`, the worktree `.venv`, authentic benchmark images,
  the real PP-DocLayoutV3 ONNX model, and the existing localhost PaddleOCR-VL
  1.6 llama.cpp service on port 8111.
- Every accepted case requested `auto` and reported active providers exactly
  `[DmlExecutionProvider, CPUExecutionProvider]`, with DirectML first. The
  layout session calls `disable_fallback()` and rejects failure to activate
  DirectML first.
- Raw JSONL, crops, payloads, responses, predictions, compat caches, and golden
  files remain under ignored `.superpowers/sdd` paths and are untracked.
- Authentic historical official scorer rows exist for every case and were
  hashed without publishing their contents. Official layout, crop, payload,
  and raw-VLM boundaries are unobservable. The current lightweight parser
  output and historical official scorer output are not a same-boundary pair,
  so `compare_inference_traces.py` was not given mismatched evidence and the
  first inference divergence remains unproven.

## TDD evidence

The first focused run failed with five expected `NameError` failures because
`_assert_trace_summary_schema` did not exist. After the minimal validator was
implemented, the same focused test file passed 9 tests. The contract requires
all 20 exact case IDs, complete boundary observations, no credential keys,
`auto`, and the exact DirectML-first provider list. Official deeper boundaries
alone may use the explicit scalar status `unobservable`.

The review fix added a regression test that loads the actual committed summary;
its first run failed because that artifact did not exist. The validator now
uses strict allowlists at every object level, exact enums and types, lowercase
64-character SHA-256 constraints, exact manifest case order/identity, an
authenticated 20-trace capture-set hash, fixed `auto`/DirectML-first providers,
and `layout_fallback_disabled=true`. Mutation tests reject arbitrary credential
or raw-content fields and require an observable official `final_output` for all
twenty cases.

## Capture diagnostics

The default port 8000 was not listening. Read-only listener and `/v1/models`
checks found an already-running contract-compatible llama.cpp service at
`127.0.0.1:8111` with the PaddleOCR-VL 1.6 GGUF and multimodal capability. No
service was launched or reconfigured.

Cases 1–3 qualified normally. The first attempt at case 4 completed inference
but hit `FileNotFoundError` while writing a long Unicode golden filename below
a 64-character case-ID directory. This was diagnosed as a Windows output-path
length failure. The evidence root was shortened and case 4 was rerun from the
start; it and cases 5–20 then qualified. No partially written case-4 evidence
was accepted.

## Evidence boundaries

- Lightweight: layout, crop, payload, raw VLM, and parser-final fingerprints
  are observable for 20/20 cases.
- Official: historical scorer output is observable for 20/20 cases.
- Official layout, crop, payload, and raw VLM: `unobservable`.
- Authentic same-boundary trace pairs: zero.
- First observable inference divergence: unproven.

The tracked machine-readable scalar summary is
`tests/fixtures/accuracy/v16-trace-capture-summary.json`. It contains exact
case IDs and full SHA-256 values, the fixed provider contract, strict boundary
observability, hashes for every raw lightweight trace, and hashes for the four
official scorer source artifacts. The Markdown table is navigation-only and
its 12-character prefixes are not authoritative.

A fresh checkout can validate the committed full-hash contract and its binding
to the committed manifest. It cannot recompute the raw capture or official-row
fingerprints without the private benchmark images and ignored JSONL, crops,
responses, predictions, and scorer artifacts.

## Review-fix verification

- Full pytest suite: 158 passed, 7 skipped.
- Focused Ruff check and format check: passed.
- `git diff --check`: passed.
- Machine summary SHA-256:
  `3094ddb24a95577466ccf1a79883b4e794308bfd0baab8fc475dc09ad620215a`.

## Scope protection

`eval/.omnidocbench` was read only for authentic official scorer observations;
it was not touched or staged. The pre-existing controller cleanup in
`.superpowers/sdd/evidence-task-2-report.md` was preserved.
