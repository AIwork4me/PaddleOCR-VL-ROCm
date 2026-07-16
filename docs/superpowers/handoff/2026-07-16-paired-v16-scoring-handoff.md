# OmniDocBench v1.6 paired scoring handoff (2026-07-16)

## Purpose and approved evaluation rule

This handoff continues the real Windows AMD paired evaluation of
`PaddleOCR-VL-ROCm`. The benchmark is **OmniDocBench v1.6**, not v1.7.

The user explicitly approved a symmetric exclusion for:

`newspaper_The Times UK_0801@magazinesclubnew_page_031.png`

PaddleOCR tracks the page as a known PEG-native problem at
https://github.com/PaddlePaddle/PaddleOCR/issues/18248. Both the Official and
Lightweight paths failed this exact page with no prediction Markdown written.
Score the shared 1,650 successful pages; do not count the page as a
Lightweight-only regression.

The exclusion is intentionally narrow:

- Official evidence must contain `peg-native` for the exact page.
- Lightweight evidence must contain `500 Server Error` for the exact page.
- Any other failed page, fallback, wrong page, or wrong signature remains a
  hard failure.

## Repository and uncommitted implementation state

Worktree:

`C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm\.worktrees\top-tier-quality`

Base commit:

`e418dc7d1e64746892550fc4174bc20ded29c82d`
(`fix(eval): handle empty Task 5 command streams`).

Uncommitted changes are deliberate and must be preserved:

- `eval/release_contract.py`
  - permits the exact v1.6 symmetric exception for engines `official` and
    `lightweight`, with engine-specific signatures described above.
- `tests/test_release_contract.py`
  - regression coverage for the accepted Lightweight 500 signature and the
    rejected timeout signature.
- `eval/symmetric_score_input.py`
  - creates immutable scorer-input mirrors by hard-linking Markdown and
    emitting an effective `_run_stats.json` plus receipt.
- `tests/test_symmetric_score_input.py`
  - verifies the Official 8-page path-repair merge does not modify the raw
    source stats.
- `eval/.omnidocbench/`
  - local, gitignored v1.6 scorer checkout; do not add it to Git.

Last verification before this handoff:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_symmetric_score_input.py tests\test_release_contract.py -q
```

Result: `18 passed`.

## Identity and raw evidence roots

Primary evidence root:

`C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-evidence\v16-2026-07-15-paired-raw-e418dc7`

Important immutable inputs and hashes:

| Item | SHA-256 |
|---|---|
| `run-config.json` | `53b28e1d34a3d1b8ec121c15193b51547d8fa49f4d775053c67b2263ff6c37c1` |
| Dataset manifest | `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496` |
| PP-DocLayoutV3 ONNX | `45bf71750b00739a41fc209f132eb104a4d6b5bb29483c9078164d8b87cf28ba` |
| PP-DocLayoutV3 YAML | `506fcfac13b3b546ae40d7886b44126420f392adb694e3f8bb6a6286a1f90fdc` |
| Main GGUF | `f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8` |
| MMProj | `204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a` |
| Worktree inference Python | `f41eab92fc06cb8c57e0a4563f747daeacf10f2393e53c2a8da1fb4552a7eb05` |
| Locked scorer Python | `18a645dc50677a66cb1fd56cec3ffe938d50dd3b133f0b9cd2db2962344e901f` |
| `eval/run_eval.py` at launch | `de1dffe73217b74b69927b06159c412159bf678c122e73630fbf9ebf1cb10ae3` |
| v1.6 scoring config | `9ffaf6d46952ed28ada22e4b675198e6bd48138e075083ee51eed781a949e49e` |

The OpenAI-compatible inference service was `http://127.0.0.1:8111/v1` with
model `PaddleOCR-VL-1.6-GGUF.gguf`.

## Inference result and repair lineage

### Official

Raw full run stats are immutable at:

`full\official\predictions\_run_stats.json`

Raw aggregate: `count=1651`, `ok=1642`, `fail=9`, `fallback=0`.
Its SHA-256 is
`ee5da8d9575daf4bf29def3fc0d33c18ac62025221a35573b0b34a332e0053f1`.

Eight failures were Windows path-length failures in
`tempfile.mkstemp()` while writing trace files for one extremely long Chinese
filename. The model result had been computed but the code writes trace before
Markdown, so neither output was retained. The ninth failure was the approved
PEG-native page.

The eight path-only pages were rerun with the same commit, model, server and
arguments through short path staging `C:\t5e418-r1`. Repair evidence:

`repairs\official-path-r1\repair-run-stats.json`

Repair aggregate: `count=8`, `ok=8`, `fail=0`, `fallback=0`.
Repair SHA-256:
`b3f4f2d323a9344dd3c212fa9dd08c3bff562bf3bc0816edecdb16b662e3a6ee`.

The 8 Markdown and 8 trace files were hash-checked before merge. The raw
full-run stats file was hash-checked before and after merge and remained
unchanged. The physical Official prediction/trace directories now each contain
1,650 page artifacts.

### Lightweight

Raw full stats:

`full\lightweight\predictions\_run_stats.json`

Aggregate: `count=1651`, `ok=1650`, `fail=1`, `fallback=0`.
SHA-256:
`a89d1cd2e293b6838ed05e3ae0b41ab899f11d733fcb4120f49e94a2414ca84d`.

The sole failure is the same approved page. Its Lightweight status is
`500 Server Error: Internal Server Error`; the client retried four times.

Long-path protection was applied from the start using the short junction root
`C:\t5e418-lw`; all eight previously problematic long-name pages have both
Markdown and trace artifacts.

## DirectML evidence

Full-run stats record:

```json
{
  "layout_provider_requested": "auto",
  "layout_providers_active": ["DmlExecutionProvider", "CPUExecutionProvider"],
  "layout_fallback_disabled": true
}
```

The full ORT profile is:

`C:\t5e418-lw\layout-profile_2026-07-15_16-23-33.json`

It is 1,174,895,601 bytes. ORT logged that its maximum number of profiling
events was reached, so do **not** treat it as a complete full-run node-share
sample or invoke the current in-memory strict parser casually.

The successful real one-page smoke run is the valid DirectML attestation for
the current model/configuration: DML nodes `1101`, CPU nodes `150`, DML share
`0.8800959232613909`, missing nodes `0`, other providers `0`; profile SHA-256
`ab3ad79364fb5ae66a52db56c9e99211716eceb9c88278e4c512245020c313f5`.
This satisfies the approved standard: DML owns a strict majority; CPU graph
partitions are allowed.

## Scorer identity and scorer input mirrors

Scorer checkout:

`eval\.omnidocbench`

Validated OmniDocBench v1.6 commit:

`147cd5ac9472002f5751221d390bf00abdbc0d2f`

Use only this locked scorer interpreter:

`C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-scorer-v16-py310\Scripts\python.exe`

It matches the locked scorer-Python SHA above and has Python `3.10.20`.

Short-path scorer input mirrors:

- `C:\t5e418-score\official`
- `C:\t5e418-score\lightweight`

They contain hard links to the 1,650 scorer-facing Markdown outputs, an
effective `_run_stats.json`, and `score-input-receipt.json`. Do not delete or
edit them before all four scores are finished.

Receipt highlights:

| Engine | Effective stats SHA-256 | Markdown manifest SHA-256 | Repair pages |
|---|---|---|---:|
| Official | `7c939ca7ed5b24c8a23d60e1405dfefad8344ff400fb02a733ef12dcd19e8e32` | `a68ef0caca8eabeef80b0a075e571c57e534a756e5791b475b463f9472981fd1` | 8 |
| Lightweight | `a89d1cd2e293b6838ed05e3ae0b41ab899f11d733fcb4120f49e94a2414ca84d` | `55df5fb47ef2d67f6c639b9e7661bb465c2c0590e98f35559c426d4c2f715df8` | 0 |

## Scoring status at handoff

All score artifacts belong under:

`results\official` and `results\lightweight` inside the primary evidence root.

Completed successfully:

`Official, non-CDM`

Artifacts:

- `results\official\metric.json`
- `results\official\run-summary.json`
- `results\official\provenance.json`

Key metrics from the official report:

| Metric | Value |
|---|---:|
| Text Edit_dist | `0.035112263905688564` |
| Table page TEDS | `0.9429171434897962` (`94.292%`) |
| Table TEDS structure-only | `0.9661806773687927` (`96.618%`) |
| Non-CDM display-formula Edit_dist | `0.09492607780185801` |

Table TEDS debug: `sample_count=665`, timeout count `0`, error count `0`.
The scorer used fallback text matching for two pages (`quick_match_timeout`),
not page timeout; this is recorded in `metric.json` and must be disclosed but
does not invalidate TEDS.

Not started yet, and must run **sequentially** because OmniDocBench shares its
`result/` directory:

1. Official CDM.
2. Lightweight non-CDM.
3. Lightweight CDM.

## Exact continuation commands

Run from the worktree. Set variables once:

```powershell
$WT = 'C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm\.worktrees\top-tier-quality'
$PY = "$WT\.venv\Scripts\python.exe"
$SCORER = 'C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-scorer-v16-py310\Scripts\python.exe'
$RAW = 'C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-evidence\v16-2026-07-15-paired-raw-e418dc7'
$DATASET = 'C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm\data\omnidocbench\v16'
Set-Location $WT
```

Official CDM:

```powershell
& $PY -m eval.run_eval --stage eval --version v16 --engine official `
  --dataset-dir $DATASET --predictions-dir 'C:\t5e418-score\official' `
  --scorer-python $SCORER --cdm `
  --copy-report "$RAW\results\official\metric-cdm.json" `
  --run-summary "$RAW\results\official\run-summary-cdm.json" `
  --provenance "$RAW\results\official\provenance-cdm.json"
```

Lightweight non-CDM:

```powershell
& $PY -m eval.run_eval --stage eval --version v16 --engine lightweight `
  --server-url 'http://127.0.0.1:8111/v1' --api-model-name 'PaddleOCR-VL-1.6-GGUF.gguf' `
  --dataset-dir $DATASET --predictions-dir 'C:\t5e418-score\lightweight' `
  --scorer-python $SCORER `
  --copy-report "$RAW\results\lightweight\metric.json" `
  --run-summary "$RAW\results\lightweight\run-summary.json" `
  --provenance "$RAW\results\lightweight\provenance.json"
```

Lightweight CDM:

```powershell
& $PY -m eval.run_eval --stage eval --version v16 --engine lightweight `
  --server-url 'http://127.0.0.1:8111/v1' --api-model-name 'PaddleOCR-VL-1.6-GGUF.gguf' `
  --dataset-dir $DATASET --predictions-dir 'C:\t5e418-score\lightweight' `
  --scorer-python $SCORER --cdm `
  --copy-report "$RAW\results\lightweight\metric-cdm.json" `
  --run-summary "$RAW\results\lightweight\run-summary-cdm.json" `
  --provenance "$RAW\results\lightweight\provenance-cdm.json"
```

After each command, require exit code 0 and verify all three requested
artifacts are non-empty. Do not run commands concurrently.

## Required post-scoring work

1. Run `eval.task5_comparison.compare_prediction_dirs` on
   `C:\t5e418-score\official` and `C:\t5e418-score\lightweight`; record the
   1,650-page normalized-output result. Do not assume it passes.
2. Run canonical trace comparison on the full trace directories. If the
   comparison reports unobservable official boundaries, preserve that result;
   do not fabricate raw Official observations.
3. Extract final Formula **CDM** and Table **TEDS** from both metric reports.
   Only CDM report values are valid for the final formula conclusion.
4. Write a final symmetric-exclusion receipt in the evidence root that cites:
   this document, both score-input receipts, raw stats SHA values, Official
   repair stats SHA, and all four metric/provenance SHA values.
5. Before claiming AMD adaptation complete, distinguish the valid one-page DML
   attestation from the capped full-run profile. Do not claim a complete
   full-run node-share ratio without a streaming/profile-capped solution.
6. Run the relevant tests and inspect `git diff`; commit only after all score
   and receipt evidence is complete and verified.

## Safety notes

- Do not overwrite the raw Official `_run_stats.json`; its 8 path failures are
  intentionally preserved as raw evidence.
- Do not remove `C:\t5e418-r1`, `C:\t5e418-lw`, or `C:\t5e418-score` until
  scoring, comparison and final receipt are complete.
- Do not start new inference or change the model/server for this paired run.
- The user’s future backend sequence is Transformers API, then vLLM, then
  llama.cpp. That is separate follow-on work; this evidence run is already
  bound to the llama.cpp-compatible service and must remain so.
