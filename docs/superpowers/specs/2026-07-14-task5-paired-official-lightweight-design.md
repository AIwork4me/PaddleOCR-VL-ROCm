# Task 5 Paired Official/Lightweight Evidence Design

- Date: 2026-07-14
- Status: Approved; DirectML majority amendment approved 2026-07-14
- Branch: `codex/top-tier-quality`
- Benchmark: OmniDocBench v1.6 at commit
  `147cd5ac9472002f5751221d390bf00abdbc0d2f`
- Evidence root:
  `C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-evidence\v16-2026-07-14-official-r7-score-recovery-py310`

## 1. Purpose

Task 5 must determine, with reproducible evidence, whether the Official and
Lightweight paths for the same PaddleOCR-VL model have equivalent inputs and
outputs on the same Windows AMD system. It must separately determine whether
the Lightweight path is successfully adapted to the AMD platform.

"100% output equivalence" means equality under the project's approved
normalized Markdown and canonical-trace contracts. It does not mean raw-file,
JSON-byte, whitespace-byte, or filesystem-metadata equality.

This task establishes a paired baseline and renders verdicts. If accuracy or
equivalence fails, causal diagnosis and any production inference correction
belong to Task 6. Task 5 must not introduce speculative output-changing fixes.

## 2. Constraints and Non-Goals

- Preserve the sealed r7 G0 evidence and its receipt byte-for-byte.
- Append Task 5 evidence under `r7/task5/`; do not rewrite r7's existing
  manifest, Official results, predictions, or G0 receipt.
- Run fresh paired Official and Lightweight inference on the same machine,
  model artifacts, dataset, inputs, and generation contract.
- Score both paths with the pinned OmniDocBench v1.6 implementation and the
  official notebook field-selection, page aggregation, and rounding rules.
- Keep the approved `peg-native` issue #18248 page visible in the 1,651-page
  corpus and scoring audit. Exclude it only from the 1,650 successful-pair
  equivalence denominator.
- Do not begin performance behavior changes. G4 remains blocked until a fresh
  configuration passes G3.
- Do not commit models, datasets, predictions, raw responses, images, crops,
  full traces, credentials, or other large/sensitive evidence.

## 3. Architecture

Task 5 uses an append-only namespace inside the existing r7 evidence root:

```text
r7/
|-- manifest.json                  # sealed G0, unchanged
|-- results/official/              # sealed G0, unchanged
`-- task5/
    |-- manifest.json
    |-- attempts/
    |   `-- <attempt-id>/
    |       |-- work/                 # raw predictions, traces, profiles
    |       |-- compact/
    |       |   |-- results/
    |       |   |   |-- official/
    |       |   |   `-- lightweight/
    |       |   `-- comparison/
    |       |       |-- input-contract.json
    |       |       |-- normalized-output.json
    |       |       |-- trace-diff.json
    |       |       |-- directml-attestation.json
    |       |       `-- decision.json
    |       |-- stage-state.json
    |       |-- snapshot-before.json
    |       |-- snapshot-after.json
    |       |-- selected-attempt.json # immutable selection candidate
    |       `-- receipt.sha256.json
    `-- selected-attempt.json         # atomic pointer copied from the candidate
```

The Task 5 manifest is an independent chain-of-custody record. It binds the r7
G0 receipt SHA-256, sealed Official artifact hashes, the Task 5 producing
commit, hardware and software environment, and every paired-run input. This
avoids falsely changing the producing commit of the already sealed G0 run.

Each attempt has a unique immutable directory. Retrying after an infrastructure
failure creates a new attempt; it never overwrites or silently repairs a failed
attempt. Only one complete, explicitly selected attempt may feed the final
comparison and receipt.

The attempt directory, not a shared root-level results directory, is the
authority for compact evidence. A failed attempt can therefore leave all of
its bytes in place without poisoning a later attempt. Root-level
`selected-attempt.json` is the only selection commit point; root-level
`results/`, `comparison/`, or `receipt.sha256.json` files are not authoritative
and are not produced by the evidence runner.

## 4. Component Boundaries

### 4.1 Task 5 manifest builder

Creates and validates the independent Task 5 manifest. It records:

- full r7 G0 receipt SHA-256 and the six sealed Official output hashes;
- repository commit and dirty-state status;
- model, mmproj, layout model, dataset, image-manifest, scorer, and config
  hashes;
- GPU, driver, OS, Python, ONNX Runtime, DirectML, Paddle/Official adapter,
  Lightweight adapter, server, and model runtime identities;
- endpoint/model name, prompt contract, image encoding, resize/crop rules,
  request ordering, seed, sampling, token limit, timeout, and retry settings;
- approved-known-failure identity and reason.

The builder rejects missing identities, hash drift, mixed commits, stale
artifacts, and Official/Lightweight contract differences before inference.

### 4.2 Paired inference orchestrator

Runs fresh Official and Lightweight inference from the same manifest. It owns
attempt selection, process lifecycle, exit-code capture, timeout handling,
orphan detection, page coverage, and failure manifests. Cross-engine fallback
is forbidden.

The two paths may use different adapters by definition, but adapter-specific
translation must be included in the trace so the canonical request contract can
be compared rather than assumed.

### 4.3 DirectML provider attestor

Proves that the Lightweight PP-DocLayoutV3 session executes on the local AMD
GPU. Accepted evidence requires all of the following:

- requested provider is `auto`;
- active providers are exactly ordered with `DmlExecutionProvider` first and
  `CPUExecutionProvider` second;
- session fallback is disabled;
- ORT profiling or equivalent per-node execution evidence attributes layout
  model node execution to `DmlExecutionProvider`;
- `DmlExecutionProvider` owns strictly more than 50% of all profiled layout
  `Node` execution events; CPU-assigned graph partitions are permitted but
  their event count and share must be reported;
- every profiled `Node` event names either `DmlExecutionProvider` or
  `CPUExecutionProvider`; missing and other-provider node counts are zero;
- provider initialization and the representative inference complete without a
  provider fallback or activation warning.

Provider-list presence alone is insufficient proof. Missing node-assignment
evidence or a DirectML event share at or below 50% is a failed AMD-adaptation
requirement, not an inferred GPU pass. Static CPU graph partitioning is not the
same as runtime fallback and does not fail AMD adaptation when DirectML owns the
majority of node events and every other requirement above passes.

### 4.4 Dual scorer

Scores each paired prediction directory independently using the pinned v1.6
checkout. Each path runs the normal evaluation plus Formula CDM and Table TEDS.
The report selects:

- `text_block.all.Edit_dist.ALL_page_avg`;
- `display_formula.page.CDM.ALL * 100`;
- `table.page.TEDS.ALL * 100`.

Each component is rounded to three decimals before computing:

```text
Overall = ((1 - TextEdit) * 100 + FormulaCDM + TableTEDS) / 3
```

Reading order is reported but excluded from Overall. Formula CDM and Table TEDS
must use equal GT-page aggregation and include zero for an unscored GT page as
specified by the approved v1.6 contract.

### 4.5 Trace normalizer and comparator

The comparator first pairs the 1,650 successful pages, then compares normalized
final Markdown for every pair. It subsequently compares every block in the
canonical boundary order:

```text
request_order -> label -> bbox -> crop_pixels -> prompt -> payload
              -> raw_result -> postprocess
```

Every boundary is classified as `equal`, `different`, or `unobservable`.
Official boundaries that cannot be authenticated remain `unobservable`; they
must never be inferred from later output or labeled equal. A proven difference
records the earliest differing boundary, page and block identity, bounded
scalar summaries, and both SHA-256 values. Full raw content remains external.

### 4.6 Decision engine and receipt generator

The decision engine emits two independent verdicts: `strict_equivalence` and
`amd_adaptation`. The receipt then hashes the shared Task 5 manifest, the
selected attempt's final state and candidate, score summaries, comparison
reports, decision, and small environment attestations.
All authoritative identities use complete lowercase SHA-256 values.

### 4.7 Attempt-local receipt and atomic selection

The receipt is generated and validated entirely within the selected attempt.
Its exact allowlist contains the shared Task 5 manifest plus that attempt's
final stage state, before/after G0 snapshots, immutable selection candidate,
score summaries, provenances, comparison reports, DirectML attestation, and
decision. Raw predictions, traces, profiles, models, datasets, and command logs
remain external and are bound only by their small manifest or stage hashes.

The final stage state and selection candidate are frozen before receipt
generation and are never modified afterward. A receipt failure is represented
by an absent or invalid attempt-local receipt and optional append-only failure
evidence; it must not rewrite a receipted candidate or any earlier stage.

After the attempt-local receipt validates, the runner creates root-level
`selected-attempt.json` atomically without overwrite. Its bytes must exactly
match `attempts/<id>/selected-attempt.json`. A root pointer is effective only
when all of the following hold:

1. the referenced attempt-local receipt exists and validates from current
   bytes;
2. that receipt hashes the attempt-local selection candidate;
3. the root pointer is byte-for-byte equal to that receipted candidate;
4. the candidate binds the Task 5 manifest, G0 closure, attempt id, and measured
   verdicts.

An interrupted pointer creation does not invalidate the already sealed
attempt. The same attempt may retry only the create-if-absent pointer operation;
it may not rerun inference, scoring, comparison, or decision. If no valid root
pointer exists, a new AttemptId may run independently even when earlier failed
or unselected attempts remain. Once a valid root pointer exists, later attempts
must fail closed rather than replace the selection.

## 5. Data Flow

1. Validate sealed r7 G0 hashes without modifying its files.
2. Create the Task 5 manifest and fail closed on any unmatched paired input.
3. Start a new immutable attempt and capture the pre-run environment.
4. Execute fresh Official inference and record predictions, failures, trace
   observability, process termination, and artifact hashes.
5. Execute Lightweight inference with DirectML-first layout and the identical
   canonical input/generation contract.
6. Attest DirectML-majority layout node execution, report CPU graph partitions,
   and prove that runtime fallback remains disabled.
7. Score Official and Lightweight independently with normal v1.6 evaluation,
   Formula CDM, and Table TEDS.
8. Pair all successful pages and compare normalized Markdown and block traces.
9. Produce independent equivalence, AMD-adaptation, and G3 decisions.
10. Freeze the attempt-local compact evidence and selection candidate, generate
    and validate its attempt-local receipt, then atomically create the root
    selection pointer.
11. Publish only small, redacted, hash-bound reports from the valid selected
    attempt in the repository.

## 6. Verdict Contracts

### 6.1 Strict equivalence

Verdict priority is `FAIL > UNKNOWN > PASS`:

- `PASS`: all 1,650 successful paired pages have equal normalized Markdown;
  every block has equal required canonical boundaries; no required boundary is
  unobservable.
- `FAIL`: any paired normalized output differs; any observable canonical
  boundary differs; or any non-approved page cannot be paired.
- `UNKNOWN`: no difference is proven, but at least one required Official or
  Lightweight canonical boundary is unobservable.

The approved issue #18248 page is excluded only from this 1,650-pair
denominator. Its prediction/failure state remains explicit in coverage and
scoring reports.

### 6.2 AMD adaptation

`amd_adaptation` is `PASS` only when:

- the public input, parameter, output, filename, and serialization contracts
  pass;
- the DirectML provider attestation in Section 4.3 passes;
- no unapproved page failure, orphan, cross-engine fallback, stale artifact,
  or manifest drift occurs;
- Formula CDM has zero timeouts and exceptions;
- Table TEDS has zero timeouts, parse errors, and exceptions;
- the fresh Lightweight notebook Overall is at least 96.13;
- Lightweight Overall and each acceptance component are no worse than the
  paired same-machine Official result.

This verdict remains independent of strict equivalence. For example, a missing
Official internal trace can make strict equivalence `UNKNOWN` while AMD
adaptation still passes. A proven output difference makes strict equivalence
fail even if Lightweight accuracy is higher.

### 6.3 G3 outcome

G3 consumes the fresh Lightweight score and all existing accuracy contracts.
It passes only when the AMD-adaptation accuracy requirements above pass. A
failed G3 admits evidence-led Task 6 diagnosis and keeps G4 blocked. It does not
authorize a fixture-specific workaround, scorer hack, ground-truth lookup, or
speculative prompt/post-processing change.

## 7. Error Handling and Attempt Integrity

- Nonzero exit, incomplete coverage, missing evidence, hash drift, mismatched
  config, scorer-version mismatch, or orphan process invalidates the attempt.
- Infrastructure retries preserve the failed attempt and start a new attempt.
- An old file may not be copied into a fresh attempt to satisfy completeness.
- A receipt or selection failure never requires deleting, moving, quarantining,
  or overwriting bytes from an earlier attempt.
- Shared root-level compact results are forbidden as evidence authority because
  multi-directory publication cannot be committed atomically on Windows.
- A scorer timeout, exception, or parse error remains visible and contributes
  according to official rules; it cannot be silently dropped from a mean.
- Logs and tracked reports redact API keys, authorization headers, signed URLs,
  absolute model paths, prompts, payload text, and raw model responses.
- Comparison reports bound diagnostic samples while retaining aggregate counts
  and complete hashes for every result category.
- `eval/.omnidocbench/` remains untracked and must not be staged or modified.

## 8. Test and Verification Strategy

### 8.1 Unit tests

Cover Markdown normalization, request/block ordering, label and bbox
canonicalization, crop/payload/raw-result hashing, boundary observability,
`FAIL > UNKNOWN > PASS`, the approved-page denominator, page-level CDM/TEDS
aggregation, three-decimal component rounding, and Overall computation.

### 8.2 Contract tests

Deliberately vary model/mmproj/layout hashes, prompt, image transforms,
generation parameters, dataset, scorer, commit, provider order, or fallback
setting. The manifest validator must reject the run before inference.

### 8.3 Fault-injection tests

Simulate zero or minority DML node execution, missing/other provider events,
DirectML initialization failure, silent provider fallback, process
interruption, orphan processes, stale files, partial page
coverage, missing trace boundaries, CDM timeout, TEDS error, and corrupted
receipt inputs. None may produce a false PASS.

### 8.4 Real acceptance run

The selected attempt must demonstrate:

- full 1,651-page corpus accounting and exactly 1,650 successful pairs under
  the approved known-failure contract;
- exhaustive normalized-output comparison for all 1,650 pairs;
- exhaustive block comparison and explicit observability status at every
  canonical boundary;
- fresh v1.6 normal, Formula CDM, and Table TEDS results for both paths;
- DirectML-majority GPU node-execution evidence for PP-DocLayoutV3, with DML
  and CPU event counts and shares;
- zero unapproved failures and zero CDM/TEDS quality errors;
- hash-bound `decision.json` and `receipt.sha256.json`.

Repository verification includes focused tests, the full test suite, Ruff,
format checks, type checks where configured, `git diff --check`, evidence
schema validation, receipt recomputation, documentation-claim validation, and
an independent read-only review before any acceptance claim.

## 9. Repository and User-Facing Deliverables

Only small, reviewable evidence summaries and hashes are tracked. The user
presentation follows progressive disclosure:

1. README comparison summary: same model, machine, inputs, and parameters.
2. Official-versus-Lightweight v1.6 metric table with Text Edit, Formula CDM,
   Table TEDS, Overall, denominators, and error counts.
3. Equivalence table with normalized page parity, canonical-trace verdict, and
   every unobservable boundary.
4. AMD proof with hardware/runtime identity, DirectML-first provider order,
   disabled fallback, and node-execution attestation.
5. Reproduction commands for Official inference, Lightweight inference,
   scoring, comparison, and receipt validation.

Public wording is generated from or checked against `decision.json`. The phrase
"100% output equivalent" is allowed only when `strict_equivalence` is `PASS`.
An `UNKNOWN` or `FAIL` result must be displayed as such; close scores cannot be
substituted for an equivalence proof. Linux CUDA paper results remain labeled
as external reference data rather than same-machine Windows AMD results.

## 10. Acceptance Summary

Task 5 is complete when an immutable selected attempt and independently
reviewed receipt support all reported verdicts. Success is not predetermined:
the task succeeds as an evidence exercise even if strict equivalence is
`UNKNOWN`/`FAIL` or G3 is blocked, provided the outcome is complete, honest,
reproducible, and routes the next work correctly.

No production accuracy correction or performance optimization is part of this
design. Those actions require the resulting evidence and their own approved
plan.
