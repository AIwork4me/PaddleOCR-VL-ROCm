# Accuracy Inference Fixes Admission Plan

**Goal:** Admit production inference fixes only after authenticated oracle attribution proves a causal boundary and measurable official v1.6 improvement.

**Architecture:** Treat authenticated attribution as a fail-closed admission gate between diagnosis and production work. The current evidence has no same-boundary official oracle inputs, so this document records a closed gate and defines the evidence required to reopen it; it contains no production task.

**Tech Stack:** Python 3.10+, JSON, SHA-256, pytest, OmniDocBench v1.6 at `147cd5ac9472002f5751221d390bf00abdbc0d2f`, ONNX Runtime DirectML.

## Global Constraints

- Fresh official-local release evidence requires `count=1651`, `ok=1650`,
  `fail=1`, `fallback=0`, and `limit_pages=null`, with only the approved issue
  #18248 `peg-native` failure and no failed-page prediction; scoring retains all
  1,651 GT pages.
- This contract admits 1,650 successful predictions and the sole approved failure.
- Formula uses `display_formula.page.CDM.ALL * 100`; Table uses `table.page.TEDS.ALL * 100`.
- Text page score is `sum(Edit_num) / sum(upper_len)`; Formula CDM and Table TEDS use equal sample means within each GT page and equal page means.
- Round Text, Formula, and Table to three decimals before computing Overall.
- Every Windows lightweight trace must record `DmlExecutionProvider` first in `layout_providers_active`; reject absent provider metadata and CPU fallback.
- Keep raw predictions, images, crops, responses, and traces untracked.
- Do not implement a prompt, crop, model, normalization, or serialization change without passing the admission gate below.

---

## File Structure

- Modify `docs/accuracy-root-cause-v16.md` only when authenticated attribution changes a cause from `unproven` to proven.
- Modify this document only after at least one cause passes every admission criterion; only then may it gain a complete TDD production task.

## Current authenticated result

| Manifest cases | Proven causes | Unproven cases | Replay calls | Synthetic contributions | Positive oracle effects | Established causal boundaries | Admitted tasks |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0 | 20 | 0 | 0 | 0 observed | 0 | 0 |

All 20 cases are `unproven`. The committed summary authenticates lightweight
boundary hashes and historical official scorer fingerprints, but contains no
official value at the same boundary needed for `crop`, `payload`, `raw_vlm`, or
`final_output` replay. Consequently no oracle score, affected-case causal count,
estimated causal Overall contribution, or earliest causal boundary exists.
Historical Formula, Table, Text, and reading-order recovery pools are not causes
and cannot satisfy this gate.

## Production-task admission gate

A cause must satisfy every criterion before a production task is written:

1. **Exact reproduction:** at least one exact manifest fixture, identified by
   its full case ID and immutable page/GT identity, reproduces the divergence
   under the authenticated DirectML-first trace contract.
2. **Official metric improvement:** a same-boundary authenticated official
   oracle value is replayed through the applicable official v1.6 scorer and
   improves a recorded before score to a recorded oracle score. A missing,
   negative, or authenticated zero effect does not pass.
3. **Named non-regressions:** neighboring gain fixtures are identified by full
   case ID and exercise the same generic behavior, with their current scores or
   boundary contracts recorded as non-regression expectations.
4. **Generic causal boundary:** ordered one-variable-at-a-time swaps establish
   the earliest positive boundary among `crop`, `payload`, `raw_vlm`, and
   `final_output`; restoration between swaps is verified, and the proposed
   correction applies generically rather than substituting a fixture-specific
   answer.

For a passing cause, the evidence record must include the exact page/GT index
and full trace hashes, before and oracle scores, affected-case count, estimated
v1.6 Overall contribution computed only after notebook component rounding, the
first causal boundary, and every named neighboring gain fixture.

## Evidence required to reopen

Admission can be reevaluated only after all of the following are available:

- authenticated official values at the same candidate boundaries for one or
  more exact manifest cases;
- an ordered attribution result that actually invokes the official-metric
  replay and records a positive effect plus the earliest positive boundary;
- retained full manifest, capture-set, trace, boundary, and scorer-source
  hashes tying the result to the existing evidence contract;
- named gain fixtures and explicit non-regression score or boundary contracts;
- an Overall estimate derived from the official notebook fields after rounding
  Text, Formula, and Table components to three decimals.

## Gate decision

No cause passes the gate. No production inference fix is authorized, and this
plan intentionally stops without production implementation or test tasks.
