# Release Gate Recovery Design

Date: 2026-07-13  
Branch: `codex/top-tier-quality`  
Decision owner: project owner

## Goal

Turn the current engineering-complete but release-blocked branch into an
evidence-backed release candidate. The work must first make the approved
OmniDocBench v1.6 upstream-page exception the single active G0 contract, then
regenerate fresh accuracy evidence, diagnose any remaining accuracy gap before
changing inference behavior, and admit performance and publication work only
after their prerequisite gates pass.

## Approved G0 contract

The project owner approved the following release contract, replacing the
earlier requirement that all 1,651 official-local inference requests succeed:

- OmniDocBench remains pinned to v1.6 commit
  `147cd5ac9472002f5751221d390bf00abdbc0d2f`.
- Scoring always retains the full 1,651-page ground-truth denominator.
- Accepted official-local inference statistics are exactly
  `count=1651`, `ok=1650`, `fail=1`, `fallback=0`, and
  `limit_pages=null`.
- The sole failure must be
  `newspaper_The Times UK_0801@magazinesclubnew_page_031.png`.
- Its status or error must contain the stable `peg-native` signature and link
  to <https://github.com/PaddlePaddle/PaddleOCR/issues/18248>.
- The failed page must have no prediction file. No fallback, synthetic
  Markdown, copied prediction, deleted GT page, or reduced denominator is
  allowed.
- Formula uses official v1.6 CDM and Table uses content-aware TEDS. Missing
  Formula/Table values, timeouts, exceptions, and denominator mismatches remain
  release failures.

This contract changes success coverage, not scoring coverage. Public wording
must say "1,650 successful predictions plus one approved known upstream
failure, scored over all 1,651 GT pages" and must not claim 1,651 successful
predictions or a PaddlePaddle maintainer resolution.

Historical plans and specifications remain useful records. They may retain
their original requirements only when clearly marked as superseded by this
approved contract. Active release, accuracy, and execution documents must use
the approved contract directly.

## Execution architecture

Work proceeds through six serial stages. Each stage produces authenticated
artifacts and has a fail-closed exit. Later stages cannot use provisional or
historical evidence to bypass a failed prerequisite.

### Stage 1: Contract convergence

Audit active specifications, plans, documentation, validators, scripts, and
tests for the obsolete `ok=1651, fail=0` requirement. Update active material to
the approved contract and annotate historical material rather than silently
rewriting history. Extend documentation-contract tests so active conflicting
wording fails CI.

Exit criteria:

- the Python release validator, PowerShell evaluation entry point, active
  plans, readiness record, and public documentation express one contract;
- exact acceptance and every rejection dimension are covered by tests;
- the failed page cannot leave a residual prediction;
- full tests, formatting, type checks, and an independent review pass.

### Stage 2: Fresh G0 and G3 evidence

Freeze the evaluated code commit and record SHA-256 values for the main GGUF,
mmproj, PP-DocLayoutV3 ONNX model/config, OmniDocBench dataset manifest,
scoring configuration, and pinned scorer source. Use new prediction and result
directories so historical artifacts remain immutable.

Run the complete official-local 1,651-page inference and validate the approved
exception before scoring. Run official v1.6 scoring, including Formula CDM and
Table TEDS quality gates. Run the lightweight path against the same dataset and
scorer contract. Generate metric JSON, run summary, provenance, alignment
diagnostics, and artifact hashes for both paths.

Notebook statistics are reconstructed exactly:

- Text page score is `sum(Edit_num) / sum(upper_len)`;
- Formula is the equal mean of sample CDM values within each GT page, then the
  equal mean across all Formula GT pages;
- Table is the equal mean of sample TEDS values within each GT page, then the
  equal mean across all Table GT pages;
- Text Edit distance, Formula percent, and Table percent are rounded to three
  decimals before Overall, which is
  `((1 - TextEdit) * 100 + FormulaPercent + TablePercent) / 3`;
- reading order is reported separately and does not enter Overall.

Exit decisions:

- if G0 fails, stop and repair evidence generation;
- if G0 passes and lightweight Overall is at least 96.13 with no component or
  contract regression, admit Stage 5;
- if G0 passes but G3 fails, admit Stage 3 and keep performance mutations
  blocked.

### Stage 3: Official same-boundary diagnosis

Add diagnostic-only observation around the official adapter for the fixed
20-case manifest. Capture authenticated official values at the same candidate
boundaries already observed on the DirectML lightweight path: crop, payload,
raw VLM output, and final output. Diagnostic capture must not alter request
payloads, decoding, parsing, outputs, or production defaults.

Keep raw images, crops, payloads, responses, predictions, and traces untracked.
Commit only strict scalar summaries, SHA-256 values, observability states, and
provider/provenance contracts. Credentials and arbitrary raw-content fields
are forbidden by schema.

Run ordered single-variable oracle replay from a restored lightweight
baseline. Formula replay must use CDM and Table replay must use content-aware
TEDS; Edit distance is not an allowed proxy. Missing or incomparable official
values remain `unproven` and never become inferred zero contributions.

Exit criteria:

- if no exact fixture establishes a positive earliest causal boundary, no
  production inference task is authorized;
- a cause passes only when an exact fixture reproduces the divergence, an
  authenticated same-boundary oracle improves the official metric, neighboring
  gain fixtures are named, and the boundary is generic rather than
  fixture-specific.

### Stage 4: Evidence-admitted accuracy fixes

Create a separate TDD plan only for causes that pass Stage 3. Every production
task must name the exact failing fixture, before score, oracle score, first
causal boundary, estimated v1.6 Overall contribution, full trace hashes, and
neighboring non-regression fixtures.

Prompt, crop, normalization, model-output, or serialization changes are
forbidden without this admission record. DirectML-first layout execution,
disabled CPU fallback, public input/output contracts, and gain fixtures are
mandatory regression gates.

After each admitted correction, rerun focused tests and the complete G0/G3
workflow. A subset or historical score cannot unlock performance.

### Stage 5: Performance optimization and acceptance

Performance behavior changes begin only after an exact configuration passes
fresh G3. Apply and independently review connection reuse, disk response cache,
slot-aware concurrency, and the paired benchmark harness as separate tasks.

The paired benchmark compares the exact accepted configuration and requires
identical output hashes and preserved v1.6 accuracy. Acceptance thresholds are:

- mean latency no greater than 13.00 seconds per page;
- P95 latency no greater than 34.82 seconds per page.

Pre-G3 stage timing remains diagnostic and cannot be presented as G4 evidence.

### Stage 6: G5 and publication

Repeat empty-cache setup on a stable public network and distinguish network
transport failures from checksum, extraction, activation, or installer
failures. Revalidate verified-cache installation, full doctor, managed-server
inference, existing-server inference, clean wheel/sdist installation, and
secret/document/repository-content audits.

Version changes, push, pull request, tag, and GitHub release remain forbidden
until G0 through G5 all pass. Publication-time GitHub authentication must be
rechecked and does not override evidence gates.

## DirectML invariant

Every accepted Windows lightweight trace and benchmark record must contain:

```json
{
  "layout_provider_requested": "auto",
  "layout_providers_active": ["DmlExecutionProvider", "CPUExecutionProvider"],
  "layout_fallback_disabled": true
}
```

The second provider is normal ORT availability, not accepted CPU execution.
DirectML must be first, session fallback must be disabled, and DirectML
activation failure or CPU-first ordering must fail closed.

## Artifact and security rules

- New runs use isolated paths and never overwrite historical evidence.
- Every release artifact records its producing commit, command, model/runtime
  hashes, dataset/scorer hashes, page counts, failures, and metric paths.
- Raw benchmark inputs and model responses remain ignored and untracked.
- Tracked scalar summaries use strict allowlists and complete lowercase
  SHA-256 values, not prefixes as the authoritative identity.
- Server diagnostics, exceptions, reports, and documentation must redact API
  keys, authorization headers, absolute model paths, payload text, and raw
  responses.
- `eval/.omnidocbench/` remains untracked and must not be staged or modified.

## Testing and review strategy

Use TDD for every code or contract change. Each independently testable task
receives a separate commit and independent review before the next task starts.
Verification is proportional to the stage:

- contract tasks: focused validators, documentation contracts, PowerShell
  entry-point tests, full pytest, Ruff, format, mypy, and `git diff --check`;
- evidence tasks: contract checks plus artifact hashes, fixed denominators,
  CDM/TEDS quality, notebook reconstruction, and provenance validation;
- inference fixes: exact failing fixtures, neighboring gain fixtures,
  DirectML/public-contract regression, then full G0/G3;
- performance: paired hashes, accepted accuracy, mean/P95 thresholds, and
  concurrency/cache correctness;
- publication: clean package installation, both server journeys, secret audit,
  repository-content audit, final branch review, and release-gate audit.

## Completion definition

The project is release-ready only when fresh artifacts demonstrate:

- G0: the approved 1,650-success/one-known-failure contract over all 1,651 GT
  pages;
- G1: compatibility and DirectML contracts pass;
- G2: any production accuracy change has authenticated causal attribution;
- G3: lightweight OmniDocBench v1.6 Overall is at least 96.13;
- G4: mean is at most 13.00 seconds/page and P95 at most 34.82 seconds/page on
  the exact G3 configuration, with unchanged outputs;
- G5: clean Windows AMD onboarding, package, documentation, and security audits
  pass, including empty-cache public-network setup.

Until then, readiness remains `BLOCKED` and no score/speed badge or release
claim is permitted.
