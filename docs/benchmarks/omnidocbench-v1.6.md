# OmniDocBench v1.6 Benchmark Facts

This is the single source of truth for public OmniDocBench v1.6 accuracy,
coverage, performance, and release-gate claims in this repository. README files,
the release-readiness record, and the results index link here instead of
maintaining independent score tables.

Last reconciled: 2026-07-17.

## Active scoring contract

The formal scoring denominator is **all 1,651 ground-truth pages**.

The approved official-local evidence contract is:

```text
count=1651
ok=1650
fail=1
fallback=0
limit_pages=null
```

The sole approved failure is
`newspaper_The Times UK_0801@magazinesclubnew_page_031.png`, with the
`peg-native` signature tracked in
[PaddleOCR issue #18248](https://github.com/PaddlePaddle/PaddleOCR/issues/18248).
There is no prediction file for the failed page. The scorer keeps that GT page
and treats the missing output as an empty prediction.

The 1,650 successful pages may be used for a paired equivalence diagnosis, but
that pairing is **not** the formal accuracy denominator and must not be
described as a “1,650-page score” or “symmetric exclusion.”

## Public accuracy records

| Record | Overall | Text Edit distance | Formula CDM | Table TEDS | Release meaning |
|---|---:|---:|---:|---:|---|
| Maintainer-accepted Windows AMD result, confirmed out of band by PaddleOCR, 2026-07-17 | **95.99** | 0.03488 | 97.36 | 94.09 | **G3 PASS**; repeat full run waived |
| Historical lightweight, Windows native, 2026-07-11 | 95.9480 | 0.034 | 96.922 | 94.322 | Reconstructed historical evidence only; not fresh G3 evidence |
| Official-local r7 score recovery, 2026-07-14 | 95.743 | 0.035 | 96.485 | 94.244 | G0 integrity evidence only; not a lightweight/G3 result |

Overall follows the OmniDocBench convention: convert text Edit distance to
accuracy, then average text accuracy, Formula CDM, and Table TEDS. The accepted
95.99 is the confirmed rounded Overall; its displayed component values have
their own reporting precision and are not inputs for reverse-engineering extra
digits. Reading-order Edit distance is 0.12882 and is excluded from Overall.

The historical lightweight row comes from:

- [raw metric artifact](../../results/omnidocbench/v16/paddleocrvl_rocm_cdm_quick_match_metric_result_windows_native_2026-07-11.json)
- [run summary and environment artifact](../../results/omnidocbench/v16/paddleocrvl_rocm_cdm_quick_match_run_summary_windows_native_2026-07-11.json)

The official-local r7 row is authenticated by the
[G0 evidence receipt](../releases/0.1.0-g0-evidence.md). Its six external r7
outputs are retained by SHA-256 in that receipt; the raw r7 files are not copied
into this repository.

The [G3 maintainer attestation](../releases/0.1.0-g3-attestation.md) records
that PaddleOCR confirmed Overall 95.99 out of band. On 2026-07-17, the project
maintainer accepted that confirmation for G3 and waived a public confirmation
artifact and another full run. This does not convert the historical files below
into a single complete raw-score artifact, and they must not be mixed to
reconstruct the accepted record. Issue #18248 is cited for the deterministic
page defect only; this repository does not claim that its public thread
confirms the score.

## Provenance and reproducibility

### Maintainer-accepted G3 record

| Required field | Bound value |
|---|---|
| Decision date | 2026-07-17 |
| Accepted Overall | 95.99 |
| Confirmation | PaddleOCR confirmation received out of band, as attested by the project maintainer |
| Public confirmation artifact | Waived by the project maintainer |
| Repeat full run | Waived by the project maintainer |
| Failed page | The single `peg-native` failure linked above |
| Evidence class | Maintainer attestation; not an independently reproducible raw-score artifact |
| Scope | G3 only; no effect on G4 or G5 |

### Historical lightweight record

| Required field | Bound value |
|---|---|
| Evaluation date | 2026-07-11 |
| Project commit | Not recorded in the tracked metric/run-summary pair |
| Model SHA-256 | Not recorded |
| mmproj SHA-256 | Not recorded |
| llama.cpp build/commit | Not recorded |
| OmniDocBench commit | `147cd5ac9472002f5751221d390bf00abdbc0d2f` in the associated evaluation record |
| Backend / quantization | Windows local llama.cpp/HIP and GGUF; quantization type not recorded |
| Hardware | Windows AMD machine; exact GPU is not bound to this score artifact |
| OS / driver / HIP | Windows build 10.0.26200 is recorded; driver and HIP versions are not recorded |
| Dataset pages | Scorer page count 1,651 |
| Success / failure / exclusion | Inference success/failure counts are not bound by this artifact pair; no scoring exclusion is allowed |
| Failure handling | All GT pages remain in scoring |
| Aggregation | Official notebook convention described above |
| Raw metric artifact | Linked above |

Because several required provenance fields are absent, this row cannot satisfy
the v0.1.0 G3 release gate.

### Official-local r7 G0 record

| Required field | Bound value |
|---|---|
| Evaluation date | 2026-07-14 |
| Project commits | r5 inference `d7fd1809568eb80818e88f674b56844d03c2de81`; reviewed r7 producing commit `fd91cb0a2d75b0a18d16b1bb34652a148cb59b9e` |
| Model SHA-256 | Not present in the tracked receipt |
| mmproj SHA-256 | Not present in the tracked receipt |
| llama.cpp build/commit | Not present in the tracked receipt |
| OmniDocBench commit | `147cd5ac9472002f5751221d390bf00abdbc0d2f` |
| Backend / quantization | Official PaddleOCR adapter using a local llama.cpp/GGUF endpoint; quantization type not recorded |
| Hardware / OS / driver / HIP | Windows AMD is established; exact GPU, driver, and HIP versions are not bound to the receipt |
| Dataset pages | 1,651 |
| Success / failure / exclusion | 1,650 success, 1 approved failure, 0 scoring exclusions |
| Failure handling | Missing failed-page output is scored as empty |
| Aggregation | Official notebook convention described above |
| Raw metric artifact | Six external outputs authenticated by the tracked G0 receipt |

The managed runtime manifest pins model, mmproj, layout, and llama.cpp resource
hashes for new installations. Those pins cannot retroactively fill missing
provenance in an older benchmark run.

## Performance status

G4 is **PASS**.

Commit history contains a 27-page diagnostic claim of 602.0 seconds to
357.2 seconds with zero structural mismatches, and an earlier commit subject
claimed a “10x inference speedup.” No tracked raw timing artifact binds either
claim to:

- a precise project/model/runtime/scorer manifest;
- VLM-stage versus end-to-end timing boundaries;
- per-page samples, mean, P50, P95, or throughput;
- the exact G3-accepted configuration.

The numerical speedup is therefore withdrawn from the README and is not a
release-gate result. The history is preserved in commits `d529cb4` and
`50ce802`; neither commit message is benchmark evidence.

Fresh artifact-backed diagnostics on the exact G3-accepted model and pipeline
show that GPU offload passes both numerical limits (mean 6.33 seconds/page,
P95 19.54 seconds/page). Eight of 27 output hashes differ from the historical
G3 baseline, and a repeated identical GPU run changed one page. The targeted
GT comparison projects those differences to the unchanged published accuracy
values 96.52 / 97.36 / 94.09 / Overall 95.99. See the
[G4 diagnostic](../releases/0.1.0-g4-diagnostic.md).

G4 acceptance still requires per-page timings, mean, P50, P95, throughput,
stage boundaries, hardware/runtime provenance, and a passing
output-equivalence check.
The sample contract is frozen in
[`eval/g4-v1.6-samples.json`](../../eval/g4-v1.6-samples.json): 27 pages,
three deterministic samples from each of nine primary document categories.
The fail-closed validator is `eval.g4_performance`.

## Gate status

| Gate | Status | Reason |
|---|---|---|
| G0 evidence integrity | PASS | Authenticated by the tracked r7 receipt |
| G1 compatibility contract | PASS | Covered by committed CLI/API/output tests |
| G3 accuracy | PASS | Maintainer accepted the PaddleOCR-confirmed Overall 95.99 result and waived another full run |
| G4 performance | PASS | Mean 6.33 s, P95 19.54 s, 0/27 failures; targeted GT projection preserves Overall 95.99 while explicitly recording raw-output differences |
| G5 launch | BLOCKED | Clean-network onboarding and the remaining prerequisite evidence are incomplete |

G2 root-cause diagnosis is not a release gate for this adaptation project.

These statuses do not authorize a release, tag, or GitHub Release publication.
