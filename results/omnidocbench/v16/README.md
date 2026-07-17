# OmniDocBench v1.6 Local Evidence Index

The authoritative score, coverage, aggregation, provenance, and gate
interpretation is maintained in the
[OmniDocBench v1.6 benchmark fact sheet](../../../docs/benchmarks/omnidocbench-v1.6.md).
Do not construct a public score by mixing values from different files in this
directory.

The accepted G3 Overall is **95.99**. Its acceptance source is the tracked
[G3 maintainer attestation](../../../docs/releases/0.1.0-g3-attestation.md),
which records PaddleOCR's out-of-band confirmation and the maintainer's waiver
of another full run. It is not reconstructed from the historical files in this
directory.

## Tracked artifacts

| Artifact family | Purpose | Release status |
|---|---|---|
| `paddleocrvl_rocm_cdm_*_windows_native_2026-07-11.json` | Historical Windows-native lightweight metric and environment evidence | Historical only; incomplete provenance; not G3 |
| `paddleocr_official_local_llamacpp_gguf_*` | Historical official-local metrics, summaries, provenance, and failure diagnostics | Historical only; not the r7 G0 output set |
| Other `paddleocrvl_rocm_*` files | Earlier comparisons and hard-case diagnostics | Diagnostic only |

The independently reviewed official-local r7 outputs remain external. Their
SHA-256 identities and scoring interpretation are bound by the
[tracked G0 receipt](../../../docs/releases/0.1.0-g0-evidence.md).

## Scoring rule

The formal denominator is all 1,651 GT pages. The approved official-local
contract contains 1,650 successful predictions and one approved failed page
with no prediction file. The scorer retains that page and treats the missing
output as empty.

A 1,650-page paired equivalence analysis is a separate diagnostic operation,
not an accuracy scoring exclusion.

## Handling

- Preserve historical failures and raw metric fields.
- Redact machine-local user paths without changing metric values.
- Name the exact artifact and aggregation convention when quoting a number.
- Do not treat a commit message, README row, or partial CDM comparison as a
  benchmark artifact.
- Do not commit datasets, full predictions, private documents, credentials, or
  unredacted logs.
