# Known Upstream Page Exception Design

## Decision

OmniDocBench v1.6 scoring continues to cover all 1,651 ground-truth pages. The
page `newspaper_The Times UK_0801@magazinesclubnew_page_031.png` remains an empty
prediction in official-local scoring; it is not removed from the denominator
and is not counted as a successful prediction.

The release coverage gate may accept exactly one official-local inference
failure when all of these facts match:

- `count=1651`, `ok=1650`, `fail=1`, `fallback=0`, `limit_pages=null`;
- the sole failed image is the filename above;
- its error contains the stable `peg-native` parser failure;
- the exception links to <https://github.com/PaddlePaddle/PaddleOCR/issues/18248>;
- no fallback prediction or synthetic Markdown is inserted.

This is a project-owner-approved, publicly traceable exception. The GitHub issue
is open and must not be described as a PaddlePaddle maintainer resolution.

## Architecture

A small `eval/release_contract.py` module owns the immutable exception and
validates run statistics. `eval/run_eval.py` and
`scripts/run_official_local_v16.ps1` both invoke that contract. Artifact summary
and provenance writers record the accepted exception explicitly, including its
filename, issue URL, and error class.

The scorer configuration and metric extraction do not change. Therefore the
reported official notebook values remain 1,651-page values with the failed page
scored as empty. A separate operational field reports `1650 successful + 1
approved known failure`.

## Failure Behavior

The gate rejects a second failure, a different filename, missing failure detail,
an error without the PEG signature, any fallback, limited inference, a changed
dataset count, or an exception applied to a non-official engine. The exception
does not waive CDM/TEDS quality checks or the G3 Overall threshold of 96.13.

## Verification

Tests cover the accepted exact record and every rejection dimension. After the
contract is implemented, run the full official-local 1,651-page inference and
scoring flow again. Conclusions, documentation, performance authorization, and
release status must use only the newly generated run stats, metric, summary, and
provenance artifacts.

