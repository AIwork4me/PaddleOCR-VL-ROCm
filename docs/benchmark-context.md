# Benchmark Context — External / Different-number References (PaddleOCR-VL-ROCm)

Numbers that are NOT the project's canonical result live here, never in
`model_card_v2.json` `results[]`.

## Different/official numbers (not auto-resolved)
| source | Overall | note |
|---|---|---|
| README "G3 PASS, maintainer-accepted out-of-band (2026-07-17)" | 95.99 | different number from the artifact-derived 95.77; attested, not recomputed from this artifact set |
| README results table | 96.33 | different number |
| Official PaddleOCR-VL evaluation (out-of-band) | 95.99 | external reference, different engine — [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |

## Canonical project result
See `model_card_v2.json` + the generated README block: **windows-hip mixed
pipeline = 95.77** (`submitted`). See `docs/migrations/rocmdoc-standard-v1.md` for
the evidence-weakness + platform-ambiguity notes.
