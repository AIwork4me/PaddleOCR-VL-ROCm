# Task 5 Report: Documentation and Local Score Posture

## Files Changed

- `README.md`
- `README.zh-CN.md`
- `eval/README.md`

## Grep Check Summary

Command run:

```powershell
rg -n "vLLM/BF16|Linux vLLM|reference-quality path|native precision aligned|PaddleOCR-VL-1.5-0.9B" README.md README.zh-CN.md eval/README.md
```

Matches remaining after the update:

- `README.md` and `README.zh-CN.md` each retain `PaddleOCR-VL-1.5-0.9B` in
  their general legacy CLI and Python API examples; these are outside the
  evaluation sections.
- `README.md` and `README.zh-CN.md` each state that local measurements are not
  claimed from a Linux vLLM/BF16 reference path; these are intentional local-only
  posture disclaimers.

No outdated model identifier remains in evaluation documentation. The local
lightweight and official evaluation examples use `PaddleOCR-VL-1.6-GGUF.gguf`.

## Review Follow-up

- Documented the local `paddleocr` dependency prerequisite for `--engine official`.
- Updated the direct inference example to use the local llama.cpp port `8111`.

## Second Re-review Follow-up

- Made `eval/README.md` self-contained for the local llama.cpp/GGUF server path,
  including a `llama-server.exe` example with the GGUF model, multimodal
  projector, and port `8111`; it no longer delegates the prerequisite to the
  top-level legacy `8000` instructions.
- Added the local `paddleocr` prerequisite next to the official-engine command
  in `README.zh-CN.md`, matching the English evaluation documentation.

## Final Review Follow-up

- Marked the official local-engine score row as pending local reproduction from
  tracked repository artifacts in both top-level READMEs. Companion setup
  evidence is identified as non-tracked context and is not presented as a
  repo-backed score.
- Restored the arXiv provenance link for the public PaddleOCR-VL-1.6 target row
  in both top-level READMEs.
- Re-ran the Task 5 grep check. Remaining matches are the intentional legacy
  CLI/API examples and the intentional Linux vLLM/BF16 reference-path
  disclaimer.
- `git diff --check` passed.
