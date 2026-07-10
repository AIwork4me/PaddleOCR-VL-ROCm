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
