# Windows AMD Doctor Evidence - 2026-07-12

This is a local environment diagnostic record, not a release gate or benchmark
artifact.

- Command: `paddleocr-vl-rocm doctor --json`
- Project commit under test: `7fbb6db` plus the uncommitted Task 6 documentation
  and diagnostic-redaction change
- Operating system: Windows 11, build `10.0.26200`
- Display adapter: `AMD Radeon(TM) 8060S Graphics`
- HIP runtime: `C:\Windows\System32\amdhip64_7.dll`
- ONNX Runtime providers available: `DmlExecutionProvider`,
  `CPUExecutionProvider`
- Existing `/v1/models` endpoint: reachable

The default managed root did not contain `config.json`, llama.cpp, model, or
layout assets during this run, so the managed runtime, resource hashes, and
DirectML layout session failed their checks. This record therefore supports
only the Windows/AMD/HIP environment statement. It does not prove a clean
managed installation, G3 accuracy, G4 performance, or release acceptance.

The command initially exposed absolute model paths returned by `/v1/models` in
successful JSON details. Task 6 added a regression test and changed successful
server details to contain only the redacted URL and model count.

