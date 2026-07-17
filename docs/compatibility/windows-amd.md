# Windows AMD Compatibility

Compatibility is reported per pipeline component. A GPU working with DirectML
does not prove that the llama.cpp HIP backend works, and an external
OpenAI-compatible server does not prove local HIP compatibility.

Status definitions:

- **Fully tested:** the named project path completed setup, Doctor, and smoke
  inference on the recorded machine. This is not a benchmark or release claim.
- **Community verified:** a reproducible community report completed the named
  path, but project maintainers have not reproduced it.
- **Expected but unverified:** code and upstream compatibility indicate that it
  may work; this project has no qualifying report.
- **Unsupported:** the project does not provide or accept that path.

## Project matrix

| Environment | DirectML layout | Local HIP VLM | Managed runtime | External server |
|---|---|---|---|---|
| Windows 11 build 10.0.26200, Radeon 8060S | Fully tested (smoke) | Fully tested (smoke) | Fully tested from verified cache | Fully tested (smoke) |
| Other AMD GPU/APU listed by AMD for the current Windows HIP SDK | Expected but unverified | Expected but unverified | Expected but unverified | Expected but unverified |
| AMD GPU/APU not listed by AMD for the current Windows HIP SDK | Expected but unverified for layout only | Unsupported by the upstream HIP support contract | Unsupported | Expected but unverified when the server is hosted elsewhere |
| Linux | Unsupported (no DirectML path) | Not managed by this project | Unsupported | Expected but unverified; CPU layout plus an external endpoint |
| macOS | Unsupported | Unsupported | Unsupported | Unsupported by the documented project scope |

AMD's current source of truth is the
[HIP SDK for Windows system-requirements table](https://rocm.docs.amd.com/projects/install-on-windows/en/develop/reference/system-requirements.html).
If a device is absent from that table, do not infer HIP support from the generic
“AMD GPU” label.

## Recorded machine

The project record identifies:

- Windows 11 build `10.0.26200`;
- `AMD Radeon(TM) 8060S Graphics`;
- `DmlExecutionProvider` before `CPUExecutionProvider`;
- the presence of `amdhip64_7.dll`;
- pinned llama.cpp build `b9884` / commit `86961efd5`;
- successful verified-cache setup and both managed/external-server smoke
  inference.

The exact AMD driver version, VRAM/shared-memory amount, and HIP runtime version
were not captured. Consequently this record does not establish performance,
accuracy, or broad hardware support. See the
[Doctor evidence](../windows-amd-doctor-evidence-2026-07-12.md) and
[Windows validation record](../releases/0.1.0-windows-validation.md).

## Reporting another device

Use the Hardware Compatibility Report issue form. Include the GPU/APU, dedicated
or shared memory, Windows build, AMD driver, HIP runtime, Python, llama.cpp
build, and redacted `paddleocr-vl-rocm doctor --json` output.

Never post tokens, private documents, endpoint credentials, user-specific
paths, or unredacted logs.
