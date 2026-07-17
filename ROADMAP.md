# Roadmap

The project is pre-release. Priorities are ordered by evidence and user impact,
not by feature count.

## Before v0.1.0

- Close G2 with authenticated same-boundary oracle evidence.
- Produce a fresh OmniDocBench v1.6 G3 run with complete provenance and Overall
  >= 96.13 without component or contract regression.
- Run an artifact-backed G4 benchmark on that exact G3 manifest, reporting
  end-to-end and VLM-stage mean, P50, P95, and throughput.
- Complete clean-network managed setup on a fresh Windows environment.
- Record exact GPU memory, Windows build, AMD driver, HIP runtime, model hashes,
  and llama.cpp commit for every accepted result.

## After v0.1.0

- Add tested managed server stop/status/cleanup commands.
- Expand community-verified Windows AMD hardware coverage.
- Improve offline installation and mirror guidance.
- Stabilize benchmark artifact schemas and automated claim validation.

Items are not release commitments. Gate status remains authoritative in
[`docs/releases/0.1.0-readiness.md`](docs/releases/0.1.0-readiness.md).
