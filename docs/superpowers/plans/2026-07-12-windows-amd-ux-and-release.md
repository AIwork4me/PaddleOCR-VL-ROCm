# Windows AMD User Experience And Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver verified one-command Windows AMD setup, actionable diagnostics, dual onboarding, polished documentation, and a reviewable GitHub release.

**Architecture:** Package a small signed-by-hash resource manifest, download large assets from upstream sources without committing them, and expose setup/doctor/run through one backward-compatible CLI. Keep the existing external-endpoint workflow first-class. Publish claims only after evidence, install, and performance gates pass.

**Tech Stack:** Python 3.10+, `requests`, `huggingface_hub`, PowerShell, llama.cpp HIP b9884, Rich, pytest, GitHub CLI.

## Global Constraints

- Windows AMD managed runtime uses llama.cpp b9884 (`86961efd5`) HIP Radeon x64.
- Main model SHA-256 is `F3AE46EC885050ACF4B3D31944431E1FD90D50664FB09126AF4A3C050BA14EE8` and size is `935769056` bytes.
- MM projector SHA-256 is `204D757D7610D9B3FAAB10D506D69E5B244E32BF765E2BAB2D0167E65E0A058A` and size is `881770560` bytes.
- Model source is `PaddlePaddle/PaddleOCR-VL-1.6-GGUF`; runtime source is the official llama.cpp b9884 Windows HIP release asset.
- English users download PP-DocLayoutV3 ONNX from `https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx`; Chinese users download it from `https://modelscope.cn/models/PaddlePaddle/PP-DocLayoutV3_onnx`.
- Every downloaded file must pass size and SHA-256 checks before activation.
- Existing `paddleocr-vl-rocm --input ...` commands remain valid.
- No hidden telemetry. Diagnostics redact secrets.
- Do not publish score or speed badges until G3 and G4 pass.

---

## File Structure

- Create `src/paddleocr_vl_rocm/assets/runtime-manifest.json`: pinned resource metadata.
- Create `src/paddleocr_vl_rocm/resources.py`: manifest parsing, resumable download, and verification.
- Create `tests/test_resources.py`: download, resume, checksum, and atomic activation tests.
- Create `src/paddleocr_vl_rocm/setup.py`: managed runtime/model/layout installation.
- Create `tests/test_setup.py`: idempotence and failure-recovery tests.
- Create `src/paddleocr_vl_rocm/doctor.py`: environment and endpoint diagnostics.
- Create `tests/test_doctor.py`: status, remediation, and redaction tests.
- Modify `src/paddleocr_vl_rocm/cli.py`: `setup`, `doctor`, and `run` subcommands with legacy compatibility.
- Modify `tests/test_cli.py`: both new and legacy journeys.
- Modify `pyproject.toml`: packaged JSON asset and download dependencies.
- Modify `README.md`, `README.zh-CN.md`, and `eval/README.md`: evidence-led launch documentation.
- Create `CONTRIBUTING.md`, `SECURITY.md`, and `.github/ISSUE_TEMPLATE/` files.
- Create `.github/workflows/ci.yml`: Windows and Linux offline quality/package checks.

### Task 1: Build and lock the runtime resource manifest

**Files:**
- Create: `scripts/build_runtime_manifest.py`
- Create: `src/paddleocr_vl_rocm/assets/runtime-manifest.json`
- Create: `tests/test_runtime_manifest.py`

**Interfaces:**
- Produces: a schema-1 JSON manifest with runtime, main GGUF, mmproj, and layout resources.

- [ ] **Step 1: Write a failing manifest-schema test**

```python
def test_runtime_manifest_is_fully_pinned():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema"] == 1
    assert manifest["runtime"]["version"] == "b9884"
    for item in manifest["resources"]:
        assert item["url"].startswith("https://")
        assert len(item["sha256"]) == 64
        assert item["size"] > 0
        assert item["destination"]
```

- [ ] **Step 2: Download the exact upstream runtime asset and hash it**

```powershell
$Url = "https://github.com/ggml-org/llama.cpp/releases/download/b9884/llama-b9884-bin-win-hip-radeon-x64.zip"
$Out = Join-Path $env:TEMP "llama-b9884-bin-win-hip-radeon-x64.zip"
Start-BitsTransfer -Source $Url -Destination $Out
Get-Item $Out | Select-Object Length
Get-FileHash -Algorithm SHA256 $Out
```

Record the returned size and hash through the builder script; do not type an
unverified value manually.

- [ ] **Step 3: Implement the manifest builder**

The builder accepts `--runtime-archive`, `--main-gguf`, `--mmproj`,
`--layout-onnx`, and `--layout-config`. The two layout files use immutable
Hugging Face resolve URLs in the machine-readable resource list. The manifest
also records the user-facing Hugging Face and ModelScope repository pages.

```python
def describe(path: Path, *, name: str, url: str, destination: str) -> dict[str, object]:
    return {
        "name": name,
        "url": url,
        "destination": destination,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
```

Use chunked hashing in the final implementation to avoid loading GGUF files in
memory. Assert the two GGUF hashes match the Global Constraints before writing.

- [ ] **Step 4: Generate and verify the manifest**

Run the builder with the downloaded runtime archive and the two locally verified
GGUF files. Run the schema test and inspect the diff for absolute local paths;
there must be none.

- [ ] **Step 5: Commit**

Commit the builder, manifest, and test with
`build: pin verified windows amd runtime assets`.

### Task 2: Implement resumable verified downloads

**Files:**
- Create: `src/paddleocr_vl_rocm/resources.py`
- Create: `tests/test_resources.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: frozen `Resource` dataclass.
- Produces: `download_resource(resource, root, session=None) -> Path`.
- Produces: `verify_resource(path, resource) -> None`.

- [ ] **Step 1: Write failing verification tests**

Test correct bytes, wrong size, wrong hash, partial `.part` resume, server without
Range support, and atomic rename. A checksum failure must delete the invalid
`.part` file but leave an existing verified destination untouched.

- [ ] **Step 2: Implement resource parsing and verification**

```python
@dataclass(frozen=True)
class Resource:
    name: str
    url: str
    destination: str
    size: int
    sha256: str


def verify_resource(path: Path, resource: Resource) -> None:
    if path.stat().st_size != resource.size:
        raise RuntimeError(f"Size mismatch for {resource.name}: {path}")
    if sha256_file(path).lower() != resource.sha256.lower():
        raise RuntimeError(f"SHA-256 mismatch for {resource.name}: {path}")
```

- [ ] **Step 3: Implement resume and atomic activation**

Write to `<destination>.part`, send `Range: bytes=<size>-` when partial data
exists, restart from zero on HTTP 200, verify, then `Path.replace(destination)`.
Print progress through a callback rather than directly to stdout.

- [ ] **Step 4: Package the manifest**

Add package-data configuration for `assets/*.json`. Keep `huggingface_hub` in
the existing `[download]` extra and provide a clear error when it is absent.

- [ ] **Step 5: Verify and commit**

Run resource tests, package build, install the wheel into a temporary venv, load
the packaged manifest, then commit with `feat: add verified resumable resource downloads`.

### Task 3: Implement idempotent managed setup

**Files:**
- Create: `src/paddleocr_vl_rocm/setup.py`
- Create: `tests/test_setup.py`

**Interfaces:**
- Produces: `SetupOptions` and `SetupResult` dataclasses.
- Produces: `setup_managed_runtime(options: SetupOptions) -> SetupResult`.
- Produces: `start_managed_server(result: SetupResult, *, port: int = 8111) -> subprocess.Popen`.

- [ ] **Step 1: Write failing setup tests**

Mock resource downloads and archive extraction. Assert first run installs all
resources, second run downloads nothing, `force=True` reinstalls, and extraction
failure leaves the previous runtime usable.

- [ ] **Step 2: Implement setup directories**

Use `%LOCALAPPDATA%/PaddleOCR-VL-ROCm` by default with `runtime/`, `models/`,
`cache/`, and `config.json`. Allow `--root` override. Never write into the
installed Python package.

- [ ] **Step 3: Install runtime and model assets**

Download the runtime archive through `download_resource`, extract into a staging
directory, verify `llama-server.exe --version` contains `9884`, then atomically
swap the runtime directory. Download and verify both GGUF resources. Reuse the
existing layout-model downloader and verify its required files.

- [ ] **Step 4: Write local configuration**

Store only local paths, server port, model basename, and manifest version. Do not
store credentials. Return every installed path in `SetupResult`.

- [ ] **Step 4A: Start and health-check the managed server**

Build the argument list without a shell:

```python
args = [
    str(result.llama_server),
    "-m", str(result.main_gguf),
    "--mmproj", str(result.mmproj),
    "--host", "127.0.0.1",
    "--port", str(port),
    "-ngl", "99",
]
process = subprocess.Popen(args, creationflags=subprocess.CREATE_NO_WINDOW)
```

Poll `/v1/models` with a 60-second deadline. On timeout, terminate the child and
raise an error that points to the captured server log. `setup --auto` starts the
server; `setup --no-start` installs only.

- [ ] **Step 5: Verify and commit**

Run setup/resource tests and a `--dry-run` smoke command. Commit with
`feat: add managed windows amd setup`.

### Task 4: Build actionable diagnostics

**Files:**
- Create: `src/paddleocr_vl_rocm/doctor.py`
- Create: `tests/test_doctor.py`
- Modify: `src/paddleocr_vl_rocm/server.py`

**Interfaces:**
- Produces: `CheckResult(name, status, message, remediation, details)`.
- Produces: `run_doctor(config, server_url=None) -> list[CheckResult]`.

- [ ] **Step 1: Write failing diagnostic tests**

Cover Windows version, AMD adapter present/absent, `amdhip64_7.dll`, disk space,
runtime version, model hashes, port availability, server `/v1/models`, and a
redacted connection error. Every failed check must have non-empty remediation.

- [ ] **Step 2: Implement checks with stable statuses**

Use `PASS`, `WARN`, and `FAIL`. Query display adapters through a PowerShell
subprocess that emits compressed JSON. Use `shutil.disk_usage`, manifest
verification, and existing server checks. Catch per-check errors so one failure
does not hide later results.

- [ ] **Step 3: Add JSON and human renderers**

`--json` emits machine-readable results. The default Rich table ends with an
exit code: 0 for no failures, 2 when any check fails.

- [ ] **Step 4: Verify and commit**

Run doctor tests and manually run against the current AMD Radeon 8060S setup.
Commit with `feat: add windows amd environment doctor`.

### Task 5: Add backward-compatible setup, doctor, and run CLI journeys

**Files:**
- Modify: `src/paddleocr_vl_rocm/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `pyproject.toml`

**Interfaces:**
- New commands: `paddleocr-vl-rocm setup`, `doctor`, and `run`.
- Legacy `paddleocr-vl-rocm --input ...` remains supported.

- [ ] **Step 1: Write parser tests for both journeys**

Assert:

```python
assert parse_args(["setup", "--auto"]).command == "setup"
assert parse_args(["doctor", "--json"]).command == "doctor"
assert parse_args(["run", "invoice.png"]).input == "invoice.png"
assert parse_args(["--input", "invoice.png"]).input == "invoice.png"
```

- [ ] **Step 2: Refactor parser construction**

Keep the existing legacy parser. When the first argument is `setup`, `doctor`,
or `run`, dispatch to a subparser; otherwise parse with the legacy parser. Map
`run invoice.png` to the same namespace and execution function as
`--input invoice.png`.

Define stable process exit codes in `cli.py`:

```python
class ExitCode(IntEnum):
    OK = 0
    USAGE = 2
    ENVIRONMENT = 10
    DOWNLOAD = 11
    SERVER = 12
    INFERENCE = 13
    PARTIAL = 14
```

Add parser/dispatch tests for every code.

- [ ] **Step 3: Wire setup and doctor**

`setup --auto` uses managed defaults. `doctor --server-url` supports an existing
endpoint without requiring managed assets. Print the exact next command after a
successful setup.

- [ ] **Step 4: Verify and commit**

Run CLI, setup, doctor, pipeline characterization, package build, and full tests.
Commit with `feat: add guided setup doctor and run commands`.

### Task 6: Polish documentation and open-source project surfaces

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `eval/README.md`
- Create: `CONTRIBUTING.md`
- Create: `SECURITY.md`
- Create: `.github/ISSUE_TEMPLATE/bug.yml`
- Create: `.github/ISSUE_TEMPLATE/hardware-report.yml`
- Create: `.github/pull_request_template.md`
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces two tested four-command onboarding paths and evidence-linked claims.

- [ ] **Step 1: Build a real demo artifact**

Run one representative example through the accepted release configuration.
Record input, rendered Markdown, structured JSON excerpt, hardware, elapsed
time, and project commit. Keep large raw assets out of Git; commit only a small
optimized preview when licensing permits.

- [ ] **Step 2: Rewrite README hierarchy**

Order: value proposition, verified evidence table, real demo, Windows AMD setup,
existing-server setup, Python API, support matrix, benchmark reproduction,
troubleshooting, contributing. Each number links to its provenance artifact.

- [ ] **Step 3: Mirror English and Chinese content**

Use the same commands, numbers, support labels, and limitations. Add a test that
searches both files for the accepted Overall, mean, P95, v1.6 commit, and runtime
version.

- [ ] **Step 4: Add contribution and support templates**

Hardware reports require GPU, driver, Windows version, llama.cpp version, model
hash, command, and redacted doctor JSON. Bug reports require a minimal input or
reproduction and expected/actual behavior.

- [ ] **Step 5: Verify and commit**

Run link checks where available, command `--help` smoke tests, package tests, and
the full local check. Commit with `docs: polish windows amd launch experience`.

- [ ] **Step 5A: Add offline CI**

Create a workflow triggered by pushes and pull requests. Use Python 3.10 and
3.13 on `windows-latest` and Python 3.10 on `ubuntu-latest`. Each job runs:

```text
pip install -e .[dev]
python -m compileall -q src/paddleocr_vl_rocm eval
ruff check src tests scripts eval
ruff format --check src tests scripts eval
mypy src
python -m pytest -q
python -m build
```

The workflow must not download models, datasets, or contact a VLM server.

### Task 7: Release and publish

**Files:**
- Modify: `pyproject.toml`
- Create: release notes under `docs/releases/`

**Interfaces:**
- Produces: a pushed review branch, draft PR, accepted release tag, and release notes.

- [ ] **Step 1: Run the final release gate**

Require G0–G5 artifacts, all local checks, clean package install, clean managed
setup, existing-server setup, and zero unintended tracked files. Confirm no
secret appears in `git diff`, tracked files, or generated evidence.

- [ ] **Step 2: Bump version and write release notes**

Name tested hardware and environment, exact v1.6 scorer commit, accuracy and
performance values, known limitations, migration notes, and legacy CLI support.

- [ ] **Step 3: Commit and push the branch**

```powershell
git status --short
git push -u origin codex/top-tier-quality
```

Expected: only intended commits are pushed to
`https://github.com/AIwork4me/PaddleOCR-VL-ROCm`.

- [ ] **Step 4: Open a draft PR**

Use the GitHub publishing workflow. Include evidence links and a checkbox for
every release gate. Keep the PR draft until hardware and benchmark checks pass.

- [ ] **Step 5: Merge and tag only after review**

After required checks and review pass, merge, create the approved version tag,
and attach release notes. Do not attach raw datasets, predictions, logs, or
unverified runtime binaries.
