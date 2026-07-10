# OmniDocBench Evaluation

End-to-end evaluation of the PaddleOCR-VL-ROCm pipeline against the
[OmniDocBench](https://github.com/opendatalab/OmniDocBench) end-to-end document
understanding benchmark. This directory holds our deliverables only: the
prediction adapter, the dataset downloader, the staged runner, and the per-version
eval configs. The OmniDocBench harness itself is pinned in a separate checkout
(see [OmniDocBench checkout](#omnidocbench-checkout-and-cdm) below).

The pipeline is split into three independent, individually-gated stages so the
heavy / environment-sensitive steps (VLM inference, eval harness) can each run
on their own:

```text
download  ──►  infer  ──►  eval
 (HF)        (our VLM)    (OmniDocBench)
```

A single missing prerequisite fails the relevant stage with a clear message
rather than crashing the whole run.

## Prerequisites

Before running any stage, you need all of the following:

1. **A local llama.cpp/GGUF VLM server is running.** The `infer` stage drives the
   adapter against an OpenAI-compatible server on the local evaluation port.
   Start `llama-server.exe` with the PaddleOCR-VL-1.6 GGUF model and its
   multimodal projector, for example:

   ```powershell
   .\llama-server.exe `
     --host 127.0.0.1 `
     --port 8111 `
     -m C:\path\to\PaddleOCR-VL-1.6-GGUF.gguf `
     --mmproj C:\path\to\PaddleOCR-VL-1.6-GGUF-mmproj.gguf
   ```

   The model and projector paths depend on your local llama.cpp/GGUF setup.
   Then confirm:

   ```powershell
   paddleocr-vl-rocm-check-server --server-url http://127.0.0.1:8111/v1
   ```

   The local engine examples below use `http://127.0.0.1:8111/v1`.

2. **The PP-DocLayoutV3 ONNX layout model downloaded.**

   ```powershell
   pip install -e .[download]
   python scripts/download_ppdoclayoutv3_onnx.py
   # -> models/PP-DocLayoutV3-onnx/
   ```

3. **The OmniDocBench evaluation environment.** The eval stage needs the
   OmniDocBench package plus the metric backends (`TEDS`, `Edit_dist`, and
   optionally `CDM`). See
   [OmniDocBench checkout and CDM](#omnidocbench-checkout-and-cdm) for the two
   supported setups (Docker, or drop CDM).

4. **`huggingface_hub`** for the `download` stage:

   ```powershell
   pip install -e .[download]
   ```

## The three stages

All commands are run from the repository root. The orchestrator is
`eval/run_eval.py`; each stage can also be invoked via its own script.

### 1. `download` — fetch the OmniDocBench dataset

Downloads the manifest JSON plus the `images/` tree for a given version from the
Hugging Face dataset `opendatalab/OmniDocBench` into
`data/omnidocbench/<version>/`:

```powershell
python eval/run_eval.py --stage download --version v16
# or directly:
python eval/download_omnidocbench.py --version v16
```

`--version` selects the dataset; `--target-dir` / `--revision` override the
defaults. Note: OmniDocBench versions are distinguished by dataset branch, not a
documented `revision=` parameter, so the default is `latest` and the downloader
warns that the revision should be pinned for reproducibility.

### 2. `infer` — generate per-page Markdown predictions

Runs the PaddleOCR-VL-ROCm adapter (`eval/PaddleOCRVLROCm_img2md.py`) over every
image in `data/omnidocbench/<version>/images/` and writes one
`<image_basename_no_ext>.md` file per page into a flat predictions directory:

```powershell
python eval/run_eval.py --stage infer --version v16
# or directly:
python eval/PaddleOCRVLROCm_img2md.py `
  --img-dir data/omnidocbench/v16/images `
  --out-dir predictions/paddleocrvl_rocm `
  --layout-model models/PP-DocLayoutV3-onnx `
  --server-url http://127.0.0.1:8111/v1
```

### Local lightweight engine

```powershell
python eval/run_eval.py --stage infer --version v16 `
  --engine lightweight `
  --vlm-backend llama-cpp-server `
  --server-url http://127.0.0.1:8111/v1 `
  --api-model-name PaddleOCR-VL-1.6-GGUF.gguf
```

### Local official engine

The official engine requires the local `paddleocr` dependency to be installed
in the environment before running this command.

```powershell
python eval/run_eval.py --stage infer --version v16 `
  --engine official `
  --server-url http://127.0.0.1:8111/v1 `
  --api-model-name PaddleOCR-VL-1.6-GGUF.gguf `
  --page-retries 1
```

This stage is **server-gated**: it first pings the VLM server and exits with a
clear message if it is unreachable. Per-page failures are caught and recorded so
a single bad page does not abort the run (a missing page scores zero in the
harness).

### 3. `eval` — run the OmniDocBench harness

Runs `python pdf_validation.py --config eval/configs/omnidocbench_<version>.yaml`
inside the OmniDocBench checkout at `eval/.omnidocbench/`:

```powershell
python eval/run_eval.py --stage eval --version v16
```

This stage is **environment-gated**: if the OmniDocBench checkout is missing, it
prints setup instructions and exits non-zero. The config points the harness at
our `predictions/paddleocrvl_rocm/` directory and the matching version's
ground-truth manifest.

### Run all three stages

```powershell
python eval/run_eval.py --stage all --version v16
```

Useful flags: `--predictions-dir`, `--config`, `--match-method`, `--server-url`,
`--layout-model`, `--api-model-name`. See `python eval/run_eval.py --help`.

## OmniDocBench checkout and CDM

The eval stage expects a pinned OmniDocBench checkout at `eval/.omnidocbench/`
(this path is gitignored). Clone and install it manually — this task does **not**
vendor OmniDocBench or its dependencies (see the note under
`[project.optional-dependencies]` in `pyproject.toml`):

```bash
git clone https://github.com/opendatalab/OmniDocBench.git eval/.omnidocbench
cd eval/.omnidocbench
git checkout <pinned-commit-or-branch>   # master = v1.6
pip install -e .
```

The `CDM` metric for `display_formula` is heavyweight: it needs **Node.js**,
**ImageMagick 7**, and **TeX Live** (texlive-full, with CJK fonts). Two supported
paths:

- **Docker (recommended).** The reproducible image
  `ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204` ships the full CDM
  toolchain. Run the `eval` stage inside that container.
- **Drop CDM.** If you have no Node/ImageMagick/TeX (e.g. on Windows/ROCm without
  Docker), remove `CDM` from the `display_formula` metric list in the config:

  ```yaml
  display_formula: { metric: [Edit_dist], cdm_workers: 13 }   # CDM dropped
  ```

  `Edit_dist` on formulas still works without any extra system deps. Do **not**
  block the run on CDM.

## v1.5 vs v1.6

v1.5 and v1.6 are **different OmniDocBench git branches plus different
datasets**, sharing the same config schema but using a different matching
algorithm. They are not a config flag.

- **v1.6** — OmniDocBench repo `master` branch, ~1,651 pages. Matching =
  Multi-Granularity Adaptive Matching (MGAM); CDM rewritten from Node.js to
  Python.
- **v1.5** — earlier git branch, ~1,355 pages. Different matching algorithm.

To score **both** versions:

1. Generate predictions once with the adapter (the adapter and the predictions
   are version-agnostic — a given page's Markdown is the same regardless of which
   manifest references it).
2. For each version, check out the matching OmniDocBench branch and run the eval
   stage against that version's dataset + config:

   ```bash
   # v1.6
   (cd eval/.omnidocbench && git checkout master)
   python eval/run_eval.py --stage eval --version v16

   # v1.5
   (cd eval/.omnidocbench && git checkout <v1.5-branch>)
   python eval/run_eval.py --stage eval --version v15
   ```

The same predictions can be scored by both, but the matching algorithm differs,
so re-run the eval stage per version.

## Where scores land

The OmniDocBench harness writes its outputs into `./result/` **inside the
OmniDocBench checkout** (it is run with that directory as its working
directory). The score table is:

```text
eval/.omnidocbench/result/<save>_metric_result.json
```

where `<save>` = `basename(prediction.data_path)_<match_method>` — for our
defaults, `paddleocrvl_rocm_quick_match_metric_result.json`. Per-element
`*_result.json` and `*_run_summary.json` files land alongside it.

Copy or symlink the score table for the record into our tracked results tree:

```text
results/omnidocbench/<version>/paddleocrvl_rocm_quick_match_metric_result.json
```

(`results/omnidocbench/` is gitignored by default; commit a score as a release
artifact separately when you want it in history.)

## Files in this directory

| File | Purpose |
|---|---|
| `PaddleOCRVLROCm_img2md.py` | Adapter: per-page image → `<basename>.md`. Mirrors OmniDocBench's `PaddleOCR_img2md.py`. |
| `download_omnidocbench.py` | HF dataset downloader for a given version. |
| `run_eval.py` | Staged orchestrator (`download` / `infer` / `eval` / `all`). |
| `configs/omnidocbench_v15.yaml` | v1.5 eval config (run against the v1.5 branch checkout). |
| `configs/omnidocbench_v16.yaml` | v1.6 eval config (run against the `master` branch checkout). |

## Status

This integration is verified **structurally** (imports, naming logic, config YAML
validity, runner `--help`). A live end-to-end run with recorded scores is
**PENDING** until a VLM server and an OmniDocBench environment are available in
this repo's CI.
