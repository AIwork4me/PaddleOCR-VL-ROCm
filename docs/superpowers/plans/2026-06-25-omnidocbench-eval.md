# OmniDocBench v1.5 / v1.6 Evaluation Chain Implementation Plan (Plan B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a reproducible OmniDocBench v1.5 + v1.6 evaluation chain to PaddleOCR-VL-ROCm: an adapter that emits per-page Markdown, dataset acquisition, a config, and a runner that produces the standard score table.

**Architecture:** OmniDocBench reads **pre-generated per-page `.md` files** from a flat directory (it never imports/calls our adapter — see `.superpowers/sdd/omnidocbench-research.md`). So our chain is three decoupled pieces, all in our repo under `eval/`: (1) an **adapter** that runs our pipeline over the dataset images and writes `<basename_no_ext>.md` per page; (2) a **dataset downloader** that fetches the OmniDocBench manifest + images from HuggingFace; (3) a **runner** that orchestrates download → infer (against our VLM server) → `pdf_validation.py --config` → score report, with a config template + version pinning for v1.5/v1.6.

**Tech Stack:** Python 3.10–3.13, `huggingface_hub`, our `paddleocr_vl_rocm` package; the OmniDocBench repo (pinned commit) + its `pdf_validation.py` as an external, separately-cloned dependency.

## Global Constraints

- **Public API unchanged.** This plan only ADDS files under `eval/` (+ README/pyproject edits). No change to `src/`.
- **No heavy deps in core.** `huggingface_hub` is already an optional `[download]` extra; reuse it. OmniDocBench's own deps (incl. CDM's Node/ImageMagick/TeX) live in the OmniDocBench checkout, NOT in our `pyproject.toml`.
- **Server-gated steps are explicitly marked.** The infer step (adapter run) needs our ROCm VLM server up; the eval step needs the OmniDocBench env (or its Docker image). Code is written + structurally verified now; the live run + score recording are deferred to when the server/env are up (user action on wake).
- **Reproducibility:** pin a specific OmniDocBench commit; datasets via HF; configs versioned (v1.5/v1.6).
- Python `>=3.10,<3.14`; new code uses `from __future__ import annotations`; passes `ruff`/`mypy`/`pytest` via `scripts/check.sh`.

## Reference

- Research brief (full citations, manifest schema, metric details, version differences):
  `.superpowers/sdd/omnidocbench-research.md`
- Design spec: `docs/superpowers/specs/2026-06-25-engineering-quality-upgrade-design.md` §3.

## File Structure

All new files under `eval/` in our repo. `eval/` is a **directory of scripts** (no `__init__.py`) —
this avoids shadowing the `eval` builtin and matches the existing `scripts/` convention, where
`tests/test_download_script.py` loads a script via `importlib`. Tests load these the same way.

| File | Responsibility |
|---|---|
| `eval/PaddleOCRVLROCm_img2md.py` | **Adapter**: iterate dataset images → run pipeline → write `<out>/<basename_no_ext>.md` per page (mirror OmniDocBench's `PaddleOCR_img2md.py`) |
| `eval/configs/omnidocbench_v16.yaml` | OmniDocBench eval config (master/v1.6 manifest, our predictions dir, metrics) |
| `eval/configs/omnidocbench_v15.yaml` | Same for v1.5 (run against the v1.5 branch checkout) |
| `eval/download_omnidocbench.py` | Fetch manifest + images from HF `opendatalab/OmniDocBench` into a managed dir |
| `eval/run_eval.py` | **Runner**: ensure OmniDocBench checkout (pinned) → download data → run adapter → run `pdf_validation.py --config` → locate + print the score report |
| `eval/README.md` | How to run v1.5/v1.6 evals; prerequisites (server, OmniDocBench env/Docker); CDM note |
| `tests/test_eval_adapter.py` | Structural test: adapter naming logic + imports (no server needed) |
| `tests/test_eval_download_defaults.py` | Test downloader defaults (HF repo id, required files) |

Modifies: `README.md`, `README.zh-CN.md` (cross-link to `eval/README.md`), `pyproject.toml` (an `eval` optional-extra note pointing to OmniDocBench — NOT vendoring its deps).

---

## Task B1: Adapter + config templates

**Files:**
- Create: `eval/__init__.py`, `eval/PaddleOCRVLROCm_img2md.py`, `eval/configs/omnidocbench_v16.yaml`, `eval/configs/omnidocbench_v15.yaml`
- Test: `tests/test_eval_adapter.py`

**Interfaces:**
- Produces: `eval.PaddleOCRVLROCm_img2md` with `process_folder(img_dir, out_dir, *, layout_model, server_url, api_model_name, vlm_backend, ...)` writing one `<basename_no_ext>.md` per image; a `main()` CLI (`--img-dir`, `--out-dir`, `--layout-model`, `--server-url`, `--api-model-name`, `--vlm-backend`).

**Read first:** `.superpowers/sdd/omnidocbench-research.md` (prediction contract: flat dir, `<basename_no_ext>.md`; no JSON needed).

- [ ] **Step 1: Write failing test `tests/test_eval_adapter.py`**

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_adapter():
    script = Path("eval/PaddleOCRVLROCm_img2md.py")
    spec = importlib.util.spec_from_file_location("paddleocrvl_rocm_img2md", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_md_name_strips_extension():
    mod = _load_adapter()
    # OmniDocBench matcher looks up <img_name[:-4]>.md first
    assert mod.expected_md_name("page_001.png") == "page_001.md"
    assert mod.expected_md_name("doc.jpeg") == "doc.md"


def test_image_extensions_lowercase():
    mod = _load_adapter()
    exts = {e.lower() for e in mod.IMAGE_EXTENSIONS}
    assert {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"} <= exts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_adapter.py -v`
Expected: FAIL (file `eval/PaddleOCRVLROCm_img2md.py` not found → importlib error).

- [ ] **Step 3: Implement the adapter**

`eval/PaddleOCRVLROCm_img2md.py` (mirror OmniDocBench's `tools/model_infer/PaddleOCR_img2md.py`):

```python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from paddleocr_vl_rocm import PaddleOCRVLROCm

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")


def expected_md_name(image_name: str) -> str:
    # OmniDocBench matcher's first lookup is <img_name[:-4]>.md (basename minus extension)
    return Path(image_name).stem + ".md"


def process_folder(
    img_dir: Path,
    out_dir: Path,
    *,
    layout_model: str = "models/PP-DocLayoutV3-onnx",
    server_url: str = "http://127.0.0.1:8000/v1",
    api_model_name: str = "PaddleOCR-VL-1.5-0.9B",
    vlm_backend: str = "vllm-server",
) -> dict:
    pipeline = PaddleOCRVLROCm(
        layout_model_dir=layout_model,
        vlm_server_url=server_url,
        api_model_name=api_model_name,
        vlm_backend=vlm_backend,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: list[dict] = []
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    for img in images:
        start = time.time()
        try:
            result = pipeline.predict(img)
            md_path = out_dir / expected_md_name(img.name)
            md_path.write_text(result.markdown_text, encoding="utf-8")
            stats.append({"image": img.name, "status": "ok", "seconds": round(time.time() - start, 2)})
        except Exception as exc:  # noqa: BLE001 - record failure, continue (page scored as empty otherwise)
            stats.append({"image": img.name, "status": f"failed: {exc}", "seconds": round(time.time() - start, 2)})
    return {"count": len(images), "ok": sum(1 for s in stats if s["status"] == "ok"), "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="PaddleOCR-VL-ROCm adapter for OmniDocBench: write per-page .md")
    parser.add_argument("--img-dir", required=True, help="Dataset images directory.")
    parser.add_argument("--out-dir", required=True, help="Output flat dir of <basename>.md predictions.")
    parser.add_argument("--layout-model", default="models/PP-DocLayoutV3-onnx")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-model-name", default="PaddleOCR-VL-1.5-0.9B")
    parser.add_argument("--vlm-backend", default="vllm-server")
    args = parser.parse_args()
    summary = process_folder(
        Path(args.img_dir),
        Path(args.out_dir),
        layout_model=args.layout_model,
        server_url=args.server_url,
        api_model_name=args.api_model_name,
        vlm_backend=args.vlm_backend,
    )
    print(summary)
```

Note: the adapter writes `.md` from `result.markdown_text` (our result already exposes it). Do NOT emit JSON for the harness. Per-page failure is caught so one bad page doesn't abort the run (a missing page scores zero).

- [ ] **Step 4: Create the config templates**

`eval/configs/omnidocbench_v16.yaml`:
```yaml
end2end_eval:
  metrics:
    text_block: { metric: [Edit_dist] }
    display_formula: { metric: [Edit_dist, CDM], cdm_workers: 13 }   # drop CDM if no TeX/ImageMagick
    table: { metric: [TEDS, Edit_dist], teds_workers: 13 }
    reading_order: { metric: [Edit_dist] }
  dataset:
    dataset_name: end2end_dataset
    ground_truth: { data_path: ./data/omnidocbench/v16/OmniDocBench.json }
    prediction: { data_path: ./predictions/paddleocrvl_rocm }
    match_method: quick_match
    match_workers: 13
```

`eval/configs/omnidocbench_v15.yaml`: identical except `ground_truth.data_path: ./data/omnidocbench/v15/OmniDocBench.json` and a top comment "Run against the OmniDocBench v1.5 git branch checkout (different matching algorithm)."

- [ ] **Step 5: Run full check + structural verification**

Run: `bash scripts/check.sh`
Expected: green. `mypy src` only covers `src/` (per Task 1 config) — `eval/` scripts are not in `files`, so they're not type-checked; that's fine (they're scripts, mirroring `scripts/`). But `ruff check`/`ruff format --check` in `check.sh` must include `eval` — **update `scripts/check.sh` and `scripts/check.ps1`** to add `eval` alongside `src tests scripts` in the ruff lines (and pytest already discovers `tests/`).

Also verify the adapter runs as a script and exposes the helper:
```bash
python eval/PaddleOCRVLROCm_img2md.py --help
python -c "import importlib.util as u; s=u.spec_from_file_location('m','eval/PaddleOCRVLROCm_img2md.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print(m.expected_md_name('page_001.png'))"
```
First prints usage; second prints `page_001.md`.

- [ ] **Step 6: Commit**

```bash
git add eval tests/test_eval_adapter.py
git commit -m "feat(eval): add OmniDocBench adapter + v1.5/v1.6 config templates"
```

---

## Task B2: Dataset downloader + runner

**Files:**
- Create: `eval/download_omnidocbench.py`, `eval/run_eval.py`
- Test: `tests/test_eval_download_defaults.py`

**Interfaces:**
- Produces: `download_omnidocbench.DEFAULT_REPO_ID`, `REQUIRED_FILES`/patterns; `main()` with `--version {v15,v16}`, `--target-dir`. `run_eval.main()` orchestrating clone-pin OmniDocBench → download → adapter → `pdf_validation.py --config` → print report path.

**Read first:** `.superpowers/sdd/omnidocbench-research.md` (HF dataset `opendatalab/OmniDocBench`; v1.5/v1.6 = repo/dataset branch checkout; `pdf_validation.py --config`; report at `./result/<save>_metric_result.json` where `<save>=basename(prediction.data_path)_<match_method>`).

- [ ] **Step 1: Write failing test `tests/test_eval_download_defaults.py`**

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(file))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_defaults():
    mod = _load("dl", "eval/download_omnidocbench.py")
    assert mod.DEFAULT_REPO_ID == "opendatalab/OmniDocBench"
    assert {"v15", "v16"} <= set(mod.VERSIONS)
```

- [ ] **Step 2: Run test to verify it fails** → FAIL (file missing).

- [ ] **Step 3: Implement `eval/download_omnidocbench.py`**

Mirror the style of `scripts/download_ppdoclayoutv3_onnx.py` (lazy-import `huggingface_hub`). Fetch the manifest JSON + the `images/` dir for the requested version into `--target-dir`. Pin a known-good HF revision if discoverable; otherwise default to latest and log a warning that the version should be pinned for reproducibility. Expose `DEFAULT_REPO_ID = "opendatalab/OmniDocBench"` and `VERSIONS = {"v15": ..., "v16": ...}` (revision/branch strings — use the repo branch names if known, else `None` with a TODO comment to pin).

- [ ] **Step 4: Implement `eval/run_eval.py`**

A thin orchestrator (NOT executing the heavy steps unless flags given — it should support a `--stage {download,infer,eval,all}` so each phase can run independently, since infer needs the server and eval needs the OmniDocBench env):
- `download`: call `download_omnidocbench.main` for the chosen version.
- `infer`: call the adapter (`PaddleOCRVLROCm_img2md.process_folder`) against the dataset images → predictions dir. **Server-gated.**
- `eval`: run `python pdf_validation.py --config eval/configs/omnidocbench_vXX.yaml` inside the OmniDocBench checkout (pin a commit; clone if absent via `git clone` + `git checkout <commit>` into a managed `eval/.omnidocbench/` dir, gitignored). **Env-gated.**
- `all`: download → infer → eval.
- After eval, locate `./result/<save>_metric_result.json` and print its path (copy/symlink into our `results/omnidocbench/<version>/`).

Keep it defensive: each stage checks its prerequisites (server reachable for infer; OmniDocBench present for eval) and fails with a clear message rather than crashing.

- [ ] **Step 5: Run full check + structural verification**

Run: `bash scripts/check.sh` → green. Verify the runner imports and `--help` works:
```bash
python eval/run_eval.py --help
```

- [ ] **Step 6: Commit**

```bash
git add eval tests/test_eval_download_defaults.py
git commit -m "feat(eval): add OmniDocBench dataset downloader + staged runner"
```

---

## Task B3: Eval docs + pyproject note

**Files:**
- Create: `eval/README.md`
- Modify: `README.md`, `README.zh-CN.md`, `pyproject.toml`, `.gitignore`

- [ ] **Step 1: Write `eval/README.md`**

Cover: prerequisites (our VLM server up; layout model downloaded; OmniDocBench env — recommend the Docker image `ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204`, or drop CDM from the config if no TeX/ImageMagick/Node); the three stages (`download` / `infer` / `eval`); how v1.5 vs v1.6 differ (branch checkout + dataset); where scores land (`results/omnidocbench/<version>/`). Provide exact commands.

- [ ] **Step 2: Cross-link from main READMEs**

Add a short "## Evaluation (OmniDocBench)" section to `README.md` and `README.zh-CN.md` pointing to `eval/README.md`.

- [ ] **Step 3: `pyproject.toml` + `.gitignore`**

- `pyproject.toml`: add an `[project.optional-dependencies] eval = [...]` note ONLY if we add a real dep — we don't (huggingface_hub already in `download`). Instead, add a comment under optional-dependencies documenting that OmniDocBench's own deps live in its checkout. If `eval/` needs to be covered by `ruff`/`mypy`, ensure `scripts/check.sh` picks it up (it already lints `src tests scripts`; add `eval` to those paths in `check.sh`/`check.ps1`).
- `.gitignore`: add `eval/.omnidocbench/` (the cloned OmniDocBench checkout) and `predictions/` and `results/omnidocbench/` (generated; or commit empty result dirs with `.gitkeep` — decide: keep generated scores OUT of git by default, document committing them as a release artifact separately).

- [ ] **Step 4: Run full check** → green. Update `scripts/check.sh`/`check.ps1` to include `eval` in ruff/mypy/pytest paths if not already.

- [ ] **Step 5: Commit**

```bash
git add eval/README.md README.md README.zh-CN.md pyproject.toml .gitignore scripts/check.sh scripts/check.ps1
git commit -m "docs(eval): document OmniDocBench v1.5/v1.6 eval workflow; include eval/ in checks"
```

---

## Notes for the implementer

- **Cannot run end-to-end now** (VLM server down + no OmniDocBench env). Verify structurally: imports, naming logic, config YAML validity (`python -c "import yaml; yaml.safe_load(open('eval/configs/omnidocbench_v16.yaml'))"`), runner `--help`. Mark the live run + score recording as PENDING in the report.
- **CDM** likely won't run on Windows/ROCm without Docker — document the Docker path and the "drop CDM" fallback. Do NOT block on CDM.
- **v1.5 vs v1.6** = different OmniDocBench git branches + different datasets, same config schema. The runner must `git checkout` the right branch for eval (infer/adapter is version-agnostic — same predictions can be scored by both, but the matching algorithm differs so re-run eval per version).
- The `eval` package name shadows a builtin — if anything is flaky, rename to `omnidocbench_eval`.
