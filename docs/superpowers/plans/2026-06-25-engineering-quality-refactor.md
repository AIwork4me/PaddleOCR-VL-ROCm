# Engineering-Quality Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 1400-line `pipeline_core.py` into single-responsibility modules while preserving byte-identical behavior, protected by a characterization safety net, and stand up local engineering scaffolding (ruff/mypy/pytest).

**Architecture:** Behavior-preserving refactor. First establish a regression safety net (characterization tests that lock current outputs). Then extract modules one at a time from `pipeline_core.py` into focused files, in dependency order (leaves first), re-running the full safety net after each move. Public API (`PaddleOCRVLROCm`, `PaddleOCRVLROCmResult`, CLI) stays unchanged as a thin facade. Local-only scaffolding; CI deferred.

**Tech Stack:** Python 3.10–3.13, pytest, ruff, mypy, ONNXRuntime, Pillow, OpenCV, NumPy, requests, rich, shapely.

## Global Constraints

- **Public API must not change** in signature or behavior: `paddleocr_vl_rocm.PaddleOCRVLROCm`, `PaddleOCRVLROCmResult`, the `paddleocr-vl-rocm` / `paddleocr-vl-rocm-check-server` entry points, and all `--cli` flags. Internal modules and private functions may be relocated freely.
- **Behavior preservation is the prime directive.** After every task the characterization tests (integration golden + per-module unit) and existing tests must be green. A refactor task that changes output is a bug, not progress.
- **Single source of truth for dependencies:** `pyproject.toml`. `requirements.txt` is deleted.
- **No CI this phase.** Scaffolding is local (`scripts/check.sh`, `scripts/check.ps1`).
- **No new dependencies.** Reuse what `pyproject.toml` already declares (add only dev tooling: `ruff`, `mypy` to the `dev` extra).
- **Python:** `>=3.10,<3.14`. New code uses `from __future__ import annotations` to match existing style.
- **Work on branch `feat/engineering-quality`**, branched from current `main`.

---

## File Structure

After this plan, `src/paddleocr_vl_rocm/` looks like:

| File | Responsibility |
|---|---|
| `constants.py` | Shared label/prompt/pixel constants (`IMAGE_LABELS`, `NON_MERGE_LABELS`, `SKIP_ORDER_LABELS`, `MARKDOWN_IGNORE_LABELS`, `DEFAULT_MIN_PIXELS`, `DEFAULT_MAX_PIXELS`, `PROMPTS`) |
| `models.py` | Shared dataclasses `LightBlock`, `TableCell` |
| `encoding.py` | image↔data-url/base64, sha256 (pure helpers) |
| `geometry.py` | bbox overlap / area / projection / polygon-overlap / `filter_overlap_boxes` |
| `imageio.py` | image read (rgb/bgr), crop, mask crop, `crop_margin`, `merge_images` |
| `content.py` | VLM text normalization, repetition truncation, CJK/newline rules |
| `table.py` | OTSL→HTML conversion (incl. OTSL constants) |
| `markdown.py` | block→Markdown serialization |
| `vlm/__init__.py`, `vlm/client.py` | `OpenAICompatibleVLMClient` (`LlamaCppClient`), payload, response parsing, cache key/load, `_prompt_for_label` |
| `preprocess.py` | `make_blocks`, `merge_blocks`, table-figure tokenization |
| `serialize.py` | PaddleOCR-VL JSON payload assembly |
| `pipeline_core.py` (slimmed) | `run_light_parser` orchestration only |
| `pipeline.py` | public `PaddleOCRVLROCm` (unchanged) |
| `result.py` | public `PaddleOCRVLROCmResult` (unchanged) |
| `layout.py`, `server.py`, `utils.py`, `cli.py`, `__init__.py` | unchanged (cli/utils may gain logging) |

Dependency order (no cycles): `constants`, `models`, `encoding` → `geometry`, `imageio`, `content` → `table` (uses `models`), `markdown` (uses `models`, `constants`) → `vlm/client` (uses `encoding`, `server`, `constants`) → `preprocess` (uses `models`, `geometry`, `imageio`, `constants`) → `serialize` (uses `models`, `layout`, `constants`) → `pipeline_core` (uses all).

---

## Task 1: Branch, scaffolding, green baseline

**Files:**
- Modify: `pyproject.toml`
- Delete: `requirements.txt`
- Create: `scripts/check.sh`, `scripts/check.ps1`

**Interfaces:**
- Produces: `dev` extra includes `ruff`, `mypy`, `pytest`; ruff/mypy/pytest configs; `scripts/check.*` runnable on Windows + Linux.

- [ ] **Step 1: Create the feature branch**

```bash
git checkout -b feat/engineering-quality
```

- [ ] **Step 2: Update `pyproject.toml`**

Replace the `[project.optional-dependencies]` and append tool config. The final relevant sections:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0.0", "ruff>=0.6.0", "mypy>=1.11.0"]
download = ["huggingface_hub>=0.25.0"]

[tool.ruff]
line-length = 100
target-version = "py310"
extend-exclude = ["models", "outputs"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["E501"]

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
check_untyped_defs = false
warn_unused_ignores = false
files = ["src/paddleocr_vl_rocm"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 3: Delete `requirements.txt`**

```bash
git rm requirements.txt
```

- [ ] **Step 4: Install dev tooling**

```bash
pip install -e ".[dev]"
```

- [ ] **Step 5: Apply formatter as its own behavior-neutral commit**

```bash
ruff format src tests scripts
git add -A
git commit -m "style: apply ruff format to existing code (behavior-neutral)"
```

- [ ] **Step 6: Resolve any lint errors ruff surfaces**

Run: `ruff check src tests scripts`
Expected: PASS, or a small list of real issues (unused imports, etc.). Fix each in code; if an error is intentional, add a targeted `# noqa: <code>  # reason`. Re-run until `ruff check` is clean.

- [ ] **Step 7: Create `scripts/check.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
python -m compileall -q src
ruff check src tests scripts
ruff format --check src tests scripts
mypy src
python -m pytest -q
```

- [ ] **Step 8: Create `scripts/check.ps1`**

```powershell
$ErrorActionPreference = "Stop"
python -m compileall -q src
ruff check src tests scripts
ruff format --check src tests scripts
mypy src
python -m pytest -q
```

- [ ] **Step 9: Run the check and confirm green**

Run: `bash scripts/check.sh` (or `pwsh scripts/check.ps1`)
Expected: compileall OK, ruff clean, mypy reports no errors for `src`, all existing tests pass.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml scripts/check.sh scripts/check.ps1
git commit -m "build: add ruff/mypy/pytest config and local check scripts"
```

---

## Task 2: Characterization safety net

**Files:**
- Create: `scripts/record_trace.py`
- Create: `tests/fixtures/.gitkeep`, `tests/fixtures/golden/.gitkeep`
- Create: `tests/test_pipeline_characterization.py`

**Interfaces:**
- Produces: a record-replay harness. `scripts/record_trace.py` captures a `compat_cache.json` + golden JSON/MD for the example images by running the live pipeline once. `tests/test_pipeline_characterization.py` replays via the compat cache (no server) and asserts byte-identity with golden, skipping gracefully when fixtures or the layout model are absent.

**Server dependency:** Step 2 (record) requires the OpenAI-compatible VLM server running once. If it is unavailable, record later — the per-module unit tests added in Tasks 3–9 are server-free and are the primary refactor gate; this integration test is the secondary, comprehensive gate.

- [ ] **Step 1: Write `scripts/record_trace.py`**

```python
"""Record a VLM compat cache + golden outputs for characterization tests.

Run once against a live OpenAI-compatible VLM server. Produces:
  tests/fixtures/compat_cache.json
  tests/fixtures/golden/<stem>.json
  tests/fixtures/golden/<stem>.md

After recording, tests/test_pipeline_characterization.py replays without a server.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import paddleocr_vl_rocm.pipeline_core as core
from paddleocr_vl_rocm.pipeline_core import (
    _jpeg_bytes,
    _png_bytes,
    _sha256_hex,
    _vlm_cache_key,
    run_light_parser,
)

REPO = Path(__file__).resolve().parent.parent
IMAGES = sorted((REPO / "examples" / "input").glob("*.png"))
FIXTURES = REPO / "tests" / "fixtures"
GOLDEN = FIXTURES / "golden"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-model-name", default="PaddleOCR-VL-1.5-0.9B")
    parser.add_argument("--layout-model", default="models/PP-DocLayoutV3-onnx")
    args = parser.parse_args()

    FIXTURES.mkdir(parents=True, exist_ok=True)
    GOLDEN.mkdir(parents=True, exist_ok=True)

    recorded: dict[str, str] = {}
    original = core.OpenAICompatibleVLMClient.complete_image

    def recording(self, prompt, image=None, image_path=None, max_new_tokens=None,
                  min_pixels=None, max_pixels=None, use_client_cache=True):
        text = original(self, prompt, image=image, image_path=image_path,
                        max_new_tokens=max_new_tokens, min_pixels=min_pixels,
                        max_pixels=max_pixels, use_client_cache=use_client_cache)
        if image is not None:
            raw = _jpeg_bytes(image) if self.backend == "vllm-server" else _png_bytes(image)
        else:
            raw = image_path.read_bytes()
        key = _vlm_cache_key(self.model, prompt=prompt, image_sha256=_sha256_hex(raw),
                             max_new_tokens=max_new_tokens, seed=self.seed)
        recorded[key] = text
        return text

    core.OpenAICompatibleVLMClient.complete_image = recording  # type: ignore[assignment]
    try:
        for img in IMAGES:
            out_dir = FIXTURES / "_tmp_record"
            out_dir.mkdir(parents=True, exist_ok=True)
            json_path = run_light_parser(
                input_path=img,
                output_dir=out_dir,
                model_dir=Path(args.layout_model),
                server_url=args.server_url,
                vlm_backend="vllm-server",
                api_model_name=args.api_model_name,
                max_new_tokens=4096,
                timeout=300.0,
                prompt_label=None,
                use_layout_detection=True,
                use_chart_recognition=False,
                use_seal_recognition=False,
                seed=1,
                threshold=0.3,
                display_input_path=str(img),
                skip_server_check=False,
            )
            stem = img.stem
            (GOLDEN / f"{stem}.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
            md = out_dir / "result.md"
            if md.exists():
                (GOLDEN / f"{stem}.md").write_text(md.read_text(encoding="utf-8"), encoding="utf-8")
    finally:
        core.OpenAICompatibleVLMClient.complete_image = original  # type: ignore[assignment]
        (FIXTURES / "_tmp_record").mkdir(parents=True, exist_ok=True)

    (FIXTURES / "compat_cache.json").write_text(
        json.dumps({"entries": recorded}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Recorded {len(recorded)} VLM responses and {len(IMAGES)} golden outputs into {FIXTURES}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Record golden fixtures (requires the VLM server up once)**

Run: `python scripts/record_trace.py --server-url http://127.0.0.1:8000/v1`
Expected: prints `Recorded N VLM responses and 7 golden outputs into .../tests/fixtures`. Creates `tests/fixtures/compat_cache.json` and `tests/fixtures/golden/*.json` + `*.md`.

If the server is unavailable now, skip this step and return to it before Task 10's final verification; the rest of the plan proceeds using the unit tests.

- [ ] **Step 3: Write `tests/test_pipeline_characterization.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paddleocr_vl_rocm.pipeline_core import run_light_parser

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
GOLDEN = FIXTURES / "golden"
COMPAT = FIXTURES / "compat_cache.json"
LAYOUT_MODEL = REPO / "models" / "PP-DocLayoutV3-onnx"
IMAGES = sorted((REPO / "examples" / "input").glob("*.png"))


@pytest.fixture(autouse=True)
def _require_fixtures():
    if not COMPAT.exists() or not LAYOUT_MODEL.exists():
        pytest.skip("compat cache or layout model not present; run scripts/record_trace.py")


@pytest.mark.parametrize("image", IMAGES, ids=[p.stem for p in IMAGES])
def test_pipeline_matches_golden(tmp_path, image):
    json_path = run_light_parser(
        input_path=image,
        output_dir=tmp_path,
        model_dir=LAYOUT_MODEL,
        server_url="http://127.0.0.1:8000/v1",
        vlm_backend="vllm-server",
        api_model_name="PaddleOCR-VL-1.5-0.9B",
        max_new_tokens=4096,
        timeout=300.0,
        prompt_label=None,
        use_layout_detection=True,
        use_chart_recognition=False,
        use_seal_recognition=False,
        seed=1,
        threshold=0.3,
        compat_cache_path=COMPAT,
        display_input_path=str(image),
        skip_server_check=True,
    )
    actual = json.loads(json_path.read_text(encoding="utf-8"))
    expected = json.loads((GOLDEN / f"{image.stem}.json").read_text(encoding="utf-8"))
    assert actual == expected
```

- [ ] **Step 4: Run the test and confirm it passes (or skips cleanly)**

Run: `python -m pytest tests/test_pipeline_characterization.py -v`
Expected: if fixtures present → 7 PASSED; if absent → SKIPPED with the recorded reason.

- [ ] **Step 5: Commit the harness + fixtures**

```bash
git add scripts/record_trace.py tests/test_pipeline_characterization.py tests/fixtures
git commit -m "test: add record-replay characterization harness for refactor safety"
```

> Keep `tests/fixtures/_tmp_record/` out of git — add it to `.gitignore` (Task 10 cleanup). The `compat_cache.json` and `golden/` ARE committed (they are the regression oracle).

---

## Task 3: Extract leaf modules (constants, models, encoding, geometry, content)

**Files:**
- Create: `src/paddleocr_vl_rocm/constants.py`, `models.py`, `encoding.py`, `geometry.py`, `content.py`
- Modify: `src/paddleocr_vl_rocm/pipeline_core.py` (remove moved symbols, add imports)
- Test: `tests/test_encoding.py`, `tests/test_geometry.py`, `tests/test_content.py`

**Interfaces:**
- Produces: pure modules importable as `from .constants import ...`, `from .models import LightBlock, TableCell`, `from .encoding import _sha256_hex, _data_url_from_bytes, ...`, `from .geometry import _overlap_ratio, _area, _filter_overlap_boxes, ...`, `from .content import _truncate_repetitive_content, _has_cjk, _normalize_vlm_result, _format_block_content, ...`. `pipeline_core.py` imports these instead of defining them.

**Move list (move verbatim — preserve exact code, only relocate):**

- `constants.py`: from `pipeline_core.py` — `IMAGE_LABELS` (line ~28), `NON_RECOGNIZED_IMAGE_LABELS` (~29), `NON_MERGE_LABELS` (~30), `SKIP_ORDER_LABELS` (~31–43), `MARKDOWN_IGNORE_LABELS` (~44–52), `MARKDOWN_IGNORE_LABEL_SET` (~53), `DEFAULT_MIN_PIXELS` (~54), `DEFAULT_MAX_PIXELS` (~55), `PROMPTS` (~56–61).
- `models.py`: `LightBlock` (~75–86), `TableCell` (~89–97). Add `from __future__ import annotations`; keep `field` import from dataclasses; `LightBlock` uses `PIL.Image` for the type hint — import it.
- `encoding.py`: `_png_bytes` (~100), `_jpeg_bytes` (~106), `_data_url_from_bytes` (~112), `_encode_png_data_url` (~117), `_image_data_url` (~122), `_sha256_hex` (~129). Imports: `base64`, `hashlib`, `mimetypes`, `io.BytesIO`, `pathlib.Path`, `PIL.Image`.
- `geometry.py`: `_area` (~556), `_overlap_ratio` (~560), `_projection_overlap_ratio` (~577), `_filter_overlap_boxes` (~597), `_polygon_overlap_ratio` (~635). For type hints only, `from .layout import LayoutBox` (layout does not import geometry → no cycle).
- `content.py`: `_has_cjk` (~1049), `_should_keep_text_newlines` (~1053), `_format_block_content` (~1067), `_normalize_vlm_result` (~1077), `_find_shortest_repeating_substring` (~1094), `_find_repeating_suffix` (~1103), `_truncate_repetitive_content` (~1116). Imports: `re`, `collections.Counter`.

- [ ] **Step 1: Write failing unit tests**

`tests/test_encoding.py`:
```python
from paddleocr_vl_rocm.encoding import _data_url_from_bytes, _sha256_hex


def test_data_url_encoding():
    assert _data_url_from_bytes(b"hi", "image/png") == "data:image/png;base64,aGk="


def test_sha256_known_empty():
    assert _sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
```

`tests/test_geometry.py`:
```python
from paddleocr_vl_rocm.geometry import _area, _overlap_ratio


def test_area_positive_and_zero():
    assert _area([0, 0, 5, 5]) == 25.0
    assert _area([5, 5, 0, 0]) == 0.0


def test_overlap_ratio_small_and_union():
    big = [0, 0, 10, 10]
    small = [2, 2, 8, 8]
    assert _overlap_ratio(big, small, mode="small") == 1.0
    assert abs(_overlap_ratio(big, small, mode="union") - 0.36) < 1e-9
```

`tests/test_content.py`:
```python
from paddleocr_vl_rocm.content import _has_cjk, _truncate_repetitive_content, _normalize_vlm_result


def test_has_cjk():
    assert _has_cjk("中文") is True
    assert _has_cjk("english") is False


def test_truncate_repetitive_lines():
    content = "\n".join(["same"] * 50)
    assert _truncate_repetitive_content(content) == "same"


def test_normalize_inline_formula_to_dollars():
    out = _normalize_vlm_result("inline", r"\(x+1\) and \(y\)")
    assert "$" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_encoding.py tests/test_geometry.py tests/test_content.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paddleocr_vl_rocm.encoding'` (etc.).

- [ ] **Step 3: Create the five modules by moving the code verbatim**

Create each file with the exact code from the move list above (copy the function bodies unchanged). Each starts with `from __future__ import annotations` and the imports noted. Do not yet delete from `pipeline_core.py`.

- [ ] **Step 4: Update `pipeline_core.py` imports**

At the top of `pipeline_core.py`, replace the now-relocated definitions with imports. Remove the moved definitions and add:

```python
from .constants import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    IMAGE_LABELS,
    MARKDOWN_IGNORE_LABELS,
    MARKDOWN_IGNORE_LABEL_SET,
    NON_MERGE_LABELS,
    NON_RECOGNIZED_IMAGE_LABELS,
    PROMPTS,
    SKIP_ORDER_LABELS,
)
from .content import (
    _format_block_content,
    _has_cjk,
    _normalize_vlm_result,
    _should_keep_text_newlines,
    _truncate_repetitive_content,
)
from .encoding import (
    _data_url_from_bytes,
    _encode_png_data_url,
    _image_data_url,
    _jpeg_bytes,
    _png_bytes,
    _sha256_hex,
)
from .geometry import _filter_overlap_boxes, _overlap_ratio, _polygon_overlap_ratio
from .models import LightBlock, TableCell
```

Keep `OTSL_*` constants in `pipeline_core.py` for now (they move in Task 5 with the table code).

- [ ] **Step 5: Run the full check**

Run: `bash scripts/check.sh`
Expected: green — ruff/mypy clean, all unit tests pass, characterization integration test passes (or skips).

- [ ] **Step 6: Commit**

```bash
git add src tests
git commit -m "refactor: extract constants/models/encoding/geometry/content from pipeline_core"
```

---

## Task 4: Extract `imageio.py`

**Files:**
- Create: `src/paddleocr_vl_rocm/imageio.py`
- Modify: `src/paddleocr_vl_rocm/pipeline_core.py`
- Test: `tests/test_imageio.py`

**Interfaces:**
- Produces: `from .imageio import _crop, _crop_from_bgr, _open_crop_source, _open_crop_source_bgr, _crop_margin, _merge_images`.

**Move list (verbatim):** `_crop` (~346), `_crop_from_bgr` (~366), `_open_crop_source` (~386), `_open_crop_source_bgr` (~402), `_crop_margin` (~413), `_merge_images` (~665). Imports: `cv2`/`numpy` lazy-imported inside functions (keep as-is), `PIL.Image`, `pathlib.Path`; `from .layout import LayoutBox` for `_crop`/`_crop_from_bgr` type hints.

- [ ] **Step 1: Write failing test `tests/test_imageio.py`**

```python
from PIL import Image

from paddleocr_vl_rocm.imageio import _crop_margin, _merge_images


def test_merge_images_stacks_vertically():
    a = Image.new("RGB", (10, 4), (1, 1, 1))
    b = Image.new("RGB", (10, 6), (2, 2, 2))
    merged = _merge_images([a, b], ["center"])
    assert merged.size == (10, 10)


def test_crop_margin_passthrough_on_uniform():
    img = Image.new("RGB", (20, 20), (255, 255, 255))
    assert _crop_margin(img).size == (20, 20)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_imageio.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Create `imageio.py` with moved code**

Move the six functions verbatim. Add `from __future__ import annotations` and the imports above.

- [ ] **Step 4: Update `pipeline_core.py`**

Remove the six definitions; add `from .imageio import _crop, _crop_from_bgr, _open_crop_source, _open_crop_source_bgr, _crop_margin, _merge_images`.

- [ ] **Step 5: Run full check**

Run: `bash scripts/check.sh`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src tests
git commit -m "refactor: extract imageio (read/crop/mask/margin/merge) from pipeline_core"
```

---

## Task 5: Extract `table.py` (OTSL→HTML)

**Files:**
- Create: `src/paddleocr_vl_rocm/table.py`
- Modify: `src/paddleocr_vl_rocm/pipeline_core.py`, `src/paddleocr_vl_rocm/models.py` (already has `TableCell`)
- Test: `tests/test_table.py`

**Interfaces:**
- Produces: `from .table import _convert_otsl_to_html`. `TableCell` lives in `models.py` (already there from Task 3); `table.py` imports it.

**Move list (verbatim):** OTSL constants `OTSL_NL`, `OTSL_FCEL`, `OTSL_ECEL`, `OTSL_LCEL`, `OTSL_UCEL`, `OTSL_XCEL`, `OTSL_TAGS`, `OTSL_FIND_PATTERN` (~62–72), `_otsl_extract_tokens_and_text` (~890), `_otsl_pad_to_square` (~897), `_otsl_parse_texts` (~934), `_convert_otsl_to_html` (~1013). Imports: `re`, `itertools`, `html`; `from .models import TableCell`.

- [ ] **Step 1: Write failing test `tests/test_table.py`**

```python
from paddleocr_vl_rocm.table import _convert_otsl_to_html


def test_otsl_to_html_basic_table():
    otsl = "<fcel>A<ecel><nl><fcel>B<ecel><nl>"
    html = _convert_otsl_to_html(otsl)
    assert html.startswith("<table>") and html.endswith("</table>")
    assert "A" in html and "B" in html
    assert html.count("<tr>") == 2


def test_otsl_to_html_trivial_input_still_returns_table_wrapper():
    # Empty input is padded and wrapped (does not crash, does not return "")
    out = _convert_otsl_to_html("")
    assert out.startswith("<table>") and out.endswith("</table>")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_table.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Create `table.py` with moved code**

Move the OTSL constants and four functions verbatim. `from .models import TableCell`.

- [ ] **Step 4: Update `pipeline_core.py`**

Remove the OTSL constants + four functions; add `from .table import _convert_otsl_to_html`.

- [ ] **Step 5: Run full check**

Run: `bash scripts/check.sh`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src tests
git commit -m "refactor: extract OTSL->HTML table conversion from pipeline_core"
```

---

## Task 6: Extract `markdown.py`

**Files:**
- Create: `src/paddleocr_vl_rocm/markdown.py`
- Modify: `src/paddleocr_vl_rocm/pipeline_core.py`
- Test: `tests/test_markdown.py`

**Interfaces:**
- Produces: `from .markdown import _markdown_from_blocks, _markdown_content_for_block`.

**Move list (verbatim):** `_markdown_from_blocks` (~833), `_collapse_soft_newlines` (~848), `_normalize_markdown_newlines` (~852), `_format_title_text` (~856), `_markdown_content_for_block` (~865). Imports: `re`; `from .constants import MARKDOWN_IGNORE_LABEL_SET`; `from .models import LightBlock`.

- [ ] **Step 1: Write failing test `tests/test_markdown.py`**

```python
from paddleocr_vl_rocm.markdown import _collapse_soft_newlines, _format_title_text
from paddleocr_vl_rocm.models import LightBlock


def test_collapse_soft_newlines():
    assert _collapse_soft_newlines("hyphen-\nated") == "hyphenated"
    assert _collapse_soft_newlines("a\nb") == "a b"


def test_format_title_text_heading_levels():
    assert _format_title_text("1.2.3 Results").startswith("#### ")
    assert _format_title_text("Introduction").startswith("## ")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_markdown.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Create `markdown.py` with moved code**

Move the five functions verbatim with the imports above.

- [ ] **Step 4: Update `pipeline_core.py`**

Remove the five definitions; add `from .markdown import _markdown_from_blocks, _markdown_content_for_block`.

- [ ] **Step 5: Run full check**

Run: `bash scripts/check.sh`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src tests
git commit -m "refactor: extract markdown serialization from pipeline_core"
```

---

## Task 7: Extract `vlm/client.py`

**Files:**
- Create: `src/paddleocr_vl_rocm/vlm/__init__.py`, `src/paddleocr_vl_rocm/vlm/client.py`
- Delete: `src/paddleocr_vl_rocm/vlm.py`
- Modify: `src/paddleocr_vl_rocm/pipeline_core.py`, `tests/test_vlm_payload.py`
- Test: `tests/test_vlm_payload.py` (update import)

**Interfaces:**
- Produces: `from .vlm.client import OpenAICompatibleVLMClient, LlamaCppClient, _completion_payload, _content_from_response, _vlm_cache_key, _load_vlm_compat_cache, _prompt_for_label`. `vlm/__init__.py` re-exports `OpenAICompatibleVLMClient`, `LlamaCppClient` (preserving `from paddleocr_vl_rocm.vlm import OpenAICompatibleVLMClient`).

**Move list (verbatim):** `_vlm_cache_key` (~133), `_load_vlm_compat_cache` (~154), `_content_from_response` (~170), `_completion_payload` (~181), class `OpenAICompatibleVLMClient` (~233), `LlamaCppClient = OpenAICompatibleVLMClient` (~331), `_prompt_for_label` (~334). Imports: `requests`, `PIL.Image`, `time`, `base64`, `json`, `pathlib.Path`, `typing.Any`; `from ..encoding import _data_url_from_bytes, _jpeg_bytes, _png_bytes, _sha256_hex`; `from ..constants import IMAGE_LABELS, PROMPTS`; `from ..server import check_openai_compatible_server, normalize_server_url`.

- [ ] **Step 1: Convert `vlm.py` → package `vlm/`**

`git mv src/paddleocr_vl_rocm/vlm.py src/paddleocr_vl_rocm/vlm/client.py` then create `vlm/__init__.py`:

```python
from __future__ import annotations

from .client import LlamaCppClient, OpenAICompatibleVLMClient

__all__ = ["OpenAICompatibleVLMClient", "LlamaCppClient"]
```

- [ ] **Step 2: Move the VLM symbols into `vlm/client.py`**

`vlm/client.py` keeps only the moved symbols (remove what Task 3 already extracted to `encoding`/`constants`; import them relatively as above). Update internal references inside the client that used the old private helpers to the imported names.

- [ ] **Step 3: Update `pipeline_core.py`**

Remove the moved VLM symbols; add:
```python
from .vlm.client import (
    LlamaCppClient,
    OpenAICompatibleVLMClient,
    _completion_payload,
    _content_from_response,
    _load_vlm_compat_cache,
    _prompt_for_label,
    _vlm_cache_key,
)
```

- [ ] **Step 4: Update `tests/test_vlm_payload.py` import**

Change:
```python
from paddleocr_vl_rocm.pipeline_core import _completion_payload
```
to:
```python
from paddleocr_vl_rocm.vlm.client import _completion_payload
```

- [ ] **Step 5: Run full check**

Run: `bash scripts/check.sh`
Expected: green, `test_vlm_payload` passes from the new location.

- [ ] **Step 6: Commit**

```bash
git add src tests
git commit -m "refactor: extract VLM client + payload into vlm/ package"
```

---

## Task 8: Extract `preprocess.py`

**Files:**
- Create: `src/paddleocr_vl_rocm/preprocess.py`
- Modify: `src/paddleocr_vl_rocm/pipeline_core.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Produces: `from .preprocess import _make_blocks, _merge_blocks, _filter_overlap_boxes_passthrough, _gather_imgs_for_table_tokens, _construct_img_path, _tokenize_figure_of_table, _untokenize_figure_of_table, _paint_token`.

**Move list (verbatim):** `_construct_img_path` (~436), `_gather_imgs_for_table_tokens` (~441), `_paint_token` (~457), `_tokenize_figure_of_table` (~496), `_untokenize_figure_of_table` (~544), `_make_blocks` (~688), `_merge_blocks` (~708). Imports: `random`, `math`, `re`, `cv2`/`numpy` lazy, `PIL.Image`, `typing.Any`; `from .constants import IMAGE_LABELS, NON_MERGE_LABELS`; `from .models import LightBlock`; `from .layout import LayoutBox`; `from .imageio import _crop, _crop_from_bgr, _merge_images`; `from .geometry import _overlap_ratio, _projection_overlap_ratio`.

- [ ] **Step 1: Write failing test `tests/test_preprocess.py`**

```python
from paddleocr_vl_rocm.preprocess import _construct_img_path


def test_construct_img_path_format():
    path = _construct_img_path("image", [10, 20, 30, 40])
    assert path == "imgs/img_in_image_box_10_20_30_40.jpg"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Create `preprocess.py` with moved code**

Move the seven functions verbatim with the imports above.

- [ ] **Step 4: Update `pipeline_core.py`**

Remove the seven definitions; add:
```python
from .preprocess import (
    _construct_img_path,
    _gather_imgs_for_table_tokens,
    _make_blocks,
    _merge_blocks,
    _paint_token,
    _tokenize_figure_of_table,
    _untokenize_figure_of_table,
)
```

- [ ] **Step 5: Run full check**

Run: `bash scripts/check.sh`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src tests
git commit -m "refactor: extract preprocess (blocks/merge/figure-tokenization) from pipeline_core"
```

---

## Task 9: Extract `serialize.py`

**Files:**
- Create: `src/paddleocr_vl_rocm/serialize.py`
- Modify: `src/paddleocr_vl_rocm/pipeline_core.py`
- Test: `tests/test_serialize.py`

**Interfaces:**
- Produces: `from .serialize import _result_payload, _layout_boxes_to_json, _blocks_with_orders, _block_to_json`.

**Move list (verbatim):** `_result_payload` (~1151), `_layout_boxes_to_json` (~1191), `_blocks_with_orders` (~1205), `_block_to_json` (~1219). Imports: `typing.Any`; `from .constants import MARKDOWN_IGNORE_LABELS, MARKDOWN_IGNORE_LABEL_SET, SKIP_ORDER_LABELS`; `from .models import LightBlock`; `from .layout import LayoutBox`.

- [ ] **Step 1: Write failing test `tests/test_serialize.py`**

```python
from paddleocr_vl_rocm.models import LightBlock
from paddleocr_vl_rocm.serialize import _block_to_json


def test_block_to_json_shape():
    block = LightBlock(label="text", content="hello", bbox=[0, 0, 10, 10], score=0.9, cls_id=23)
    out = _block_to_json(block, idx=0, order=1)
    assert out["block_label"] == "text"
    assert out["block_content"] == "hello"
    assert out["block_bbox"] == [0, 0, 10, 10]
    assert out["block_id"] == 0
    assert out["block_order"] == 1
    assert out["block_polygon_points"][0] == [0.0, 0.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_serialize.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Create `serialize.py` with moved code**

Move the four functions verbatim with the imports above.

- [ ] **Step 4: Update `pipeline_core.py`**

Remove the four definitions; add:
```python
from .serialize import _blocks_with_orders, _block_to_json, _layout_boxes_to_json, _result_payload
```

- [ ] **Step 5: Run full check**

Run: `bash scripts/check.sh`
Expected: green. `pipeline_core.py` should now contain essentially only `run_light_parser` plus imports.

- [ ] **Step 6: Commit**

```bash
git add src tests
git commit -m "refactor: extract JSON payload serialization from pipeline_core"
```

---

## Task 10: Slim `pipeline_core.py`, logging, README, final verification

**Files:**
- Modify: `src/paddleocr_vl_rocm/pipeline_core.py` (final import cleanup), `src/paddleocr_vl_rocm/utils.py` (logger helper), `src/paddleocr_vl_rocm/vlm/client.py` (log retries), `.gitignore`, `README.md`, `README.zh-CN.md`
- Test: existing suite (no new test; verification task)

**Interfaces:**
- Produces: `pipeline_core.py` reduced to `run_light_parser` orchestration + imports. A `utils.get_logger()` helper. VLM retries emit `logging.debug`. README documents the dev workflow.

- [ ] **Step 1: Add a logger helper to `utils.py`**

Append to `src/paddleocr_vl_rocm/utils.py`:
```python
import logging


def get_logger(name: str = "paddleocr_vl_rocm") -> logging.Logger:
    return logging.getLogger(name)
```

- [ ] **Step 2: Log VLM retries in `vlm/client.py`**

In `OpenAICompatibleVLMClient.complete_image`, around the retry loop, add at the top of `vlm/client.py`:
```python
from ..utils import get_logger

_logger = get_logger(__name__)
```
and inside the retry loop where it sleeps on 429/5xx, add:
```python
_logger.warning("VLM request %s (attempt %d), retrying in %.1fs", response.status_code, attempt + 1, 1.5 * (attempt + 1))
```
and in the `except requests.RequestException` branch before sleeping:
```python
_logger.warning("VLM request error (attempt %d): %s", attempt + 1, exc)
```
Do not change retry counts, backoff, or return values.

- [ ] **Step 3: Confirm `pipeline_core.py` is now slim**

Open `src/paddleocr_vl_rocm/pipeline_core.py`. It should contain only: module imports, the `run_light_parser` function, and nothing else substantive (all helpers now imported). Remove any now-unused top-level imports flagged by `ruff check` (e.g., `html`, `itertools`, `Counter`, `random` if no longer referenced). Run `ruff check src` and remove unused imports it reports.

- [ ] **Step 4: Update `.gitignore`**

Append:
```
tests/fixtures/_tmp_record/
```

- [ ] **Step 5: Add a Development section to `README.md` (after the Tests section)**

```markdown
## Development

Install with dev tooling and run the full local check:

```powershell
pip install -e .[dev]
./scripts/check.ps1   # Linux/macOS: bash scripts/check.sh
```

The check runs `compileall`, `ruff check`, `ruff format --check`, `mypy src`, and `pytest`.

To establish the characterization fixtures (requires the VLM server once):

```powershell
python scripts/record_trace.py --server-url http://127.0.0.1:8000/v1
```

This records `tests/fixtures/compat_cache.json` and golden outputs so `tests/test_pipeline_characterization.py` can replay the pipeline byte-for-byte without a server. The test skips automatically if fixtures or the layout model are absent.
```

Mirror the equivalent section in `README.zh-CN.md`.

- [ ] **Step 6: Run the full check one final time**

Run: `bash scripts/check.sh`
Expected: fully green — compileall, ruff (check + format), mypy, and all tests including the 7 characterization cases (if fixtures recorded) passing.

- [ ] **Step 7: Commit**

```bash
git add src .gitignore README.md README.zh-CN.md
git commit -m "refactor: slim pipeline_core to orchestration, add logging, document dev workflow"
```

---

## Notes for the implementer

- **Behavior is sacred.** If a characterization test (integration or unit) fails after a move, the move changed behavior — fix the move, do not "fix" the test. Tests added in this plan lock the *current* behavior on purpose.
- **One module per commit.** Keep the per-task commits; they make review and bisection trivial.
- **Line numbers are approximate** (the formatter ran in Task 1, shifting lines). Locate symbols by name, not by exact line.
- **`from __future__ import annotations`** goes at the top of every new module to match the existing style and keep type hints cheap.
- After Task 10, the branch is ready for Plan B (OmniDocBench v1.5/v1.6 evaluation chain), which will re-use this stable, official-compatible public API.
