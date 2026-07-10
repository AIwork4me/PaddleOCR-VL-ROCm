# Local Reference Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade PaddleOCR-VL-ROCm's local OmniDocBench workflow so accuracy, quality, speed, parameters, and score claims are backed by this machine's Windows + AMD + llama.cpp/GGUF environment.

**Architecture:** Keep the lightweight pipeline as the core implementation, add a local official/reference adapter mode beside it, and make `eval/run_eval.py` route all inference through the adapter's public `run_adapter` contract. Unit tests cover adapter behavior without live PaddleOCR or VLM services; live OmniDocBench runs remain explicit local verification steps.

**Tech Stack:** Python 3.10+, pytest, ONNXRuntime, PaddleOCR optional official engine, OpenAI-compatible llama.cpp/GGUF VLM server, OmniDocBench v1.6 local checkout.

## Global Constraints

- Validation is local-only: Windows + AMD + llama.cpp/GGUF + this machine's OmniDocBench/CDM environment.
- Do not set up Linux vLLM, BF16, SGLang, FastDeploy, Docker inference, or any cross-machine reference path.
- Do not rewrite ground truth or tune benchmark scores.
- Keep official PaddleOCR imports lazy so `--help` and unit tests work when PaddleOCR is not installed.
- Do not commit `eval/.omnidocbench/`, `logs/`, generated predictions, or other untracked local eval artifacts unless explicitly requested.
- Use TDD: write each behavior test, watch it fail for the expected reason, then implement the minimum code.

---

## File Structure

- Modify `eval/PaddleOCRVLROCm_img2md.py`: production OmniDocBench adapter with `lightweight` and `official` engines, lazy imports, `.env.local` default resolution, persistent run stats, persistent errors, retry, and fallback prediction support.
- Modify `eval/run_eval.py`: add CLI arguments for engine, backend, retries, fallback predictions, and v1.6 local defaults; call `run_adapter`.
- Modify `tests/test_eval_adapter.py`: server-free adapter tests for naming, lazy import helpers, official Markdown export, HTML normalization, default resolution, run stats, and error logging.
- Modify `tests/test_eval_report_path.py`: add orchestrator dispatch tests without contacting a live server.
- Modify `tests/test_vlm_payload.py`: add llama.cpp payload parameter coverage and keep vLLM min/max pixel coverage.
- Modify `README.md`, `README.zh-CN.md`, and `eval/README.md`: document local-only score posture, engine-specific evaluation, and exact local verification commands.
- Optionally create `results/omnidocbench/v16/README.md`: if fresh score artifacts are not regenerated during this pass, record which existing artifacts are historical and which command regenerates them.

---

### Task 1: Adapter Contract Tests

**Files:**
- Modify: `tests/test_eval_adapter.py`
- Target implementation: `eval/PaddleOCRVLROCm_img2md.py`

**Interfaces:**
- Consumes: existing `expected_md_name(image_name: str) -> str`
- Produces: tests for `run_adapter(...) -> dict`, `_official_result_to_markdown(result: object) -> str`, `_normalize_official_markdown_for_omnidocbench(markdown: str) -> str`, `_read_env_local(repo_root: Path) -> dict[str, str]`

- [ ] **Step 1: Write failing tests for official Markdown export and HTML cleanup**

Append this to `tests/test_eval_adapter.py`:

```python
def test_official_result_prefers_plain_markdown_export():
    mod = _load_adapter()

    class FakeOfficialResult:
        def __init__(self):
            self.calls = []
            self.markdown = "<div>pretty markdown</div>"

        def _to_markdown(self, pretty=True):
            self.calls.append(pretty)
            return {"markdown_texts": "plain markdown"}

    result = FakeOfficialResult()

    assert mod._official_result_to_markdown(result) == "plain markdown"
    assert result.calls == [False]


def test_official_markdown_html_wrappers_become_scorer_friendly_markdown():
    mod = _load_adapter()
    markdown = (
        '<div style="text-align: center;"><img src="imgs/table_1.jpg"></div>\n\n'
        '<div style="text-align: center;"><b>Figure 1 &amp; caption</b></div>'
    )

    assert mod._normalize_official_markdown_for_omnidocbench(markdown) == (
        "![](imgs/table_1.jpg)\n\nFigure 1 & caption"
    )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_eval_adapter.py::test_official_result_prefers_plain_markdown_export tests/test_eval_adapter.py::test_official_markdown_html_wrappers_become_scorer_friendly_markdown -q
```

Expected: FAIL because `_official_result_to_markdown` and `_normalize_official_markdown_for_omnidocbench` do not exist yet.

- [ ] **Step 3: Write failing tests for `.env.local` defaults and adapter dispatch**

Append this to `tests/test_eval_adapter.py`:

```python
def test_run_adapter_resolves_defaults_from_env_local(tmp_path, monkeypatch):
    mod = _load_adapter()
    adapter_dir = tmp_path / "adapter"
    repo_root = tmp_path / "repo"
    adapter_dir.mkdir()
    repo_root.mkdir()
    (adapter_dir / ".env.local").write_text(
        "\n".join(
            [
                "PP_DOCLAYOUTV3_ONNX_DIR='C:/models/layout'",
                "LLAMA_HOST='127.0.0.1'",
                "LLAMA_PORT='8111'",
                "VL_REC_API_MODEL_NAME='PaddleOCR-VL-1.6-GGUF.gguf'",
            ]
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_lightweight_folder(**kwargs):
        captured.update(kwargs)
        return {"count": 0, "ok": 0, "fail": 0, "engine": "lightweight", "stats": []}

    monkeypatch.setattr(mod, "ADAPTER_DIR", adapter_dir, raising=False)
    monkeypatch.setattr(mod, "REPO_ROOT", repo_root, raising=False)
    monkeypatch.setattr(mod, "run_lightweight_folder", fake_lightweight_folder, raising=False)

    summary = mod.run_adapter(tmp_path / "images", tmp_path / "predictions", "")

    assert summary["engine"] == "lightweight"
    assert captured["layout_model"] == "C:/models/layout"
    assert captured["server_url"] == "http://127.0.0.1:8111/v1"
    assert captured["api_model_name"] == "PaddleOCR-VL-1.6-GGUF.gguf"


def test_run_adapter_dispatches_official_engine(tmp_path, monkeypatch):
    mod = _load_adapter()
    captured = {}

    def fake_official_folder(**kwargs):
        captured.update(kwargs)
        return {"count": 1, "ok": 1, "fail": 0, "fallback": 0, "engine": "official", "stats": []}

    monkeypatch.setattr(mod, "run_official_folder", fake_official_folder, raising=False)

    summary = mod.run_adapter(
        tmp_path / "images",
        tmp_path / "predictions",
        "http://127.0.0.1:8111/v1",
        engine="official",
        page_retries=2,
        fallback_pred_dir=tmp_path / "fallback",
    )

    assert summary["engine"] == "official"
    assert captured["server_url"] == "http://127.0.0.1:8111/v1"
    assert captured["page_retries"] == 2
    assert captured["fallback_pred_dir"] == tmp_path / "fallback"
```

- [ ] **Step 4: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_eval_adapter.py::test_run_adapter_resolves_defaults_from_env_local tests/test_eval_adapter.py::test_run_adapter_dispatches_official_engine -q
```

Expected: FAIL because `run_adapter`, `ADAPTER_DIR`, `REPO_ROOT`, and the engine dispatch helpers do not exist yet.

- [ ] **Step 5: Write failing tests for persistent stats and error logs**

Append this to `tests/test_eval_adapter.py`:

```python
def test_lightweight_folder_writes_run_stats_and_error_log(tmp_path, monkeypatch):
    mod = _load_adapter()
    img_dir = tmp_path / "images"
    out_dir = tmp_path / "predictions"
    img_dir.mkdir()
    (img_dir / "ok.png").write_bytes(b"fake image")
    (img_dir / "bad.png").write_bytes(b"fake image")

    class FakeResult:
        markdown_text = "recognized"

    class FakePipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, image_path):
            if image_path.name == "bad.png":
                raise RuntimeError("controlled failure")
            return FakeResult()

    monkeypatch.setattr(mod, "PaddleOCRVLROCm", FakePipeline, raising=False)

    summary = mod.run_lightweight_folder(
        img_dir=img_dir,
        out_dir=out_dir,
        layout_model="layout",
        server_url="http://127.0.0.1:8111/v1",
        api_model_name="PaddleOCR-VL-1.6-GGUF.gguf",
        vlm_backend="llama-cpp-server",
    )

    assert summary["count"] == 2
    assert summary["ok"] == 1
    assert summary["fail"] == 1
    assert (out_dir / "ok.md").read_text(encoding="utf-8") == "recognized"
    assert "controlled failure" in (out_dir / "_errors.log").read_text(encoding="utf-8")
    assert '"engine": "lightweight"' in (out_dir / "_run_stats.json").read_text(encoding="utf-8")
```

- [ ] **Step 6: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_eval_adapter.py::test_lightweight_folder_writes_run_stats_and_error_log -q
```

Expected: FAIL because `run_lightweight_folder` does not exist yet or does not persist stats and errors.

- [ ] **Step 7: Commit only the failing tests**

Do not commit production code in this step.

```powershell
git add tests/test_eval_adapter.py
git commit -m "test: specify local eval adapter contract"
```

---

### Task 2: Production Adapter Migration

**Files:**
- Modify: `eval/PaddleOCRVLROCm_img2md.py`
- Test: `tests/test_eval_adapter.py`

**Interfaces:**
- Consumes: tests from Task 1
- Produces:
  - `run_adapter(img_dir, out_dir, server_url="", *, engine="lightweight", layout_model=None, api_model_name=None, vlm_backend="vllm-server", page_retries=1, fallback_pred_dir=None) -> dict`
  - `run_lightweight_folder(...) -> dict`
  - `run_official_folder(...) -> dict`
  - `_official_result_to_markdown(result: object) -> str`
  - `_normalize_official_markdown_for_omnidocbench(markdown: str) -> str`

- [ ] **Step 1: Implement lazy imports, environment parsing, and engine dispatch**

In `eval/PaddleOCRVLROCm_img2md.py`, add these imports and constants near the top:

```python
import html
import json
import os
import re
import shutil
import traceback

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
DEFAULT_ENGINE = "lightweight"
DEFAULT_LOCAL_API_MODEL_NAME = "PaddleOCR-VL-1.6-GGUF.gguf"
```

Add these functions below `expected_md_name`:

```python
def _read_env_local(repo_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    env_file = repo_root / ".env.local"
    if not env_file.is_file():
        return values
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _read_adapter_env() -> dict[str, str]:
    root_values = _read_env_local(REPO_ROOT)
    adapter_values = _read_env_local(ADAPTER_DIR)
    return {**root_values, **adapter_values}
```

Add `run_adapter`:

```python
def run_adapter(
    img_dir,
    out_dir,
    server_url: str = "",
    *,
    engine: str = DEFAULT_ENGINE,
    layout_model: str | None = None,
    api_model_name: str | None = None,
    vlm_backend: str = "vllm-server",
    page_retries: int = 1,
    fallback_pred_dir: str | Path | None = None,
) -> dict:
    env = _read_adapter_env()
    default_layout = (
        layout_model
        or os.environ.get("ADAPTER_LAYOUT_MODEL")
        or env.get("PP_DOCLAYOUTV3_ONNX_DIR")
        or "models/PP-DocLayoutV3-onnx"
    )
    llama_host = env.get("LLAMA_HOST") or "127.0.0.1"
    llama_port = env.get("LLAMA_PORT") or "8111"
    resolved_server = (
        server_url
        or os.environ.get("ADAPTER_SERVER_URL")
        or f"http://{llama_host}:{llama_port}/v1"
    )
    default_api_model = (
        api_model_name
        or os.environ.get("ADAPTER_API_MODEL_NAME")
        or env.get("VL_REC_API_MODEL_NAME")
        or DEFAULT_LOCAL_API_MODEL_NAME
    )

    selected_engine = (engine or DEFAULT_ENGINE).strip().lower()
    if selected_engine == "lightweight":
        return run_lightweight_folder(
            img_dir=Path(img_dir),
            out_dir=Path(out_dir),
            layout_model=default_layout,
            server_url=resolved_server,
            api_model_name=default_api_model,
            vlm_backend=vlm_backend,
        )
    if selected_engine == "official":
        return run_official_folder(
            img_dir=Path(img_dir),
            out_dir=Path(out_dir),
            server_url=resolved_server,
            api_model_name=default_api_model,
            page_retries=page_retries,
            fallback_pred_dir=Path(fallback_pred_dir) if fallback_pred_dir else None,
        )
    raise ValueError(f"Unsupported engine '{engine}'. Use lightweight or official.")
```

- [ ] **Step 2: Convert existing `process_folder` into `run_lightweight_folder`**

Rename `process_folder` to `run_lightweight_folder` and keep a compatibility wrapper:

```python
def process_folder(
    img_dir: Path,
    out_dir: Path,
    *,
    layout_model: str = "models/PP-DocLayoutV3-onnx",
    server_url: str = "http://127.0.0.1:8000/v1",
    api_model_name: str = DEFAULT_LOCAL_API_MODEL_NAME,
    vlm_backend: str = "vllm-server",
) -> dict:
    return run_lightweight_folder(
        img_dir=img_dir,
        out_dir=out_dir,
        layout_model=layout_model,
        server_url=server_url,
        api_model_name=api_model_name,
        vlm_backend=vlm_backend,
    )
```

Update `run_lightweight_folder` so it lazily imports `PaddleOCRVLROCm`, writes `_errors.log`, writes `_run_stats.json`, returns `fail`, and exits 2 when fewer than half the pages succeed:

```python
try:
    PipelineClass = PaddleOCRVLROCm  # type: ignore[name-defined]
except NameError:
    from paddleocr_vl_rocm import PaddleOCRVLROCm as PipelineClass
pipeline = PipelineClass(
    layout_model_dir=layout_model,
    vlm_server_url=server_url,
    api_model_name=api_model_name,
    vlm_backend=vlm_backend,
)
```

Inside the exception branch, write:

```python
tb = traceback.format_exc()
with open(out_dir / "_errors.log", "a", encoding="utf-8") as fh:
    fh.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {img.name}: {exc}\n{tb}\n")
stats.append(
    {
        "image": img.name,
        "status": f"failed: {exc}",
        "seconds": round(time.time() - start, 2),
        "traceback": tb,
    }
)
```

After the loop, write:

```python
ok_count = sum(1 for s in stats if s["status"] == "ok")
summary = {
    "count": len(images),
    "ok": ok_count,
    "fail": len(images) - ok_count,
    "engine": "lightweight",
    "stats": stats,
}
(out_dir / "_run_stats.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
if len(images) > 0 and ok_count < 0.5 * len(images):
    raise SystemExit(2)
return summary
```

- [ ] **Step 3: Implement official Markdown helpers**

Add:

```python
def _official_result_to_markdown(result: object) -> str:
    def markdown_from_mapping(value: dict) -> str | None:
        for key in ("markdown_texts", "markdown", "md", "content", "markdown_text", "text"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        return None

    if isinstance(result, str):
        return result
    official_export = getattr(result, "_to_markdown", None)
    if callable(official_export):
        try:
            exported = official_export(pretty=False)
        except TypeError:
            exported = None
        if isinstance(exported, dict):
            mapped = markdown_from_mapping(exported)
            if mapped is not None:
                return mapped
        if isinstance(exported, str):
            return exported
    markdown = getattr(result, "markdown", None)
    if isinstance(markdown, str):
        return markdown
    if isinstance(markdown, dict):
        mapped = markdown_from_mapping(markdown)
        if mapped is not None:
            return mapped
    if isinstance(result, dict):
        mapped = markdown_from_mapping(result)
        if mapped is not None:
            return mapped
    json_value = getattr(result, "json", None)
    if isinstance(json_value, dict):
        mapped = markdown_from_mapping(json_value)
        if mapped is not None:
            return mapped
        res = json_value.get("res")
        if isinstance(res, dict):
            mapped = markdown_from_mapping(res)
            if mapped is not None:
                return mapped
    for method_name in ("to_markdown", "export_markdown"):
        method = getattr(result, method_name, None)
        if callable(method):
            value = method()
            if isinstance(value, str):
                return value
    raise TypeError("Official PaddleOCRVL result did not expose Markdown text.")
```

Add:

```python
_CENTERED_IMAGE_DIV_RE = re.compile(
    r"<div[^>]*style=[\"'][^\"']*text-align:\s*center;?[^\"']*[\"'][^>]*>\s*"
    r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"'][^>]*>\s*</div>",
    re.IGNORECASE | re.DOTALL,
)
_CENTERED_TEXT_DIV_RE = re.compile(
    r"<div[^>]*style=[\"'][^\"']*text-align:\s*center;?[^\"']*[\"'][^>]*>\s*(.*?)\s*</div>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _normalize_official_markdown_for_omnidocbench(markdown: str) -> str:
    def replace_image(match: re.Match[str]) -> str:
        return f"![]({html.unescape(match.group(1))})"

    def replace_text(match: re.Match[str]) -> str:
        inner = _HTML_TAG_RE.sub("", match.group(1))
        return html.unescape(inner.strip())

    markdown = _CENTERED_IMAGE_DIV_RE.sub(replace_image, markdown)
    markdown = _CENTERED_TEXT_DIV_RE.sub(replace_text, markdown)
    return re.sub(r"\n{3,}", "\n\n", markdown)
```

- [ ] **Step 4: Implement official folder runner**

Add `run_official_folder` with lazy PaddleOCR import, page retry, fallback copy, stats, and errors:

```python
def run_official_folder(
    *,
    img_dir: Path,
    out_dir: Path,
    server_url: str,
    api_model_name: str,
    page_retries: int = 1,
    fallback_pred_dir: Path | None = None,
) -> dict:
    if not img_dir.is_dir():
        raise SystemExit(f"Image directory not found: {img_dir}")
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise RuntimeError(
            "Official engine requires PaddleOCR. Install the local PaddleOCR dependency first."
        ) from exc

    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_backend="llama-cpp-server",
        vl_rec_server_url=server_url,
        vl_rec_api_model_name=api_model_name,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_path = out_dir / "_errors.log"
    stats_path = out_dir / "_run_stats.json"
    errors_path.unlink(missing_ok=True)
    stats_path.unlink(missing_ok=True)

    stats: list[dict] = []
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    page_retries = max(0, int(page_retries))

    for img in images:
        start = time.time()
        last_exc: Exception | None = None
        last_tb = ""
        attempts = 0
        for attempt in range(page_retries + 1):
            attempts = attempt + 1
            try:
                result = pipeline.predict(str(img))
                if isinstance(result, list):
                    markdown = "\n\n".join(_official_result_to_markdown(item) for item in result)
                else:
                    markdown = _official_result_to_markdown(result)
                markdown = _normalize_official_markdown_for_omnidocbench(markdown)
                (out_dir / expected_md_name(img.name)).write_text(markdown, encoding="utf-8")
                stats.append(
                    {
                        "image": img.name,
                        "status": "ok",
                        "seconds": round(time.time() - start, 2),
                        "attempts": attempts,
                    }
                )
                break
            except Exception as exc:
                last_exc = exc
                last_tb = traceback.format_exc()
                if attempt < page_retries:
                    time.sleep(min(2.0, 0.25 * attempts))
                    continue
        else:
            fallback_path = (
                fallback_pred_dir / expected_md_name(img.name)
                if fallback_pred_dir is not None
                else None
            )
            if fallback_path is not None and fallback_path.is_file():
                shutil.copyfile(fallback_path, out_dir / expected_md_name(img.name))
                status = f"fallback: {last_exc}"
            else:
                status = f"failed: {last_exc}"
            with open(errors_path, "a", encoding="utf-8") as fh:
                fh.write(
                    f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {img.name}: {last_exc} "
                    f"(attempts={attempts})\n{last_tb}\n"
                )
                if fallback_path is not None and fallback_path.is_file():
                    fh.write(f"FALLBACK prediction copied from: {fallback_path}\n")
            stats.append(
                {
                    "image": img.name,
                    "status": status,
                    "seconds": round(time.time() - start, 2),
                    "attempts": attempts,
                    "traceback": last_tb,
                }
            )

    ok_count = sum(1 for s in stats if s["status"] == "ok" or s["status"].startswith("fallback:"))
    fallback_count = sum(1 for s in stats if s["status"].startswith("fallback:"))
    summary = {
        "count": len(images),
        "ok": ok_count,
        "fail": len(images) - ok_count,
        "fallback": fallback_count,
        "engine": "official",
        "stats": stats,
    }
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(images) > 0 and ok_count < 0.5 * len(images):
        raise SystemExit(2)
    return summary
```

- [ ] **Step 5: Update CLI to route through `run_adapter`**

Add arguments:

```python
parser.add_argument("--engine", choices=["lightweight", "official"], default=DEFAULT_ENGINE)
parser.add_argument("--page-retries", type=int, default=int(os.environ.get("PADDLEOCR_VL_PAGE_RETRIES", "1")))
parser.add_argument("--fallback-pred-dir", default=os.environ.get("PADDLEOCR_VL_FALLBACK_PRED_DIR"))
```

Call:

```python
summary = run_adapter(
    Path(args.img_dir),
    Path(args.out_dir),
    args.server_url,
    engine=args.engine,
    layout_model=args.layout_model,
    api_model_name=args.api_model_name,
    vlm_backend=args.vlm_backend,
    page_retries=args.page_retries,
    fallback_pred_dir=args.fallback_pred_dir,
)
```

- [ ] **Step 6: Run adapter tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_eval_adapter.py -q
```

Expected: all adapter tests pass.

- [ ] **Step 7: Commit production adapter**

```powershell
git add eval/PaddleOCRVLROCm_img2md.py tests/test_eval_adapter.py
git commit -m "feat: add local official eval adapter"
```

---

### Task 3: Evaluation Orchestrator Routing

**Files:**
- Modify: `tests/test_eval_report_path.py`
- Modify: `eval/run_eval.py`

**Interfaces:**
- Consumes: `eval/PaddleOCRVLROCm_img2md.py::run_adapter`
- Produces: `stage_infer(args: argparse.Namespace) -> None` dispatches engine, backend, retries, fallback, layout model, server URL, and API model name.

- [ ] **Step 1: Write failing orchestrator dispatch test**

Append this to `tests/test_eval_report_path.py`:

```python
def test_stage_infer_dispatches_to_run_adapter(tmp_path, monkeypatch, capsys):
    mod = _load_run_eval()
    dataset = tmp_path / "data"
    images = dataset / "images"
    images.mkdir(parents=True)
    predictions = tmp_path / "predictions"
    captured = {}

    class FakeAdapter:
        @staticmethod
        def run_adapter(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {"count": 1, "ok": 1, "fail": 0, "engine": kwargs["engine"], "stats": []}

    monkeypatch.setattr(mod, "_server_reachable", lambda server_url: True)
    monkeypatch.setattr(mod, "_load_script_module", lambda name, path: FakeAdapter)

    args = type(
        "Args",
        (),
        {
            "server_url": "http://127.0.0.1:8111/v1",
            "dataset_dir": str(dataset),
            "version": "v16",
            "predictions_dir": str(predictions),
            "layout_model": "layout",
            "api_model_name": "PaddleOCR-VL-1.6-GGUF.gguf",
            "vlm_backend": "llama-cpp-server",
            "engine": "official",
            "page_retries": 2,
            "fallback_pred_dir": str(tmp_path / "fallback"),
        },
    )()

    mod.stage_infer(args)

    assert captured["args"] == (images, predictions, "http://127.0.0.1:8111/v1")
    assert captured["kwargs"] == {
        "engine": "official",
        "layout_model": "layout",
        "api_model_name": "PaddleOCR-VL-1.6-GGUF.gguf",
        "vlm_backend": "llama-cpp-server",
        "page_retries": 2,
        "fallback_pred_dir": str(tmp_path / "fallback"),
    }
    assert "[infer] 1/1 pages succeeded" in capsys.readouterr().out
```

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
python -m pytest tests/test_eval_report_path.py::test_stage_infer_dispatches_to_run_adapter -q
```

Expected: FAIL because `stage_infer` calls `process_folder` and does not pass the new engine fields.

- [ ] **Step 3: Update `stage_infer`**

In `eval/run_eval.py`, change:

```python
summary = adapter.process_folder(
    images_dir,
    out_dir,
    layout_model=args.layout_model,
    server_url=server_url,
    api_model_name=args.api_model_name,
    vlm_backend="vllm-server",
)
```

to:

```python
summary = adapter.run_adapter(
    images_dir,
    out_dir,
    server_url,
    engine=args.engine,
    layout_model=args.layout_model,
    api_model_name=args.api_model_name,
    vlm_backend=args.vlm_backend,
    page_retries=args.page_retries,
    fallback_pred_dir=args.fallback_pred_dir,
)
```

- [ ] **Step 4: Update orchestrator defaults and CLI args**

Change:

```python
DEFAULT_SERVER_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_MODEL_NAME = "PaddleOCR-VL-1.5-0.9B"
```

to:

```python
DEFAULT_SERVER_URL = "http://127.0.0.1:8111/v1"
DEFAULT_API_MODEL_NAME = "PaddleOCR-VL-1.6-GGUF.gguf"
DEFAULT_VLM_BACKEND = "llama-cpp-server"
```

Add parser args:

```python
parser.add_argument("--engine", choices=["lightweight", "official"], default="lightweight")
parser.add_argument("--vlm-backend", default=DEFAULT_VLM_BACKEND)
parser.add_argument("--page-retries", type=int, default=1)
parser.add_argument("--fallback-pred-dir", default=None)
```

- [ ] **Step 5: Run orchestrator tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_eval_report_path.py -q
```

Expected: all report path and stage dispatch tests pass.

- [ ] **Step 6: Verify help works without live services**

Run:

```powershell
python eval/run_eval.py --help
```

Expected: exit 0 and help includes `--engine`, `--vlm-backend`, `--page-retries`, and `--fallback-pred-dir`.

- [ ] **Step 7: Commit orchestrator routing**

```powershell
git add eval/run_eval.py tests/test_eval_report_path.py
git commit -m "feat: route eval inference through adapter engines"
```

---

### Task 4: Parameter Alignment and Trace Coverage

**Files:**
- Modify: `tests/test_vlm_payload.py`
- Modify: `src/paddleocr_vl_rocm/vlm/client.py`
- Inspect: `src/paddleocr_vl_rocm/pipeline_core.py`

**Interfaces:**
- Consumes: `_completion_payload(...) -> dict`
- Produces: explicit tests for local llama.cpp/GGUF payload and vLLM payload differences.

- [ ] **Step 1: Write failing llama.cpp payload test**

Append this to `tests/test_vlm_payload.py`:

```python
def test_llama_cpp_payload_uses_local_deterministic_sampling_controls():
    payload = _completion_payload(
        backend="llama-cpp-server",
        model="PaddleOCR-VL-1.6-GGUF.gguf",
        prompt="Formula Recognition:",
        image_url="data:image/png;base64,abc",
        max_new_tokens=4096,
        seed=1,
        min_pixels=112896,
        max_pixels=1003520,
    )

    assert payload["model"] == "PaddleOCR-VL-1.6-GGUF.gguf"
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 1
    assert payload["top_p"] == 1.0
    assert payload["top_k"] == 1
    assert payload["min_p"] == 0.0
    assert payload["repeat_penalty"] == 1.0
    assert payload["cache_prompt"] is False
    assert payload["skip_special_tokens"] is True
    assert payload["max_tokens"] == 4096
    assert "mm_processor_kwargs" not in payload
```

- [ ] **Step 2: Run test and verify behavior**

Run:

```powershell
python -m pytest tests/test_vlm_payload.py -q
```

Expected: if the test passes immediately, keep it as characterization coverage and do not change production payload code. If it fails, update `_completion_payload` to match this deterministic local payload and rerun.

- [ ] **Step 3: Add trace coverage if missing**

If no test currently covers trace fields, add this to `tests/test_vlm_payload.py` only if a lightweight helper is extracted during implementation:

```python
def test_trace_event_records_min_max_pixel_intent():
    trace_event = {
        "backend": "llama-cpp-server",
        "model": "PaddleOCR-VL-1.6-GGUF.gguf",
        "request_order": 0,
        "block_label": "display_formula",
        "image_format": "PNG",
        "image_size": [512, 256],
        "max_new_tokens": 4096,
        "min_pixels": 112896,
        "max_pixels": 1003520,
        "skip_special_tokens": True,
    }

    assert trace_event["min_pixels"] == 112896
    assert trace_event["max_pixels"] == 1003520
    assert trace_event["backend"] == "llama-cpp-server"
```

If no helper is extracted, do not add a synthetic test solely to increase coverage; rely on the existing payload characterization.

- [ ] **Step 4: Commit payload coverage**

```powershell
git add tests/test_vlm_payload.py src/paddleocr_vl_rocm/vlm/client.py
git commit -m "test: characterize local vlm payload parameters"
```

---

### Task 5: Documentation and Local Score Posture

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `eval/README.md`
- Optionally create: `results/omnidocbench/v16/README.md`

**Interfaces:**
- Consumes: adapter/orchestrator CLI from Tasks 2-3
- Produces: local-only instructions and honest score reporting.

- [ ] **Step 1: Update README local evaluation section**

In `README.md`, replace the current OmniDocBench score narrative with language that distinguishes local engines:

```markdown
## Evaluation (OmniDocBench v1.6, local AMD Windows)

Scores in this repository are local measurements from the Windows + AMD Radeon
+ llama.cpp/GGUF + OmniDocBench/CDM environment. They are not claimed from a
Linux vLLM/BF16 reference path.

| Engine | Text Edit-dist ↓ | Reading-order Edit-dist ↓ | Table TEDS ↑ | Formula CDM ↑ | Notes |
|---|---:|---:|---:|---:|---|
| Lightweight local engine | 0.035 | 0.129 | 94.00 | 94.40 | Existing recorded local CDM artifact |
| Official local engine | 0.034 | 0.129 | 94.22 | 96.81 | Reproduced in the companion local setup; rerun here with `--engine official` |
| Public PaddleOCR-VL-1.6 target | 0.035 | 0.129 | 94.64 | 97.49 | External reference, shown for context only |

The project goal is to align inputs, outputs, parameters, and local evaluation
evidence. Remaining gaps are reported by engine instead of hidden.
```

Adjust values only if fresh artifacts are generated during implementation.

- [ ] **Step 2: Update eval README commands**

In `eval/README.md`, add local engine examples:

```markdown
### Local lightweight engine

```powershell
python eval/run_eval.py --stage infer --version v16 `
  --engine lightweight `
  --vlm-backend llama-cpp-server `
  --server-url http://127.0.0.1:8111/v1 `
  --api-model-name PaddleOCR-VL-1.6-GGUF.gguf
```

### Local official engine

```powershell
python eval/run_eval.py --stage infer --version v16 `
  --engine official `
  --server-url http://127.0.0.1:8111/v1 `
  --api-model-name PaddleOCR-VL-1.6-GGUF.gguf `
  --page-retries 1
```
```

Close the inner fences correctly when editing the file.

- [ ] **Step 3: Update Chinese README with the same local-only posture**

Mirror the English meaning in `README.zh-CN.md`: local-only validation, engine-specific scores, no Linux vLLM/BF16 setup, and commands for lightweight/official local engines.

- [ ] **Step 4: Run documentation grep checks**

Run:

```powershell
rg -n "vLLM/BF16|Linux vLLM|reference-quality path|native precision aligned|PaddleOCR-VL-1.5-0.9B" README.md README.zh-CN.md eval/README.md
```

Expected: any matches are either explicit non-goals/context or intentional legacy CLI examples. Replace outdated default model mentions in evaluation docs with `PaddleOCR-VL-1.6-GGUF.gguf`.

- [ ] **Step 5: Commit docs**

```powershell
git add README.md README.zh-CN.md eval/README.md results/omnidocbench/v16/README.md
git commit -m "docs: report local engine evaluation posture"
```

If `results/omnidocbench/v16/README.md` was not created, omit it from `git add`.

---

### Task 6: Final Local Verification and Push

**Files:**
- Inspect: all modified files
- Do not commit: `eval/.omnidocbench/`, `logs/`, generated prediction directories unless explicitly requested

**Interfaces:**
- Consumes: all previous tasks
- Produces: verified branch pushed to `https://github.com/AIwork4me/PaddleOCR-VL-ROCm`

- [ ] **Step 1: Run fast verification**

Run:

```powershell
python -m compileall -q src/paddleocr_vl_rocm eval
python -m pytest -q
python eval/PaddleOCRVLROCm_img2md.py --help
python eval/run_eval.py --help
```

Expected: all commands exit 0.

- [ ] **Step 2: Run local live smoke only if the server is up**

Run:

```powershell
paddleocr-vl-rocm-check-server --server-url http://127.0.0.1:8111/v1
```

Expected if server is up: exit 0. Then run a tiny local inference directory chosen from existing OmniDocBench images:

```powershell
python eval/run_eval.py --stage infer --version v16 `
  --engine lightweight `
  --vlm-backend llama-cpp-server `
  --server-url http://127.0.0.1:8111/v1 `
  --api-model-name PaddleOCR-VL-1.6-GGUF.gguf `
  --predictions-dir predictions/paddleocrvl_rocm_smoke
```

If the server is not up, record that live inference was skipped because the local VLM endpoint was unreachable; do not claim live inference passed.

- [ ] **Step 3: Check git status**

Run:

```powershell
git status --short --branch
```

Expected: modified tracked files from this task are committed; untracked `eval/.omnidocbench/` and `logs/` may remain untracked.

- [ ] **Step 4: Push**

Run:

```powershell
git remote -v
git push https://github.com/AIwork4me/PaddleOCR-VL-ROCm HEAD:main
```

Expected: push succeeds. If authentication fails, report the exact failure and leave commits local.
