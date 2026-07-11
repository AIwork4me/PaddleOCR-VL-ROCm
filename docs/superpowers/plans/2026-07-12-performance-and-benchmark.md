# Performance And Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure and reduce lightweight inference latency to mean ≤13.00 seconds/page and P95 ≤34.82 seconds/page while preserving the accepted v1.6 accuracy result.

**Architecture:** Add stage-level timing first, establish cold and warm baselines, then apply only output-preserving optimizations. Reuse HTTP connections per worker, make deterministic disk caching optional, and size VLM concurrency from explicit server capability data. Bind every speed report to output hashes and a v1.6 score artifact.

**Tech Stack:** Python 3.10+, `time.perf_counter`, `requests`, SQLite, `ThreadPoolExecutor`, pytest, llama.cpp HIP.

## Global Constraints

- Run only after the accuracy fix plan has passed the ≥96.13 G3 gate.
- Baseline hardware, driver, power mode, model, runtime, and dataset must remain fixed.
- Official baseline: mean 18.57 seconds/page, P95 46.42 seconds/page.
- Release target: mean ≤13.00 seconds/page, P95 ≤34.82 seconds/page.
- Default deterministic outputs must remain byte-identical for scheduling, connection, and cache changes.
- Encoding changes require a fresh full OmniDocBench v1.6 run.
- Disk cache is opt-in and must include all request-affecting values in its key.

---

## File Structure

- Create `src/paddleocr_vl_rocm/timing.py`: timer and percentile helpers.
- Create `tests/test_timing.py`: deterministic timing-summary tests.
- Modify `src/paddleocr_vl_rocm/pipeline_core.py`: stage timing observer.
- Modify `src/paddleocr_vl_rocm/pipeline.py`: expose page timing.
- Modify `eval/PaddleOCRVLROCm_img2md.py`: timing summaries in `_run_stats.json`.
- Modify `src/paddleocr_vl_rocm/vlm/client.py`: thread-local connection reuse.
- Modify `tests/test_vlm_payload.py`: session-reuse tests.
- Create `src/paddleocr_vl_rocm/cache.py`: optional SQLite response cache.
- Create `tests/test_cache.py`: key, atomicity, and corruption tests.
- Modify `src/paddleocr_vl_rocm/cli.py`: cache and concurrency controls.
- Create `scripts/benchmark_inference.py`: cold/warm benchmark harness.
- Create `tests/test_benchmark_harness.py`: summary and quality-pairing tests.

### Task 1: Add deterministic stage timing and summaries

**Files:**
- Create: `src/paddleocr_vl_rocm/timing.py`
- Create: `tests/test_timing.py`
- Modify: `src/paddleocr_vl_rocm/pipeline_core.py`
- Modify: `src/paddleocr_vl_rocm/pipeline.py`
- Modify: `eval/PaddleOCRVLROCm_img2md.py`

**Interfaces:**
- Produces: `summarize_seconds(values: Sequence[float]) -> dict[str, float | int]`.
- `run_light_parser(..., timing_events: list[dict[str, float]] | None = None)` records decode, layout, crop/encode, VLM, finalize, and total seconds.

- [ ] **Step 1: Write failing summary tests**

```python
def test_summarize_seconds_uses_nearest_rank_percentiles():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert summarize_seconds(values) == {
        "count": 5,
        "mean": 22.0,
        "p50": 3.0,
        "p95": 100.0,
        "p99": 100.0,
        "max": 100.0,
    }


def test_summarize_seconds_empty():
    assert summarize_seconds([]) == {"count": 0}
```

- [ ] **Step 2: Run tests and verify RED**

Run `python -m pytest tests/test_timing.py -q`.

Expected: module import failure.

- [ ] **Step 3: Implement the timing helper**

```python
from __future__ import annotations

import math
from collections.abc import Sequence


def _nearest_rank(values: list[float], percentile: float) -> float:
    index = max(0, math.ceil(percentile * len(values)) - 1)
    return values[index]


def summarize_seconds(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": _nearest_rank(ordered, 0.50),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "max": ordered[-1],
    }
```

- [ ] **Step 4: Instrument pipeline stages**

Use `perf_counter()` around image open, layout prediction, block/crop creation,
the executor section, and result serialization. Append one dictionary with all
stage values and `total_seconds`. Do not change control flow.

- [ ] **Step 5: Add adapter timing summaries**

Preserve existing per-page `seconds`. Add:

```python
summary["timing"] = summarize_seconds(
    [float(item["seconds"]) for item in stats if item["status"] == "ok"]
)
```

Store stage summaries when pipeline timing is available.

- [ ] **Step 6: Verify and commit**

Run timing tests, adapter tests, characterization tests, and the full suite.
Commit with `feat: add stage-level inference timing`.

### Task 2: Reuse HTTP connections safely per worker

**Files:**
- Modify: `src/paddleocr_vl_rocm/vlm/client.py`
- Modify: `tests/test_vlm_payload.py`

**Interfaces:**
- Each worker thread lazily owns one `requests.Session`.
- No `Session` object is shared across concurrent worker threads.

- [ ] **Step 1: Write a failing same-thread reuse test**

Monkeypatch `requests.Session` with a factory that records instances. Make two
uncached calls from one thread and assert one session and two `.post()` calls.
Make calls from two threads and assert two sessions.

- [ ] **Step 2: Run test and verify RED**

Expected: current code calls module-level `requests.post` and creates no session.

- [ ] **Step 3: Implement thread-local sessions**

```python
import threading


class OpenAICompatibleVLMClient:
    def __init__(self, ...):
        ...
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session
```

Replace `requests.post(...)` with `self._session().post(...)` and change no
payload, timeout, or retry behavior.

- [ ] **Step 4: Verify output invariants and commit**

Run payload, characterization, and full tests. Commit with
`perf: reuse vlm http connections per worker`.

### Task 3: Add an opt-in deterministic disk response cache

**Files:**
- Create: `src/paddleocr_vl_rocm/cache.py`
- Create: `tests/test_cache.py`
- Modify: `src/paddleocr_vl_rocm/vlm/client.py`
- Modify: `src/paddleocr_vl_rocm/pipeline.py`
- Modify: `src/paddleocr_vl_rocm/cli.py`

**Interfaces:**
- Produces: `request_cache_key(payload: dict, image_sha256: str) -> str`.
- Produces: `SQLiteResponseCache(path: Path)` with `get(key)` and `put(key, value)`.
- Cache defaults to disabled.

- [ ] **Step 1: Write failing cache tests**

Assert the key changes for backend, model, prompt, image hash, seed, sampling,
token limit, and pixel settings. Assert two cache instances can read the same
value and a malformed database raises `RuntimeError` naming the path.

- [ ] **Step 2: Implement the complete request key**

```python
def request_cache_key(payload: dict[str, Any], image_sha256: str) -> str:
    value = {"schema": 1, "payload": payload, "image_sha256": image_sha256}
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
```

- [ ] **Step 3: Implement SQLite cache storage**

Create table `responses(key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at
TEXT NOT NULL)`. Use `INSERT OR REPLACE`, commit each write, and set a five-second
busy timeout. Never store server URL or credentials.

- [ ] **Step 4: Wire optional cache settings**

Add `cache_path: str | Path | None = None` to `PaddleOCRVLROCm`. Add
`--cache-path` to the CLI. The client checks memory cache, compatibility cache,
then disk cache; it writes disk cache only after a successful response.

- [ ] **Step 5: Verify and commit**

Run cache, payload, CLI, characterization, and full tests. Commit with
`perf: add deterministic persistent response cache`.

### Task 4: Make VLM concurrency explicit and evidence-backed

**Files:**
- Modify: `src/paddleocr_vl_rocm/server.py`
- Modify: `src/paddleocr_vl_rocm/pipeline.py`
- Modify: `src/paddleocr_vl_rocm/cli.py`
- Modify: `tests/test_cli.py`
- Create: `tests/test_server_capabilities.py`

**Interfaces:**
- Produces: `detect_llama_cpp_slots(server_url: str, timeout: float = 10.0) -> int | None`.
- `vlm_max_workers=0` means auto; positive values remain explicit.

- [ ] **Step 1: Write capability tests**

Mock `GET /slots` returning a list of four slots and assert `4`. Assert 404,
invalid JSON, and connection failure return `None` without leaking credentials.

- [ ] **Step 2: Implement detection**

Call `<base>/slots`, require a JSON list, and return its length when positive.
Do not retry capability discovery.

- [ ] **Step 3: Implement conservative auto mode**

Resolve workers as:

```python
def resolve_vlm_workers(requested: int, detected_slots: int | None) -> int:
    if requested > 0:
        return requested
    return max(1, detected_slots or 1)
```

Keep the public default at `1` for compatibility. Expose `0` as documented auto
mode and include requested/resolved values in traces and run stats.

- [ ] **Step 4: Verify and commit**

Run server, CLI, payload, characterization, and full tests. Commit with
`feat: add explicit llama cpp slot-aware concurrency`.

### Task 5: Build the paired quality/performance benchmark harness

**Files:**
- Create: `scripts/benchmark_inference.py`
- Create: `tests/test_benchmark_harness.py`
- Modify: `eval/README.md`

**Interfaces:**
- Produces a JSON artifact containing environment, config, output manifest hash, timing summary, and paired v1.6 score artifact path.

- [ ] **Step 1: Write failing acceptance tests**

Given synthetic official and lightweight summaries, assert:

```python
assert evaluate_targets(
    official={"mean": 18.57, "p95": 46.42},
    candidate={"mean": 12.9, "p95": 34.0},
    quality={"overall": 96.13, "quality_gate": True},
)["accepted"] is True
```

Assert rejection for mean `13.01`, P95 `34.83`, missing quality artifact, or
output-manifest mismatch.

- [ ] **Step 2: Implement benchmark modes**

Support `--mode cold`, `--mode warm`, and `--mode corpus`. Require explicit
input directory, server URL, model, workers, output directory, and quality
artifact. Record Git commit, GPU/driver data, llama-server version, and all
inference settings.

- [ ] **Step 3: Run the baseline matrix**

Run workers `1`, `2`, and detected-slot auto with cache disabled. Select the
fastest configuration that preserves output manifest hashes and the accepted
v1.6 score. Use the cache-enabled run only as a separately labeled repeated-input
result.

- [ ] **Step 4: Apply the stop rule**

If the measured lightweight baseline already meets mean and P95 targets, do not
add encoding or scheduling complexity. If it fails, use stage timing to select
only the dominant remaining stage and write a focused follow-up plan.

- [ ] **Step 5: Verify and commit**

Run the harness tests and full suite. Commit code and documentation, but keep
raw benchmark outputs untracked. Commit a small accepted summary only after its
paired quality gate passes.
