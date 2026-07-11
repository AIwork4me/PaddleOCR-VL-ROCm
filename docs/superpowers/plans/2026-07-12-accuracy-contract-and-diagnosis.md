# Accuracy Contract And Root-Cause Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce canonical input/request/output traces and an evidence-backed page/region diagnosis that identifies the actual causes of the remaining 0.182-point accuracy gap.

**Architecture:** Add a stable trace schema around the existing layout, crop, payload, raw-response, normalization, and serialization boundaries. Compare lightweight traces and score artifacts against same-machine official outputs without changing production behavior. End with a ranked root-cause report whose accepted fixes receive a separate, evidence-specific TDD plan before implementation.

**Tech Stack:** Python 3.10+, dataclasses, JSON, SHA-256, pytest, ONNXRuntime, llama.cpp HIP, OmniDocBench v1.6.

## Global Constraints

- Run this plan only after `2026-07-12-v16-evidence-and-scoring.md` passes.
- Keep OmniDocBench pinned to v1.6 commit `147cd5ac9472002f5751221d390bf00abdbc0d2f`.
- Do not change prompts, crops, normalization, or serialization during diagnosis.
- Do not add filename-specific or ground-truth-dependent behavior.
- Redact server credentials and authorization data from every trace.
- Preserve existing JSON and Markdown output byte-for-byte while adding observability.

---

## File Structure

- Create `src/paddleocr_vl_rocm/contracts.py`: canonical request and trace dataclasses.
- Create `tests/test_contracts.py`: canonicalization, hashing, and redaction tests.
- Modify `src/paddleocr_vl_rocm/vlm/client.py`: payload observer hook.
- Modify `src/paddleocr_vl_rocm/pipeline_core.py`: complete block trace fields.
- Modify `tests/test_vlm_payload.py`: payload-to-contract tests.
- Create `tests/fixtures/contracts/v16-native-output.json`: versioned output-key and default-parameter snapshot.
- Create `tests/test_native_compat_contract.py`: CLI, Python, JSON, Markdown, and filename compatibility tests.
- Modify `scripts/record_trace.py`: JSONL trace export and v1.6 defaults.
- Create `scripts/compare_inference_traces.py`: structural trace comparison.
- Create `tests/test_compare_inference_traces.py`: diff classification tests.
- Create `scripts/analyze_omnidocbench_deltas.py`: page and component delta ranking.
- Create `tests/test_analyze_omnidocbench_deltas.py`: score extraction and ranking tests.
- Create `docs/accuracy-root-cause-v16.md`: generated evidence report.

### Task 1: Define the canonical inference contract

**Files:**
- Create: `src/paddleocr_vl_rocm/contracts.py`
- Create: `tests/test_contracts.py`

**Interfaces:**
- Produces: `canonical_json(value: object) -> str`.
- Produces: `fingerprint(value: object) -> str`.
- Produces: `redact(value: object) -> object`.
- Produces: frozen `VLMRequestContract` and `BlockTrace` dataclasses.

- [ ] **Step 1: Write failing canonicalization and redaction tests**

```python
def test_contract_fingerprint_is_order_independent():
    assert fingerprint({"b": 2, "a": 1}) == fingerprint({"a": 1, "b": 2})


def test_redact_removes_credentials_recursively():
    value = {
        "Authorization": "Bearer secret",
        "url": "https://host/path?token=secret",
        "nested": {"api_key": "secret"},
    }
    assert redact(value) == {
        "Authorization": "<redacted>",
        "url": "https://host/path?token=%3Credacted%3E",
        "nested": {"api_key": "<redacted>"},
    }
```

- [ ] **Step 2: Run tests and verify RED**

Run `python -m pytest tests/test_contracts.py -q`.

Expected: module import failure.

- [ ] **Step 3: Implement canonical helpers and dataclasses**

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SECRET_KEYS = {"authorization", "api_key", "apikey", "token", "access_token"}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if key.lower() in SECRET_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str) and "://" in value:
        parts = urlsplit(value)
        query = [(key, "<redacted>" if key.lower() in SECRET_KEYS else item) for key, item in parse_qsl(parts.query)]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    return value


@dataclass(frozen=True)
class VLMRequestContract:
    backend: str
    model: str
    prompt: str
    image_format: str
    image_sha256: str
    image_size: tuple[int, int]
    payload: dict[str, Any]

    def fingerprint(self) -> str:
        return fingerprint(redact(asdict(self)))


@dataclass(frozen=True)
class BlockTrace:
    request_order: int
    label: str
    bbox: tuple[float, float, float, float]
    request: VLMRequestContract
    raw_result_sha256: str
    final_result_sha256: str
```

- [ ] **Step 4: Run tests and commit**

Run `python -m pytest tests/test_contracts.py -q` and `python -m pytest -q`.

Commit:

```powershell
git add src/paddleocr_vl_rocm/contracts.py tests/test_contracts.py
git commit -m "feat: define inference compatibility contracts"
```

### Task 2: Observe exact VLM payloads without changing behavior

**Files:**
- Modify: `src/paddleocr_vl_rocm/vlm/client.py`
- Modify: `src/paddleocr_vl_rocm/pipeline_core.py`
- Modify: `tests/test_vlm_payload.py`

**Interfaces:**
- `OpenAICompatibleVLMClient.__init__` accepts `request_observer: Callable[[VLMRequestContract], None] | None`.
- Existing callers remain source-compatible.

- [ ] **Step 1: Write a failing observer test**

Mock `requests.post`, call `complete_image`, and assert the observer receives the
same deterministic llama.cpp payload sent to the server, including:

```python
assert observed.payload["temperature"] == 0.0
assert observed.payload["seed"] == 1
assert observed.payload["top_k"] == 1
assert observed.payload["top_p"] == 1.0
assert observed.payload["min_p"] == 0.0
assert observed.payload["repeat_penalty"] == 1.0
assert observed.payload["cache_prompt"] is False
assert observed.payload["max_tokens"] == 4096
```

- [ ] **Step 2: Run test and verify RED**

Run `python -m pytest tests/test_vlm_payload.py -q`.

Expected: constructor rejects `request_observer`.

- [ ] **Step 3: Add the observer**

Store the callback in `__init__`. Immediately after `_completion_payload`, call:

```python
if self._request_observer is not None:
    self._request_observer(
        VLMRequestContract(
            backend=self.backend,
            model=self.model,
            prompt=prompt,
            image_format="JPEG" if self.backend == "vllm-server" else "PNG",
            image_sha256=image_sha256,
            image_size=image.size if image is not None else (0, 0),
            payload=redact(payload),
        )
    )
```

In `pipeline_core.py`, use the observed contract to populate the existing trace
event instead of reconstructing a partial payload independently.

- [ ] **Step 4: Verify characterization invariants**

Run:

```powershell
python -m pytest tests/test_vlm_payload.py tests/test_pipeline_characterization.py -q
python -m pytest -q
```

Expected: all golden outputs remain unchanged.

- [ ] **Step 5: Commit**

```powershell
git add src/paddleocr_vl_rocm/vlm/client.py src/paddleocr_vl_rocm/pipeline_core.py tests/test_vlm_payload.py
git commit -m "feat: capture canonical vlm request traces"
```

### Task 3: Export and compare block traces

**Files:**
- Modify: `scripts/record_trace.py`
- Create: `scripts/compare_inference_traces.py`
- Create: `tests/test_compare_inference_traces.py`

**Interfaces:**
- Produces: one JSONL event per block with canonical request and result hashes.
- Produces: `compare_traces(reference: list[dict], candidate: list[dict]) -> dict[str, object]`.

- [ ] **Step 1: Write failing comparison tests**

```python
def test_compare_traces_classifies_crop_before_payload():
    reference = [{"request_order": 0, "label": "formula", "bbox": [1, 2, 3, 4], "image_sha256": "a", "payload_fingerprint": "x"}]
    candidate = [{"request_order": 0, "label": "formula", "bbox": [1, 2, 3, 4], "image_sha256": "b", "payload_fingerprint": "y"}]
    report = compare_traces(reference, candidate)
    assert report["differences"][0]["first_divergence"] == "crop_pixels"


def test_compare_traces_classifies_postprocess_difference():
    reference = [{"raw_result_sha256": "a", "final_result_sha256": "b"}]
    candidate = [{"raw_result_sha256": "a", "final_result_sha256": "c"}]
    report = compare_traces(reference, candidate)
    assert report["differences"][0]["first_divergence"] == "postprocess"
```

- [ ] **Step 2: Implement ordered classification**

Compare in this exact order: request count, request order, label, bbox, crop
hash, prompt, payload fingerprint, raw result, final result. Return summary
counts and the first 100 detailed differences.

- [ ] **Step 3: Extend `record_trace.py`**

Change defaults to `PaddleOCR-VL-1.6-GGUF.gguf` and
`llama-cpp-server`. Add `--trace-jsonl` and write each redacted event with:

```python
with trace_path.open("w", encoding="utf-8") as stream:
    for event in trace_events:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
```

- [ ] **Step 4: Verify and commit**

Run:

```powershell
python -m pytest tests/test_compare_inference_traces.py tests/test_vlm_payload.py -q
python scripts/compare_inference_traces.py --help
git diff --check
```

Commit the three files with message `feat: add inference trace differential`.

### Task 3A: Lock the public input, parameter, and output contract

**Files:**
- Create: `tests/fixtures/contracts/v16-native-output.json`
- Create: `tests/test_native_compat_contract.py`
- Modify: `src/paddleocr_vl_rocm/cli.py`
- Modify: `src/paddleocr_vl_rocm/pipeline.py`

**Interfaces:**
- Produces one versioned snapshot covering constructor defaults, CLI defaults,
  required JSON keys, Markdown filename rules, and model settings.

- [ ] **Step 1: Record the contract fixture from a representative official-local result**

Store only field names, types, defaults, and hashes; do not commit the source
image or full prediction. The fixture must include:

```json
{
  "schema": 1,
  "json_keys": ["input_path", "width", "height", "layout_det_res", "parsing_res_list", "model_settings"],
  "markdown_suffix": ".md",
  "json_suffix": "_res.json",
  "defaults": {
    "api_model_name": "PaddleOCR-VL-1.6-GGUF.gguf",
    "vlm_backend": "llama-cpp-server",
    "max_new_tokens": 4096,
    "seed": 1,
    "threshold": 0.3,
    "vlm_max_workers": 1
  }
}
```

- [ ] **Step 2: Write failing compatibility tests**

Assert `build_parser()` and `PaddleOCRVLROCm.__init__` expose the fixture defaults.
Replay one existing golden input and assert required JSON keys, value types,
output suffixes, and Markdown hash.

- [ ] **Step 3: Align stale defaults without changing payload semantics**

Replace the v1.5 CLI/Python model default with
`PaddleOCR-VL-1.6-GGUF.gguf` and the backend default with
`llama-cpp-server`. Keep explicit vLLM arguments supported.

- [ ] **Step 4: Verify and commit**

Run CLI, native-contract, payload, characterization, and full tests. Commit with
`fix: align public inference defaults with v16 contract`.

### Task 4: Rank OmniDocBench page and component deltas

**Files:**
- Create: `scripts/analyze_omnidocbench_deltas.py`
- Create: `tests/test_analyze_omnidocbench_deltas.py`

**Interfaces:**
- Produces: `load_component_samples(result_dir: Path) -> list[dict[str, object]]`.
- Produces: `rank_deltas(reference, candidate) -> dict[str, object]`.

- [ ] **Step 1: Write failing ranking tests**

Use synthetic formula and table records for two pages. Assert the report ranks
the largest candidate-minus-reference loss first and keeps Formula CDM and
Table TEDS separate.

- [ ] **Step 2: Implement normalized sample keys**

Use `(component, img_id, gt_idx)` as the key. Store official score, lightweight
score, delta, GT, both predictions, and error metadata. Aggregate by page with
equal sample weights to mirror v1.6.

- [ ] **Step 3: Add CLI output**

Support:

```text
--official-result-dir
--lightweight-result-dir
--out-json
--top 100
```

Write a deterministic JSON report sorted by `(delta, page, gt_idx)`.

- [ ] **Step 4: Verify and commit**

Run the focused tests, `python scripts/analyze_omnidocbench_deltas.py --help`,
the full suite, and commit with `feat(eval): rank official-lightweight metric deltas`.

### Task 5: Produce the root-cause report and next fix plan

**Files:**
- Create: `docs/accuracy-root-cause-v16.md`
- Create after diagnosis: `docs/superpowers/plans/2026-07-12-accuracy-root-cause-fixes.md`

**Interfaces:**
- Consumes: corrected v1.6 artifacts, trace differentials, and top page/component deltas.
- Produces: a ranked, quantified root-cause report and a concrete TDD fix plan.

- [ ] **Step 1: Generate the top-delta report**

Run the analyzer over the corrected official-local and lightweight result
directories. Record counts by Formula CDM, Table TEDS, text, reading order,
layout/crop, payload, raw VLM output, and post-processing.

- [ ] **Step 2: Capture paired traces for representative cases**

Select at least five cases from each material loss category. Capture lightweight
traces and the closest observable official inputs/outputs. Do not change code or
predictions during capture.

- [ ] **Step 3: Write the evidence report**

For every claimed root cause, include page, component, score delta, first trace
divergence, affected sample count, estimated Overall contribution, and the
smallest generic correction boundary.

- [ ] **Step 4: Write the evidence-specific fix plan**

Use `superpowers:writing-plans` again. Every production change in that plan must
name the exact failing fixture and expected metric or contract improvement.
Do not implement a speculative normalization, crop, or prompt change from this
diagnostic plan.

- [ ] **Step 5: Commit diagnostic evidence**

Commit only the small report, analyzer, tests, and the new fix plan. Keep raw
traces and predictions untracked.
