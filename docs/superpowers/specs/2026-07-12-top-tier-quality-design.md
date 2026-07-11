# PaddleOCR-VL-ROCm Top-Tier Quality Design

- Date: 2026-07-12
- Status: Approved
- Delivery branch: `codex/top-tier-quality`
- Target repository: `https://github.com/AIwork4me/PaddleOCR-VL-ROCm`
- Benchmark standard: OmniDocBench v1.6 only

## 1. Objective

Turn PaddleOCR-VL-ROCm into a top-tier open-source project for Windows users
with AMD GPUs. The project must provide:

1. Auditable benchmark evidence that follows the official OmniDocBench v1.6
   computation and aggregation rules exactly.
2. Lightweight inference accuracy that is no worse than the same-machine
   official PaddleOCR path and is within 0.20 points of the public 96.33 paper
   result.
3. Strict, versioned input, request-parameter, output, and serialization
   compatibility contracts.
4. Mean latency at least 30% lower and P95 latency at least 25% lower than the
   same-machine official path, without an accuracy regression.
5. A polished experience for both new Windows AMD users and developers who
   already run an OpenAI-compatible VLM endpoint.
6. A reviewable release with reproducible evidence pushed to the target GitHub
   repository.

The public paper result was measured on a Linux CUDA path. It is an external
target, not a locally reproduced ROCm score. This project aims to approach it
on Windows and AMD hardware without misrepresenting the environment.

## 2. Benchmark Version Policy

OmniDocBench v1.6 is the industry baseline for this project. The implementation
and result schema must be pinned to official commit:

`147cd5ac9472002f5751221d390bf00abdbc0d2f`

The later v1.7 update is a minor follow-on version. It may be observed for
future compatibility but must not affect this release's dataset, score claims,
acceptance thresholds, or comparison tables.

Every published result must record:

- OmniDocBench commit and dirty-state status;
- dataset manifest path, dataset hash, and page count;
- prediction directory hash or per-file manifest;
- match method and complete evaluation configuration;
- Python, TeX Live, ImageMagick, Ghostscript, CDM, and TEDS environment details;
- model, quantization, llama.cpp build, GPU, driver, and project Git commit;
- page, formula-sample, and table-sample denominators;
- timeout, exception, missing-page, and fallback counts.

## 3. Official OmniDocBench v1.6 Scoring Contract

### 3.1 Matching and dataset

- Evaluate the full 1651-page OmniDocBench v1.6 dataset.
- Use the official end-to-end evaluation path and `quick_match` configuration.
- Use identical ground truth, matcher configuration, and scorer checkout for
  the official-local and lightweight prediction directories.
- A benchmark run is not release-grade unless all 1651 predictions exist.
- The official-local baseline may retry its failed page only through the same
  official engine. It must never use a lightweight prediction as fallback.

### 3.2 Formula CDM

For each matched display-formula sample:

1. Compute the official CDM character-detection-matching result.
2. Use `F1_score` as the sample's CDM score.
3. Treat a rendering or evaluation exception as zero and record it.
4. Average formula-sample scores within each GT page.
5. Assign zero to a GT formula page that has no scored prediction.
6. Average the page means equally over all GT pages containing display formulas.
7. Multiply the result by 100 for the leaderboard value.

The leaderboard field is:

`display_formula.page.CDM.ALL * 100`

`display_formula.all.CDM.all * 100` is the sample-level mean. It remains in the
evidence artifact for audit but must not be shown as the official leaderboard
Formula CDM value.

### 3.3 Table TEDS

For each matched table sample:

1. Compute official HTML tree-edit-distance similarity, including cell content.
2. Treat an empty prediction, parsing error, timeout, or exception as zero and
   record it.
3. Average table-sample scores within each GT page.
4. Assign zero to a GT table page that has no scored prediction.
5. Average the page means equally over all GT pages containing tables.
6. Multiply the result by 100 for the leaderboard value.

The leaderboard field is:

`table.page.TEDS.ALL * 100`

`table.all.TEDS.all * 100` is the sample-level mean. It remains available for
audit but must not be shown as the official leaderboard Table TEDS value.

TEDS-S is reported separately and does not enter Overall.

### 3.4 Text, reading order, rounding, and Overall

The leaderboard Text Edit Distance comes from:

`text_block.all.Edit_dist.ALL_page_avg`

Reading-order Edit Distance comes from:

`reading_order.all.Edit_dist.ALL_page_avg`

Reading order is reported but does not enter Overall.

The official notebook constructs a table and applies `.round(3)` before
computing Overall. The project must reproduce that order exactly:

```text
text = round(text_block.all.Edit_dist.ALL_page_avg, 3)
formula = round(display_formula.page.CDM.ALL * 100, 3)
table = round(table.page.TEDS.ALL * 100, 3)
overall = ((1 - text) * 100 + formula + table) / 3
```

The project must test both the selected fields and the rounding order. A
high-precision recomputation that skips the notebook's `.round(3)` step may be
kept as a diagnostic, but it must not be labeled the official notebook score.

### 3.5 Corrected current evidence

Applying the official notebook algorithm to the currently tracked artifacts
produces:

| Engine | Text Edit | Formula CDM | Table TEDS | Official notebook Overall |
|---|---:|---:|---:|---:|
| Same-machine official | 0.034 | 96.502 | 94.239 | 95.7803 |
| Lightweight ROCm | 0.034 | 96.922 | 94.322 | 95.9480 |

These values remain provisional until the official path has 1651/1651 valid
predictions and both paths are regenerated by the pinned, clean scoring flow.
The current README claim that 95.7657 is the official-notebook official-local
score is incorrect because it computes Overall from unrounded components.

## 4. Architecture and Release Gates

Work proceeds through six ordered gates. A later gate cannot hide a regression
from an earlier gate.

### G0: Evidence integrity

- Pin OmniDocBench v1.6 and the dataset.
- Repair the one failed official-local page with the same engine.
- Re-score both engines with the exact contract in Section 3.
- Generate immutable provenance, denominators, metric-quality, and environment
  artifacts.

### G1: Compatibility contract

- Capture layout labels, boxes, scores, merge decisions, crop order, crop size,
  and crop image hashes.
- Capture backend, model, prompt, image format, generation parameters, token
  limit, seed, sampling controls, and request order.
- Capture raw-response hashes, normalization stages, final block content, JSON,
  and Markdown.
- Version the CLI, Python API, configuration, JSON schema, Markdown rules, and
  filename contract.

### G2: Root-cause diagnosis

Diagnose differences in this order:

1. dataset, prediction coverage, matcher, aggregation, and rounding;
2. layout detection, overlap filtering, merge rules, ordering, and crops;
3. prompt, image encoding, resize policy, sampling, and token limits;
4. raw VLM output and backend/runtime divergence;
5. formula/table normalization, Markdown generation, and serialization.

Use canonical trace diffs, region-level comparisons, raw-response replay, and
oracle swaps to measure the score contribution of each layer. Production code
changes require a reproducible case, a focused failing test, and a generic
fix. Filename-specific rules, ground-truth lookup, scorer-specific hacks, and
semantic formula rewriting are forbidden.

### G3: Accuracy acceptance

The lightweight engine passes only when:

- official notebook Overall is at least 96.13;
- Overall is no worse than the corrected same-machine official path;
- all component metrics are published, including regressions;
- all 1651 pages succeed;
- CDM and TEDS have zero timeouts and zero exceptions;
- no input/output or default-parameter contract regresses.

### G4: Performance acceptance

The existing official-local timing baseline is:

- mean: 18.57 seconds per page;
- P95: 46.42 seconds per page.

The lightweight release target is therefore:

- mean no greater than 13.00 seconds per page;
- P95 no greater than 34.82 seconds per page;
- paired accuracy no worse than the accepted G3 result.

### G5: Launch readiness

- Both onboarding paths pass on a clean Windows environment.
- Runtime and model downloads are pinned and checksum-verified.
- Doctor output and common failure recovery are verified.
- README claims link directly to versioned evidence.
- Packaging, documentation, examples, contribution guidance, and release notes
  pass review.

## 5. Inference Components and Data Flow

### 5.1 Configuration contract

CLI flags, Python arguments, environment variables, and config files map into
one typed configuration model. Precedence is explicit and testable. The model
contains layout, crop, VLM, sampling, concurrency, cache, timeout, output, and
runtime settings.

Backend adapters translate this canonical configuration to llama.cpp HIP or an
external OpenAI-compatible endpoint. Unsupported backend capabilities are
reported explicitly rather than silently ignored. Secrets never appear in
traces or logs.

### 5.2 Layout and crop planner

The ONNX layout component emits an ordered, inspectable plan before VLM work
starts. It owns box filtering, merging, table-figure tokenization, formula
margin cropping, and crop hashing. Its output can be replayed independently of
the layout model.

### 5.3 Bounded VLM scheduler

The scheduler owns concurrency, backpressure, connection reuse, retries, and
timing. It preserves deterministic final block order regardless of completion
order. Default concurrency is derived conservatively from detected llama.cpp
parallel slots and memory headroom.

### 5.4 Normalization and serialization

Raw VLM responses pass through label-specific normalization and table/formula
handling before stable JSON and Markdown serialization. Each step is replayable
from stored traces. The default output is native-compatible; optional pretty
rendering cannot affect benchmark output.

### 5.5 Managed Windows AMD runtime

The repository stores only manifests and tooling, not large binaries or model
weights. The setup command downloads pinned llama.cpp HIP, VLM weights, and
layout assets with:

- SHA-256 verification;
- resumable downloads;
- cache reuse;
- proxy and mirror overrides;
- offline-manifest support;
- atomic installation and clear corruption recovery.

## 6. Accuracy Improvement Workflow

Create a fixed, representative development set from identified failure
categories: formulas, tables, text, reading order, layout, empty output,
truncation, and long-tail latency. Use it for fast iteration, but accept a fix
only after a full 1651-page run.

For every proposed change:

1. record the failing page and smallest reproducible region;
2. classify the responsible pipeline layer;
3. add a failing unit, golden, or contract test;
4. implement the smallest generic correction;
5. replay affected and neighboring categories;
6. run full OmniDocBench v1.6 before publishing the change.

The report must distinguish true model/runtime divergence from adapter,
preprocessing, post-processing, and aggregation defects.

## 7. Performance Design

Instrumentation records image decode, layout, crop/encode, queue wait, VLM,
normalization, serialization, and total page time. VLM telemetry records time
to first token when available. Reports include cold start, warm single-page,
sustained corpus, mean, P50, P95, P99, pages per minute, failures, retries, and
peak memory.

Optimization order:

1. reuse HTTP connections and match in-flight requests to actual server slots;
2. overlap CPU layout and encoding with GPU inference under bounded
   backpressure;
3. add a deterministic cache keyed by runtime/model manifest, prompt, image
   hash, and all generation parameters;
4. remove redundant image copies and conversions;
5. tune image transport only after crop-level and full-suite quality checks;
6. generate hardware-aware llama.cpp presets with a conservative fallback.

Deterministic mode must produce byte-identical final outputs before and after a
scheduling or cache optimization. Every published speed artifact is paired
with a quality artifact from the same configuration and Git commit.

## 8. User Experience

### 8.1 New Windows AMD user

The intended four-command journey is:

```powershell
pip install paddleocr-vl-rocm
paddleocr-vl-rocm setup --auto
paddleocr-vl-rocm doctor
paddleocr-vl-rocm run invoice.png
```

The exact command layout may reuse existing entry points where that preserves
compatibility, but the journey must remain no longer or less discoverable.

### 8.2 Existing endpoint developer

Developers can install the Python package, validate their server, run the CLI,
or construct `PaddleOCRVLROCm` with an explicit endpoint. CLI and Python use the
same configuration and output contract.

### 8.3 README and trust

The README leads with the Windows AMD value proposition, a real input-to-output
demo, two onboarding choices, and direct links to benchmark evidence. Final
score and speed badges appear only after G3 and G4 pass.

The support matrix separates tested hardware, community-reported hardware, and
unknown hardware. The project makes no broad ROCm compatibility claim without
evidence. English and Chinese documentation remain equivalent.

### 8.4 Diagnostics and recovery

`doctor` checks GPU, driver, disk, RAM/VRAM, ports, runtime, server
capabilities, model files, and hashes. Each failure gives a concrete repair
command. Batch runs preserve successful pages and write a structured failure
manifest. Local operation is the default; there is no hidden telemetry.

## 9. Error Handling

Use stable error categories and exit codes for configuration, dependency,
download verification, service connectivity, memory pressure, inference,
serialization, and evaluation failures.

Interactive use may retry transient network and server errors with bounded
backoff. Benchmark mode records retries but forbids cross-engine fallback.
Partial batch results are explicit and resumable. Logs redact API keys,
authorization headers, signed URLs, and environment secrets.

## 10. Test and Verification Matrix

### Offline CI

- compilation, lint, formatting, and type checks;
- unit and boundary tests;
- serialization and schema tests;
- golden pipeline replay;
- configuration-precedence and CLI tests;
- exact OmniDocBench v1.6 field-selection and rounding tests;
- synthetic CDM/TEDS page-aggregation tests, including missing pages;
- cache-key, atomic-write, retry, and redaction tests;
- package build and install smoke tests.

### Live integration

- official/lightweight canonical trace comparison;
- managed llama.cpp HIP setup and server check;
- deterministic repeated inference;
- representative hard-case quality suite;
- clean Windows onboarding for both user paths.

### Release verification

- corrected official-local 1651-page inference and scoring;
- lightweight 1651-page inference and scoring;
- zero CDM/TEDS timeouts and exceptions;
- exact notebook report generated from pinned sources;
- cold and warm performance runs on the same hardware;
- evidence, documentation, and claim consistency audit;
- full local check suite and clean Git diff check.

## 11. Delivery

Work is committed in reviewable phases on `codex/top-tier-quality`. Only small,
auditable evidence and summaries are committed. Local datasets, model files,
raw prediction directories, logs, temporary CDM renders, and secrets remain
untracked.

After all gates pass:

1. push the branch to `AIwork4me/PaddleOCR-VL-ROCm`;
2. open a reviewable pull request containing the evidence summary;
3. merge only after required checks pass;
4. tag the release and publish release notes that name tested hardware and
   known limitations.

## 12. Risks and Mitigations

- **Benchmark version drift:** pin v1.6 commit, dataset hash, and config.
- **Windows CDM implementation drift:** keep platform patches isolated, record
  their diff, and verify known per-sample fixtures against official behavior.
- **Score inflation from missing pages:** require 1651/1651 predictions and
  explicit GT-page denominators.
- **Benchmark overfitting:** forbid sample-specific rules and require generic
  tests plus full-suite regression evidence.
- **Aggressive concurrency causing failures:** use bounded scheduling and memory
  checks with a conservative fallback.
- **Unverifiable marketing claims:** link every number to a versioned artifact
  and label external CUDA results separately.
- **Large downloads and fragile setup:** use resumable, checksum-verified,
  cached downloads with mirror and offline support.

## 13. Acceptance Summary

The release is complete only when all of the following are true:

- OmniDocBench is pinned to v1.6 commit `147cd5ac`.
- Both engines are scored with the exact official page-level and notebook
  rounding rules.
- Lightweight Overall is at least 96.13 and no worse than official-local.
- All 1651 pages succeed and CDM/TEDS have no timeout or exception.
- Mean latency is at most 13.00 seconds and P95 is at most 34.82 seconds on the
  same-machine benchmark.
- Input, parameter, JSON, Markdown, and filename contracts pass.
- Both onboarding journeys pass on a clean Windows AMD setup.
- README claims, evidence, English/Chinese docs, and release notes agree.
- The reviewed work is pushed to the target GitHub repository.
