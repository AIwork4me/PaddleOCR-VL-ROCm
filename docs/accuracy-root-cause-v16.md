# OmniDocBench v1.6 Accuracy Root-Cause Report

- Date: 2026-07-12
- Benchmark contract: OmniDocBench v1.6, commit `147cd5ac9472002f5751221d390bf00abdbc0d2f`
- Status: historical diagnosis complete; fresh release evidence blocked

## Executive finding

The fresh corrected 1,651-page comparison does not exist. The sole missing
official page, `newspaper_The Times UK_0801@magazinesclubnew_page_031.png`,
reliably reaches llama.cpp inference and then fails the official engine at the
PEG chat-output parser with HTTP 500. Builds 9884 and 9637, including the
publisher's minimal launch, reproduced the failure. The official run remains
`count=1651`, `ok=1650`, `fail=1`, `fallback=0`; no lightweight fallback was
used. Therefore none of the numbers below is fresh release evidence.

Authentic historical scorer artifacts do show enough recoverable page-level
loss to cover the 0.182-point target gap: Formula CDM loss pages contain at
most 0.2020 Overall points of historical headroom, Table TEDS loss pages
0.0618, and text loss pages 0.0285. Their sum, 0.2924, is an upper bound that
assumes every loss is recovered with no regression on the more numerous gains.
It is not a forecast. The current historical lightweight result is already
0.1677 Overall points better than the historical official result.

No authentic canonical official/lightweight inference traces were found.
Consequently the scorer records prove final matched-output differences, but
they do not prove whether the first inference divergence is layout, crop,
payload, raw VLM output, or post-processing. Those causes remain unproven and
no production inference change is authorized by this report.

## Evidence classes and provenance

### Fresh evidence

- Official repair attempts: deterministic HTTP 500 at the PEG parser on the
  sole missing page; no prediction was emitted or merged.
- DirectML contract smoke: the managed Windows `auto` path recorded
  `layout_provider_requested=auto` and
  `layout_providers_active=[DmlExecutionProvider, CPUExecutionProvider]`, with
  DirectML first and fallback disabled. The real PP-DocLayoutV3 smoke returned
  12 boxes. This proves the current provider contract, not the provider used by
  the historical benchmark predictions.

### Historical/reconstructed evidence

The official side is the authentic per-sample scorer output at:

`C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm\eval\.omnidocbench\result\paddleocr_official_local_llamacpp_gguf_v16_quick_match_*_result.json`

The lightweight side is the later, CDM-populated authentic scorer output at:

`C:\Users\rocm\Desktop\omnidocbench-amd-windows\eval-infra\01-omnidocbench\OmniDocBench\result\paddleocrvl_rocm_cdm_quick_match_*_result.json`

Artifact SHA-256 values:

| Component | Official | Lightweight |
|---|---|---|
| metric | `524ab3fe6bc2a12dc38dc0df99d09acb09507cc72b0346eaab9eb55f484f8d91` | `f2fc8e85dc26980cc09c68a6903b70d6d3ccdb62edfe3444b09d07de4b3b3ae4` |
| Formula CDM samples | `89c7c1d9fc16622d510a0429f93021e4afec10adfec1662a68084d0c87b8cde8` | `3ed359c6d8f71cf836f03f9bfd9c1d2b0d2fa6f700aff8d581fa864ed9cc4353` |
| Table TEDS samples | `e94b314d3b4a44d518f81430c602c785c49bf8e874caf27cef1dc528d314716a` | `81f3975146b990638193eb8b81bf3e24dc968b53ad7ade21142fe4fe93cac759` |
| text samples | `b0d2e1871e8f107341a888f79a1dc9f7ce89367ae534d598335d18c4716590a3` | `9811a06b747a3628828a7266f6abc2fcc23b1ac0cc2bff469b87aef35bae7eb9` |
| reading order | `a82fe11b9463c17ea75460a405a3df24409c81987617d7abf178977f2253c2ad` | `11ae78bcdb059e573e4081be09e094763f8c936c69a55265fb8bfff1d34aa606` |

An older file at
`C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm\result\paddleocrvl_rocm_cdm_quick_match_display_formula_result.json`
contains 2,352 zero CDM values and its companion metric reports page CDM zero.
It is a genuine historical intermediate, but it is not paired with the later
96.922 metric and is excluded. Mixing that file with the later metric would
create a false 32.307-point Overall loss. This is the highest-ranked proven
evidence-integrity hazard.

## Official v1.6 notebook reconstruction

The observable metric JSON contains the official notebook fields. Components
were rounded to three decimals before Overall:

| Engine | Text Edit | Formula CDM | Table TEDS | Reading order | Overall |
|---|---:|---:|---:|---:|---:|
| Historical official | 0.034 | 96.502 | 94.239 | 0.129487 | 95.7803 |
| Historical lightweight | 0.034 | 96.922 | 94.322 | 0.128238 | 95.9480 |
| Lightweight minus official | 0.000 | +0.420 | +0.083 | -0.001249 | +0.1677 |

Formula CDM was independently reconstructed as the equal mean of sample CDM
within each of 313 GT formula pages and then the equal mean of page scores.
Table TEDS was reconstructed identically over 458 GT table pages. Text used
the v1.6 `sum(Edit_num)/sum(upper_len)` calculation within each of 1,557 pages
and then the equal mean of page scores. Reading order used its 1,638 page
records. All four reconstructions reproduce the metric JSON fields.

## Current delta analyzer result

The current `scripts/analyze_omnidocbench_deltas.py` was run on isolated copies
of the selected Formula CDM and Table TEDS artifacts with `--top 100`. It
reported 2,945 stable matched sample keys: 2,280 formulas and all 665 tables.
Formula matching also reported 72 official-only and 72 lightweight-only keys;
these are excluded from paired sample attribution. Because the analyzer's
component mean covers only matched keys, authoritative component values above
come from full official v1.6 page reconstruction, not the analyzer's subset.
On that matched subset, its signed `official_score - lightweight_score` page
means were -0.0044675 for Formula CDM and -0.0008284 for Table TEDS (negative
means the lightweight side was better).

| Component | Matched samples | Prediction differs | Loss samples | Gain samples | Equal samples |
|---|---:|---:|---:|---:|---:|
| Formula CDM | 2,280 | 295 | 49 | 104 | 2,127 |
| Table TEDS | 665 | 77 | 32 | 39 | 594 |
| Text Edit | 19,657 | 983 | 190 | 180 | 19,287 |
| Reading-order Edit | 1,638 | 82 | 28 | 32 | 1,578 |

The text and reading-order rows were reconstructed with the same v1.6
statistics after stable `(img_id, gt_idx)` alignment; they are not emitted by
the current Formula/Table analyzer CLI.

## Ranked findings

### 0. Evidence pairing can manufacture a false CDM regression

- Affected samples: 2,352 in the excluded all-zero intermediate.
- Observable divergence: artifact/metric consistency boundary.
- Quantified impact if mispaired: Formula 96.922 becomes 0, falsely reducing
  Overall by 32.307 points.
- Smallest generic correction boundary: validate that per-sample page
  reconstruction equals the companion metric before delta ranking.
- Status: proven evidence-integrity defect; not an inference defect.

### 1. Formula loss pages are the largest historical recovery pool

- Loss/gain/equal pages: 31/65/217 of 313.
- Maximum historical Overall contribution from loss recovery: 0.2020 points.
- First observable divergence: final scorer-matched formula prediction/CDM.
- Earliest inference divergence and smallest production correction boundary:
  unproven until paired canonical traces and oracle replays exist.

Representative paired cases:

| Page / GT index | Official CDM | Lightweight CDM | Page delta | Observation |
|---|---:|---:|---:|---|
| `page-7dfc88d8-6d95-446c-b910-2410e8552f76.png` / 1 | 0.965 | 0.061 | 0.9040 | Both predictions non-empty and differ. |
| `page-21967f5d-667d-488e-a5b3-76b9d6f53656.png` / 21 | 1.000 | 0.000 | 0.2531 | Lightweight matched prediction is empty for this sample. |
| `page-dad0f4e5-290f-496f-bbdd-099ad75c6ff0.png` / 15 | 1.000 | 0.000 | 0.2000 | Both predictions non-empty and differ. |
| `page-05746fc5-2045-4dea-94e7-4bbab648d702.png` / 12 | 0.981 | 0.000 | 0.1635 | Both predictions non-empty and differ. |
| `book_en_国外数学教材-数论-Melvyn B. Nathanson—Elementary Methods in Number Theory_0451.png` / 7 | 0.956 | 0.000 | 0.0797 | Both predictions non-empty and differ. |

The two empty lightweight loss samples are observations, not proof of
truncation, layout loss, or post-processing deletion.

### 2. Table structure/content losses form a smaller recovery pool

- Loss/gain/equal pages: 32/37/389 of 458.
- Maximum historical Overall contribution from loss recovery: 0.0618 points.
- First observable divergence: final scorer-matched HTML/TEDS.
- Earliest inference divergence and smallest production correction boundary:
  unproven. Visible rowspan/colspan and content differences do not establish
  whether the model or a downstream transformation introduced them.

| Page / GT index | Official TEDS | Lightweight TEDS | Page delta |
|---|---:|---:|---:|
| `yanbaor2_yanbaoPPT_2098.jpg` / 0 | 0.9750 | 0.6341 | 0.3409 |
| `yanbaor2_3e1be78252e2fdfe1adf12bba38ec2a7b30699e152d61269aa6e5827f5adcc35.pdf_13.jpg` / 0 | 0.7385 | 0.5846 | 0.1538 |
| `docstructbench_llm-raw-the-eye-o.O-TraneGuide.pdf_7.jpg` / 0 | 0.8548 | 0.7080 | 0.1468 |
| `eastmoney_ea59610b9b1a8f0df46f7a89da1116cbf256c772e1148f26017991e28c8bca21.pdf_18.jpg` / 1 | 0.9742 | 0.9085 | 0.0329 |
| `docstructbench_enbook-zlib-o.O-17761417.pdf_894.jpg` / 0 | 0.8704 | 0.8398 | 0.0306 |

### 3. Text loss pages provide limited but measurable headroom

- Loss/gain/equal pages: 115/86/1,356 of 1,557.
- Maximum historical Overall contribution from loss recovery: 0.0285 points.
- First observable divergence: final scorer-matched normalized text.
- Earliest inference divergence and smallest production correction boundary:
  unproven.

| Page / representative GT index | Official Edit | Lightweight Edit | Page loss |
|---|---:|---:|---:|
| `jiaocaineedrop_jiaocai_needrop_en_349.jpg` / 19 | 0.0570 | 1.0000 | 0.2251 |
| `PPT_lecture1_page_005.png` / 3 | 0.0000 | 0.1776 | 0.1267 |
| `magazine_TheEconomist.2023.12.09_page_048.png` / 20 | 0.0048 | 0.9592 | 0.0736 |
| `page-2329f04a-41b3-435b-993a-a0652294b07d.png` / 4 | 0.7294 | 0.7550 | 0.0656 |
| `docstructbench_llm-raw-the-eye-o.O-Player%27s%20Options%20-%20Halflings.pdf_11.jpg` / 8 | 0.0000 | 0.2605 | 0.0635 |

### 4. Reading-order losses are visible but do not enter Overall

- Loss/gain/equal pages: 28/32/1,578 of 1,638.
- Overall contribution: exactly zero under the v1.6 notebook formula.
- First observable divergence: final matched order list.
- Representative loss pages: `page-2329f04a-41b3-435b-993a-a0652294b07d.png`
  (0.2857), `page-21967f5d-667d-488e-a5b3-76b9d6f53656.png`
  (0.1852), `page-268266af-56c0-4b3b-9d07-73c6e50feb58.png`
  (0.1667), `page-4319d401-c9e8-4326-9869-7572cf2e0e96.png`
  (0.1429), and `color_textbook_教材全解1+1二年级下册英语上海牛津版_page_006.png`
  (0.1333).

## Trace-boundary accounting

| Boundary | Classified count | Evidence conclusion |
|---|---:|---|
| Dataset/coverage | 1 official failed page | Fresh release comparison blocked. |
| Formula CDM final output | 49 matched loss samples | Measured; 72 keys per side remain unmatched. |
| Table TEDS final output | 32 matched loss samples | Measured with complete 665-key alignment. |
| Text final output | 190 matched loss samples | Measured on stable keys. |
| Reading-order final output | 28 matched loss samples | Measured; excluded from Overall. |
| Layout / crop | unproven | No authentic paired canonical trace. |
| Payload | unproven | No authentic paired canonical trace. |
| Raw VLM output | unproven | No authentic paired canonical trace. |
| Post-processing | unproven | No raw-to-final paired trace. |

Counts above are not silently converted to zero. “Unproven” means the
observable does not exist.

## Decision

The next work must first make evidence pairing fail closed, capture canonical
traces for the named cases with `DmlExecutionProvider` active on every Windows
lightweight trace, and perform crop/payload/raw/final oracle swaps. Only a
subsequent evidence-specific plan may change inference behavior. The current
historical loss pools are suitable for prioritization and fixtures, not for
claiming a production root cause or a fresh 96.13 acceptance result.
