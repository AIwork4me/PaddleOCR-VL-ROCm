# OmniDocBench v1.6 Accuracy Root-Cause Report

> Historical diagnostic record. Current public score, coverage, aggregation,
> provenance, and gate interpretation lives in
> [`docs/benchmarks/omnidocbench-v1.6.md`](benchmarks/omnidocbench-v1.6.md).

- Date: 2026-07-12
- Benchmark contract: OmniDocBench v1.6, commit `147cd5ac9472002f5751221d390bf00abdbc0d2f`
- Status: DirectML-qualified lightweight traces captured; paired official trace diagnosis incomplete

## Executive finding

The fresh corrected comparison must be regenerated under the approved known
failure contract. The sole missing official page,
`newspaper_The Times UK_0801@magazinesclubnew_page_031.png`, reliably reaches
llama.cpp inference and then fails the official engine at the PEG chat-output
parser with HTTP 500. Builds 9884 and 9637 reproduced the failure, now publicly
tracked in PaddleOCR issue #18248. The accepted operational shape is exactly
`count=1651`, `ok=1650`, `fail=1`, `fallback=0`; no fallback or synthetic output
is allowed. Scoring retains all 1,651 GT pages and treats this page as empty.
The historical numbers below are still not fresh release evidence.

Authentic historical scorer artifacts do show enough candidate page-level
loss to cover the 0.182-point target gap: Formula CDM loss pages contain at
most 0.2020 Overall points of historical headroom, Table TEDS loss pages
0.0618, and text loss pages 0.0285. Their sum, 0.2924, is an upper bound that
assumes every loss is recovered with no regression on the more numerous gains.
It is not a forecast. The current historical lightweight result is already
0.1677 Overall points better than the historical official result.

Twenty authentic canonical lightweight inference traces are now captured with
DirectML active first. No matching official inference traces were found, so
Task 5 root-cause diagnosis remains incomplete.
Consequently the scorer records prove final matched-output differences, but
they do not prove whether the first inference divergence is layout, crop,
payload, raw VLM output, or post-processing. Those causes remain unproven and
no production inference change is authorized by this report.

The authenticated attribution run evaluated all 20 manifest cases. All 20
returned `status="unproven"`; the replay callback was invoked zero times, no
oracle score or synthetic contribution was produced, and no first causal
boundary was established.

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

`<repo>\eval\.omnidocbench\result\paddleocr_official_local_llamacpp_gguf_v16_quick_match_*_result.json`

The lightweight side is the later, CDM-populated authentic scorer output at:

`<omnidocbench-worktree>\eval-infra\01-omnidocbench\OmniDocBench\result\paddleocrvl_rocm_cdm_quick_match_*_result.json`

Artifact SHA-256 values:

| Component | Official | Lightweight |
|---|---|---|
| metric | `524ab3fe6bc2a12dc38dc0df99d09acb09507cc72b0346eaab9eb55f484f8d91` | `f2fc8e85dc26980cc09c68a6903b70d6d3ccdb62edfe3444b09d07de4b3b3ae4` |
| Formula CDM samples | `89c7c1d9fc16622d510a0429f93021e4afec10adfec1662a68084d0c87b8cde8` | `3ed359c6d8f71cf836f03f9bfd9c1d2b0d2fa6f700aff8d581fa864ed9cc4353` |
| Table TEDS samples | `e94b314d3b4a44d518f81430c602c785c49bf8e874caf27cef1dc528d314716a` | `81f3975146b990638193eb8b81bf3e24dc968b53ad7ade21142fe4fe93cac759` |
| text samples | `b0d2e1871e8f107341a888f79a1dc9f7ce89367ae534d598335d18c4716590a3` | `9811a06b747a3628828a7266f6abc2fcc23b1ac0cc2bff469b87aef35bae7eb9` |
| reading order | `a82fe11b9463c17ea75460a405a3df24409c81987617d7abf178977f2253c2ad` | `11ae78bcdb059e573e4081be09e094763f8c936c69a55265fb8bfff1d34aa606` |

An older file at
`<repo>\result\paddleocrvl_rocm_cdm_quick_match_display_formula_result.json`
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

The corrected `scripts/analyze_omnidocbench_deltas.py` was run on isolated copies
of the selected Formula CDM and Table TEDS artifacts with `--top 100`. It
reported 3,015 stable matched sample keys: 2,350 formulas and all 665 tables.
Formula identity is `(img_id, canonical gt_position, SHA-256(raw gt))`, not the
scorer-local `gt_idx`. Formula matching reported two official-only and two
lightweight-only keys;
these are excluded from paired sample attribution. Because the analyzer's
component mean covers only matched keys, authoritative component values above
come from full official v1.6 page reconstruction, not the analyzer's subset.
On that matched subset, its signed `official_score - lightweight_score` page
means were -0.0037960 for Formula CDM and -0.0008284 for Table TEDS (negative
means the lightweight side was better).

Before ranking, the analyzer now fails closed unless each side contains exactly
2,352 Formula samples over 313 pages and 665 Table samples over 458 pages, and
unless those fixed-denominator page reconstructions equal the companion metric
fields within `1e-12`. A consistently incomplete sample file plus a matching
incomplete metric can no longer pass.

| Component | Matched samples | Prediction differs | Loss samples | Gain samples | Equal samples |
|---|---:|---:|---:|---:|---:|
| Formula CDM | 2,350 | 294 | 56 | 102 | 2,192 |
| Table TEDS | 665 | 77 | 32 | 39 | 594 |
| Text Edit | 19,657 | 983 | 190 | 180 | 19,287 |
| Reading-order Edit | 1,638 | 82 | 28 | 32 | 1,578 |

The text and reading-order rows were reconstructed with the same v1.6
statistics after stable `(img_id, gt_idx)` alignment; they are not emitted by
the current Formula/Table analyzer CLI.

## Prioritized scorer observables (not diagnosed inference root causes)

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

| Page / official GT index / source position | Official CDM | Lightweight CDM | Page delta | Observation |
|---|---:|---:|---:|---|
| `page-7dfc88d8-6d95-446c-b910-2410e8552f76.png` / 1 / `[1]` | 0.965 | 0.061 | 0.9040 | Both predictions non-empty and differ. |
| `page-dad0f4e5-290f-496f-bbdd-099ad75c6ff0.png` / 15 / `[15]` | 1.000 | 0.000 | 0.2000 | Both predictions non-empty and differ. |
| `page-05746fc5-2045-4dea-94e7-4bbab648d702.png` / 12 / `[12]` | 0.981 | 0.000 | 0.1635 | Both predictions non-empty and differ. |
| `book_en_国外数学教材-数论-Melvyn B. Nathanson—Elementary Methods in Number Theory_0451.png` / 7 / `[6]` | 0.956 | 0.000 | 0.0797 | Both predictions non-empty and differ. |
| `yanbaopptmerge_9081a70ff98b3e7d640660a9412c447d.pdf_1287.jpg` / 26 / `[52]` | 0.788 | 0.381 | 0.0139 | Both predictions non-empty and differ. |

Three of the 56 aligned Formula loss samples have empty lightweight matched
predictions. That is an observation, not proof of truncation, layout loss, or
post-processing deletion.

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
| Dataset/coverage | 1 approved official PEG failure | Full 1,651-page scoring denominator retained; fresh comparison must be regenerated. |
| Formula CDM final output | 56 matched loss samples | Measured; two source-identity keys per side remain unmatched. |
| Table TEDS final output | 32 matched loss samples | Measured with complete 665-key alignment. |
| Text final output | 190 matched loss samples | Measured on stable keys. |
| Reading-order final output | 28 matched loss samples | Measured; excluded from Overall. |
| Layout / crop | unproven | No authentic paired canonical trace. |
| Payload | unproven | No authentic paired canonical trace. |
| Raw VLM output | unproven | No authentic paired canonical trace. |
| Post-processing | unproven | No raw-to-final paired trace. |

Counts above are not silently converted to zero. “Unproven” means the
observable does not exist.

## Authenticated oracle attribution and production admission

The attribution input is the committed, authenticated capture summary described
below. It supplies lightweight boundary hashes and historical official scorer
fingerprints, but it supplies no official value at the same `crop`, `payload`,
`raw_vlm`, or `final_output` boundary. The ordered attribution therefore cannot
perform a valid swap.

| Manifest cases | Proven causes | Unproven cases | Replay calls | Synthetic contributions | Oracle improvements | First causal boundaries | Authorized production tasks |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0 | 20 | 0 | 0 | 0 observed | 0 established | 0 |

There is no proven-cause row for which an exact page/GT index, before score,
oracle score, affected-case count, estimated v1.6 Overall contribution, first
causal boundary, and trace hashes can all be recorded. Inventing missing values
or copying values from the historical recovery pools would misclassify scorer
observations as causal inference evidence. The historical Formula, Table, Text,
and reading-order pools above remain unproven prioritization evidence only.

A production task is admitted only when one cause satisfies all four conditions:

1. At least one exact manifest fixture reproduces the divergence under the
   authenticated DirectML-first trace contract.
2. A same-boundary authenticated official oracle swap improves the applicable
   official v1.6 metric from a recorded before score to a recorded oracle score.
3. Named neighboring gain fixtures exercise the same generic behavior and are
   designated as non-regression cases.
4. The earliest causal boundary is established from ordered, one-variable-at-a-
   time swaps, identifying a generic correction boundary rather than a
   case-specific output substitution.

No cause currently satisfies all four conditions. In particular, the official
same-boundary oracle inputs needed to establish metric improvement and the
earliest causal boundary are absent. To reopen admission,
capture or otherwise authenticate the missing official boundary values for an
exact manifest case, retain the existing case and trace hashes, run the ordered
oracle replay with the official v1.6 scorer, record a positive metric delta and
the earliest positive boundary, and name gain fixtures that must not regress.
Until that evidence exists, prompt, crop, model, normalization, and serialization
changes are not authorized.

## DirectML-qualified canonical capture summary

On 2026-07-12, `scripts/record_trace.py` captured each of the twenty manifest
cases against the already-running local PaddleOCR-VL 1.6 llama.cpp service.
Every accepted trace recorded `layout_provider_requested=auto` and active
providers exactly `[DmlExecutionProvider, CPUExecutionProvider]`; the layout
session disables fallback and DirectML was first. Raw JSONL, crops, payloads,
responses, and predictions remain ignored and untracked.

The authoritative scalar record is
`tests/fixtures/accuracy/v16-trace-capture-summary.json`. It binds every exact
64-character case ID to full 64-character SHA-256 values for the raw trace
artifact, every observable lightweight boundary, and the official final scorer
rows. It also authenticates the manifest, the complete lightweight capture
set, and all four official scorer source artifacts. A fresh checkout can
validate this committed full-hash contract, but cannot recompute it without the
private benchmark images and ignored raw capture/scorer artifacts.

`LW final` fingerprints the current lightweight parser output. `Official
scorer` fingerprints the authentic historical official scorer rows for that
page. These are not the same boundary, so they were not passed to
`compare_inference_traces.py` and no first inference divergence is claimed.
Official layout, crop, payload, and raw VLM boundaries are explicitly
`unobservable`; the closest observable official boundary is the historical
scorer output. The abbreviated values below are for visual navigation only and
are not authoritative evidence; validation relies exclusively on the full
hashes in the machine-readable summary.

| Case ID prefix | Trace | Layout | Crop | Payload | Raw VLM | LW final | Official scorer |
|---|---|---|---|---|---|---|---|
| `b93ac0c74e16` | `6c94cfa1f1fa` | `33cf437eafd1` | `cbdef18a56ac` | `8e6cc4cd0766` | `cea435df2c76` | `09713e6742d8` | `fe877e9a21bc` |
| `e5a0e8987c11` | `388750e68dee` | `0ab09ed7328a` | `17b537001e7b` | `7e4d2829ba13` | `c02a0d6a17b9` | `3f7ca6d4341f` | `5e3e88703088` |
| `13d7060250dd` | `156d577dd309` | `a745462dc978` | `29d70c2ac417` | `10a5bffe127f` | `2b9b46cf061e` | `945ad8276470` | `730af95877b2` |
| `06f6f1ff2842` | `8f7f43cbeba3` | `3ed5806f06d6` | `813dff4cb7c9` | `e6f1f98853f0` | `ed827b34ea60` | `6af85bec81dc` | `a778915b53d9` |
| `8602994a7678` | `6fd067e0c102` | `997e95064a5d` | `ba3594136d1a` | `73fc586fd2dd` | `1a746b354998` | `d6c577598b9e` | `a1f48597f042` |
| `d18aa15a556c` | `9156aea16a1a` | `c132f0989f8d` | `6ab50133f469` | `c6bbc2223e80` | `c69252f881f0` | `a734baf8be5a` | `bbf19cd4f026` |
| `a963d6c05bac` | `6659bce8f3b8` | `cda77781d7da` | `d8114c7a096f` | `837c3f8a80c1` | `fa8fa0cf0795` | `e6fa81a81665` | `3fcf912a634d` |
| `e0d93191a7c7` | `49ed7ea8a511` | `51251451ec48` | `42e7a49ba5a7` | `31d6d9cd113f` | `7ba72bfba742` | `42360a2c1c66` | `4882a07041e7` |
| `5aa11596200d` | `4fde689b2fed` | `a1e3a747afce` | `a93d93188f16` | `bd7977688b61` | `664853e3340c` | `3dc3f1cbfa49` | `533acc587f59` |
| `ed2a0570f69a` | `bb4f37527ca3` | `8d4edd8c4a7d` | `9c8734f4e47c` | `ebe981e642ee` | `655ef0fa3eab` | `c5211f5f83a3` | `371f57d6aa19` |
| `dba1aec09157` | `87bddd1498ad` | `1aa79571c7e3` | `89b46a7e613a` | `0c7992539425` | `6aef8bf8669d` | `e87adaf466ef` | `e078fd48f236` |
| `d34e246c22c4` | `ad1203c81cbe` | `b3421ec0c678` | `c8c05df35536` | `97f1624e6565` | `f92ba21b7c03` | `e1a29fccbe00` | `7fbd776373d0` |
| `a1b6c1a7af8a` | `985763956e6d` | `26323b85593b` | `327522ae50e4` | `26300566bbe1` | `23ce08c03de5` | `8cb6ac02be34` | `a62006d9d4e4` |
| `fc98de317a28` | `db473ae614f5` | `83ebced010ce` | `9a7778f4fae8` | `1d69566c26ab` | `ac9c06444d07` | `dcc5caff6441` | `3801f4ff9651` |
| `9bb616a6d1c4` | `cc438206e680` | `a6dca95d0c41` | `5f5ee170b78a` | `6add63eb3117` | `79d202b1960e` | `1821e1fd0047` | `43d990fdf86c` |
| `4d791bbfe0ae` | `09a25e2d3c76` | `83ebced010ce` | `9a7778f4fae8` | `1d69566c26ab` | `ba047a696093` | `8a82154130fc` | `ee7a8673874b` |
| `2f98a52ddf07` | `317fb06ed0a0` | `5d72b0f2af19` | `2baa7c9f3bae` | `2cfca4b08760` | `072dcfbe633d` | `ecb12f4e3913` | `5ee38cb8508d` |
| `1251d0294f22` | `4d2ca15eaf13` | `c2986fea0ca6` | `58bb6439368d` | `59afba2be6a6` | `0740fe08ca8a` | `2260f025ecb0` | `be4d1d8d6b66` |
| `427790a859a7` | `2ab227368545` | `dd12be69d75f` | `6aa34d110c99` | `54e98a3b3639` | `c8292d90b6e2` | `8bc8eff63d9c` | `c93b58fda613` |
| `c868f6459ab2` | `c766b18c2766` | `d2f0398aebce` | `99a33ed73f43` | `b8076f7aba2b` | `80350088820e` | `14f74f4a60b5` | `75856e887868` |

## Decision

The production-plan admission gate authorizes zero inference-fix tasks. The
separate inference-fix plan records this closed decision and stops. Evidence
completion may resume only through authenticated same-boundary official oracle
inputs and the gate criteria above; it does not authorize speculative prompt,
crop, model, normalization, or serialization work. The historical loss pools
remain suitable for prioritization, not for claiming a production cause or a
fresh 96.13 acceptance result.
