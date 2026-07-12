# DirectML Provider Metadata Design

## Goal

Persist the requested and active layout execution providers through inference
traces, lightweight evaluation run statistics, run summaries, and benchmark
provenance without changing the native PaddleOCR result JSON schema.

## Stable contract

Every supported metadata surface uses these exact fields:

- `layout_provider_requested`: the user-facing selection such as `auto`,
  `directml`, or `cpu`.
- `layout_providers_active`: the ordered provider list reported by the live ONNX
  Runtime session. DirectML evidence requires `DmlExecutionProvider` first.

The PP-DocLayout model and managed pipeline expose these stable attributes.
Existing `active_providers`, `layout_provider`, and `active_layout_providers`
attributes remain as compatibility aliases.

## Data flow

The managed pipeline resolves and constructs the layout session, then passes
the two stable metadata values explicitly to `run_light_parser`. Each block
trace event receives both fields when it is created. Result serialization is
unchanged.

The lightweight folder adapter initializes the layout session before processing
pages so Windows `auto` fails closed at startup. It then reads the stable fields
from the initialized pipeline and stores them at the top level of both its
returned summary and `_run_stats.json`.

Artifact generation copies the same top-level fields from run stats into the
published run summary and benchmark provenance. No provider value is inferred
from another field, preserving the distinction between `auto` and an explicit
`directml` request.

## Error and compatibility behavior

DirectML initialization errors propagate before page processing and before a
misleading benchmark run can begin. Provider ordering is preserved verbatim.
Legacy callers that do not supply trace metadata receive empty/default provider
metadata only outside the managed path; existing result and trace fields remain
unchanged.

## Verification

Tests cover provider fields on block traces, initialized lightweight adapter
summaries and `_run_stats.json`, artifact run summaries, and provenance. The
DirectML path additionally asserts `DmlExecutionProvider` is first. Existing
DirectML resolution/session tests, trace tests, adapter tests, artifact tests,
the real local GPU smoke, and the full suite remain required.
