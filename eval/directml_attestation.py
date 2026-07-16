"""Fail-closed attestation of DirectML node execution from an ORT profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

DML_PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON value: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_strict_json(path: Path, raw: bytes | None = None) -> object:
    source = Path(path)
    if raw is None:
        raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid JSON in {source}: invalid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source}: {exc.msg}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc


def attest_directml_profile(
    profile_path: Path, run_stats: Mapping[str, object]
) -> dict[str, object]:
    """Summarize ORT node providers and attest the strict DirectML evidence contract."""
    profile = Path(profile_path).resolve(strict=True)
    raw = profile.read_bytes()
    events = _load_strict_json(profile, raw)
    if not isinstance(events, list):
        raise ValueError("ORT profile must be a JSON event list")
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("Every ORT profile event must be a JSON object")

    providers: list[str] = []
    missing_provider_nodes = 0
    for event in events:
        if event.get("cat") != "Node":
            continue
        args = event.get("args")
        provider = args.get("provider") if isinstance(args, Mapping) else None
        if not isinstance(provider, str) or not provider:
            missing_provider_nodes += 1
        else:
            providers.append(provider)

    dml_nodes = providers.count(DML_PROVIDER)
    cpu_nodes = providers.count(CPU_PROVIDER)
    provider_nodes = dml_nodes + cpu_nodes
    dml_node_share = dml_nodes / provider_nodes if provider_nodes else 0.0
    cpu_node_share = cpu_nodes / provider_nodes if provider_nodes else 0.0
    other_providers = sorted(set(providers) - {DML_PROVIDER, CPU_PROVIDER})
    other_provider_nodes = sum(provider in other_providers for provider in providers)
    requested = run_stats.get("layout_provider_requested")
    active = run_stats.get("layout_providers_active")
    fallback_disabled = run_stats.get("layout_fallback_disabled")
    passed = (
        type(requested) is str
        and requested == "auto"
        and type(active) is list
        and active == [DML_PROVIDER, CPU_PROVIDER]
        and type(fallback_disabled) is bool
        and fallback_disabled is True
        and dml_nodes > 0
        and dml_node_share > 0.5
        and missing_provider_nodes == 0
        and not other_providers
    )
    return {
        "dml_node_events": dml_nodes,
        "cpu_node_events": cpu_nodes,
        "dml_node_share": dml_node_share,
        "cpu_node_share": cpu_node_share,
        "missing_provider_node_events": missing_provider_nodes,
        "other_provider_node_events": other_provider_nodes,
        "node_providers": sorted(set(providers)),
        "other_providers": other_providers,
        "profile_sha256": hashlib.sha256(raw).hexdigest(),
        "profile_bytes": len(raw),
        "verdict": "PASS" if passed else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--allow-fail-verdict", action="store_true")
    args = parser.parse_args()
    try:
        stats = _load_strict_json(args.stats)
        if not isinstance(stats, dict):
            raise ValueError("Run stats must be a JSON object")
        report = attest_directml_profile(args.profile, stats)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2))
    if report["verdict"] == "FAIL" and not args.allow_fail_verdict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
