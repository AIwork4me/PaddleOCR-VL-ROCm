"""Fail-closed attestation of DirectML node execution from an ORT profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

DML_PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


def attest_directml_profile(
    profile_path: Path, run_stats: Mapping[str, object]
) -> dict[str, object]:
    """Summarize ORT node providers and attest the strict DirectML evidence contract."""
    profile = Path(profile_path).resolve(strict=True)
    raw = profile.read_bytes()
    events = json.loads(raw)
    if not isinstance(events, list):
        raise ValueError("ORT profile must be a JSON event list")

    providers: list[str] = []
    missing_provider_nodes = 0
    for event in events:
        if not isinstance(event, Mapping) or event.get("cat") != "Node":
            continue
        args = event.get("args")
        provider = args.get("provider") if isinstance(args, Mapping) else None
        if not isinstance(provider, str) or not provider:
            missing_provider_nodes += 1
        else:
            providers.append(provider)

    dml_nodes = providers.count(DML_PROVIDER)
    cpu_nodes = providers.count(CPU_PROVIDER)
    other_providers = sorted(set(providers) - {DML_PROVIDER, CPU_PROVIDER})
    other_provider_nodes = sum(provider in other_providers for provider in providers)
    requested = run_stats.get("layout_provider_requested")
    active = run_stats.get("layout_providers_active")
    fallback_disabled = run_stats.get("layout_fallback_disabled")
    passed = (
        requested == "auto"
        and active == [DML_PROVIDER, CPU_PROVIDER]
        and fallback_disabled is True
        and dml_nodes > 0
        and cpu_nodes == 0
        and missing_provider_nodes == 0
        and not other_providers
    )
    return {
        "dml_node_events": dml_nodes,
        "cpu_node_events": cpu_nodes,
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
    args = parser.parse_args()
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    if not isinstance(stats, Mapping):
        raise SystemExit("Run stats must be a JSON object")
    print(json.dumps(attest_directml_profile(args.profile, stats), indent=2))


if __name__ == "__main__":
    main()
