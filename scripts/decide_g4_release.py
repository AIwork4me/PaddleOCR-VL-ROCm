"""Produce the independent final G4 decision and receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.g4_release import build_final_receipt, decide_g4_release, validate_source_receipts


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--performance-root", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite final G4 evidence: {output_root}")
    output_root.mkdir(parents=True)
    performance = args.performance_root.resolve()
    quality = args.quality_root.resolve()
    performance_artifact = performance / "g4-run-artifact.json"
    performance_decision = performance / "g4-decision.json"
    performance_receipt = performance / "g4-receipt.json"
    quality_artifact = quality / "g4-quality-artifact.json"
    quality_decision = quality / "g4-quality-decision.json"
    quality_receipt = quality / "g4-quality-receipt.json"
    quality_contract = quality / "g4-quality-scorer-contract.json"
    subset_gt = quality / "g4-quality-subset-gt.json"
    validate_source_receipts(
        manifest_path=args.manifest,
        performance_artifact_path=performance_artifact,
        performance_decision_path=performance_decision,
        performance_receipt=_read(performance_receipt),
        quality_artifact_path=quality_artifact,
        quality_decision_path=quality_decision,
        quality_receipt=_read(quality_receipt),
        quality_contract_path=quality_contract,
        subset_gt_path=subset_gt,
    )
    decision = decide_g4_release(
        manifest=_read(args.manifest),
        performance_artifact=_read(performance_artifact),
        performance_decision=_read(performance_decision),
        quality_artifact=_read(quality_artifact),
        quality_decision=_read(quality_decision),
    )
    decision_path = output_root / "g4-final-decision.json"
    receipt_path = output_root / "g4-final-receipt.json"
    _write(decision_path, decision)
    _write(
        receipt_path,
        build_final_receipt(
            {
                "sample_manifest": args.manifest,
                "performance_artifact": performance_artifact,
                "performance_decision": performance_decision,
                "performance_receipt": performance_receipt,
                "quality_artifact": quality_artifact,
                "quality_decision": quality_decision,
                "quality_receipt": quality_receipt,
                "final_decision": decision_path,
            }
        ),
    )
    print(json.dumps(decision, indent=2))
    return 0 if decision["g4"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
