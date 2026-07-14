from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.artifact_utils import sha256_file

TASK5_SCHEMA = 1
APPROVED_G0_RECEIPT_SHA256 = (
    "d0b7fcbe389e03439b5ba65126008fa5ee828a59e358ae0347c5bb6a51648a04"
)
OFFICIAL_OUTPUTS = (
    "results/official/metric.json",
    "results/official/metric-cdm.json",
    "results/official/provenance.json",
    "results/official/provenance-cdm.json",
    "results/official/run-summary.json",
    "results/official/run-summary-cdm.json",
)
TOP_LEVEL_KEYS = {
    "schema",
    "task5_root",
    "git_commit",
    "g0",
    "inputs",
    "environment",
    "contracts",
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
LOGICAL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Evidence input must be a regular file: {path}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def snapshot_sealed_g0(r7_root: Path, receipt_path: Path) -> dict[str, object]:
    root = r7_root.resolve(strict=True)
    return {
        "receipt": file_identity(receipt_path),
        "manifest": file_identity(root / "manifest.json"),
        "official_outputs": {
            relative: file_identity(root / relative) for relative in OFFICIAL_OUTPUTS
        },
    }


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_task5_manifest(
    *,
    r7_root: Path,
    receipt_path: Path,
    git_commit: str,
    inputs: Mapping[str, Path],
    environment: Mapping[str, object],
    contracts: Mapping[str, object],
) -> dict[str, object]:
    if GIT_COMMIT.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a full lowercase 40-character commit")
    if not inputs:
        raise ValueError("Task 5 inputs must be a non-empty object")
    input_identities: dict[str, object] = {}
    for name in inputs:
        if not isinstance(name, str) or LOGICAL_NAME.fullmatch(name) is None:
            raise ValueError(f"Invalid Task 5 input logical name: {name!r}")
    for name in sorted(inputs):
        input_identities[name] = file_identity(inputs[name])
    return {
        "schema": TASK5_SCHEMA,
        "task5_root": str((r7_root / "task5").resolve(strict=False)),
        "git_commit": git_commit,
        "g0": snapshot_sealed_g0(r7_root, receipt_path),
        "inputs": input_identities,
        "environment": _copy_json_object(environment, label="environment"),
        "contracts": _copy_json_object(contracts, label="contracts"),
    }


def validate_task5_manifest(
    manifest: Mapping[str, object], *, task5_root: Path
) -> None:
    if set(manifest) != TOP_LEVEL_KEYS:
        raise ValueError("Task 5 manifest has incorrect top-level keys")
    if (
        not isinstance(manifest["schema"], int)
        or isinstance(manifest["schema"], bool)
        or manifest["schema"] != TASK5_SCHEMA
    ):
        raise ValueError(f"Task 5 manifest schema must be {TASK5_SCHEMA}")
    git_commit = manifest["git_commit"]
    if not isinstance(git_commit, str) or GIT_COMMIT.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a full lowercase 40-character commit")

    g0 = _require_mapping(manifest["g0"], label="g0")
    if set(g0) != {"receipt", "manifest", "official_outputs"}:
        raise ValueError("g0 must contain exactly receipt, manifest, and official_outputs")
    sealed_manifest = _require_mapping(g0["manifest"], label="g0 manifest")
    manifest_path = _identity_path(sealed_manifest, label="g0 manifest")
    r7_root = manifest_path.parent
    if manifest_path != r7_root / "manifest.json":
        raise ValueError("Sealed G0 manifest path must be exactly r7/manifest.json")

    expected_root = r7_root / "task5"
    recorded_root_value = manifest["task5_root"]
    if not isinstance(recorded_root_value, str) or not recorded_root_value:
        raise ValueError("task5_root must be a non-empty absolute path")
    recorded_root = Path(recorded_root_value)
    supplied_absolute = task5_root.absolute()
    if (
        str(recorded_root) != str(expected_root)
        or str(supplied_absolute) != str(expected_root)
        or supplied_absolute.resolve(strict=False) != supplied_absolute
    ):
        raise ValueError("Task 5 root must be exactly r7/task5")

    receipt = _require_mapping(g0["receipt"], label="g0 receipt")
    receipt_digest = receipt.get("sha256")
    if receipt_digest != APPROVED_G0_RECEIPT_SHA256:
        raise ValueError("Manifest does not bind the approved G0 receipt SHA-256")
    _validate_identity(receipt, label="g0 receipt")
    _validate_identity(
        sealed_manifest,
        label="g0 manifest",
        expected_path=r7_root / "manifest.json",
    )

    official_outputs = _require_mapping(g0["official_outputs"], label="official_outputs")
    if set(official_outputs) != set(OFFICIAL_OUTPUTS):
        raise ValueError("g0 official_outputs must contain exactly the six approved outputs")
    for relative in OFFICIAL_OUTPUTS:
        identity = _require_mapping(official_outputs[relative], label=relative)
        _validate_identity(identity, label=relative, expected_path=r7_root / relative)

    inputs = _require_mapping(manifest["inputs"], label="inputs")
    if not inputs:
        raise ValueError("Task 5 inputs must be a non-empty object")
    for name in inputs:
        if not isinstance(name, str) or LOGICAL_NAME.fullmatch(name) is None:
            raise ValueError(f"Invalid Task 5 input logical name: {name!r}")
    if list(inputs) != sorted(inputs):
        raise ValueError("Task 5 input logical names must be sorted")
    for name, raw_identity in inputs.items():
        identity = _require_mapping(raw_identity, label=f"input {name!r}")
        _validate_identity(identity, label=f"input {name!r}")

    _copy_json_object(
        _require_mapping(manifest["environment"], label="environment"),
        label="environment",
    )
    _copy_json_object(
        _require_mapping(manifest["contracts"], label="contracts"),
        label="contracts",
    )


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Task 5 manifest {label} must be an object")
    return value


def _identity_path(identity: Mapping[str, object], *, label: str) -> Path:
    path_value = identity.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"Evidence {label} path must be a non-empty string")
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"Evidence {label} path must be absolute")
    return path


def _validate_identity(
    identity: Mapping[str, object],
    *,
    label: str,
    expected_path: Path | None = None,
) -> None:
    if set(identity) != {"path", "bytes", "sha256"}:
        raise ValueError(f"Evidence {label} identity has incorrect keys")
    path = _identity_path(identity, label=label)
    byte_count = identity["bytes"]
    digest = identity["sha256"]
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
        raise ValueError(f"Evidence {label} bytes must be a nonnegative integer")
    if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
        raise ValueError(f"Evidence {label} must use a full lowercase SHA-256")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Evidence {label} must be an existing regular file") from exc
    if str(resolved) != str(path):
        raise ValueError(f"Evidence {label} resolved path has changed")
    if expected_path is not None and path != expected_path:
        raise ValueError(f"Evidence {label} path has changed")
    if not resolved.is_file():
        raise ValueError(f"Evidence {label} must be an existing regular file")
    if sha256_file(resolved) != digest:
        raise ValueError(f"Evidence {label} SHA-256 has changed")
    if resolved.stat().st_size != byte_count:
        raise ValueError(f"Evidence {label} byte size has changed")


def _copy_json_object(
    value: Mapping[str, object], *, label: str
) -> dict[str, object]:
    copied = _copy_json_value(value, label=label)
    if not isinstance(copied, dict):
        raise ValueError(f"{label} must be a JSON object")
    return copied


def _copy_json_value(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain only finite JSON numbers")
        return value
    if isinstance(value, list):
        return [_copy_json_value(item, label=label) for item in value]
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} JSON object keys must be strings")
            copied[key] = _copy_json_value(item, label=label)
        return copied
    raise ValueError(f"{label} must contain only JSON scalar, list, and object values")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _json_object_argument(value: str) -> dict[str, object]:
    candidate = Path(value)
    if not value.lstrip().startswith("{") and candidate.is_file():
        parsed: object = json.loads(
            candidate.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    else:
        parsed = json.loads(value, parse_constant=_reject_json_constant)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object or a path to one")
    return parsed


def _input_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("inputs must use NAME=PATH")
    return name, Path(path)


def _unique_inputs(values: list[tuple[str, Path]]) -> dict[str, Path]:
    inputs: dict[str, Path] = {}
    for name, path in values:
        if name in inputs:
            raise ValueError(f"Duplicate Task 5 input logical name: {name!r}")
        inputs[name] = path
    return inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and validate Task 5 evidence.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--r7-root", type=Path, required=True)
    create.add_argument("--receipt", type=Path, required=True)
    create.add_argument("--git-commit", required=True)
    create.add_argument("--input", action="append", type=_input_argument, required=True)
    create.add_argument(
        "--environment",
        "--environment-json",
        dest="environment",
        type=_json_object_argument,
        required=True,
    )
    create.add_argument(
        "--contracts",
        "--contracts-json",
        dest="contracts",
        type=_json_object_argument,
        required=True,
    )
    create.add_argument("--output", type=Path)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--task5-root", type=Path, required=True)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--r7-root", type=Path, required=True)
    snapshot.add_argument("--receipt", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            manifest = build_task5_manifest(
                r7_root=args.r7_root,
                receipt_path=args.receipt,
                git_commit=args.git_commit,
                inputs=_unique_inputs(args.input),
                environment=args.environment,
                contracts=args.contracts,
            )
            output = args.output or args.r7_root / "task5" / "manifest.json"
            validate_task5_manifest(manifest, task5_root=output.parent)
            atomic_write_json(output, manifest)
        elif args.command == "validate":
            validate_task5_manifest(
                _load_json_object(args.manifest),
                task5_root=args.task5_root,
            )
        else:
            print(
                json.dumps(
                    snapshot_sealed_g0(args.r7_root, args.receipt),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
