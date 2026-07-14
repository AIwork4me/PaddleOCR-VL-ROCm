from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import eval.task5_manifest as task5_manifest
from eval.artifact_utils import sha256_file
from eval.task5_manifest import (
    OFFICIAL_OUTPUTS,
    build_task5_manifest,
    snapshot_sealed_g0,
    validate_task5_manifest,
)

ROOT = Path(__file__).parents[1]
G0_RECEIPT = ROOT / "docs" / "releases" / "0.1.0-g0-evidence.md"
EXPECTED_OFFICIAL_OUTPUTS = set(OFFICIAL_OUTPUTS)
EXPECTED_AUTHORIZED_OUTPUT_DIGESTS = {
    "results/official/metric.json": (
        "99645675151d6bdad5cd912600c0884c5c0febf3c0a0da4f9f331281294699a1"
    ),
    "results/official/metric-cdm.json": (
        "cd93fa7a540edbd69e9562178bb1887eea75803b0ee358cb2c47da23eecab5e3"
    ),
    "results/official/provenance.json": (
        "9749bba95ab651ab8446bac5230f2456f8999a43755ded3f22f6d214210a59cb"
    ),
    "results/official/provenance-cdm.json": (
        "364c04c05e82f4fdd0eee10c91c4983c8f7d7731fd57dc9d3a9e53dc6d1ba0a6"
    ),
    "results/official/run-summary.json": (
        "c78f610f86f19c07009966e3ec7449b3cd7b16d70b7d8433114b1aa7e2fac895"
    ),
    "results/official/run-summary-cdm.json": (
        "30c3965b3fa0e922e0ac6a9eddf28f480d9179c08ba56ee42f7b2c0aa499c0fc"
    ),
}
PRODUCTION_OUTPUT_DIGESTS = getattr(
    task5_manifest, "APPROVED_G0_OUTPUT_SHA256", None
)
TEST_OUTPUT_DIGESTS = {
    relative: hashlib.sha256(
        (json.dumps({"output": relative}) + "\n").encode()
    ).hexdigest()
    for relative in OFFICIAL_OUTPUTS
}


@pytest.fixture(autouse=True)
def use_test_output_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        task5_manifest,
        "APPROVED_G0_OUTPUT_SHA256",
        TEST_OUTPUT_DIGESTS,
        raising=False,
    )


def make_sealed_r7(tmp_path: Path) -> tuple[Path, Path]:
    r7 = tmp_path / "r7"
    (r7 / "results" / "official").mkdir(parents=True, exist_ok=True)
    (r7 / "manifest.json").write_text('{"sealed": true}\n', encoding="utf-8")
    for relative in OFFICIAL_OUTPUTS:
        output = r7 / relative
        output.write_bytes((json.dumps({"output": relative}) + "\n").encode())
    return r7, G0_RECEIPT


def valid_manifest(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    r7, receipt = make_sealed_r7(tmp_path)
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}\n", encoding="utf-8")
    manifest = build_task5_manifest(
        r7_root=r7,
        receipt_path=receipt,
        git_commit="2" * 40,
        inputs={"dataset": dataset},
        environment={"os": "Windows", "providers": ["DML", "CPU"]},
        contracts={"benchmark": "OmniDocBench-v1.6", "pair_pages": 1650},
    )
    return manifest, r7, dataset


def run_manifest_cli(
    command: str,
    tmp_path: Path,
    manifest_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    r7 = tmp_path / "r7"
    authority = repr(TEST_OUTPUT_DIGESTS)
    bootstrap = (
        "import eval.task5_manifest as m;"
        f"m.APPROVED_G0_OUTPUT_SHA256={authority};"
        "raise SystemExit(m.main())"
    )
    arguments = [sys.executable, "-c", bootstrap, command]
    if command == "create":
        dataset = tmp_path / "dataset.json"
        dataset.write_text("{}\n", encoding="utf-8")
        make_sealed_r7(tmp_path)
        arguments.extend(
            [
                "--r7-root",
                str(r7),
                "--receipt",
                str(G0_RECEIPT),
                "--git-commit",
                "2" * 40,
                "--input",
                f"dataset={dataset}",
                "--environment",
                '{"os":"Windows","gpu":"AMD"}',
                "--contracts",
                '{"benchmark":"OmniDocBench-v1.6","pair_pages":1650}',
            ]
        )
    elif command == "validate":
        assert manifest_path is not None
        arguments.extend(
            ["--manifest", str(manifest_path), "--task5-root", str(r7 / "task5")]
        )
    else:
        arguments.extend(["--r7-root", str(r7), "--receipt", str(G0_RECEIPT)])
    return subprocess.run(arguments, text=True, capture_output=True, check=False, cwd=ROOT)


def test_manifest_binds_receipt_r7_manifest_and_six_outputs(tmp_path: Path) -> None:
    r7, receipt = make_sealed_r7(tmp_path)
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}\n", encoding="utf-8")
    manifest = build_task5_manifest(
        r7_root=r7,
        receipt_path=receipt,
        git_commit="2" * 40,
        inputs={"dataset": dataset},
        environment={"os": "Windows", "gpu": "AMD"},
        contracts={"benchmark": "OmniDocBench-v1.6", "pair_pages": 1650},
    )
    assert manifest["g0"]["receipt"]["sha256"] == sha256_file(receipt)
    assert manifest["g0"]["manifest"]["sha256"] == sha256_file(r7 / "manifest.json")
    assert set(manifest["g0"]["official_outputs"]) == EXPECTED_OFFICIAL_OUTPUTS


def test_official_output_digest_contract_is_exact() -> None:
    assert PRODUCTION_OUTPUT_DIGESTS == EXPECTED_AUTHORIZED_OUTPUT_DIGESTS


def test_manifest_rejects_task5_outside_exact_r7_child(tmp_path: Path) -> None:
    r7, receipt = make_sealed_r7(tmp_path)
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}\n", encoding="utf-8")
    manifest = build_task5_manifest(
        r7_root=r7,
        receipt_path=receipt,
        git_commit="2" * 40,
        inputs={"dataset": dataset},
        environment={"os": "Windows"},
        contracts={"benchmark": "OmniDocBench-v1.6", "pair_pages": 1650},
    )
    with pytest.raises(ValueError, match="exactly r7/task5"):
        validate_task5_manifest(manifest, task5_root=tmp_path / "task5-copy")


def test_revalidation_detects_any_sealed_g0_mutation(tmp_path: Path) -> None:
    r7, receipt = make_sealed_r7(tmp_path)
    snapshot_sealed_g0(r7, receipt)
    (r7 / "results/official/metric.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="approved SHA-256"):
        snapshot_sealed_g0(r7, receipt)


def test_rebuilt_identity_cannot_authorize_replaced_official_output(
    tmp_path: Path,
) -> None:
    manifest, r7, _ = valid_manifest(tmp_path)
    output = r7 / OFFICIAL_OUTPUTS[0]
    output.write_bytes(b"replacement")
    manifest["g0"]["official_outputs"][OFFICIAL_OUTPUTS[0]] = (
        task5_manifest.file_identity(output)
    )
    with pytest.raises(ValueError, match="approved SHA-256"):
        validate_task5_manifest(manifest, task5_root=r7 / "task5")


def test_build_sorts_inputs_and_copies_json_values(tmp_path: Path) -> None:
    r7, receipt = make_sealed_r7(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    environment: dict[str, object] = {"nested": {"providers": ["DML", "CPU"]}}
    contracts: dict[str, object] = {"pair_pages": 1650}
    manifest = build_task5_manifest(
        r7_root=r7,
        receipt_path=receipt,
        git_commit="a" * 40,
        inputs={"z": second, "a": first},
        environment=environment,
        contracts=contracts,
    )
    environment["nested"]["providers"].append("mutated")
    contracts["pair_pages"] = 0
    assert list(manifest["inputs"]) == ["a", "z"]
    assert manifest["environment"] == {"nested": {"providers": ["DML", "CPU"]}}
    assert manifest["contracts"] == {"pair_pages": 1650}


@pytest.mark.parametrize("section", ["environment", "contracts"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_build_rejects_nonfinite_json_numbers(
    tmp_path: Path, section: str, value: float
) -> None:
    r7, receipt = make_sealed_r7(tmp_path)
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    keywords = {
        "r7_root": r7,
        "receipt_path": receipt,
        "git_commit": "a" * 40,
        "inputs": {"dataset": dataset},
        "environment": {"os": "Windows"},
        "contracts": {"pair_pages": 1650},
    }
    keywords[section] = {"invalid": value}
    with pytest.raises(ValueError, match="finite JSON"):
        build_task5_manifest(**keywords)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_validate_rejects_nonexact_top_level_schema(tmp_path: Path, mutation: str) -> None:
    manifest, r7, _ = valid_manifest(tmp_path)
    if mutation == "missing":
        manifest.pop("contracts")
    else:
        manifest["extra"] = True
    with pytest.raises(ValueError, match="top-level keys"):
        validate_task5_manifest(manifest, task5_root=r7 / "task5")


def test_validate_rejects_boolean_schema(tmp_path: Path) -> None:
    manifest, r7, _ = valid_manifest(tmp_path)
    manifest["schema"] = True
    with pytest.raises(ValueError, match="schema"):
        validate_task5_manifest(manifest, task5_root=r7 / "task5")


def test_build_rejects_nonstring_input_name_before_sorting(tmp_path: Path) -> None:
    r7, receipt = make_sealed_r7(tmp_path)
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="logical name"):
        build_task5_manifest(
            r7_root=r7,
            receipt_path=receipt,
            git_commit="a" * 40,
            inputs={1: dataset, "dataset": dataset},
            environment={"os": "Windows"},
            contracts={"pair_pages": 1650},
        )


def test_validate_rejects_nonstring_input_name_before_sorting(tmp_path: Path) -> None:
    manifest, r7, dataset = valid_manifest(tmp_path)
    manifest["inputs"][1] = manifest["inputs"]["dataset"]
    with pytest.raises(ValueError, match="logical name"):
        validate_task5_manifest(manifest, task5_root=r7 / "task5")
    assert dataset.is_file()


@pytest.mark.parametrize("target", ["input", "g0"])
def test_validate_rehashes_every_recorded_file(tmp_path: Path, target: str) -> None:
    manifest, r7, dataset = valid_manifest(tmp_path)
    if target == "input":
        dataset.write_text("mutated", encoding="utf-8")
    else:
        (r7 / OFFICIAL_OUTPUTS[0]).write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_task5_manifest(manifest, task5_root=r7 / "task5")


def test_validate_requires_full_lowercase_sha256(tmp_path: Path) -> None:
    manifest, r7, _ = valid_manifest(tmp_path)
    manifest["inputs"]["dataset"]["sha256"] = "A" * 64
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_task5_manifest(manifest, task5_root=r7 / "task5")


def test_validate_rejects_recorded_symlink_path(tmp_path: Path) -> None:
    manifest, r7, dataset = valid_manifest(tmp_path)
    link = tmp_path / "dataset-link.json"
    try:
        link.symlink_to(dataset)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    manifest["inputs"]["dataset"]["path"] = str(link)
    with pytest.raises(ValueError, match="resolved path has changed"):
        validate_task5_manifest(manifest, task5_root=r7 / "task5")


def test_validate_binds_approved_receipt_digest(tmp_path: Path) -> None:
    manifest, r7, _ = valid_manifest(tmp_path)
    manifest["g0"]["receipt"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="approved G0 receipt"):
        validate_task5_manifest(manifest, task5_root=r7 / "task5")


def test_cli_create_then_validate_rehashes_inputs(tmp_path: Path) -> None:
    completed = run_manifest_cli("create", tmp_path)
    assert completed.returncode == 0, completed.stderr
    manifest_path = tmp_path / "r7/task5/manifest.json"
    assert run_manifest_cli("validate", tmp_path, manifest_path).returncode == 0
    dataset = tmp_path / "dataset.json"
    dataset.write_text("mutated", encoding="utf-8")
    failed = run_manifest_cli("validate", tmp_path, manifest_path)
    assert failed.returncode != 0
    assert "SHA-256" in failed.stderr


def test_cli_snapshot_emits_sealed_g0_json(tmp_path: Path) -> None:
    make_sealed_r7(tmp_path)
    completed = run_manifest_cli("snapshot", tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert set(json.loads(completed.stdout)["official_outputs"]) == EXPECTED_OFFICIAL_OUTPUTS


def test_atomic_write_ignores_prepositioned_predictable_hardlink(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    victim = tmp_path / "victim.json"
    victim.write_bytes(b"do-not-touch")
    os.link(victim, output.with_suffix(".json.tmp"))
    task5_manifest.atomic_write_json(output, {"safe": True})
    assert victim.read_bytes() == b"do-not-touch"
    assert json.loads(output.read_text(encoding="utf-8")) == {"safe": True}


def test_atomic_write_rejects_nonregular_output(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    output.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        task5_manifest.atomic_write_json(output, {"safe": True})


def test_atomic_write_rejects_output_symlink_when_available(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("target", encoding="utf-8")
    output = tmp_path / "manifest.json"
    try:
        output.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink"):
        task5_manifest.atomic_write_json(output, {"safe": True})


def test_atomic_write_fails_closed_on_parent_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def drifting_identity(parent: Path) -> tuple[str, int]:
        nonlocal calls
        calls += 1
        return (str(parent), calls)

    monkeypatch.setattr(
        task5_manifest, "_parent_identity", drifting_identity, raising=False
    )
    with pytest.raises(ValueError, match="parent.*changed"):
        task5_manifest.atomic_write_json(tmp_path / "manifest.json", {"safe": True})
    assert calls >= 2


def test_atomic_write_cleans_owned_temporary_after_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_fsync(fd: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="simulated fsync failure"):
        task5_manifest.atomic_write_json(tmp_path / "manifest.json", {"safe": True})
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


@pytest.mark.parametrize("collision", ["input", "g0"])
def test_cli_create_rejects_output_hardlinked_to_recorded_identity(
    tmp_path: Path, collision: str
) -> None:
    r7, _ = make_sealed_r7(tmp_path)
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}\n", encoding="utf-8")
    task5 = r7 / "task5"
    task5.mkdir()
    target = dataset if collision == "input" else r7 / "manifest.json"
    os.link(target, task5 / "manifest.json")
    completed = run_manifest_cli("create", tmp_path)
    assert completed.returncode != 0
    assert "same file as recorded evidence" in completed.stderr


def test_cli_create_reloads_and_revalidates_persisted_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    r7, _ = make_sealed_r7(tmp_path)
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}\n", encoding="utf-8")
    output = r7 / "task5" / "manifest.json"
    real_atomic_write = task5_manifest.atomic_write_json

    def write_then_tamper(
        path: Path, value: object, *, expected_parent: Path | None = None
    ) -> None:
        real_atomic_write(path, value, expected_parent=expected_parent)
        path.write_text('{"schema": true}\n', encoding="utf-8")

    monkeypatch.setattr(task5_manifest, "atomic_write_json", write_then_tamper)
    with pytest.raises(SystemExit) as raised:
        task5_manifest.main(
            [
                "create",
                "--r7-root",
                str(r7),
                "--receipt",
                str(G0_RECEIPT),
                "--git-commit",
                "2" * 40,
                "--input",
                f"dataset={dataset}",
                "--environment",
                '{"os":"Windows"}',
                "--contracts",
                '{"benchmark":"OmniDocBench-v1.6","pair_pages":1650}',
                "--output",
                str(output),
            ]
        )
    assert raised.value.code == 2
    assert "top-level keys" in capsys.readouterr().err
