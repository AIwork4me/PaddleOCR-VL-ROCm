from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

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


def make_sealed_r7(tmp_path: Path) -> tuple[Path, Path]:
    r7 = tmp_path / "r7"
    (r7 / "results" / "official").mkdir(parents=True)
    (r7 / "manifest.json").write_text('{"sealed": true}\n', encoding="utf-8")
    for relative in OFFICIAL_OUTPUTS:
        output = r7 / relative
        output.write_text(json.dumps({"output": relative}) + "\n", encoding="utf-8")
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
    arguments = [sys.executable, "-m", "eval.task5_manifest", command]
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
    before = snapshot_sealed_g0(r7, receipt)
    (r7 / "results/official/metric.json").write_text("changed", encoding="utf-8")
    assert snapshot_sealed_g0(r7, receipt) != before


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
