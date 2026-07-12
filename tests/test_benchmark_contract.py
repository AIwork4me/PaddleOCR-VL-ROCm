from pathlib import Path

import pytest

from eval import benchmark_contract as contract


def test_validate_checkout_accepts_expected_commit_and_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        contract,
        "SCORING_BLOBS",
        {
            "tools/generate_result_tables.ipynb": (
                "72fb7a5c7d40bb6f1b2b839fc33d31856c756ee8"
            )
        },
    )
    monkeypatch.setattr(
        contract,
        "_git",
        lambda *args: {
            ("rev-parse", "HEAD"): contract.OMNIDOCBENCH_V16_COMMIT,
            ("rev-parse", "HEAD:tools/generate_result_tables.ipynb"): (
                contract.SCORING_BLOBS["tools/generate_result_tables.ipynb"]
            ),
        }[args],
    )

    result = contract.validate_checkout(tmp_path)

    assert result["commit"] == contract.OMNIDOCBENCH_V16_COMMIT


def test_validate_checkout_rejects_v17_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(contract, "_git", lambda *args: "0c7db667")

    with pytest.raises(RuntimeError, match="OmniDocBench v1.6"):
        contract.validate_checkout(tmp_path)


def test_validate_checkout_returns_blob_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = "src/core/metrics.py"
    blob = "6039ff87c463be88c988e7ec017860b8f0687b2a"
    monkeypatch.setattr(contract, "SCORING_BLOBS", {path: blob})
    monkeypatch.setattr(
        contract,
        "_git",
        lambda *args: {
            ("rev-parse", "HEAD"): contract.OMNIDOCBENCH_V16_COMMIT,
            ("rev-parse", f"HEAD:{path}"): blob,
        }[args],
    )

    assert contract.validate_checkout(tmp_path) == {
        "commit": contract.OMNIDOCBENCH_V16_COMMIT,
        "blobs": {path: blob},
    }


def test_validate_checkout_rejects_changed_scoring_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = "src/core/metrics.py"
    monkeypatch.setattr(contract, "SCORING_BLOBS", {path: "expected"})
    monkeypatch.setattr(
        contract,
        "_git",
        lambda *args: {
            ("rev-parse", "HEAD"): contract.OMNIDOCBENCH_V16_COMMIT,
            ("rev-parse", f"HEAD:{path}"): "changed",
        }[args],
    )

    with pytest.raises(RuntimeError, match="scoring blob"):
        contract.validate_checkout(tmp_path)


def test_sha256_file_hashes_file_contents(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"OmniDocBench v1.6\n")

    assert contract.sha256_file(sample) == (
        "4784121b9c0e4788c6cf6b4731e8bfa21e4587b44b6131fe3a1682101e3bffaf"
    )


def test_main_returns_two_when_contract_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def reject(_checkout: Path) -> dict[str, object]:
        raise RuntimeError("not OmniDocBench v1.6")

    monkeypatch.setattr(contract, "validate_checkout", reject)

    assert contract.main(["--checkout", str(tmp_path)]) == 2
    assert "not OmniDocBench v1.6" in capsys.readouterr().err


def test_prepare_script_pins_and_validates_v16_checkout() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "prepare_omnidocbench_v16.ps1"
    ).read_text(encoding="utf-8")

    assert contract.OMNIDOCBENCH_V16_COMMIT in script
    assert '$Checkout = "eval/.omnidocbench"' in script
    assert "git clone https://github.com/opendatalab/OmniDocBench.git $Checkout" in script
    assert "git -C $Checkout fetch origin $Commit" in script
    assert "git -C $Checkout checkout --detach $Commit" in script
    assert "git -C $Checkout apply --check $Patch" in script
    assert "git -C $Checkout apply $Patch" in script
    assert "python eval/benchmark_contract.py --checkout $Checkout" in script
    assert script.count("$LASTEXITCODE") >= 6
