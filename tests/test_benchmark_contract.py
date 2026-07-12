from pathlib import Path
import shutil
import subprocess

import pytest

from eval import benchmark_contract as contract


REPO_ROOT = Path(__file__).parents[1]


def _run_git(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


@pytest.fixture
def checkout_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _run_git(checkout, "init")
    _run_git(checkout, "config", "user.email", "tests@example.com")
    _run_git(checkout, "config", "user.name", "Contract Tests")

    (checkout / "scoring.py").write_text("official scoring\n", encoding="utf-8")
    (checkout / "windows_a.py").write_text("before a\n", encoding="utf-8")
    (checkout / "windows_b.py").write_text("before b\n", encoding="utf-8")
    (checkout / "unrelated.py").write_text("unchanged\n", encoding="utf-8")
    _run_git(checkout, "add", ".")
    _run_git(checkout, "commit", "-m", "fixture")

    commit = _run_git(checkout, "rev-parse", "HEAD")
    scoring_blob = _run_git(checkout, "rev-parse", "HEAD:scoring.py")
    (checkout / "windows_a.py").write_text("after a\n", encoding="utf-8")
    (checkout / "windows_b.py").write_text("after b\n", encoding="utf-8")
    tracked_patch = tmp_path / "windows.patch"
    tracked_patch.write_text(
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "diff",
                "--",
                "windows_a.py",
                "windows_b.py",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout,
        encoding="utf-8",
    )
    _run_git(checkout, "reset", "--hard", "HEAD")

    monkeypatch.setattr(contract, "OMNIDOCBENCH_V16_COMMIT", commit)
    monkeypatch.setattr(contract, "SCORING_BLOBS", {"scoring.py": scoring_blob})
    monkeypatch.setattr(
        contract,
        "WINDOWS_CDM_PATHS",
        ("windows_a.py", "windows_b.py"),
        raising=False,
    )
    monkeypatch.setattr(contract, "WINDOWS_CDM_PATCH", tracked_patch, raising=False)
    return checkout, tracked_patch


def _ensure_windows_patch(checkout: Path, patch: Path) -> subprocess.CompletedProcess[str]:
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is required for preparation-script tests")
    script = REPO_ROOT / "scripts" / "prepare_omnidocbench_v16.ps1"
    command = (
        f". '{script}'; "
        f"Ensure-WindowsCdmPatch -Checkout '{checkout}' -Patch '{patch}'"
    )
    return subprocess.run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def test_validate_checkout_accepts_expected_commit_and_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = "tools/generate_result_tables.ipynb"
    empty_patch = tmp_path / "empty.patch"
    empty_patch.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        contract,
        "SCORING_BLOBS",
        {path: "72fb7a5c7d40bb6f1b2b839fc33d31856c756ee8"},
    )
    monkeypatch.setattr(contract, "WINDOWS_CDM_PATHS", ())
    monkeypatch.setattr(contract, "WINDOWS_CDM_PATCH", empty_patch)
    monkeypatch.setattr(
        contract,
        "_git",
        lambda *args: {
            ("rev-parse", "HEAD"): contract.OMNIDOCBENCH_V16_COMMIT,
            ("rev-parse", f"HEAD:{path}"): contract.SCORING_BLOBS[path],
            ("hash-object", "--", path): contract.SCORING_BLOBS[path],
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("diff", "--no-ext-diff", "--"): "",
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
    empty_patch = tmp_path / "empty.patch"
    empty_patch.write_text("", encoding="utf-8")
    monkeypatch.setattr(contract, "SCORING_BLOBS", {path: blob})
    monkeypatch.setattr(contract, "WINDOWS_CDM_PATHS", ())
    monkeypatch.setattr(contract, "WINDOWS_CDM_PATCH", empty_patch)
    monkeypatch.setattr(
        contract,
        "_git",
        lambda *args: {
            ("rev-parse", "HEAD"): contract.OMNIDOCBENCH_V16_COMMIT,
            ("rev-parse", f"HEAD:{path}"): blob,
            ("hash-object", "--", path): blob,
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("diff", "--no-ext-diff", "--"): "",
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


def test_validate_checkout_rejects_contaminated_scoring_file(
    checkout_fixture: tuple[Path, Path],
) -> None:
    checkout, patch = checkout_fixture
    _run_git(checkout, "apply", str(patch))
    (checkout / "scoring.py").write_text("contaminated scoring\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="working-tree scoring blob"):
        contract.validate_checkout(checkout)


def test_validate_checkout_rejects_unexpected_dirty_path(
    checkout_fixture: tuple[Path, Path],
) -> None:
    checkout, patch = checkout_fixture
    _run_git(checkout, "apply", str(patch))
    (checkout / "unrelated.py").write_text("contaminated\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dirty state"):
        contract.validate_checkout(checkout)


def test_prepare_patch_applies_to_clean_checkout(
    checkout_fixture: tuple[Path, Path],
) -> None:
    checkout, patch = checkout_fixture

    result = _ensure_windows_patch(checkout, patch)

    assert result.returncode == 0, result.stderr
    assert (checkout / "windows_a.py").read_text(encoding="utf-8") == "after a\n"
    assert contract.validate_checkout(checkout)["commit"]


def test_prepare_patch_accepts_exact_already_applied_patch(
    checkout_fixture: tuple[Path, Path],
) -> None:
    checkout, patch = checkout_fixture
    _run_git(checkout, "apply", str(patch))

    result = _ensure_windows_patch(checkout, patch)

    assert result.returncode == 0, result.stderr
    assert contract.validate_checkout(checkout)["commit"]


def test_prepare_patch_rejects_partial_patch_state(
    checkout_fixture: tuple[Path, Path],
) -> None:
    checkout, patch = checkout_fixture
    _run_git(checkout, "apply", str(patch))
    _run_git(checkout, "checkout", "--", "windows_a.py")

    result = _ensure_windows_patch(checkout, patch)

    assert result.returncode != 0
    assert "partial or corrupt" in result.stderr
