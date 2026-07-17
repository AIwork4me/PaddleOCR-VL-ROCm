import hashlib
import re
import subprocess
from pathlib import Path

import pytest
import tomllib
import yaml

from eval.release_contract import KNOWN_V16_OFFICIAL_FAILURE

ROOT = Path(__file__).parents[1]
README_EN = ROOT / "README.md"
README_ZH = ROOT / "README.zh-CN.md"
ROADMAP = ROOT / "ROADMAP.md"
EVAL_README = ROOT / "eval" / "README.md"
EVIDENCE_README = ROOT / "results" / "omnidocbench" / "v16" / "README.md"
BENCHMARK_FACTS = ROOT / "docs" / "benchmarks" / "omnidocbench-v1.6.md"
COMPATIBILITY = ROOT / "docs" / "compatibility" / "windows-amd.md"
CI = ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_READINESS = ROOT / "docs" / "releases" / "0.1.0-readiness.md"
G0_EVIDENCE = ROOT / "docs" / "releases" / "0.1.0-g0-evidence.md"
G3_ATTESTATION = ROOT / "docs" / "releases" / "0.1.0-g3-attestation.md"
WINDOWS_VALIDATION = ROOT / "docs" / "releases" / "0.1.0-windows-validation.md"
G5_ATTESTATION = ROOT / "docs" / "releases" / "0.1.0-g5-attestation.md"
G5_CLOSEOUT = ROOT / "docs" / "releases" / "0.1.0-g5-closeout.md"
PUBLICATION_HANDOFF = ROOT / "docs" / "releases" / "0.1.0-handoff.md"
PATCH_RELEASE = ROOT / "docs" / "releases" / "0.1.1-release.md"
CHANGELOG = ROOT / "CHANGELOG.md"
PYPROJECT = ROOT / "pyproject.toml"

OMNIDOCBENCH_V16_COMMIT = "147cd5ac9472002f5751221d390bf00abdbc0d2f"
LAYOUT_HF = "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx"
LAYOUT_MODELSCOPE = "https://modelscope.cn/models/PaddlePaddle/PP-DocLayoutV3_onnx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_quality_ci_contract(workflow: str) -> None:
    parsed = yaml.safe_load(workflow)
    assert isinstance(parsed, dict)
    quality = parsed["jobs"]["quality"]
    matrix = {(entry["os"], entry["python"]) for entry in quality["strategy"]["matrix"]["include"]}
    assert ("windows-latest", "3.10") in matrix
    assert ("ubuntu-latest", "3.10") in matrix
    setup_python = [
        step
        for step in quality["steps"]
        if step.get("uses", "").startswith("actions/setup-python@")
    ]
    assert len(setup_python) == 1
    assert setup_python[0]["with"]["python-version"] == "${{ matrix.python }}"
    pytest_step = next(
        (s for s in quality["steps"] if s.get("run", "").startswith("python -m pytest")), None
    )
    assert pytest_step is not None
    assert pytest_step["run"].startswith("python -m pytest -q -m")


def test_bilingual_readmes_publish_accepted_release_gates() -> None:
    for path in (README_EN, README_ZH):
        text = _read(path)
        for accepted in ("95.99", "97.36", "94.09"):
            assert accepted in text
        for withdrawn in ("602.0", "357.2", "1.7x"):
            assert withdrawn not in text
        assert "docs/benchmarks/omnidocbench-v1.6.md" in text
        assert "docs/releases/0.1.0-g3-attestation.md" in text
        assert "G3" in text and "PASS" in text and "G4" in text
        assert "G5" in text and "PASS" in text
        assert "0.1.0-g5-attestation.md" in text


def test_patch_release_aligns_version_docs_without_rewriting_v010() -> None:
    metadata = tomllib.loads(_read(PYPROJECT))["project"]
    english = _read(README_EN)
    chinese = _read(README_ZH)
    changelog = _read(CHANGELOG)
    contract = re.sub(r"\s+", " ", _read(PATCH_RELEASE))

    assert metadata["version"] == "0.1.1"
    assert "v0.1.1 is READY" in english
    assert "v0.1.1 已就绪" in chinese
    assert "## 0.1.1 - 2026-07-17" in changelog
    for value in (
        "Status: **READY**",
        "evidence-alignment patch release",
        "no inference, scoring, runtime, model, resource-manifest, or public API",
        "v0.1.0 tag and Release remain unchanged",
        "afc9b65cb0c2f8d8effb1a4d22b8323bed1640ec",
        "Overall 95.99",
        "does not claim a successful empty-cache public-network",
    ):
        assert value in contract


def test_bilingual_readmes_have_both_four_command_journeys() -> None:
    for path in (README_EN, README_ZH):
        text = _read(path)
        for command in (
            "git clone https://github.com/AIwork4me/PaddleOCR-VL-ROCm.git",
            "cd PaddleOCR-VL-ROCm",
            "py -3.11 -m venv .venv",
            r".venv\Scripts\Activate.ps1",
            'pip install -e ".[download]"',
            "paddleocr-vl-rocm setup --auto",
            "paddleocr-vl-rocm doctor",
            "paddleocr-vl-rocm run examples/input/magazine.png",
            "paddleocr-vl-rocm doctor --server-url http://127.0.0.1:8111/v1",
            "paddleocr-vl-rocm run examples/input/magazine.png --server-url http://127.0.0.1:8111/v1",
        ):
            assert command in text, f"{path.name} must contain {command}"


def test_bilingual_python_examples_use_the_public_api() -> None:
    for path in (README_EN, README_ZH):
        text = _read(path)
        assert "from paddleocr_vl_rocm import PaddleOCRVLROCm" in text
        assert 'vlm_server_url="http://127.0.0.1:8111/v1"' in text
        assert "print(result.markdown_text)" in text
        assert "from paddleocr_vl_rocm import PaddleOCRVL\n" not in text


def test_layout_download_sources_are_language_specific() -> None:
    english = _read(README_EN)
    chinese = _read(README_ZH)
    evaluation = _read(EVAL_README)

    assert LAYOUT_HF in english
    assert LAYOUT_MODELSCOPE in chinese
    assert LAYOUT_HF in evaluation
    assert LAYOUT_MODELSCOPE in evaluation
    for text in (english, chinese, evaluation):
        assert "AlexTransformer/PP-DocLayoutV3-onnx" not in text


def test_tracked_evidence_index_delegates_numbers_to_fact_sheet() -> None:
    text = _read(EVIDENCE_README)

    assert "docs/benchmarks/omnidocbench-v1.6.md" in text
    assert "Do not construct a public score by mixing values" in text
    assert "95.99" in text
    assert "0.1.0-g3-attestation.md" in text


def test_readmes_label_demo_and_benchmarks_as_non_release_evidence() -> None:
    english = _read(README_EN).lower()
    chinese = _read(README_ZH)

    assert "compatibility demo" in english
    assert "not release evidence" in english
    assert "g3" in english and "g4" in english
    assert "兼容性演示" in chinese
    assert "不是发布证据" in chinese
    assert "G3" in chinese and "G4" in chinese


def test_bilingual_readmes_document_the_single_page_exception_without_score_inflation() -> None:
    issue = "https://github.com/PaddlePaddle/PaddleOCR/issues/18248"
    filename = "newspaper_The Times UK_0801@magazinesclubnew_page_031.png"
    facts = _read(BENCHMARK_FACTS)

    assert issue in facts
    assert filename in facts
    assert "peg-native" in facts
    assert "formal scoring denominator is **all 1,651 ground-truth pages**" in facts
    assert "ok=1650" in facts and "fail=1" in facts
    assert "0 scoring exclusions" in facts
    assert "1,650-page score" in facts
    assert "PaddlePaddle maintainer confirmed" not in facts

    evaluation = re.sub(r"\s+", " ", _read(EVAL_README))
    assert "no prediction file" in evaluation
    assert "treats the missing output as empty for scoring" in evaluation
    assert "failed page is an empty prediction" not in evaluation
    assert "PaddlePaddle 维护者已确认" not in _read(README_ZH)


def test_offline_ci_covers_supported_python_matrix_and_quality_gates() -> None:
    workflow = _read(CI)

    _assert_quality_ci_contract(workflow)

    for value in (
        "windows-latest",
        "ubuntu-latest",
        '"3.10"',
        '"3.13"',
        "python -m compileall -q src/paddleocr_vl_rocm eval",
        "ruff check src tests scripts eval",
        "ruff format --check src tests scripts eval",
        "mypy src",
        "python -m pytest -q",
        "python -m build",
        "permissions:",
        "contents: read",
        "timeout-minutes: 20",
        "cancel-in-progress: true",
    ):
        assert value in workflow
    for forbidden in ("download_omnidocbench", "setup --auto", "run --server-url"):
        assert forbidden not in workflow


def test_ci_contract_rejects_full_pytest_moved_to_another_job() -> None:
    workflow = re.sub(r"      - run: python -m pytest.*\n", "", _read(CI))
    workflow += (
        "\n  tests-elsewhere:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: python -m pytest -q\n"
    )

    with pytest.raises(AssertionError):
        _assert_quality_ci_contract(workflow)


def test_ci_contract_rejects_setup_python_fixed_to_313() -> None:
    workflow = _read(CI).replace("${{ matrix.python }}", "3.13")

    with pytest.raises(AssertionError):
        _assert_quality_ci_contract(workflow)


def test_release_readiness_records_all_gates_passed() -> None:
    text = _read(RELEASE_READINESS)

    assert "Status: READY" in text
    for gate in ("G0", "G1", "G3", "G4", "G5"):
        assert gate in text
    for evidence in (
        "1,650 successes",
        "1,651 GT pages",
        "95.99",
        "0.1.0-g3-attestation.md",
        "13.00",
        "34.82",
        "gh auth status",
        "does not authorize bypassing any evidence gate",
        "0.1.0-g5-attestation.md",
        "public-network",
    ):
        assert evidence in text
    assert "All release gates pass" in text


def test_g0_readiness_binds_independently_reviewed_r7_receipt() -> None:
    assert G0_EVIDENCE.is_file(), "tracked r7 G0 receipt must exist"
    receipt = _read(G0_EVIDENCE)
    normalized_receipt = re.sub(r"\s+", " ", receipt)
    receipt_bytes = G0_EVIDENCE.read_bytes().replace(b"\r\n", b"\n")
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    readiness = _read(RELEASE_READINESS)
    evidence_index = _read(EVIDENCE_README)

    assert not re.search(r"\b[A-Za-z]:\\", receipt)
    for value in (
        "v16-2026-07-13-official-r5",
        "v16-2026-07-14-official-r7-score-recovery-py310",
        "recovery-task-4b-portable-gs-score-20260714-045911",
        "fd91cb0a2d75b0a18d16b1bb34652a148cb59b9e",
        "count=1651",
        "ok=1650",
        "fail=1",
        "fallback=0",
        "limit_pages=null",
        KNOWN_V16_OFFICIAL_FAILURE["image"],
        KNOWN_V16_OFFICIAL_FAILURE["error_signature"],
        KNOWN_V16_OFFICIAL_FAILURE["issue_url"],
        "no failed-page prediction file",
        "Text Edit distance: 0.035",
        "display_formula.page.CDM.ALL: 96.485%",
        "sample_count=2352, timeout_case_count=0, exception_case_count=0",
        "table.page.TEDS.ALL: 94.244%",
        "sample_count=665, timeout_case_count=0, error_case_count=0",
        "Overall: 95.743",
        "Reading order is excluded",
        "runner exit code: 0",
        "no orphan",
        "adapter_command",
        "original r5 inference source",
        "r7 did not run inference",
    ):
        assert value in normalized_receipt

    assert "Audit date: 2026-07-17" in readiness
    assert re.search(r"\| G0 evidence integrity \| PASS \|", readiness)
    assert "Status: READY" in readiness
    assert re.search(r"\| G4 .* \| PASS \|", readiness)
    assert re.search(r"\| G5 .* \| PASS \|", readiness)
    assert re.search(r"\| G3 accuracy acceptance \| PASS \|", readiness)
    assert not re.search(r"\| G2 .* \|", readiness)
    for text in (readiness, evidence_index):
        assert "0.1.0-g0-evidence.md" in text
    assert receipt_sha256 in readiness


def test_benchmark_fact_sheet_is_the_only_active_public_score_table() -> None:
    facts = _read(BENCHMARK_FACTS)
    for value in (
        "95.9480",
        "95.743",
        "95.99",
        "97.36",
        "94.09",
        "count=1651",
        "ok=1650",
        "fail=1",
        "fallback=0",
        "limit_pages=null",
        "0 scoring exclusions",
        "G4 is **PASS**",
    ):
        assert value in facts
    assert "G3 accuracy | PASS" in facts
    assert "confirmed Overall 95.99 out of band" in facts
    assert "does not claim that its public thread" in facts
    assert "no tracked raw timing artifact" in facts.lower()
    assert "d529cb4" in facts and "50ce802" in facts


def test_g3_attestation_records_manual_acceptance_scope() -> None:
    assert G3_ATTESTATION.is_file()
    text = re.sub(r"\s+", " ", _read(G3_ATTESTATION))
    for value in (
        "G3 PASS",
        "95.99",
        "97.36",
        "94.09",
        "confirmed the 95.99 result out of band",
        "waives both a public confirmation artifact and another full benchmark run",
        "does not claim",
        "G4 or G5",
    ):
        assert value in text


def test_g5_attestation_records_network_waiver_without_false_success() -> None:
    assert G5_ATTESTATION.is_file()
    text = re.sub(r"\s+", " ", _read(G5_ATTESTATION))
    for value in (
        "G5 PASS",
        "explicitly waived",
        "does not call that attempt successful",
        "verified cache",
        "managed-server journey",
        "already-running-server journey",
        "44 blocks",
        "wheel",
        "source distribution",
        "full test suite passed",
    ):
        assert value in text


def test_g5_closeout_closes_only_unwaived_evidence_items() -> None:
    assert G5_CLOSEOUT.is_file()
    closeout = re.sub(r"\s+", " ", _read(G5_CLOSEOUT))
    handoff = re.sub(r"\s+", " ", _read(PUBLICATION_HANDOFF))
    for value in (
        "Status: **CLOSED**",
        "does not claim that path succeeded",
        "afc9b65cb0c2f8d8effb1a4d22b8323bed1640ec",
        "exit code 0 in 875.3 seconds",
        "full pytest: PASS",
        "Twine 6.2.0",
        "6/6 PASS",
        "not a draft",
        "not a prerelease",
        "G5 has no remaining unwaived validation or publication item",
    ):
        assert value in closeout
    assert "Closeout status: **CLOSED**" in handoff
    assert "0.1.0-g5-closeout.md" in handoff
    assert "explicitly waived" in handoff


def test_g2_is_not_an_active_release_gate() -> None:
    for path in (README_EN, README_ZH, ROADMAP, RELEASE_READINESS, BENCHMARK_FACTS):
        text = _read(path)
        assert not re.search(r"\| G2 .* \|", text)
    assert "G2 root-cause diagnosis is not a release gate" in _read(BENCHMARK_FACTS)


def test_compatibility_matrix_separates_pipeline_components_and_evidence_levels() -> None:
    text = _read(COMPATIBILITY)
    for value in (
        "Fully tested",
        "Community verified",
        "Expected but unverified",
        "Unsupported",
        "DirectML layout",
        "Local HIP VLM",
        "Managed runtime",
        "External server",
        "AMD Radeon(TM) 8060S Graphics",
        "driver version",
        "HIP runtime version",
        "rocm.docs.amd.com",
    ):
        assert value in text


def test_tracked_files_do_not_disclose_personal_workspace_paths() -> None:
    raw_personal_root = b"C:" + b"\\Users\\" + b"rocm"
    json_personal_root = b"C:" + b"\\\\Users\\\\" + b"rocm"
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("git metadata is unavailable")
    for relative in completed.stdout.decode("utf-8").split("\0"):
        if not relative:
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        raw = path.read_bytes()
        assert raw_personal_root not in raw, relative
        assert json_personal_root not in raw, relative


def test_windows_validation_distinguishes_cached_install_from_network_setup() -> None:
    text = re.sub(r"\s+", " ", _read(WINDOWS_VALIDATION))

    for evidence in (
        "AMD Radeon(TM) 8060S Graphics",
        "9884 (86961efd5)",
        "Verified 5 pinned resources",
        "DirectML first and fallback disabled",
        "44",
        "8123",
        "pre-verified local cache",
        "9,437,184-byte partial file",
        "explicitly waived empty-cache public-network transport",
        "non-successful network attempt",
        "No clean public-network download is claimed",
    ):
        assert evidence in text


def _assert_approved_v16_contract(text: str) -> None:
    match = re.search(r"Release contract:\s*(.+?)(?:\n\s*\n|\Z)", text, re.DOTALL)
    assert match, "active document must contain a Release contract block"
    contract = re.sub(r"[`\s]+", " ", match.group(1)).strip()

    assignments = {
        name: re.findall(rf"\b{name}\s*=\s*([^,;\s]+)", contract)
        for name in ("count", "ok", "fail", "fallback", "limit_pages")
    }
    assert assignments == {
        "count": ["1651"],
        "ok": ["1650"],
        "fail": ["1"],
        "fallback": ["0"],
        "limit_pages": ["null"],
    }
    assert KNOWN_V16_OFFICIAL_FAILURE["issue_url"] in contract
    assert KNOWN_V16_OFFICIAL_FAILURE["image"] in contract
    assert KNOWN_V16_OFFICIAL_FAILURE["error_signature"] in contract
    assert re.search(r"\b(?:sole|single|exactly one)\b", contract, re.IGNORECASE)
    assert re.search(r"no (?:failed-page )?prediction file", contract, re.IGNORECASE)
    assert re.search(r"all 1,651 GT pages (?:are )?scored", contract, re.IGNORECASE)
    normalized_text = re.sub(r"[`\s]+", " ", text)
    obsolete_assignment_pair = re.compile(
        r"(?:\bok\s*=\s*1651\b.{0,120}\bfail\s*=\s*0\b"
        r"|\bfail\s*=\s*0\b.{0,120}\bok\s*=\s*1651\b)",
        re.IGNORECASE,
    )
    assert not obsolete_assignment_pair.search(normalized_text)
    assert not re.search(
        r"(?:1,?651\s+(?:successful|success(?:es)?)|all\s+1,?651\s+pages\s+succeed)",
        normalized_text,
        re.IGNORECASE,
    )


def test_active_release_documents_use_approved_v16_exception() -> None:
    active = [
        ROOT / "docs/superpowers/plans/2026-07-12-accuracy-inference-fixes.md",
        ROOT / "docs/releases/0.1.0-readiness.md",
        ROOT / "eval/README.md",
    ]
    for path in active:
        _assert_approved_v16_contract(path.read_text(encoding="utf-8"))


VALID_CONTRACT = f"""Release contract: `count=1651`, `ok=1650`, `fail=1`,
`fallback=0`, and `limit_pages=null`; the sole approved failure is
{KNOWN_V16_OFFICIAL_FAILURE["image"]} with
{KNOWN_V16_OFFICIAL_FAILURE["error_signature"]} tracked at
{KNOWN_V16_OFFICIAL_FAILURE["issue_url"]}. There is no failed-page prediction
file, and all 1,651 GT pages are scored.
"""


@pytest.mark.parametrize(
    "mutated",
    [
        VALID_CONTRACT.replace("ok=1650", "ok = 1651"),
        VALID_CONTRACT.replace("fail=1", "fail = 0"),
        VALID_CONTRACT.replace("limit_pages=null", "limit_pages = 1651"),
        VALID_CONTRACT.replace("sole approved", "approved"),
        VALID_CONTRACT.replace(KNOWN_V16_OFFICIAL_FAILURE["image"], "other.png"),
        VALID_CONTRACT.replace("no failed-page prediction\nfile", "an empty prediction file"),
        VALID_CONTRACT.replace("all 1,651 GT pages are scored", "1,650 GT pages are scored"),
        VALID_CONTRACT + " 1,651 successful predictions are required.",
        VALID_CONTRACT + "\n\nOfficial evidence requires ok=1651, fail=0.",
        VALID_CONTRACT + "\n\nObsolete gate: `count=1651`, **ok = 1651**; `fail = 0`.",
        VALID_CONTRACT + "\n\nLegacy requirement: fail\n=\n0 / ok\t=\t1651.",
    ],
)
def test_release_contract_parser_rejects_realistic_regressions(mutated: str) -> None:
    with pytest.raises(AssertionError):
        _assert_approved_v16_contract(mutated)
