import re
from pathlib import Path

import pytest

from eval.release_contract import KNOWN_V16_OFFICIAL_FAILURE

ROOT = Path(__file__).parents[1]
README_EN = ROOT / "README.md"
README_ZH = ROOT / "README.zh-CN.md"
EVAL_README = ROOT / "eval" / "README.md"
EVIDENCE_README = ROOT / "results" / "omnidocbench" / "v16" / "README.md"
CI = ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_READINESS = ROOT / "docs" / "releases" / "0.1.0-readiness.md"
WINDOWS_VALIDATION = ROOT / "docs" / "releases" / "0.1.0-windows-validation.md"

OMNIDOCBENCH_V16_COMMIT = "147cd5ac9472002f5751221d390bf00abdbc0d2f"
LAYOUT_HF = "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx"
LAYOUT_MODELSCOPE = "https://modelscope.cn/models/PaddlePaddle/PP-DocLayoutV3_onnx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_bilingual_readmes_lock_verified_historical_claims() -> None:
    for path in (README_EN, README_ZH):
        text = _read(path)
        for value in (
            "95.7803",
            "95.9480",
            "96.502",
            "96.922",
            "94.239",
            "94.322",
            OMNIDOCBENCH_V16_COMMIT,
            "b9884",
            "86961efd5",
        ):
            assert value in text, f"{path.name} must contain {value}"
        assert "95.7657" not in text
        assert "95.9475" not in text
        assert "96.13" not in text


def test_bilingual_readmes_have_both_four_command_journeys() -> None:
    for path in (README_EN, README_ZH):
        text = _read(path)
        for command in (
            "pip install -e .[download]",
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


def test_tracked_evidence_index_uses_official_notebook_rounding() -> None:
    text = _read(EVIDENCE_README)

    assert "95.9480" in text
    assert "95.7803" in text
    assert "95.9475" not in text
    assert "95.7657" not in text


def test_readmes_label_demo_and_benchmarks_as_non_release_evidence() -> None:
    english = _read(README_EN).lower()
    chinese = _read(README_ZH)

    assert "compatibility demo" in english
    assert "historical evidence" in english
    assert "g3" in english and "g4" in english
    assert "兼容性演示" in chinese
    assert "历史证据" in chinese
    assert "G3" in chinese and "G4" in chinese


def test_bilingual_readmes_document_the_single_page_exception_without_score_inflation() -> None:
    issue = "https://github.com/PaddlePaddle/PaddleOCR/issues/18248"
    filename = "newspaper_The Times UK_0801@magazinesclubnew_page_031.png"
    english = _read(README_EN)
    chinese = _read(README_ZH)

    for text in (english, chinese):
        assert issue in text
        assert filename in text
        assert "peg-native" in text
    assert "1,650" in english and "1,651" in english
    assert "1,650" in chinese and "1,651" in chinese
    assert "all 1,651 ground-truth pages" in english
    assert "empty prediction" in english
    assert "全部 1,651 个 GT 页面" in chinese
    assert "空预测" in chinese
    assert "PaddlePaddle maintainer confirmed" not in english

    evaluation = re.sub(r"\s+", " ", _read(EVAL_README))
    assert "no prediction file" in evaluation
    assert "treats the missing output as empty for scoring" in evaluation
    assert "failed page is an empty prediction" not in evaluation
    assert "PaddlePaddle 维护者已确认" not in chinese


def test_offline_ci_covers_supported_python_matrix_and_quality_gates() -> None:
    workflow = _read(CI)

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
    ):
        assert value in workflow
    for forbidden in ("download_omnidocbench", "setup --auto", "run --server-url"):
        assert forbidden not in workflow


def test_release_readiness_fails_closed_until_all_gates_pass() -> None:
    text = _read(RELEASE_READINESS)

    assert "Status: BLOCKED" in text
    for gate in ("G0", "G1", "G2", "G3", "G4", "G5"):
        assert gate in text
    for evidence in (
        "1,650 successes",
        "1,651 GT pages",
        "20/20",
        "95.9480",
        "96.13",
        "13.00",
        "34.82",
        "gh auth status",
        "does not authorize bypassing any evidence gate",
    ):
        assert evidence in text
    assert "Do not bump" in text


def test_windows_validation_distinguishes_cached_install_from_network_setup() -> None:
    text = _read(WINDOWS_VALIDATION)

    for evidence in (
        "AMD Radeon(TM) 8060S Graphics",
        "9884 (86961efd5)",
        "Verified 5 pinned resources",
        "DirectML first and fallback disabled",
        "44",
        "8123",
        "pre-verified local cache",
        "release-assets.githubusercontent.com",
        "not a clean network-download acceptance",
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
    ],
)
def test_release_contract_parser_rejects_realistic_regressions(mutated: str) -> None:
    with pytest.raises(AssertionError):
        _assert_approved_v16_contract(mutated)
