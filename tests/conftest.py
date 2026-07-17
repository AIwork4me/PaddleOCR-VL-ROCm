"""CI-aware pytest configuration: skip GPU/network/integration tests in CI.

Tests are classified by filename pattern, so no test file modifications needed.
"""

import os

import pytest

# Tests that directly import onnxruntime or do ONNX/DirectML inference
_GPU_TESTS = {
    "test_pipeline_characterization",
    "test_directml_attestation",
}

# Tests that make HTTP requests or download from HuggingFace
_NETWORK_TESTS = {
    "test_doctor",
    "test_server",
    "test_setup",
    "test_download_script",
}

# Tests that run subprocess scripts (slow, may need GPU/network in the subprocess)
_INTEGRATION_TESTS = {
    "test_run_official_local_v16_script",
    "test_run_release_evidence_v16_script",
    "test_run_task5_paired_v16_script",
    "test_release_evidence",
    "test_scorer_preflight",
}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: tests that need onnxruntime / DirectML / GPU")
    config.addinivalue_line("markers", "network: tests that make HTTP requests")
    config.addinivalue_line("markers", "integration: tests that run external scripts or subprocess")


def _classify(nodeid: str) -> set[str]:
    """Derive marker set from nodeid filename (no markers needed in test files)."""
    markers: set[str] = set()
    file_stem = nodeid.split("::")[0].replace(".py", "").split("/")[-1]
    if file_stem in _GPU_TESTS:
        markers.add("gpu")
    if file_stem in _NETWORK_TESTS:
        markers.add("network")
    if file_stem in _INTEGRATION_TESTS:
        markers.add("integration")
    return markers


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    in_ci = os.environ.get("CI", "").lower() == "true"
    if not in_ci:
        return

    for item in items:
        classified = _classify(item.nodeid)
        if classified:
            reason = f"CI: requires {', '.join(sorted(classified))}"
            item.add_marker(pytest.mark.skip(reason=reason))
