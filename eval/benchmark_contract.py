"""Validate the pinned OmniDocBench v1.6 scoring checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from contextvars import ContextVar
from pathlib import Path

OMNIDOCBENCH_V16_COMMIT = "147cd5ac9472002f5751221d390bf00abdbc0d2f"
SCORING_BLOBS = {
    "tools/generate_result_tables.ipynb": "72fb7a5c7d40bb6f1b2b839fc33d31856c756ee8",
    "src/core/metrics.py": "6039ff87c463be88c988e7ec017860b8f0687b2a",
    "src/metrics/cal_metric.py": "8993efdc2f55769e96d04f634645a00de7d5b900",
    "src/metrics/table_metric.py": "705e294919bb1ff96cf1a69655b1267958a66407",
    "src/metrics/cdm_metric.py": "c82d5a405f92cf7493e6cf9201b4ba5531759ba8",
    "src/dataset/end2end_dataset.py": "633a28a2629d7cd30d9d49c10cecc619b57519ac",
}
WINDOWS_CDM_PATHS = (
    "src/metrics/cdm/modules/latex2bbox_color.py",
    "src/metrics/cdm/modules/texlive_env.py",
)
WINDOWS_CDM_PATCH = Path(__file__).parent / "patches" / "omnidocbench-v16-windows-cdm.patch"

_GIT_CHECKOUT: ContextVar[Path | None] = ContextVar("git_checkout", default=None)


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    """Run a Git command in the checkout currently being validated."""
    checkout = _GIT_CHECKOUT.get()
    if checkout is None:
        raise RuntimeError("Git checkout context is not configured")
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.rstrip()


def validate_checkout(checkout: Path) -> dict[str, object]:
    """Validate the v1.6 commit and scoring blobs and return provenance."""
    token = _GIT_CHECKOUT.set(checkout)
    try:
        commit = _git("rev-parse", "HEAD")
        if commit != OMNIDOCBENCH_V16_COMMIT:
            raise RuntimeError(
                "OmniDocBench v1.6 checkout required: "
                f"expected {OMNIDOCBENCH_V16_COMMIT}, found {commit}"
            )

        blobs: dict[str, str] = {}
        for path, expected_blob in SCORING_BLOBS.items():
            blob = _git("rev-parse", f"HEAD:{path}")
            if blob != expected_blob:
                raise RuntimeError(
                    f"OmniDocBench v1.6 scoring blob mismatch for {path}: "
                    f"expected {expected_blob}, found {blob}"
                )
            worktree_blob = _git("hash-object", "--", path)
            if worktree_blob != expected_blob:
                raise RuntimeError(
                    f"OmniDocBench v1.6 working-tree scoring blob mismatch for {path}: "
                    f"expected {expected_blob}, found {worktree_blob}"
                )
            blobs[path] = blob

        status = _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
        expected_status = sorted(f" M {path}" for path in WINDOWS_CDM_PATHS)
        if sorted(status) != expected_status:
            raise RuntimeError(
                "OmniDocBench v1.6 dirty state must contain exactly the tracked "
                f"Windows CDM patch paths; found {status!r}"
            )

        worktree_patch = _git("diff", "--no-ext-diff", "--", *WINDOWS_CDM_PATHS)
        expected_patch = WINDOWS_CDM_PATCH.read_text(encoding="utf-8").strip()
        if worktree_patch != expected_patch:
            raise RuntimeError(
                "OmniDocBench v1.6 Windows CDM worktree diff does not match the tracked patch"
            )
    finally:
        _GIT_CHECKOUT.reset(token)

    return {"commit": commit, "blobs": blobs}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        provenance = validate_checkout(args.checkout)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 2

    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
