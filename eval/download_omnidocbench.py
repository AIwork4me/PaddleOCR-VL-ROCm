"""OmniDocBench dataset downloader.

Fetches the Hugging Face dataset ``opendatalab/OmniDocBench`` (the manifest JSON
plus the ``images/`` directory) for a given version into a managed
``data/omnidocbench/<version>/`` directory. Mirrors the lazy-import style of
``scripts/download_ppdoclayoutv3_onnx.py``.

OmniDocBench versions are distinguished by dataset branch, not a documented
``revision=`` parameter: v1.6 is the current default dataset (~1,651 pages,
matching the OmniDocBench repo's ``master`` branch), v1.5 is the earlier branch
(~1,355 pages). Pin a known-good revision below if one becomes discoverable;
until then we default to the dataset's latest state and log a warning that the
version should be pinned for reproducibility.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

DEFAULT_REPO_ID = "opendatalab/OmniDocBench"

# Fetch the manifest JSON plus the images/ directory. The manifest filename has
# historically been ``OmniDocBench.json``; pull any top-level json plus the
# whole images/ tree to be robust to renames.
ALLOW_PATTERNS = ["*.json", "images/*"]

# Pinned Hugging Face revisions per version. None = latest (warn to pin).
# v1.6 = current default dataset (~1,651 pages); v1.5 = earlier branch
# (~1,355 pages). TODO: pin revision for reproducibility.
VERSIONS: dict[str, str | None] = {
    "v15": None,  # TODO: pin revision
    "v16": None,  # TODO: pin revision
}

log = logging.getLogger("download_omnidocbench")


def download_dataset(
    repo_id: str,
    target: Path,
    revision: str | None,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Download the manifest + images into ``target`` via ``snapshot_download``.

    Returns the resolved local directory holding the manifest.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            "huggingface_hub is required to download OmniDocBench. "
            "Install it with: pip install -e .[download]"
        ) from exc

    if revision is None:
        log.warning(
            "No pinned Hugging Face revision for OmniDocBench; fetching latest. "
            "Pin a revision in eval/download_omnidocbench.py VERSIONS for reproducibility."
        )

    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS,
        local_dir=str(target),
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    resolved = Path(local_dir)
    if not resolved.exists():
        raise SystemExit(f"snapshot_download reported a missing directory: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the OmniDocBench dataset (manifest + images) from Hugging Face."
    )
    parser.add_argument(
        "--version",
        choices=sorted(VERSIONS),
        default="v16",
        help="OmniDocBench version to fetch. Default: v16.",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Target directory. Default: data/omnidocbench/<version>.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Hugging Face revision/branch/commit. Overrides VERSIONS pin. "
            "Default: VERSIONS[version] (may be None = latest)."
        ),
    )
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    revision = args.revision if args.revision is not None else VERSIONS[args.version]
    target = (
        Path(args.target_dir).expanduser().resolve()
        if args.target_dir
        else Path("data/omnidocbench") / args.version
    )
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None

    resolved = download_dataset(args.repo_id, target, revision=revision, cache_dir=cache_dir)
    print(f"OmniDocBench {args.version} ready: {resolved}")


if __name__ == "__main__":
    main()
