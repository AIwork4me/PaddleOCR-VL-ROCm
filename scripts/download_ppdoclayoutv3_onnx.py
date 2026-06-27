from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_REPO_ID = "AlexTransformer/PP-DocLayoutV3-onnx"
REQUIRED_FILES = ["inference.onnx", "inference.yml"]


def copy_model(source: Path, target: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (source / name).exists()]
    if missing:
        raise SystemExit(f"Source model directory is missing {missing}: {source}")
    target.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        shutil.copy2(source / name, target / name)
    print(f"PP-DocLayoutV3 ONNX model ready: {target}")


def download_model(
    repo_id: str, target: Path, revision: str | None = None, cache_dir: Path | None = None
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for direct downloads. "
            "Install it with: pip install -e .[download]"
        ) from exc

    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=REQUIRED_FILES,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    copy_model(Path(local_dir), target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the PP-DocLayoutV3 ONNX layout model directory."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face model repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--revision", default=None, help="Optional Hugging Face revision, tag, or commit id."
    )
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional local directory containing inference.onnx and inference.yml.",
    )
    parser.add_argument("--target-dir", default="models/PP-DocLayoutV3-onnx")
    args = parser.parse_args()
    target = Path(args.target_dir).expanduser().resolve()
    if args.source_dir:
        copy_model(Path(args.source_dir).expanduser().resolve(), target)
    else:
        cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
        download_model(args.repo_id, target, revision=args.revision, cache_dir=cache_dir)


if __name__ == "__main__":
    main()
