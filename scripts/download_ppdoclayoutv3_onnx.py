from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_model(source: Path, target: Path) -> None:
    required = ["inference.onnx", "inference.yml"]
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise SystemExit(f"Source model directory is missing {missing}: {source}")
    target.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copy2(source / name, target / name)
    print(f"PP-DocLayoutV3 ONNX model ready: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the PP-DocLayoutV3 ONNX layout model directory.")
    parser.add_argument("--source-dir", required=True, help="Directory containing inference.onnx and inference.yml.")
    parser.add_argument("--target-dir", default="models/PP-DocLayoutV3-onnx")
    args = parser.parse_args()
    copy_model(Path(args.source_dir).expanduser().resolve(), Path(args.target_dir).expanduser().resolve())


if __name__ == "__main__":
    main()

