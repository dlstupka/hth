"""Canonical detector-configuration discovery."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

def load_detector_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def detector_is_automatic(path: Path) -> bool:
    return bool(load_detector_config(path).get("automatic", True))

def configured_detector_paths(detector_dir: Path, *, automatic_only: bool = False) -> list[Path]:
    paths=sorted(path for path in detector_dir.glob("*.json") if path.is_file())
    return [path for path in paths if detector_is_automatic(path)] if automatic_only else paths

def configured_detectors(detector_dir: Path, *, automatic_only: bool = False) -> list[str]:
    return [path.stem for path in configured_detector_paths(detector_dir, automatic_only=automatic_only)]

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    listing=sub.add_parser("list")
    listing.add_argument("--dir",type=Path,required=True)
    listing.add_argument("--automatic-only",action="store_true")
    args=parser.parse_args()
    for path in configured_detector_paths(args.dir,automatic_only=args.automatic_only):
        print(path)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
