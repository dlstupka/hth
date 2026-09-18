"""Safely extract an immutable release asset and resolve its collection root."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath


def extract_collection(
    asset: Path,
    destination: Path,
    marker: str,
    required_directory: str = "",
) -> tuple[Path, Path]:
    """Extract one ZIP and resolve the sole collection containing ``marker``."""
    if not asset.is_file():
        raise ValueError(f"Immutable release asset does not exist: {asset}")
    if not marker or PurePosixPath(marker).name != marker:
        raise ValueError("Collection marker must be one filename")
    if required_directory and PurePosixPath(required_directory).name != required_directory:
        raise ValueError("Required collection directory must be one directory name")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    extraction_root = destination.resolve()
    with zipfile.ZipFile(asset) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe immutable release member: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symbolic links are not allowed in immutable releases: {member.filename}")
            target = (extraction_root / Path(*path.parts)).resolve()
            try:
                target.relative_to(extraction_root)
            except ValueError as exc:
                raise ValueError(f"Unsafe immutable release member: {member.filename}") from exc
        archive.extractall(extraction_root)

    markers = sorted(extraction_root.rglob(marker))
    if len(markers) != 1:
        raise ValueError(
            f"Expected exactly one {marker} beneath {extraction_root}, found {len(markers)}"
        )
    marker_path = markers[0]
    collection_root = marker_path.parent
    if required_directory and not (collection_root / required_directory).is_dir():
        raise ValueError(
            f"Collection rooted at {collection_root} has no {required_directory} directory"
        )
    return collection_root, marker_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract-collection")
    extract.add_argument("--asset", type=Path, required=True)
    extract.add_argument("--destination", type=Path, required=True)
    extract.add_argument("--marker", required=True)
    extract.add_argument("--required-directory", default="")
    extract.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args(argv)
    collection_root, marker_path = extract_collection(
        args.asset, args.destination, args.marker, args.required_directory
    )
    with args.github_output.open("a", encoding="utf-8") as handle:
        handle.write(f"collection_root={collection_root}{os.linesep}")
        handle.write(f"marker_path={marker_path}{os.linesep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
