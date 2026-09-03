from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "source-release-manifest.json"
DEFAULT_NOTES = (
    "Immutable HTH source DOCX masters. "
    "See source-release-manifest.json for SHA-256 provenance."
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "release-only"


def resolve_assets(images: Path | None, files: list[Path]) -> tuple[list[Path], Path | None]:
    if images is not None and files:
        raise SystemExit("Use either --images or explicit file paths, not both.")

    if images is not None:
        images = images.resolve()
        if not images.is_dir():
            raise SystemExit(f"Images folder not found: {images}")

        docs = sorted(images.glob("*.docx"), key=lambda x: x.name.lower())
        if not docs:
            raise SystemExit(f"No DOCX masters found beneath {images}")
        return docs, images

    if not files:
        raise SystemExit("Provide --images FOLDER or one or more explicit source files.")

    assets = [path.resolve() for path in files]
    missing = [path for path in assets if not path.is_file()]
    if missing:
        raise SystemExit(
            "Source file(s) not found:\n" + "\n".join(f"  {path}" for path in missing)
        )

    names = [path.name.lower() for path in assets]
    if len(names) != len(set(names)):
        raise SystemExit("Source asset filenames must be unique within a release.")

    return assets, None


def main() -> int:
    p = argparse.ArgumentParser(
        description="Publish HTH source masters as an immutable GitHub Release"
    )
    p.add_argument("--repository", required=True, help="owner/repository")
    p.add_argument("--tag", required=True, help="immutable source tag, e.g. HTH-SOURCE-0002")
    p.add_argument(
        "--images",
        type=Path,
        help="backward-compatible mode: folder containing DOCX masters",
    )
    p.add_argument("--collection-id", default="")
    p.add_argument(
        "--source-root",
        type=Path,
        help="optional source git repo root used only to record its HEAD",
    )
    p.add_argument(
        "--source-commit",
        default="",
        help="explicit source commit/provenance value; overrides --source-root HEAD",
    )
    p.add_argument("--title", default="")
    p.add_argument(
        "--notes",
        default=DEFAULT_NOTES,
        help="GitHub release notes",
    )
    p.add_argument(
        "--target",
        default="",
        help="optional GitHub branch/commit the new release tag should target",
    )
    p.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="one or more explicit source files to publish",
    )
    args = p.parse_args()

    assets, images = resolve_assets(args.images, args.files)

    if args.source_commit:
        source_commit = args.source_commit
    else:
        if args.source_root is not None:
            source_root = args.source_root.resolve()
        elif images is not None:
            source_root = images.parent
        else:
            source_root = Path.cwd()
        source_commit = git_commit(source_root)

    manifest = {
        "schema_version": 1,
        "collection_id": args.collection_id,
        "source_repository": args.repository,
        "release_tag": args.tag,
        "source_commit": source_commit,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "assets": [
            {
                "name": asset.name,
                "path": f"images/{asset.name}",
                "size": asset.stat().st_size,
                "sha256": sha256(asset),
            }
            for asset in assets
        ],
    }

    # Keep SOURCE-0001's manifest schema/asset path contract, while avoiding
    # writing temporary provenance metadata into the source folder itself.
    manifest_path = Path.cwd() / MANIFEST_NAME
    if manifest_path.exists():
        raise SystemExit(
            f"Refusing to overwrite existing {manifest_path}. "
            "Move/remove it and rerun."
        )

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    title = args.title or f"HTH source collection {args.tag}"
    cmd = [
        "gh",
        "release",
        "create",
        args.tag,
        "--repo",
        args.repository,
        "--title",
        title,
        "--notes",
        args.notes,
    ]
    if args.target:
        cmd.extend(["--target", args.target])

    cmd.extend([str(manifest_path), *[str(asset) for asset in assets]])

    print("Publishing immutable source release:")
    print(" ".join(cmd[:8]), "<manifest + source assets>")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit("GitHub CLI 'gh' was not found. Install it and run 'gh auth login' first.")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
    finally:
        manifest_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
