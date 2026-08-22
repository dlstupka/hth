from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "source-release-manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "release-only"


def main() -> int:
    p = argparse.ArgumentParser(description="Publish HTH DOCX masters as an immutable GitHub Release")
    p.add_argument("--repository", required=True, help="owner/repository")
    p.add_argument("--tag", required=True, help="immutable source tag, e.g. HTH-SOURCE-0001")
    p.add_argument("--images", required=True, type=Path, help="folder containing DOCX masters")
    p.add_argument("--collection-id", default="")
    p.add_argument("--source-root", type=Path, help="optional source git repo root used only to record its HEAD")
    p.add_argument("--title", default="")
    args = p.parse_args()

    images = args.images.resolve()
    docs = sorted(images.glob("*.docx"), key=lambda x: x.name.lower())
    if not docs:
        raise SystemExit(f"No DOCX masters found beneath {images}")

    source_root = (args.source_root or images.parent).resolve()
    manifest = {
        "schema_version": 1,
        "collection_id": args.collection_id,
        "source_repository": args.repository,
        "release_tag": args.tag,
        "source_commit": git_commit(source_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "assets": [
            {"name": doc.name, "size": doc.stat().st_size, "sha256": sha256(doc)}
            for doc in docs
        ],
    }
    manifest_path = images / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    title = args.title or f"HTH source collection {args.tag}"
    cmd = [
        "gh", "release", "create", args.tag,
        "--repo", args.repository,
        "--title", title,
        "--notes", "Immutable HTH source DOCX masters. See source-release-manifest.json for SHA-256 provenance.",
        str(manifest_path),
        *[str(doc) for doc in docs],
    ]
    print("Publishing immutable source release:")
    print(" ".join(cmd[:8]), "<manifest + DOCX assets>")
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
