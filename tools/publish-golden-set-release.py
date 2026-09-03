from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hth.golden_set_release import build_image_bundle, verify_and_extract


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_assets(
    *, repository: str, golden_set: Path, freeze: Path
) -> tuple[str, str, str]:
    golden_set = golden_set.resolve()
    freeze = freeze.resolve()
    if not golden_set.is_file() or not freeze.is_file():
        raise ValueError("Golden Set and freeze manifest must both exist")

    freeze_data = json.loads(freeze.read_text(encoding="utf-8"))
    golden_data = json.loads(golden_set.read_text(encoding="utf-8"))
    golden_id = str(freeze_data.get("golden_set_id") or "").strip()
    if not golden_id:
        raise ValueError("freeze manifest is missing golden_set_id")
    if str(golden_data.get("collection_id") or golden_data.get("id") or "") != golden_id:
        raise ValueError("Golden Set identity does not match freeze manifest")

    actual_sha = sha256(golden_set)
    if actual_sha != str(freeze_data.get("golden_set_sha256") or ""):
        raise ValueError("Golden Set bytes do not match freeze manifest SHA-256")

    release = freeze_data.get("canonical_release") or {}
    if release.get("repository") != repository:
        raise ValueError("repository does not match canonical_release.repository")
    suffix = golden_id.removeprefix("HTH-")
    expected = {
        "tag": f"HTH-GOLDEN-{suffix}",
        "golden_set_asset": f"{golden_id}.golden-set.json",
        "freeze_asset": f"{golden_id}.freeze.json",
    }
    for field, value in expected.items():
        if release.get(field) != value:
            raise ValueError(
                f"canonical_release.{field} must be {value!r}, got {release.get(field)!r}"
            )
    return expected["tag"], expected["golden_set_asset"], expected["freeze_asset"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and publish an immutable HTH Golden Set GitHub Release"
    )
    parser.add_argument("--repository", required=True, help="owner/source-repository")
    parser.add_argument("--golden-set", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path, help="processed image root containing selected ordinals")
    parser.add_argument("--source-release-tag", required=True)
    parser.add_argument("--source-release-manifest-sha256", required=True)
    parser.add_argument("--finalized-freeze", required=True, type=Path, help="write the release-ready freeze manifest here")
    parser.add_argument("--target", default="", help="source repository branch or commit")
    parser.add_argument("--title", default="")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and show the release without publishing"
    )
    args = parser.parse_args()
    digest = args.source_release_manifest_sha256.lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SystemExit("--source-release-manifest-sha256 must be a 64-character SHA-256")

    tag, golden_name, freeze_name = release_assets(
        repository=args.repository,
        golden_set=args.golden_set,
        freeze=args.freeze,
    )
    golden_sha = sha256(args.golden_set)
    title = args.title or f"HTH Golden Set {tag.removeprefix('HTH-GOLDEN-')}"
    with tempfile.TemporaryDirectory(prefix="hth-golden-release-") as temporary:
        staging = Path(temporary)
        staged_golden = staging / golden_name
        staged_freeze = staging / freeze_name
        bundle_name = f"{json.loads(args.freeze.read_text(encoding='utf-8'))['golden_set_id']}.images.zip"
        staged_bundle = staging / bundle_name
        # read_bytes/write_bytes deliberately preserves the approved JSON bytes exactly.
        staged_golden.write_bytes(args.golden_set.read_bytes())
        image_bundle = build_image_bundle(args.golden_set, args.image_root, staged_bundle)
        freeze_data = json.loads(args.freeze.read_text(encoding="utf-8"))
        freeze_data["source_release"] = {"repository": args.repository, "tag": args.source_release_tag, "manifest_sha256": args.source_release_manifest_sha256, "image_identity": "global_ordinal"}
        freeze_data["image_bundle"] = image_bundle
        freeze_data["canonical_release"]["image_bundle_asset"] = bundle_name
        staged_freeze.write_bytes(
            (json.dumps(freeze_data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        )
        args.finalized_freeze.write_bytes(staged_freeze.read_bytes())
        freeze_sha = sha256(staged_freeze)
        notes = ("Immutable HTH Golden Set release.\n\n" f"- Golden Set SHA-256: `{golden_sha}`\n" f"- Freeze manifest SHA-256: `{freeze_sha}`\n" f"- Image bundle SHA-256: `{image_bundle['sha256']}`\n\n" "Do not replace any asset. Publish a new Golden Set ID for any substantive change.")
        print(f"Repository: {args.repository}")
        print(f"Tag: {tag}")
        print(f"Golden Set asset: {golden_name} ({golden_sha})")
        print(f"Freeze asset: {freeze_name} ({freeze_sha})")
        print(f"Image bundle: {bundle_name} ({image_bundle['sha256']}; {len(image_bundle['images'])} images)")
        print(f"Finalized freeze: {args.finalized_freeze}")
        if args.dry_run:
            print("Dry run complete; nothing was published.")
            return 0
        command = [
            "gh", "release", "create", tag,
            "--repo", args.repository,
            "--title", title,
            "--notes", notes,
        ]
        if args.target:
            command.extend(["--target", args.target])
        command.extend([str(staged_golden), str(staged_freeze), str(staged_bundle)])
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise SystemExit(
                "GitHub CLI 'gh' was not found. Install it and run 'gh auth login' first."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(exc.returncode) from exc
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
