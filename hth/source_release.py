from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MANIFEST_NAME = "source-release-manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _request(url: str, *, token: str = "", accept: str = "application/vnd.github+json") -> urllib.request.Request:
    headers = {
        "Accept": accept,
        "User-Agent": "hth-source-release/1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _read_json(url: str, *, token: str = "") -> dict[str, Any]:
    try:
        with urllib.request.urlopen(_request(url, token=token), timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API request failed ({exc.code}) for {url}: {detail}") from exc


def _download_asset(asset: dict[str, Any], destination: Path, *, token: str = "") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    req = _request(str(asset["url"]), token=token, accept="application/octet-stream")
    try:
        with urllib.request.urlopen(req, timeout=300) as response, destination.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Release asset download failed ({exc.code}) for {asset.get('name')}: {detail}") from exc


def _validate_manifest(manifest: dict[str, Any], *, repository: str, tag: str) -> list[dict[str, Any]]:
    if int(manifest.get("schema_version", 0)) != 1:
        raise RuntimeError("Unsupported source release manifest schema")
    if manifest.get("source_repository") != repository:
        raise RuntimeError(
            f"Manifest source_repository {manifest.get('source_repository')!r} does not match {repository!r}"
        )
    if manifest.get("release_tag") != tag:
        raise RuntimeError(f"Manifest release_tag {manifest.get('release_tag')!r} does not match {tag!r}")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise RuntimeError("Source release manifest has no assets")
    for record in assets:
        name = str(record.get("name") or "")
        if not name.lower().endswith(".docx"):
            raise RuntimeError(f"Source release manifest contains non-DOCX source asset: {name!r}")
        digest = str(record.get("sha256") or "")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
            raise RuntimeError(f"Invalid SHA-256 for source asset {name!r}")
    return assets


def download_release(repository: str, tag: str, destination: Path, token: str = "") -> dict[str, Any]:
    api = f"https://api.github.com/repos/{repository}/releases/tags/{tag}"
    release = _read_json(api, token=token)
    release_assets = {str(a.get("name")): a for a in release.get("assets", [])}
    manifest_asset = release_assets.get(MANIFEST_NAME)
    if manifest_asset is None:
        raise RuntimeError(
            f"Release {repository}@{tag} does not contain required {MANIFEST_NAME}; "
            "publish the source release with tools/publish-source-release.py first"
        )

    root = destination
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_NAME
    _download_asset(manifest_asset, manifest_path, token=token)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _validate_manifest(manifest, repository=repository, tag=tag)

    expected_names = {str(r["name"]) for r in expected}
    missing = sorted(expected_names - set(release_assets))
    if missing:
        raise RuntimeError(f"Release is missing manifest-declared source assets: {', '.join(missing)}")

    for record in expected:
        name = str(record["name"])
        target = images / name
        _download_asset(release_assets[name], target, token=token)
        actual_size = target.stat().st_size
        expected_size = int(record.get("size", -1))
        if expected_size >= 0 and actual_size != expected_size:
            raise RuntimeError(f"Size mismatch for {name}: expected {expected_size}, got {actual_size}")
        actual_sha = _sha256(target)
        if actual_sha.lower() != str(record["sha256"]).lower():
            raise RuntimeError(f"SHA-256 mismatch for {name}: expected {record['sha256']}, got {actual_sha}")

    manifest_sha = _sha256(manifest_path)
    return {
        "repository": repository,
        "release_tag": tag,
        "release_id": release.get("id"),
        "release_url": release.get("html_url"),
        "source_commit": str(manifest.get("source_commit") or "release-only"),
        "manifest_sha256": manifest_sha,
        "docx_count": len(expected),
        "destination": str(root),
    }


def _write_github_output(values: dict[str, Any], path: str) -> None:
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            if value is not None:
                handle.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify immutable HTH source DOCX masters from a GitHub Release")
    parser.add_argument("--repository", required=True, help="owner/repository containing the source release")
    parser.add_argument("--tag", required=True, help="immutable source release tag")
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--token", default=os.environ.get("HTH_SOURCE_TOKEN", ""))
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))
    args = parser.parse_args()

    info = download_release(args.repository, args.tag, args.destination, args.token)
    _write_github_output(info, args.github_output)
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
