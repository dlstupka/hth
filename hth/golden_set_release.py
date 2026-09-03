from __future__ import annotations

import hashlib
import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_image(root: Path, ordinal: int) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"):
        for candidate in (root / f"fs_{ordinal:04d}{suffix}", root / "raw" / f"fs_{ordinal:04d}{suffix}"):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No processed image found for Golden Set ordinal {ordinal}")


def build_image_bundle(golden_set: Path, image_root: Path, output: Path) -> dict[str, Any]:
    golden = json.loads(golden_set.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for page in golden.get("pages") or []:
            ordinal = int(page["global_ordinal"])
            source = find_image(image_root, ordinal)
            data = source.read_bytes()
            name = f"raw/fs_{ordinal:04d}{source.suffix.lower()}"
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
            records.append({"global_ordinal": ordinal, "path": name, "size": len(data), "sha256": sha256_bytes(data)})
    bundle = output.read_bytes()
    return {"asset": output.name, "format": "zip/store", "size": len(bundle), "sha256": sha256_bytes(bundle), "images": records}


def verify_and_extract(bundle: Path, metadata: dict[str, Any], destination: Path) -> None:
    data = bundle.read_bytes()
    if len(data) != int(metadata.get("size", -1)) or sha256_bytes(data) != metadata.get("sha256"):
        raise ValueError("Golden Set image bundle size or SHA-256 mismatch")
    expected = {str(item["path"]): item for item in metadata.get("images") or []}
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if set(names) != set(expected) or any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("Golden Set image bundle membership mismatch")
        for name in names:
            content = archive.read(name)
            record = expected[name]
            if len(content) != int(record["size"]) or sha256_bytes(content) != record["sha256"]:
                raise ValueError(f"Golden Set image integrity failure: {name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)


def download_release_bundle(repository: str, tag: str, freeze: Path, destination: Path, token: str = "") -> Path:
    metadata = json.loads(freeze.read_text(encoding="utf-8"))["image_bundle"]
    asset = str(metadata["asset"])
    url = f"https://api.github.com/repos/{repository}/releases/tags/{tag}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "hth-golden-set/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        release = json.load(response)
    match = next((item for item in release.get("assets", []) if item.get("name") == asset), None)
    if match is None:
        raise ValueError(f"Golden Set release is missing {asset}")
    asset_headers = dict(headers); asset_headers["Accept"] = "application/octet-stream"
    bundle = destination / asset
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(match["url"], headers=asset_headers), timeout=300) as response:
        bundle.write_bytes(response.read())
    verify_and_extract(bundle, metadata, destination)
    return destination / "raw"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify an HTH Golden Set image bundle")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--token", default=os.environ.get("HTH_SOURCE_TOKEN") or os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))
    args = parser.parse_args()
    raw = download_release_bundle(args.repository, args.tag, args.freeze, args.destination, args.token)
    if args.github_output:
        with Path(args.github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"image_root={raw}\n")
    print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
