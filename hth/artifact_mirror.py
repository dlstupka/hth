"""Verified, non-authoritative GitHub Release artifact mirror support."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request


@dataclass(frozen=True)
class MirrorArtifact:
    repository: str
    tag: str
    asset_name: str
    artifact_id: str
    authoritative_repository: str
    authoritative_reference: str
    license: str

    @property
    def asset_url(self) -> str:
        return f"https://github.com/{self.repository}/releases/download/{self.tag}/{self.asset_name}"

    @property
    def manifest_url(self) -> str:
        return f"{self.asset_url}.manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    spec: MirrorArtifact,
    target: Path,
    *,
    fetch: Callable[[str, Path], None],
    validator: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Download and verify a mirror artifact against its immutable manifest."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        temp_root = Path(temp)
        manifest_path = temp_root / "manifest.json"
        artifact_path = temp_root / spec.asset_name
        fetch(spec.manifest_url, manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "schema_version": "1",
            "artifact_id": spec.artifact_id,
            "mirror_repository": spec.repository,
            "mirror_tag": spec.tag,
            "asset_name": spec.asset_name,
            "authoritative_repository": spec.authoritative_repository,
            "authoritative_reference": spec.authoritative_reference,
            "license": spec.license,
            "trust_role": "non-authoritative redundancy mirror",
        }
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise RuntimeError(
                    f"Mirror manifest {key} mismatch: expected {expected!r}, got {manifest.get(key)!r}"
                )
        expected_sha = str(manifest.get("sha256") or "")
        if len(expected_sha) != 64:
            raise RuntimeError("Mirror manifest has no valid SHA-256")
        fetch(spec.asset_url, artifact_path)
        actual_sha = _sha256(artifact_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Mirror artifact SHA-256 mismatch: expected={expected_sha} actual={actual_sha}"
            )
        if validator is not None:
            validator(artifact_path)
        artifact_path.replace(target)
    return {"site": "HTH non-authoritative mirror", "url": spec.asset_url,
            "reference": spec.tag, "tier": "mirror", "manifest": manifest}


def _github_json(request: urllib.request.Request) -> dict[str, Any]:
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _upload(url: str, name: str, content: bytes, content_type: str, token: str) -> None:
    separator = "&" if "?" in url else "?"
    request = urllib.request.Request(
        f"{url}{separator}{urllib.parse.urlencode({'name': name})}",
        data=content,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request):
        pass


def publish(
    spec: MirrorArtifact,
    artifact: Path,
    *,
    authoritative_source: dict[str, Any],
    token: str | None = None,
) -> str:
    """Best-effort publication after a verified authoritative acquisition."""
    token = token or os.environ.get("HTH_MIRROR_TOKEN")
    if not token:
        return "skipped-no-token"
    artifact = Path(artifact)
    api = f"https://api.github.com/repos/{spec.repository}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        release = _github_json(urllib.request.Request(f"{api}/releases/tags/{spec.tag}", headers=headers))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        release = _github_json(urllib.request.Request(
            f"{api}/releases",
            data=json.dumps({
                "tag_name": spec.tag,
                "name": spec.tag,
                "body": "Immutable non-authoritative HTH redundancy artifact. See attached manifest.",
                "draft": False,
                "prerelease": False,
            }).encode("utf-8"),
            method="POST",
            headers={**headers, "Content-Type": "application/json"},
        ))
    existing = {str(asset.get("name")) for asset in release.get("assets", [])}
    manifest_name = f"{spec.asset_name}.manifest.json"
    if spec.asset_name in existing and manifest_name in existing:
        return "already-present"
    if spec.asset_name in existing or manifest_name in existing:
        raise RuntimeError(
            f"Immutable mirror release {spec.tag} is incomplete; refusing to modify it"
        )
    manifest = {
        "schema_version": "1",
        "artifact_id": spec.artifact_id,
        "mirror_repository": spec.repository,
        "mirror_tag": spec.tag,
        "asset_name": spec.asset_name,
        "size": artifact.stat().st_size,
        "sha256": _sha256(artifact),
        "license": spec.license,
        "authoritative_repository": spec.authoritative_repository,
        "authoritative_reference": spec.authoritative_reference,
        "authoritative_source": authoritative_source,
        "trust_role": "non-authoritative redundancy mirror",
    }
    upload_url = str(release["upload_url"]).split("{", 1)[0]
    if spec.asset_name not in existing:
        _upload(upload_url, spec.asset_name, artifact.read_bytes(), "application/octet-stream", token)
    if manifest_name not in existing:
        _upload(
            upload_url, manifest_name,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            "application/json", token,
        )
    return "published"
