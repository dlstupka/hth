"""Verified immutable GitHub Release storage for collection-derived caches."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request
import zipfile


CACHE_MANIFEST_SCHEMA_VERSION = "1"
CACHE_TRUST_ROLE = "non-authoritative collection cache"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_evidence_id(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _release_component(value: str) -> str:
    return "-".join(part for part in value.upper().replace("_", "-").split("-") if part)


@dataclass(frozen=True)
class EvidenceCacheArtifact:
    repository: str
    detector: str
    evidence_id: str
    identity: dict[str, Any]

    @property
    def tag(self) -> str:
        return f"HTH-EVIDENCE-{_release_component(self.detector)}-{self.evidence_id.upper()}"

    @property
    def asset_name(self) -> str:
        return f"{self.detector}-{self.evidence_id}.zip"

    @property
    def asset_url(self) -> str:
        return f"https://github.com/{self.repository}/releases/download/{self.tag}/{self.asset_name}"

    @property
    def manifest_url(self) -> str:
        return f"{self.asset_url}.manifest.json"


def _validate_evidence_payload(
    payload: Any,
    *,
    manifest_sha256: str,
    size_bytes: int,
    source: str,
    index_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Learned-evidence manifest is not an object: {source}")
    persistence = payload.get("persistence")
    if not isinstance(persistence, dict):
        raise RuntimeError(f"Learned-evidence persistence metadata is missing: {source}")
    identity = persistence.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError(f"Learned-evidence identity is missing: {source}")
    detector = str(payload.get("detector") or "")
    evidence_id = str(persistence.get("evidence_id") or "")
    if not detector or str(identity.get("detector") or "") != detector:
        raise RuntimeError(f"Learned-evidence detector identity mismatch: {source}")
    expected_id = canonical_evidence_id(identity)
    if evidence_id != expected_id:
        raise RuntimeError(
            f"Learned-evidence identity hash mismatch: expected={expected_id} actual={evidence_id}"
        )
    records = payload.get("records")
    image_keys = identity.get("image_keys")
    if not isinstance(records, list) or not isinstance(image_keys, list):
        raise RuntimeError(f"Learned-evidence records or image keys are missing: {source}")
    record_keys = [str(row.get("image_key") or "") for row in records if isinstance(row, dict)]
    if len(record_keys) != len(records) or record_keys != [str(value) for value in image_keys]:
        raise RuntimeError(f"Learned-evidence record order does not match its identity: {source}")
    if int(payload.get("page_count") or -1) != len(records):
        raise RuntimeError(f"Learned-evidence page count does not match its records: {source}")
    if index_entry is not None:
        expected = {
            "evidence_id": evidence_id,
            "manifest_sha256": manifest_sha256,
            "size_bytes": size_bytes,
            "page_count": len(records),
            "image_keys": image_keys,
        }
        for key, value in expected.items():
            if index_entry.get(key) != value:
                raise RuntimeError(
                    f"Learned-evidence index {key} mismatch: expected={value!r} "
                    f"actual={index_entry.get(key)!r}"
                )
    return {
        "payload": payload,
        "detector": detector,
        "evidence_id": evidence_id,
        "identity": identity,
        "manifest_sha256": manifest_sha256,
        "size_bytes": size_bytes,
        "page_count": len(records),
    }


def validate_evidence_manifest(path: Path, *, index_entry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate a persisted learned-evidence manifest and its canonical identity."""
    path = Path(path)
    raw = path.read_bytes()
    return _validate_evidence_payload(
        json.loads(raw.decode("utf-8")),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        source=str(path),
        index_entry=index_entry,
    )


def deterministic_evidence_bundle(manifest: Path, destination: Path) -> None:
    """Package a complete evidence directory reproducibly for release storage."""
    manifest = Path(manifest)
    root = manifest.parent
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    members = {manifest}
    for record in payload.get("records", []):
        if not isinstance(record, dict):
            continue
        for key in ("file", "probability_file"):
            relative = record.get(key)
            if relative:
                path = root / str(relative)
                if not path.is_file() or root.resolve() not in path.resolve().parents:
                    raise RuntimeError(f"Learned-evidence asset is missing or unsafe: {relative}")
                members.add(path)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(members):
            name = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def validate_evidence_bundle(bundle: Path, *, expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    with zipfile.ZipFile(bundle) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if not names or "manifest.json" not in names or len(names) != len(set(names)):
            raise RuntimeError("Learned-evidence bundle must contain one manifest.json")
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename or member.is_dir():
                raise RuntimeError(f"Unsafe learned-evidence bundle member: {member.filename}")
        raw = archive.read("manifest.json")
        # Reading every member verifies the ZIP CRC without writing a complete
        # throwaway extraction to disk. Materialization remains the only disk
        # extraction performed for a cache hit.
        for member in members:
            if member.filename == "manifest.json":
                continue
            with archive.open(member) as source:
                while source.read(1024 * 1024):
                    pass
    actual = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and actual != expected_manifest_sha256:
        raise RuntimeError(
            f"Bundled evidence manifest SHA-256 mismatch: expected={expected_manifest_sha256} actual={actual}"
        )
    validated = _validate_evidence_payload(
        json.loads(raw.decode("utf-8")),
        manifest_sha256=actual,
        size_bytes=len(raw),
        source=f"{bundle}!manifest.json",
    )
    payload = validated["payload"]
    archived_files = set(names)
    for record in payload.get("records", []):
        if not isinstance(record, dict):
            continue
        for key in ("file", "probability_file"):
            relative = record.get(key)
            if relative:
                path = PurePosixPath(str(relative))
                if path.is_absolute() or ".." in path.parts or path.as_posix() not in archived_files:
                    raise RuntimeError(f"Bundled learned-evidence asset is missing or unsafe: {relative}")
    return validated


def fetch_url(url: str, target: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/octet-stream", "User-Agent": "HTH-collection-cache/1"},
    )
    with urllib.request.urlopen(request) as response, Path(target).open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)


def materialize(
    bundle: Path,
    output: Path,
    *,
    expected_identity: dict[str, Any] | None = None,
) -> Path:
    """Validate and atomically hydrate a complete evidence directory."""
    validated = validate_evidence_bundle(bundle)
    if expected_identity is not None and validated["identity"] != expected_identity:
        raise RuntimeError("Materialized learned-evidence identity mismatch")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "evidence"
        staged.mkdir()
        with zipfile.ZipFile(bundle) as archive:
            archive.extractall(staged)
        if output.exists():
            if output.is_dir():
                import shutil
                shutil.rmtree(output)
            else:
                output.unlink()
        os.replace(staged, output)
    return output / "manifest.json"


def _required_manifest(spec: EvidenceCacheArtifact) -> dict[str, Any]:
    return {
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": "learned-evidence",
        "cache_repository": spec.repository,
        "cache_tag": spec.tag,
        "asset_name": spec.asset_name,
        "detector": spec.detector,
        "evidence_id": spec.evidence_id,
        "evidence_identity": spec.identity,
        "trust_role": CACHE_TRUST_ROLE,
    }


def _validate_release_manifest(payload: dict[str, Any], spec: EvidenceCacheArtifact) -> None:
    for key, expected in _required_manifest(spec).items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"Collection-cache manifest {key} mismatch: expected {expected!r}, got {payload.get(key)!r}"
            )
    if len(str(payload.get("sha256") or "")) != 64:
        raise RuntimeError("Collection-cache manifest has no valid asset SHA-256")
    if len(str(payload.get("evidence_manifest_sha256") or "")) != 64:
        raise RuntimeError("Collection-cache manifest has no valid evidence-manifest SHA-256")


def download(
    spec: EvidenceCacheArtifact,
    target: Path,
    *,
    fetch: Callable[[str, Path], None],
) -> dict[str, Any]:
    """Download and fully validate an immutable collection-cache artifact."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        temp_root = Path(temp)
        release_manifest_path = temp_root / "release-manifest.json"
        artifact_path = temp_root / spec.asset_name
        fetch(spec.manifest_url, release_manifest_path)
        release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
        _validate_release_manifest(release_manifest, spec)
        fetch(spec.asset_url, artifact_path)
        actual_sha256 = sha256_file(artifact_path)
        if actual_sha256 != release_manifest["sha256"]:
            raise RuntimeError(
                "Collection-cache asset SHA-256 mismatch: "
                f"expected={release_manifest['sha256']} actual={actual_sha256}"
            )
        evidence = validate_evidence_bundle(
            artifact_path,
            expected_manifest_sha256=release_manifest["evidence_manifest_sha256"],
        )
        for key, expected in (
            ("detector", spec.detector),
            ("evidence_id", spec.evidence_id),
            ("identity", spec.identity),
        ):
            if evidence[key] != expected:
                raise RuntimeError(
                    f"Bundled learned-evidence {key} mismatch: "
                    f"expected={expected!r} actual={evidence[key]!r}"
                )
        artifact_path.replace(target)
    return release_manifest


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
    spec: EvidenceCacheArtifact,
    artifact: Path,
    *,
    evidence_manifest_sha256: str,
    migration_source: dict[str, Any] | None = None,
    token: str | None = None,
) -> str:
    """Publish a complete immutable cache release, refusing partial mutation."""
    token = token or os.environ.get("HTH_CACHE_TOKEN") or os.environ.get("HTH_RESULTS_TOKEN")
    if not token:
        return "skipped-no-token"
    artifact = Path(artifact)
    api = f"https://api.github.com/repos/{spec.repository}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    raced_creation = False
    try:
        release = _github_json(urllib.request.Request(f"{api}/releases/tags/{spec.tag}", headers=headers))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        try:
            release = _github_json(urllib.request.Request(
                f"{api}/releases",
                data=json.dumps({
                    "tag_name": spec.tag,
                    "name": spec.tag,
                    "body": (
                        "Immutable non-authoritative HTH collection-cache artifact. "
                        "See the attached machine-readable manifest."
                    ),
                    "draft": False,
                    "prerelease": False,
                }).encode("utf-8"),
                method="POST",
                headers={**headers, "Content-Type": "application/json"},
            ))
        except urllib.error.HTTPError as create_exc:
            # Independent pipelines may miss the same immutable artifact at the
            # same time. GitHub returns 422 to all but the release winner; use
            # the winner's release instead of treating that harmless race as a
            # cache publication failure.
            if create_exc.code != 422:
                raise
            raced_creation = True
            release = _github_json(urllib.request.Request(
                f"{api}/releases/tags/{spec.tag}", headers=headers
            ))
    existing = {str(asset.get("name")) for asset in release.get("assets", [])}
    manifest_name = f"{spec.asset_name}.manifest.json"
    if spec.asset_name in existing and manifest_name in existing:
        return "already-present"
    if (spec.asset_name in existing or manifest_name in existing) and not raced_creation:
        raise RuntimeError(f"Immutable cache release {spec.tag} is incomplete; refusing to modify it")
    release_manifest = {
        **_required_manifest(spec),
        "size_bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
        "evidence_manifest_sha256": evidence_manifest_sha256,
        "migration_source": migration_source,
    }
    upload_url = str(release["upload_url"]).split("{", 1)[0]
    uploads = (
        (spec.asset_name, artifact.read_bytes(), "application/zip"),
        (
            manifest_name,
            (json.dumps(release_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            "application/json",
        ),
    )
    for name, content, content_type in uploads:
        if name in existing:
            continue
        try:
            _upload(upload_url, name, content, content_type, token)
        except urllib.error.HTTPError as upload_exc:
            if upload_exc.code != 422:
                raise
            refreshed = _github_json(urllib.request.Request(
                f"{api}/releases/tags/{spec.tag}", headers=headers
            ))
            refreshed_assets = {str(asset.get("name")) for asset in refreshed.get("assets", [])}
            if name not in refreshed_assets:
                raise
        existing.add(name)
    return "published"
