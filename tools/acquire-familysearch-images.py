#!/usr/bin/env python3
"""Acquire historical-record images from FamilySearch using the official API.

The downloader deliberately does not scrape the viewer or reuse browser cookies.
It consumes FamilySearch's authenticated Historical Records Image resources,
follows image/media links returned by FamilySearch, verifies downloaded image
bytes, and persists provenance/checksums for resumable acquisition.

An authenticated FamilySearch access token must be supplied through the
FAMILYSEARCH_ACCESS_TOKEN environment variable (or --access-token-env).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - repository runtime provides Pillow
    raise SystemExit("Pillow is required: install repository dependencies first.") from exc


API_BASE = "https://api.familysearch.org"
IMAGE_PATH = "/platform/records/images/{iid}"
GEDCOMX_ACCEPT = "application/x-fs-v1+json"
USER_AGENT = "HTH-FamilySearch-Acquisition/0.1 (+https://github.com/dlstupka/hth)"
DEFAULT_TOKEN_ENV = "FAMILYSEARCH_ACCESS_TOKEN"
DEFAULT_MANIFEST = "familysearch-source-manifest.json"

IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tif",
    "image/x-tiff": ".tif",
    "image/webp": ".webp",
    "image/jp2": ".jp2",
    "image/jpeg2000": ".jp2",
}
KNOWN_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".jp2"}

IID_PATTERN = re.compile(r"(3:1:[A-Za-z0-9-]+)")
ARK_PATTERN = re.compile(r"/ark:/61903/(3:1:[A-Za-z0-9-]+)")

# Link relations are ranked.  Unknown links with obvious image extensions are
# considered only after explicit image/download/artifact relations.
PREFERRED_MEDIA_RELS = (
    "original",
    "download",
    "image-original",
    "image",
    "artifact",
    "media",
    "content",
    "digital-artifact",
    "digitalArtifact",
)
METADATA_RELS = ("image-metadata", "metadata")
NEXT_RELS = ("next", "image-next")


@dataclass(frozen=True)
class ImageFacts:
    width: int
    height: int
    format: str
    mode: str


class AcquisitionError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - verification against source metadata only
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_image(path: Path) -> ImageFacts:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return ImageFacts(
                width=int(image.width),
                height=int(image.height),
                format=str(image.format or ""),
                mode=str(image.mode or ""),
            )
    except Exception as exc:
        raise AcquisitionError(f"{path} is not a valid readable image: {exc}") from exc


def normalize_content_type(value: str | None) -> str:
    return (value or "").split(";", 1)[0].strip().lower()


def extension_for(content_type: str, url: str, image_format: str | None = None) -> str:
    content_type = normalize_content_type(content_type)
    if content_type in IMAGE_EXTENSIONS:
        return IMAGE_EXTENSIONS[content_type]
    if image_format:
        fmt = image_format.upper()
        return {
            "JPEG": ".jpg",
            "PNG": ".png",
            "TIFF": ".tif",
            "WEBP": ".webp",
            "JPEG2000": ".jp2",
            "JP2": ".jp2",
        }.get(fmt, ".img")
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in KNOWN_IMAGE_SUFFIXES:
        return ".jpg" if suffix == ".jpeg" else ".tif" if suffix == ".tiff" else suffix
    guessed = mimetypes.guess_extension(content_type) if content_type else None
    return guessed or ".img"


def sanitize_url(url: str) -> str:
    """Strip query/fragment so signed URLs or tokens are never persisted."""
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))


def extract_seed(seed: str) -> tuple[str, dict[str, str]]:
    """Return FamilySearch image id plus any traversal context from a URL/id."""
    seed = seed.strip()
    if IID_PATTERN.fullmatch(seed):
        return seed, {}

    match = ARK_PATTERN.search(seed) or IID_PATTERN.search(seed)
    if not match:
        raise AcquisitionError(
            "Could not find a FamilySearch historical-image id (3:1:...) in seed."
        )
    iid = match.group(1)
    query = parse_qs(urlparse(seed).query)
    context: dict[str, str] = {}
    # The Image resource explicitly documents cc/wc/from. Preserve those when
    # they are present in the viewer/API URL. groupId is retained as provenance,
    # but is not silently re-labeled as waypoint context.
    for key in ("cc", "wc", "from", "groupId"):
        values = query.get(key)
        if values and values[0]:
            context[key] = values[0]
    return iid, context


def iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def extract_iid(document: Any, fallback: str | None = None) -> str:
    """Find the current 3:1 image id in an Image-resource response."""
    if isinstance(document, dict):
        for key in ("id", "imageId", "image_id"):
            value = document.get(key)
            if isinstance(value, str):
                match = IID_PATTERN.search(value)
                if match:
                    return match.group(1)

    for node in iter_dicts(document):
        for value in node.values():
            if not isinstance(value, str):
                continue
            match = ARK_PATTERN.search(value) or IID_PATTERN.search(value)
            if match:
                return match.group(1)
    if fallback:
        return fallback
    raise AcquisitionError("FamilySearch response did not identify the current image id.")


def _relation_href(link_value: Any) -> str | None:
    if isinstance(link_value, str) and link_value.startswith(("http://", "https://")):
        return link_value
    if isinstance(link_value, dict):
        href = link_value.get("href")
        if isinstance(href, str) and href.startswith(("http://", "https://")):
            return href
    return None


def links_by_relation(document: Any) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for node in iter_dicts(document):
        links = node.get("links")
        if not isinstance(links, dict):
            continue
        for rel, value in links.items():
            href = _relation_href(value)
            if href:
                found.setdefault(str(rel), []).append(href)
    return found


def media_candidates(document: Any) -> list[tuple[str, str]]:
    """Extract likely image-byte URLs, strongest FamilySearch relations first."""
    links = links_by_relation(document)
    result: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(rel: str, url: str) -> None:
        if url not in seen:
            seen.add(url)
            result.append((rel, url))

    for rel in PREFERRED_MEDIA_RELS:
        for url in links.get(rel, []):
            add(rel, url)

    # Some API representations use namespaced/custom relation names.
    for rel, urls in links.items():
        rel_lower = rel.lower()
        if any(token in rel_lower for token in ("download", "original", "artifact", "media")):
            for url in urls:
                add(rel, url)

    # Last resort: a FamilySearch-returned href with an obvious image suffix.
    for rel, urls in links.items():
        for url in urls:
            if Path(urlparse(url).path).suffix.lower() in KNOWN_IMAGE_SUFFIXES:
                add(rel, url)
    return result


def metadata_candidates(document: Any, iid: str) -> list[str]:
    links = links_by_relation(document)
    result: list[str] = []
    for rel in METADATA_RELS:
        result.extend(links.get(rel, []))
    default = f"{API_BASE}/platform/records/images/{iid}/metadata"
    if default not in result:
        result.append(default)
    return result


def next_link(document: Any) -> str | None:
    links = links_by_relation(document)
    for rel in NEXT_RELS:
        urls = links.get(rel)
        if urls:
            return urls[0]
    return None


def flatten_scalar_pairs(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, (dict, list)):
                yield from flatten_scalar_pairs(child, path)
            else:
                yield path, child
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_scalar_pairs(child, f"{prefix}[{index}]")


def source_expectations(metadata: Any) -> dict[str, Any]:
    """Extract only source facts we can compare without assuming one schema."""
    result: dict[str, Any] = {}
    for path, value in flatten_scalar_pairs(metadata):
        leaf = path.rsplit(".", 1)[-1].lower()
        if isinstance(value, bool):
            continue
        if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
            n = int(value)
            if leaf in {"width", "imagewidth", "image_width", "widthpx", "width_px"}:
                result.setdefault("width", n)
            elif leaf in {"height", "imageheight", "image_height", "heightpx", "height_px"}:
                result.setdefault("height", n)
            elif leaf in {"filesize", "file_size", "size", "sizebytes", "size_bytes"}:
                result.setdefault("size", n)
        if isinstance(value, str):
            v = value.strip().lower()
            if leaf in {"md5", "checksum", "hash", "messagedigest"} and re.fullmatch(
                r"[0-9a-f]{32}", v
            ):
                result.setdefault("md5", v)
            elif leaf in {"sha256", "sha_256"} and re.fullmatch(r"[0-9a-f]{64}", v):
                result.setdefault("sha256", v)
    return result


def verify_against_source(
    path: Path, facts: ImageFacts, expectations: dict[str, Any]
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "image_readable": True,
        "width": facts.width,
        "height": facts.height,
    }
    mismatches: list[str] = []
    if "width" in expectations:
        checks["source_width"] = expectations["width"]
        if facts.width != expectations["width"]:
            mismatches.append(
                f"width {facts.width} != FamilySearch metadata {expectations['width']}"
            )
    if "height" in expectations:
        checks["source_height"] = expectations["height"]
        if facts.height != expectations["height"]:
            mismatches.append(
                f"height {facts.height} != FamilySearch metadata {expectations['height']}"
            )
    if "size" in expectations:
        actual = path.stat().st_size
        checks["source_size"] = expectations["size"]
        if actual != expectations["size"]:
            mismatches.append(
                f"size {actual} != FamilySearch metadata {expectations['size']}"
            )
    if "sha256" in expectations:
        actual_sha = sha256_file(path)
        checks["source_sha256"] = expectations["sha256"]
        if actual_sha.lower() != expectations["sha256"].lower():
            mismatches.append("SHA-256 does not match FamilySearch metadata")
    if "md5" in expectations:
        actual_md5 = md5_file(path)
        checks["source_md5"] = expectations["md5"]
        if actual_md5.lower() != expectations["md5"].lower():
            mismatches.append("MD5 does not match FamilySearch metadata")
    if mismatches:
        raise AcquisitionError(f"{path}: " + "; ".join(mismatches))
    checks["source_metadata_checks"] = sorted(expectations)
    return checks


class FamilySearchClient:
    def __init__(
        self,
        token: str,
        *,
        timeout: int = 60,
        retries: int = 5,
        backoff: float = 2.0,
    ):
        self.token = token
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def _request(
        self,
        url: str,
        *,
        accept: str,
        range_header: str | None = None,
    ):
        headers = {
            "Accept": accept,
            "Authorization": f"Bearer {self.token}",
            "User-Agent": USER_AGENT,
        }
        if range_header:
            headers["Range"] = range_header
        return Request(url, headers=headers)

    def open(self, url: str, *, accept: str, range_header: str | None = None):
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                return urlopen(
                    self._request(url, accept=accept, range_header=range_header),
                    timeout=self.timeout,
                )
            except HTTPError as exc:
                last = exc
                # Auth/rights/not-found errors are not transient.
                if exc.code in {400, 401, 403, 404}:
                    raise AcquisitionError(
                        f"FamilySearch request failed {exc.code} for {sanitize_url(url)}"
                    ) from exc
                if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                    raise AcquisitionError(
                        f"FamilySearch request failed {exc.code} for {sanitize_url(url)}"
                    ) from exc
            except URLError as exc:
                last = exc
            if attempt < self.retries:
                time.sleep(self.backoff * attempt)
        raise AcquisitionError(
            f"FamilySearch request failed after {self.retries} attempts: "
            f"{sanitize_url(url)} ({last})"
        )

    def json(self, url: str) -> tuple[dict[str, Any], dict[str, str]]:
        with self.open(url, accept=GEDCOMX_ACCEPT) as response:
            content_type = normalize_content_type(response.headers.get("Content-Type"))
            if "json" not in content_type:
                raise AcquisitionError(
                    f"Expected JSON from {sanitize_url(url)}, got {content_type or 'unknown'}"
                )
            payload = json.load(response)
            headers = {k.lower(): v for k, v in response.headers.items()}
        if not isinstance(payload, dict):
            raise AcquisitionError(f"Expected JSON object from {sanitize_url(url)}")
        return payload, headers

    def download_image(self, url: str, destination: Path) -> tuple[str, int]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(destination.name + ".part")
        try:
            with self.open(url, accept="image/*") as response:
                content_type = normalize_content_type(response.headers.get("Content-Type"))
                if not content_type.startswith("image/"):
                    raise AcquisitionError(
                        f"FamilySearch media link did not return an image "
                        f"({content_type or 'unknown'}): {sanitize_url(url)}"
                    )
                size = 0
                with temp.open("wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        size += len(chunk)
            if size == 0:
                raise AcquisitionError(f"FamilySearch returned an empty image: {sanitize_url(url)}")
            # The caller may rename after Pillow confirms the actual format.
            return content_type, size
        except Exception:
            temp.unlink(missing_ok=True)
            raise


def image_resource_url(iid: str, context: dict[str, str], *, seek: str = "current") -> str:
    params: dict[str, str] = {"seek": seek}
    for key in ("cc", "wc", "from"):
        if context.get(key):
            params[key] = context[key]
    return f"{API_BASE}{IMAGE_PATH.format(iid=iid)}?{urlencode(params)}"


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "source_system": "FamilySearch",
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "images": [],
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise AcquisitionError(f"Invalid acquisition manifest: {path}")
    return data


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = utc_now()
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def manifest_entry_by_ordinal(manifest: dict[str, Any], ordinal: int) -> dict[str, Any] | None:
    for entry in manifest.get("images", []):
        if entry.get("ordinal") == ordinal:
            return entry
    return None


def existing_named_image(images_dir: Path, stem: str) -> Path | None:
    matches = [
        path
        for path in images_dir.glob(stem + ".*")
        if path.is_file() and path.suffix.lower() in KNOWN_IMAGE_SUFFIXES
    ]
    if len(matches) > 1:
        raise AcquisitionError(
            f"Multiple existing images use logical name {stem}: "
            + ", ".join(str(p.name) for p in matches)
        )
    return matches[0] if matches else None


def fetch_metadata(client: FamilySearchClient, document: Any, iid: str) -> tuple[Any, str | None]:
    for url in metadata_candidates(document, iid):
        try:
            payload, _ = client.json(url)
            return payload, sanitize_url(url)
        except AcquisitionError:
            continue
    return {}, None


def resolve_media(
    client: FamilySearchClient,
    document: dict[str, Any],
    iid: str,
    metadata: dict[str, Any],
) -> tuple[str, str]:
    candidates = media_candidates(document) + media_candidates(metadata)
    seen: set[str] = set()
    errors: list[str] = []
    for rel, url in candidates:
        if url in seen:
            continue
        seen.add(url)
        # Do not download during discovery. A tiny range request rejects JSON
        # links cheaply; the real download follows only after a media candidate.
        try:
            with client.open(url, accept="image/*", range_header="bytes=0-0") as response:
                content_type = normalize_content_type(response.headers.get("Content-Type"))
                if content_type.startswith("image/"):
                    return rel, url
                errors.append(f"{rel}: {content_type or 'unknown content type'}")
        except AcquisitionError as exc:
            errors.append(f"{rel}: {exc}")
    detail = "; ".join(errors[-5:]) if errors else "no image/download/artifact link was returned"
    raise AcquisitionError(
        f"FamilySearch image {iid} exposes metadata but no downloadable image representation "
        f"through the official API response ({detail}). The collection may prohibit downloads, "
        "or this API representation may not expose image bytes."
    )


def verify_existing(
    path: Path,
    entry: dict[str, Any] | None,
    *,
    iid: str,
    expectations: dict[str, Any],
) -> tuple[ImageFacts, dict[str, Any]]:
    facts = inspect_image(path)
    checks = verify_against_source(path, facts, expectations)
    if entry:
        if entry.get("familysearch_image_id") and entry["familysearch_image_id"] != iid:
            raise AcquisitionError(
                f"{path}: manifest image id {entry['familysearch_image_id']} != current {iid}"
            )
        expected_sha = entry.get("sha256")
        if expected_sha:
            actual_sha = sha256_file(path)
            if actual_sha.lower() != str(expected_sha).lower():
                raise AcquisitionError(f"{path}: SHA-256 no longer matches acquisition manifest")
            checks["manifest_sha256"] = True
    return facts, checks


def acquire(args: argparse.Namespace) -> int:
    token = os.environ.get(args.access_token_env, "").strip()
    if not token:
        raise AcquisitionError(
            f"Authenticated FamilySearch token not found in ${args.access_token_env}. "
            "Use FamilySearch OAuth Authorization Code flow and export the resulting access token."
        )

    iid, seed_context = extract_seed(args.seed)
    context = dict(seed_context)
    if args.collection_context:
        context["cc"] = args.collection_context
    if args.waypoint_context:
        context["wc"] = args.waypoint_context
    if args.from_context:
        context["from"] = args.from_context

    images_dir = args.images_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    manifest.update(
        {
            "collection_id": args.collection_id,
            "collection_title": args.collection_title,
            "seed": {
                "image_id": iid,
                "url": sanitize_url(args.seed) if "://" in args.seed else None,
                "context": context,
            },
            "authentication": "FamilySearch OAuth bearer token (token not persisted)",
            "acquisition_method": "FamilySearch Historical Records Image API",
        }
    )

    client = FamilySearchClient(
        token,
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
    )

    seen_iids: set[str] = set()
    ordinal = args.start_ordinal
    acquired = 0
    skipped = 0

    while True:
        if args.count is not None and ordinal >= args.start_ordinal + args.count:
            break

        resource_url = image_resource_url(iid, context, seek="current")
        document, resource_headers = client.json(resource_url)
        current_iid = extract_iid(document, fallback=iid)
        if current_iid in seen_iids:
            raise AcquisitionError(
                f"Traversal repeated FamilySearch image id {current_iid} at ordinal {ordinal}"
            )
        seen_iids.add(current_iid)

        metadata, metadata_url = fetch_metadata(client, document, current_iid)
        expectations = source_expectations(metadata)
        stem = args.name_template.format(ordinal=ordinal, iid=current_iid)
        prior = manifest_entry_by_ordinal(manifest, ordinal)
        existing = existing_named_image(images_dir, stem)

        if existing is not None:
            facts, checks = verify_existing(
                existing, prior, iid=current_iid, expectations=expectations
            )
            print(
                f"{ordinal:04d}: exists/verified {existing.name} "
                f"{facts.width}x{facts.height} iid={current_iid}"
            )
            skipped += 1
            entry = prior or {}
            entry.update(
                {
                    "ordinal": ordinal,
                    "filename": existing.name,
                    "familysearch_image_id": current_iid,
                    "familysearch_ark": f"https://www.familysearch.org/ark:/61903/{current_iid}",
                    "sha256": sha256_file(existing),
                    "size": existing.stat().st_size,
                    "width": facts.width,
                    "height": facts.height,
                    "format": facts.format,
                    "verification": checks,
                    "metadata_url": metadata_url,
                    "status": "existing-verified",
                }
            )
        else:
            rel, media_url = resolve_media(client, document, current_iid, metadata)
            # Download to an extension-neutral temporary file first; Pillow is
            # authoritative for the actual format used in the final filename.
            temp_dest = images_dir / f"{stem}.download"
            content_type, _ = client.download_image(media_url, temp_dest)
            part = temp_dest.with_name(temp_dest.name + ".part")
            # download_image writes .part so incomplete bytes can never acquire a
            # final source name.
            facts = inspect_image(part)
            ext = extension_for(content_type, media_url, facts.format)
            destination = images_dir / f"{stem}{ext}"
            if destination.exists():
                part.unlink(missing_ok=True)
                raise AcquisitionError(
                    f"Refusing to overwrite existing source image {destination}"
                )
            checks = verify_against_source(part, facts, expectations)
            part.replace(destination)
            digest = sha256_file(destination)
            print(
                f"{ordinal:04d}: downloaded {destination.name} "
                f"{facts.width}x{facts.height} {destination.stat().st_size:,} B "
                f"sha256={digest[:12]}... iid={current_iid}"
            )
            acquired += 1
            entry = {
                "ordinal": ordinal,
                "filename": destination.name,
                "familysearch_image_id": current_iid,
                "familysearch_ark": f"https://www.familysearch.org/ark:/61903/{current_iid}",
                "sha256": digest,
                "size": destination.stat().st_size,
                "width": facts.width,
                "height": facts.height,
                "format": facts.format,
                "mime_type": content_type,
                "media_relation": rel,
                "media_url": sanitize_url(media_url),
                "metadata_url": metadata_url,
                "verification": checks,
                "downloaded_at_utc": utc_now(),
                "status": "downloaded-verified",
            }

        if prior is None:
            manifest["images"].append(entry)
        else:
            index = manifest["images"].index(prior)
            manifest["images"][index] = entry
        manifest["images"].sort(key=lambda row: row.get("ordinal", 0))
        write_manifest(args.manifest, manifest)

        if args.probe_only:
            print("Probe successful; stopping after one image.")
            break

        # Prefer a FamilySearch-provided next link. Otherwise use the documented
        # seek=next operation on the current Image resource.
        direct_next = next_link(document)
        if direct_next:
            next_document, _ = client.json(direct_next)
        else:
            next_document, _ = client.json(
                image_resource_url(current_iid, context, seek="next")
            )
        next_iid = extract_iid(next_document, fallback=None)
        if next_iid == current_iid:
            if args.count is None:
                print(f"Reached end of FamilySearch traversal at {current_iid}.")
                break
            raise AcquisitionError(
                f"FamilySearch seek=next did not advance after {current_iid}; "
                "supply the collection/waypoint context from the image viewer URL."
            )

        iid = next_iid
        ordinal += 1

    manifest["summary"] = {
        "records": len(manifest.get("images", [])),
        "downloaded_this_run": acquired,
        "existing_verified_this_run": skipped,
        "unique_familysearch_image_ids": len(
            {row.get("familysearch_image_id") for row in manifest.get("images", [])}
        ),
    }
    if args.expected_count is not None:
        actual = len(manifest.get("images", []))
        manifest["summary"]["expected_count"] = args.expected_count
        manifest["summary"]["expected_count_match"] = actual == args.expected_count
        if actual != args.expected_count:
            write_manifest(args.manifest, manifest)
            raise AcquisitionError(
                f"Acquisition manifest has {actual} images; expected {args.expected_count}"
            )

    # Collection-level verification: no duplicate image ids, names, or hashes.
    ids = [row.get("familysearch_image_id") for row in manifest["images"]]
    names = [row.get("filename") for row in manifest["images"]]
    duplicates = {
        "familysearch_image_ids": sorted({x for x in ids if x and ids.count(x) > 1}),
        "filenames": sorted({x for x in names if x and names.count(x) > 1}),
    }
    manifest["collection_verification"] = {
        "duplicate_familysearch_image_ids": duplicates["familysearch_image_ids"],
        "duplicate_filenames": duplicates["filenames"],
        "all_files_present": all((images_dir / name).is_file() for name in names if name),
        "all_images_readable": True,
    }
    write_manifest(args.manifest, manifest)
    if duplicates["familysearch_image_ids"] or duplicates["filenames"]:
        raise AcquisitionError(f"Duplicate source identities detected: {duplicates}")

    print(
        f"Done: {acquired} downloaded, {skipped} existing verified, "
        f"{len(manifest['images'])} manifest records."
    )
    print(f"Manifest: {args.manifest}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Download historical-record images through the authenticated "
            "FamilySearch Image API without browser/AHK capture."
        )
    )
    p.add_argument(
        "seed",
        help=(
            "First FamilySearch image id (3:1:...) or viewer/API URL containing it. "
            "A viewer URL is preferred because cc/wc context is retained when present."
        ),
    )
    p.add_argument("--images-dir", type=Path, default=Path("images"))
    p.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    p.add_argument("--collection-id", default="HTH-SOURCE-0002")
    p.add_argument(
        "--collection-title",
        default="Baptisms: San Antonio. Baptism Records 1788-1824, 1858-1898",
    )
    p.add_argument("--start-ordinal", type=int, default=1)
    p.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of sequential FamilySearch images to acquire; omit to traverse to end.",
    )
    p.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Fail at completion unless the manifest contains exactly this many images.",
    )
    p.add_argument(
        "--name-template",
        default="fs_{ordinal:04d}",
        help="Filename stem template. Available fields: ordinal, iid.",
    )
    p.add_argument("--collection-context", help="FamilySearch Image API cc context.")
    p.add_argument("--waypoint-context", help="FamilySearch Image API wc context.")
    p.add_argument("--from-context", help="FamilySearch Image API from context.")
    p.add_argument("--access-token-env", default=DEFAULT_TOKEN_ENV)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--backoff", type=float, default=2.0)
    p.add_argument(
        "--probe-only",
        action="store_true",
        help="Resolve/verify only the seed image, then stop.",
    )
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        return acquire(args)
    except AcquisitionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
