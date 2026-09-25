"""Serve the reference director and proxy public Golden Set release assets locally.

Run ``python tools/reference-collection-director.py`` and enter the source
repository URL in the opened page. No GitHub token or third-party dependency is
needed for a public source repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


TOOLS = Path(__file__).resolve().parent
SOURCE_RE = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/?$", re.I)
GOLDEN_SET_RE = re.compile(r"^HTH-GOLDEN-\d{4,}$")
MAX_JSON_BYTES = 2_000_000
MAX_BUNDLE_BYTES = 128_000_000
CACHE_ROOT = Path(os.environ.get("HTH_REFERENCE_CACHE_ROOT") or
                  (Path(os.environ["LOCALAPPDATA"]) / "HTH" / "reference-collection-cache"
                   if os.environ.get("LOCALAPPDATA") else Path.home() / ".cache" / "hth" / "reference-collection"))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def source_repository(value: str) -> str:
    match = SOURCE_RE.fullmatch(value.strip())
    if not match:
        raise ValueError("Enter a source repository URL such as https://github.com/owner/repository")
    return f"{match.group(1)}/{match.group(2)}"


def results_checkout(repository: str) -> Path:
    """Use the sibling checkout named by the source-repository convention."""
    return TOOLS.parent.parent / f"{repository.split('/')[1]}-results"


def update_results_checkout(repository: str) -> dict[str, str]:
    checkout = results_checkout(repository)
    if not checkout.is_dir():
        raise FileNotFoundError(f"No local results checkout at {checkout}. The source release can still be loaded.")
    expected = f"github.com:{repository}-results"

    def git(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-c", f"safe.directory={checkout.as_posix()}", "-C", str(checkout), *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )

    remote = git("remote", "get-url", "origin")
    if remote.returncode:
        raise ValueError(f"Results checkout has no readable origin: {checkout}")
    actual = remote.stdout.strip().removesuffix(".git")
    if actual not in (f"git@{expected}", f"https://github.com/{repository}-results"):
        raise ValueError(f"Results checkout origin does not match {repository}-results: {actual}")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty.returncode or dirty.stdout.strip():
        raise ValueError("Results checkout has tracked local changes; export or commit them before updating.")
    before = git("rev-parse", "HEAD")
    if before.returncode:
        raise ValueError("Results checkout has no current commit")
    pulled = git("pull", "--ff-only", timeout=600)
    if pulled.returncode:
        raise ValueError(f"git pull --ff-only failed: {(pulled.stderr or pulled.stdout).strip()[-500:]}")
    after = git("rev-parse", "HEAD")
    return {
        "repository": f"{repository}-results", "checkout": str(checkout),
        "before": before.stdout.strip(), "after": after.stdout.strip(),
        "status": "updated" if before.stdout != after.stdout else "already current",
    }


def _open(url: str, accept: str = "application/vnd.github+json"):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "hth-reference-director/1"}),
        timeout=60,
    )


def _read_limited(url: str, limit: int = MAX_JSON_BYTES) -> bytes:
    with _open(url) as response:
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("Release metadata exceeds the accepted size limit")
    return payload


def _cache_path(sha256: str) -> Path:
    if not SHA256_RE.fullmatch(sha256):
        raise ValueError("Invalid immutable asset SHA-256")
    return CACHE_ROOT / "sha256" / sha256[:2] / sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_cached_asset(sha256: str, size: int | None = None) -> Path | None:
    path = _cache_path(sha256)
    if not path.is_file():
        return None
    if (size is not None and path.stat().st_size != size) or _sha256_file(path) != sha256:
        path.unlink(missing_ok=True)
        return None
    return path


def _read_cached_json_asset(asset: dict[str, Any], expected_sha256: str | None = None) -> bytes:
    digest = expected_sha256 or str(asset.get("digest", "")).removeprefix("sha256:")
    trusted_digest = digest if SHA256_RE.fullmatch(digest) else None
    size = int(asset.get("size") or 0)
    cached = _verified_cached_asset(trusted_digest, size if size > 0 else None) if trusted_digest else None
    if cached:
        if cached.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("Cached release metadata exceeds the accepted size limit")
        return cached.read_bytes()
    payload = _read_limited(str(asset["browser_download_url"]))
    if trusted_digest and hashlib.sha256(payload).hexdigest() != trusted_digest:
        raise ValueError(f"Release asset SHA-256 mismatch: {asset['name']}")
    if size > 0 and len(payload) != size:
        raise ValueError(f"Release asset size mismatch: {asset['name']}")
    if trusted_digest:
        target = _cache_path(trusted_digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".asset-", delete=False) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        temporary.replace(target)
    return payload


def _release_asset(assets: dict[str, dict[str, Any]], name: str, repository: str, tag: str) -> dict[str, Any]:
    asset = assets.get(name)
    if asset is None:
        raise ValueError(f"Release {tag} is missing {name}")
    url = str(asset.get("browser_download_url", ""))
    expected = f"https://github.com/{repository}/releases/download/{tag}/{name}"
    if url != expected:
        raise ValueError(f"Release asset URL does not match {repository}/{tag}/{name}")
    return asset


def resolve_release(repository: str, tag: str) -> dict[str, Any]:
    if not GOLDEN_SET_RE.fullmatch(tag):
        raise ValueError("Invalid Golden Set release tag")
    api = f"https://api.github.com/repos/{repository}/releases/tags/{tag}"
    release = json.loads(_read_limited(api))
    if release.get("tag_name") != tag:
        raise ValueError("GitHub returned a different Golden Set release")
    assets = {asset["name"]: asset for asset in release.get("assets", [])}
    freeze_asset = _release_asset(assets, f"{tag}.freeze.json", repository, tag)
    golden_asset = _release_asset(assets, f"{tag}.golden-set.json", repository, tag)
    freeze_bytes = _read_cached_json_asset(freeze_asset)
    freeze = json.loads(freeze_bytes)
    golden_bytes = _read_cached_json_asset(golden_asset, freeze.get("golden_set_sha256"))
    golden_set = json.loads(golden_bytes)
    digest = hashlib.sha256(golden_bytes).hexdigest()
    if freeze.get("golden_set_id") != tag or freeze.get("state") != "frozen":
        raise ValueError("The selected Golden Set is not frozen under the requested tag")
    if freeze.get("golden_set_sha256") != digest:
        raise ValueError("Golden Set JSON SHA-256 does not match its freeze record")
    if golden_set.get("golden_set_workflow", {}).get("golden_set_id") != tag:
        raise ValueError("Golden Set JSON identity does not match its release tag")
    ordinals = [int(page["global_ordinal"]) for page in golden_set.get("pages", [])]
    membership = freeze.get("membership", {})
    if (len(ordinals) != membership.get("page_count") or
            ordinals != membership.get("global_ordinals") or
            len(ordinals) != len(set(ordinals))):
        raise ValueError("Frozen Golden Set page membership does not match its JSON")
    canonical = freeze.get("canonical_release", {})
    if canonical.get("repository") != repository or canonical.get("tag") != tag:
        raise ValueError("Freeze record names a different canonical source release")
    if canonical.get("golden_set_asset") != golden_asset["name"] or canonical.get("freeze_asset") != freeze_asset["name"]:
        raise ValueError("Freeze record names different Golden Set assets")
    bundle = freeze.get("image_bundle", {})
    bundle_asset = _release_asset(assets, str(bundle.get("asset", "")), repository, tag)
    size = int(bundle.get("size", 0))
    if (canonical.get("image_bundle_asset") != bundle_asset["name"] or
            bundle.get("format") != "zip/store" or
            size <= 0 or size > MAX_BUNDLE_BYTES or
            bundle_asset.get("size") != size or
            bundle_asset.get("digest") != f"sha256:{bundle.get('sha256')}"):
        raise ValueError("Image bundle metadata does not match the immutable release")
    if ([int(image["global_ordinal"]) for image in bundle.get("images", [])] != ordinals or
            any(image.get("sha256") != page.get("image_sha256")
                for image, page in zip(bundle["images"], golden_set["pages"]))):
        raise ValueError("Image bundle membership or digests differ from the frozen Golden Set")
    return {
        "repository": repository,
        "tag": tag,
        "golden_set_sha256": digest,
        "golden_set": golden_set,
        "bundle": bundle,
        "bundle_url": bundle_asset["browser_download_url"],
    }


def list_golden_sets(repository: str) -> list[str]:
    api = f"https://api.github.com/repos/{repository}/releases?per_page=100"
    releases = json.loads(_read_limited(api))
    if not isinstance(releases, list):
        raise ValueError("GitHub did not return a release list")
    return [tag for item in releases if GOLDEN_SET_RE.fullmatch(tag := str(item.get("tag_name", "")))]


class DirectorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int]):
        super().__init__(address, DirectorHandler)
        self.releases: dict[tuple[str, str], dict[str, Any]] = {}
        self.release_lock = threading.Lock()
        self.bundle_lock = threading.Lock()
        self.update_lock = threading.Lock()

    def release(self, repository: str, tag: str) -> dict[str, Any]:
        key = (repository, tag)
        with self.release_lock:
            if key not in self.releases:
                self.releases[key] = resolve_release(repository, tag)
            return self.releases[key]


class DirectorHandler(SimpleHTTPRequestHandler):
    server: DirectorServer

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, directory=str(TOOLS), **kwargs)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        request = urllib.parse.urlsplit(self.path)
        if request.path not in ("/api/golden-sets", "/api/reference-release", "/api/image-bundle"):
            return super().do_GET()
        headers_sent = False
        try:
            params = urllib.parse.parse_qs(request.query, strict_parsing=True)
            repository = source_repository(params["source_repo"][0])
            if request.path == "/api/golden-sets":
                return self._json(200, {"repository": repository, "golden_sets": list_golden_sets(repository)})
            tag = params["golden_set_id"][0]
            release = self.server.release(repository, tag)
            if request.path == "/api/reference-release":
                return self._json(200, {key: release[key] for key in ("repository", "tag", "golden_set_sha256", "golden_set", "bundle")})
            bundle = release["bundle"]
            expected_size, expected_sha256 = int(bundle["size"]), str(bundle["sha256"])
            with self.server.bundle_lock:
                cached = _verified_cached_asset(expected_sha256, expected_size)
                if cached:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Length", str(expected_size))
                    self.send_header("X-HTH-Cache", "hit")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    headers_sent = True
                    with cached.open("rb") as handle:
                        while chunk := handle.read(256 * 1024):
                            self.wfile.write(chunk)
                    return
                target = _cache_path(expected_sha256)
                target.parent.mkdir(parents=True, exist_ok=True)
                with _open(release["bundle_url"], "application/octet-stream") as upstream:
                    if upstream.headers.get("Content-Length") not in (None, str(expected_size)):
                        raise ValueError("Downloaded image bundle size differs from the freeze record")
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".bundle-", delete=False) as handle:
                            temporary = Path(handle.name)
                            digest, received = hashlib.sha256(), 0
                            self.send_response(200)
                            self.send_header("Content-Type", "application/zip")
                            self.send_header("Content-Length", str(expected_size))
                            self.send_header("X-HTH-Cache", "miss")
                            self.send_header("Cache-Control", "no-store")
                            self.end_headers()
                            headers_sent = True
                            while chunk := upstream.read(256 * 1024):
                                received += len(chunk)
                                if received > expected_size:
                                    raise ValueError("Downloaded image bundle exceeds the freeze record size")
                                digest.update(chunk)
                                handle.write(chunk)
                                self.wfile.write(chunk)
                            if received != expected_size or digest.hexdigest() != expected_sha256:
                                raise ValueError("Downloaded image bundle SHA-256 or size differs from the freeze record")
                    except Exception:
                        if temporary is not None:
                            temporary.unlink(missing_ok=True)
                        raise
                    temporary.replace(target)
        except (KeyError, IndexError, ValueError) as error:
            if not headers_sent:
                self._json(400, {"error": str(error)})
        except urllib.error.HTTPError as error:
            if not headers_sent:
                self._json(502, {"error": f"GitHub returned HTTP {error.code}; check the public source repository and release tag"})
        except (OSError, urllib.error.URLError) as error:
            # Once streaming begins the browser detects a truncated response.
            if not headers_sent:
                self._json(502, {"error": f"Could not retrieve the public release: {error}"})

    def do_POST(self) -> None:
        request = urllib.parse.urlsplit(self.path)
        if request.path != "/api/results-update":
            return self._json(404, {"error": "Unknown update operation"})
        try:
            if self.headers.get("Origin") != f"http://127.0.0.1:{self.server.server_port}":
                raise ValueError("Results update must originate from the local director page")
            if int(self.headers.get("Content-Length", "0")) != 0:
                raise ValueError("Results update does not accept a request body")
            params = urllib.parse.parse_qs(request.query, strict_parsing=True)
            repository = source_repository(params["source_repo"][0])
            with self.server.update_lock:
                result = update_results_checkout(repository)
            self._json(200, result)
        except (KeyError, IndexError, ValueError, FileNotFoundError) as error:
            self._json(400, {"error": str(error)})
        except subprocess.TimeoutExpired:
            self._json(504, {"error": "git pull timed out; inspect the results checkout before retrying"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = DirectorServer(("127.0.0.1", args.port))
    url = f"http://127.0.0.1:{server.server_port}/reference-collection-director.html"
    print(f"Reference Collection Director: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
