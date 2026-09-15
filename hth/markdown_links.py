"""Stable Markdown links for HTH provenance summaries."""
from __future__ import annotations

from urllib.parse import quote, urlparse


def code(value: object) -> str:
    """Render a value with the code styling used by HTH manifests."""
    text = str(value).strip() if value is not None else ""
    return f"`{text or 'unknown'}`"


def link(label: str, url: str) -> str:
    """Link an already-rendered label when an authoritative URL is known."""
    return f"[{label}]({url})" if url else label


def code_link(value: object, url: str) -> str:
    """Preserve a manifest's code-formatted value while making it navigable."""
    return link(code(value), url)


def github_repository_url(repository: str, *, server_url: str = "https://github.com") -> str:
    repository = repository.strip()
    if repository.startswith(("http://", "https://")):
        repository = urlparse(repository).path.strip("/")
    elif repository.startswith("git@") and ":" in repository:
        repository = repository.split(":", 1)[1]
    repository = repository.strip("/").removesuffix(".git")
    return f"{server_url.rstrip('/')}/{repository}" if repository else ""


def github_release_url(
    repository: str,
    tag: str,
    *,
    server_url: str = "https://github.com",
) -> str:
    base = github_repository_url(repository, server_url=server_url)
    return f"{base}/releases/tag/{quote(tag.strip(), safe='')}" if base and tag.strip() else ""


def github_commit_url(
    repository: str,
    commit: str,
    *,
    server_url: str = "https://github.com",
) -> str:
    base = github_repository_url(repository, server_url=server_url)
    return f"{base}/commit/{quote(commit.strip(), safe='')}" if base and commit.strip() else ""


def github_blob_url(
    repository: str,
    ref: str,
    path: str,
    *,
    server_url: str = "https://github.com",
) -> str:
    base = github_repository_url(repository, server_url=server_url)
    if not base or not ref.strip() or not path.strip():
        return ""
    encoded_ref = quote(ref.strip(), safe="")
    encoded_path = quote(path.strip().lstrip("/"), safe="/")
    return f"{base}/blob/{encoded_ref}/{encoded_path}"


def github_raw_url(
    repository: str,
    ref: str,
    path: str,
    *,
    server_url: str = "https://github.com",
) -> str:
    """Return GitHub's direct-file URL for large downloadable evidence."""
    base = github_repository_url(repository, server_url=server_url)
    if not base or not ref.strip() or not path.strip():
        return ""
    encoded_ref = quote(ref.strip(), safe="")
    encoded_path = quote(path.strip().lstrip("/"), safe="/")
    return f"{base}/raw/{encoded_ref}/{encoded_path}"


def github_tree_url(
    repository: str,
    ref: str,
    path: str,
    *,
    server_url: str = "https://github.com",
) -> str:
    base = github_repository_url(repository, server_url=server_url)
    if not base or not ref.strip() or not path.strip():
        return ""
    encoded_ref = quote(ref.strip(), safe="")
    encoded_path = quote(path.strip().lstrip("/"), safe="/")
    return f"{base}/tree/{encoded_ref}/{encoded_path}"
