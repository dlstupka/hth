"""Render canonical release provenance in GitHub Actions summaries."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from hth.markdown_links import (
    code,
    code_link,
    github_commit_url,
    github_release_url,
    github_repository_url,
)


def input_release_lines(
    *,
    title: str,
    repository: str,
    release_tag: str,
    asset: str,
    asset_sha256: str,
    cache_source: str,
    server_url: str = "https://github.com",
) -> list[str]:
    """Return provenance for an immutable release asset used as stage input."""
    required = {
        "title": title,
        "repository": repository,
        "release_tag": release_tag,
        "asset": asset,
        "asset_sha256": asset_sha256,
        "cache_source": cache_source,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise ValueError(f"Release input provenance is missing: {', '.join(missing)}")
    release_url = github_release_url(repository, release_tag, server_url=server_url)
    repository_url = github_repository_url(repository, server_url=server_url)
    return [
        "",
        f"## {title}",
        "",
        f"- Release: {code_link(release_tag, release_url)}",
        f"- Repository: {code_link(repository, repository_url)}",
        f"- Asset: {code(asset)}",
        f"- Asset SHA-256: {code(asset_sha256)}",
        f"- Resolution: {code(cache_source)}",
    ]


def summary_lines(
    *,
    title: str,
    result_label: str,
    result_identity: str,
    repository: str,
    release_tag: str,
    release_activity: str,
    server_url: str = "https://github.com",
    results_commit: str = "",
    upstream_label: str = "",
    upstream_identity: str = "",
    upstream_url: str = "",
) -> list[str]:
    """Return the shared durable/reusable release provenance block."""
    required = {
        "title": title,
        "result_label": result_label,
        "result_identity": result_identity,
        "repository": repository,
        "release_tag": release_tag,
        "release_activity": release_activity,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise ValueError(f"Release provenance is missing: {', '.join(missing)}")
    upstream_values = (upstream_label, upstream_identity, upstream_url)
    if any(upstream_values) and not all(upstream_values):
        raise ValueError("Upstream provenance requires label, identity, and URL")

    release_url = github_release_url(repository, release_tag, server_url=server_url)
    lines = ["", f"## {title}", ""]
    if all(upstream_values):
        lines.append(f"- {upstream_label}: {code_link(upstream_identity, upstream_url)}")
    lines.extend(
        [
            f"- {result_label}: {code_link(result_identity, release_url)}",
            f"- Release: {code_link(release_tag, release_url)}",
            f"- Release activity: {code(release_activity)}",
        ]
    )
    if results_commit:
        commit_url = github_commit_url(repository, results_commit, server_url=server_url)
        lines.append(f"- Results commit: {code_link(results_commit, commit_url)}")
    return lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True)
    parser.add_argument("--result-label", default="")
    parser.add_argument("--result-identity", default="")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--release-activity", default="")
    parser.add_argument("--results-commit", default="")
    parser.add_argument("--upstream-label", default="")
    parser.add_argument("--upstream-identity", default="")
    parser.add_argument("--upstream-url", default="")
    parser.add_argument("--input-asset", default="")
    parser.add_argument("--input-asset-sha256", default="")
    parser.add_argument("--cache-source", default="")
    parser.add_argument("--server-url", default=os.environ.get("GITHUB_SERVER_URL", "https://github.com"))
    parser.add_argument("--github-summary", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.input_asset:
        rendered = input_release_lines(
            title=args.title,
            repository=args.repository,
            release_tag=args.release_tag,
            asset=args.input_asset,
            asset_sha256=args.input_asset_sha256,
            cache_source=args.cache_source,
            server_url=args.server_url,
        )
    else:
        rendered = summary_lines(
            title=args.title,
            result_label=args.result_label,
            result_identity=args.result_identity,
            repository=args.repository,
            release_tag=args.release_tag,
            release_activity=args.release_activity,
            server_url=args.server_url,
            results_commit=args.results_commit,
            upstream_label=args.upstream_label,
            upstream_identity=args.upstream_identity,
            upstream_url=args.upstream_url,
        )
    with Path(args.github_summary).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(rendered) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
