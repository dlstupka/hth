#!/usr/bin/env python3
"""Resolve canonical frozen Golden Sets from operator-facing release identity."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from hth.validate_golden_set_freeze import validate_freeze


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _repository_slug(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^(?:https?://github\.com/|git@github\.com:)", "", text)
    return text.removesuffix(".git").strip("/")


def _frozen_manifests(freeze_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(Path(freeze_root).glob("*.freeze.json")):
        payload = _read_json(path)
        if str(payload.get("state") or "").strip().lower() == "frozen":
            manifests.append((path, payload))
    return manifests


def resolve_golden_set_for_source(
    freeze_root: Path,
    *,
    repository_root: Path,
    source_repository: str,
    source_release_tag: str,
    source_release_manifest_sha256: str,
) -> str:
    """Resolve the frozen Golden Set tied to one immutable source release."""
    repository = _repository_slug(source_repository)
    tag = str(source_release_tag or "").strip()
    manifest_sha = str(source_release_manifest_sha256 or "").strip().lower()
    if not repository or not tag or not manifest_sha:
        raise SystemExit(
            "Source repository, release tag, and release-manifest SHA-256 are "
            "required to resolve the compatible Golden Set"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha):
        raise SystemExit("Source release manifest SHA-256 must be 64 hexadecimal characters")

    matches: list[tuple[str, Path]] = []
    for path, payload in _frozen_manifests(freeze_root):
        source = payload.get("source_release")
        source = source if isinstance(source, dict) else {}
        if (
            _repository_slug(str(source.get("repository") or "")) == repository
            and str(source.get("tag") or "").strip() == tag
            and str(source.get("manifest_sha256") or "").strip().lower() == manifest_sha
        ):
            golden_set_id = str(payload.get("golden_set_id") or "").strip()
            if golden_set_id:
                try:
                    validate_freeze(
                        freeze_path=path,
                        repository_root=Path(repository_root).resolve(),
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    raise SystemExit(
                        f"Frozen Golden Set mapping is invalid: {path}: {exc}"
                    ) from exc
                matches.append((golden_set_id, path))

    if not matches:
        raise SystemExit(
            "No frozen Golden Set matches source release "
            f"{repository}@{tag} ({manifest_sha})"
        )
    if len(matches) != 1:
        evidence = ", ".join(f"{golden_id} ({path})" for golden_id, path in matches)
        raise SystemExit(
            "Source release maps to multiple frozen Golden Sets; refusing an "
            f"ambiguous production detector selection: {evidence}"
        )
    return matches[0][0]


def canonical_release_for_golden_set(
    freeze_root: Path,
    *,
    golden_set_id: str,
) -> dict[str, str]:
    """Return the persisted canonical release identity for one Golden Set."""
    target = str(golden_set_id or "").strip()
    matches: list[dict[str, str]] = []
    for _path, payload in _frozen_manifests(freeze_root):
        if str(payload.get("golden_set_id") or "").strip() != target:
            continue
        release = payload.get("canonical_release")
        release = release if isinstance(release, dict) else {}
        repository = str(release.get("repository") or "").strip()
        tag = str(release.get("tag") or "").strip()
        if repository and tag:
            matches.append({"repository": repository, "tag": tag})
    if len(matches) > 1:
        raise SystemExit(f"Golden Set ID {target} has multiple canonical releases")
    return matches[0] if matches else {}


def resolve_golden_set_release(
    freeze_root: Path,
    *,
    repository_root: Path,
    release_tag: str,
) -> dict[str, str]:
    """Resolve and validate one canonical Golden Set release tag."""
    tag = str(release_tag or "").strip()
    if not tag:
        raise SystemExit("Golden Set release tag is required")

    matches: list[tuple[Path, dict[str, Any]]] = []
    for path, payload in _frozen_manifests(freeze_root):
        release = payload.get("canonical_release")
        release = release if isinstance(release, dict) else {}
        if str(release.get("tag") or "").strip() == tag:
            matches.append((path, payload))

    if not matches:
        raise SystemExit(f"No frozen Golden Set matches release tag {tag}")
    if len(matches) != 1:
        evidence = ", ".join(str(path) for path, _ in matches)
        raise SystemExit(f"Golden Set release tag {tag} is ambiguous: {evidence}")

    freeze_path, payload = matches[0]
    repository_root_input = Path(repository_root)
    repository_root_resolved = repository_root_input.resolve()
    relative = Path(str(payload.get("golden_set_path") or ""))
    golden_set_path = (repository_root_resolved / relative).resolve()
    try:
        golden_set_path.relative_to(repository_root_resolved)
    except ValueError as exc:
        raise SystemExit(f"Frozen Golden Set path escapes repository: {relative}") from exc
    validate_freeze(freeze_path=freeze_path, repository_root=repository_root_resolved)
    return {
        "golden_set_id": str(payload.get("golden_set_id") or ""),
        "golden_set_path": (repository_root_input / relative).as_posix(),
        "freeze_path": freeze_path.as_posix(),
        "release_tag": tag,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)

    resolved = resolve_golden_set_release(
        args.freeze_root,
        repository_root=args.repository_root,
        release_tag=args.release_tag,
    )
    print(json.dumps(resolved, sort_keys=True))
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as handle:
            for name, value in resolved.items():
                handle.write(f"{name}={value}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
