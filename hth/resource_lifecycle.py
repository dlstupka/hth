#!/usr/bin/env python3
"""Audit CBE cache and immutable-release liveness without deleting anything."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from hth.canonical_build_evidence import SCOPE_EVIDENCE_PATHS, canonical_hash, load_evidence_store


SCHEMA_VERSION = "1.0"
REPORT_TYPE = "canonical-resource-lifecycle"


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _release_key(repository: str, release: str) -> str:
    return f"{repository}@{release}"


def _register_release(
    releases: dict[str, dict[str, Any]],
    *,
    repository: str,
    release: str,
    manifest_sha256: str | None,
) -> dict[str, Any]:
    key = _release_key(repository, release)
    item = releases.setdefault(key, {
        "kind": "immutable-release",
        "repository": repository,
        "release": release,
        "release_manifest_sha256": manifest_sha256,
        "active_consumers": [],
        "historical_consumers": [],
        "authoritative_producers": [],
    })
    existing = str(item.get("release_manifest_sha256") or "")
    incoming = str(manifest_sha256 or "")
    if existing and incoming and existing != incoming:
        raise ValueError(f"Immutable release identity collision for {key}: {existing} != {incoming}")
    if not existing and incoming:
        item["release_manifest_sha256"] = incoming
    return item


def _release_inventory(path: Path | None, default_repository: str) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _read_json(path)
    rows = payload.get("releases") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Release inventory must be a list or contain a releases list")
    inventory: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Release inventory contains a non-object entry")
        repository = str(row.get("repository") or default_repository).strip()
        release = str(row.get("release") or row.get("tag") or row.get("tagName") or "").strip()
        if not repository or not release:
            raise ValueError("Release inventory entry has no repository or release tag")
        inventory[_release_key(repository, release)] = {**row, "repository": repository, "release": release}
    return inventory


def _declared_mirror_tags(repository_root: Path | None) -> set[str]:
    if repository_root is None:
        return set()
    tags: set[str] = set()
    source_root = Path(repository_root) / "hth"
    if not source_root.is_dir():
        return tags
    pattern = re.compile(r"HTH-MIRROR-[A-Z0-9][A-Z0-9-]*")
    for path in sorted(source_root.rglob("*.py")):
        tags.update(pattern.findall(path.read_text(encoding="utf-8")))
    return tags


def _release_family(item: dict[str, Any]) -> str | None:
    if item.get("release_kind") in {"mirror", "learned-evidence-cache"}:
        return None
    match = re.match(
        r"^(HTH-(?:PHOTOMETRIC|TONAL|CHROMATIC|DENOISING|SHARPENING|BINARIZATION))-",
        str(item.get("release") or ""),
    )
    return match.group(1) if match else None


def _integrity_label(item: dict[str, Any], inventory_row: dict[str, Any] | None) -> str:
    declared = str(
        (inventory_row or {}).get("integrity")
        or (inventory_row or {}).get("validationStatus")
        or ""
    ).strip().lower()
    if declared in {"bad", "corrupt", "failed", "invalid", "rejected"}:
        return "bad"
    if declared in {"good", "valid", "verified"}:
        return "good"
    digest = str(item.get("release_manifest_sha256") or "")
    return "good" if len(digest) == 64 else "unknown"


def build_report(
    results_root: Path,
    *,
    release_inventory: Path | None = None,
    default_release_repository: str = "",
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Build a conservative liveness report from every registered CBE store."""
    results_root = Path(results_root)
    builds: list[dict[str, Any]] = []
    cache_elements: list[dict[str, Any]] = []
    release_uses: dict[str, dict[str, Any]] = {}

    for scope, relative in sorted(SCOPE_EVIDENCE_PATHS.items()):
        store_path = results_root / relative
        if not store_path.is_file():
            continue
        store = load_evidence_store(store_path, scope=scope)
        authoritative = str(store.get("authoritative_identity") or "")
        for identity, record in sorted(store["records"].items()):
            active = identity == authoritative
            result_identity = str((record.get("canonical_result") or {}).get("identity") or "")
            builds.append({
                "scope": scope,
                "effective_build_identity": identity,
                "canonical_result_identity": result_identity,
                "authoritative": active,
                "evidence_path": relative,
                "resource_utilization": record.get("resource_utilization"),
            })
            cache_lineage = "current" if active else "previous"
            cache_elements.append({
                "kind": "canonical-build-evidence-cache",
                "scope": scope,
                "identity": identity,
                "canonical_result_identity": result_identity,
                "path": relative,
                "integrity": "good",
                "lineage": cache_lineage,
                "dirty": not active,
                "cleanup_eligible": False,
                "cleanup_reason": (
                    "authoritative-build-evidence"
                    if active
                    else "historical-audit-provenance-retained"
                ),
                "labels": ["good", cache_lineage, "dirty" if not active else "clean"],
            })

            source = (record.get("effective_inputs") or {}).get("source") or {}
            repository = str(source.get("repository") or "").strip()
            release = str(source.get("release") or "").strip()
            if repository and release:
                item = _register_release(
                    release_uses,
                    repository=repository,
                    release=release,
                    manifest_sha256=source.get("release_manifest_sha256"),
                )
                consumer = {"scope": scope, "effective_build_identity": identity}
                item["active_consumers" if active else "historical_consumers"].append(consumer)

        if authoritative:
            record = store["records"].get(authoritative) or {}
            for artifact in (record.get("canonical_result") or {}).get("artifacts") or []:
                if not isinstance(artifact, dict) or artifact.get("logical_name") != "release-record":
                    continue
                release_path = results_root / str(artifact.get("published_path") or "")
                if not release_path.is_file():
                    continue
                release_record = _read_json(release_path)
                release = str(release_record.get("tag") or "").strip()
                repository = default_release_repository.strip()
                if not release or not repository:
                    continue
                item = _register_release(
                    release_uses,
                    repository=repository,
                    release=release,
                    manifest_sha256=release_record.get("asset_sha256"),
                )
                item["authoritative_producers"].append({
                    "scope": scope,
                    "effective_build_identity": authoritative,
                    "canonical_result_identity": (record.get("canonical_result") or {}).get("identity"),
                })

    inventory = _release_inventory(release_inventory, default_release_repository)
    declared_mirrors = _declared_mirror_tags(repository_root)
    for key, row in inventory.items():
        _register_release(
            release_uses,
            repository=row["repository"],
            release=row["release"],
            manifest_sha256=None,
        )

    for item in release_uses.values():
        if item["release"].startswith("HTH-MIRROR-") or item["repository"].endswith("/hth-mirror"):
            item["release_kind"] = "mirror"
        elif item["release"].startswith("HTH-EVIDENCE-") or item["repository"].endswith("-cache"):
            item["release_kind"] = "learned-evidence-cache"
        else:
            item["release_kind"] = "regular"
        item["declared_by_current_code"] = (
            item["release_kind"] == "mirror" and item["release"] in declared_mirrors
        )

    current_families = {
        family
        for item in release_uses.values()
        if (item["active_consumers"] or item["authoritative_producers"] or item["declared_by_current_code"])
        for family in [_release_family(item)]
        if family
    }

    release_elements = []
    for key, item in sorted(release_uses.items()):
        active = bool(
            item["active_consumers"]
            or item["authoritative_producers"]
            or item["declared_by_current_code"]
        )
        historical = bool(item["historical_consumers"])
        inventoried = key in inventory
        inventory_row = inventory.get(key)
        family = _release_family(item)
        if active:
            lineage = "current"
        elif historical:
            lineage = "previous"
        elif family and family in current_families:
            lineage = "superseded"
        else:
            lineage = "unreferenced"
        integrity = _integrity_label(item, inventory_row)
        publication = "latest" if bool((inventory_row or {}).get("isLatest")) else "not-latest"
        dirty = integrity != "good" or lineage != "current" or bool((inventory_row or {}).get("isDraft"))
        cleanup_authority = item["release_kind"] != "learned-evidence-cache"
        cleanup_eligible = (
            cleanup_authority
            and inventoried
            and lineage in {"superseded", "unreferenced"}
            and not historical
        )
        if active:
            reason = "referenced-by-authoritative-build"
        elif historical:
            reason = "historical-audit-provenance-reference"
        elif not cleanup_authority:
            reason = "learned-evidence-utilization-ledger-required"
        elif lineage == "superseded":
            reason = "superseded-unreferenced-inventory-release"
        elif inventoried:
            reason = "unreferenced-inventory-release"
        else:
            reason = "referenced-release-not-present-in-inventory"
        release_elements.append({
            **item,
            "inventoried": inventoried,
            "integrity": integrity,
            "lineage": lineage,
            "publication": publication,
            "dirty": dirty,
            "cleanup_authority": cleanup_authority,
            "cleanup_eligible": cleanup_eligible,
            "cleanup_reason": reason,
            "labels": [integrity, lineage, publication, "dirty" if dirty else "clean"],
        })

    canonical_state = {
        "builds": builds,
        "cache_elements": cache_elements,
        "release_elements": release_elements,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "resource_state_identity": canonical_hash(canonical_state),
        "summary": {
            "build_records": len(builds),
            "cache_elements": len(cache_elements),
            "dirty_cache_elements": sum(1 for item in cache_elements if item["dirty"]),
            "cleanup_eligible_cache_elements": sum(1 for item in cache_elements if item["cleanup_eligible"]),
            "release_elements": len(release_elements),
            "dirty_release_elements": sum(1 for item in release_elements if item["dirty"]),
            "cleanup_eligible_release_elements": sum(1 for item in release_elements if item["cleanup_eligible"]),
        },
        **canonical_state,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_summary(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    summary = report["summary"]
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(
            "\n### Cache and release lifecycle\n\n"
            f"- Build records: `{summary['build_records']}`\n"
            f"- Dirty cache elements: `{summary['dirty_cache_elements']}`\n"
            f"- Cleanup-eligible cache elements: `{summary['cleanup_eligible_cache_elements']}`\n"
            f"- Dirty release elements: `{summary['dirty_release_elements']}`\n"
            f"- Cleanup-eligible release elements: `{summary['cleanup_eligible_release_elements']}`\n"
            "- Cleanup action: `none` (eligibility report only)\n"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--release-inventory", type=Path)
    parser.add_argument("--default-release-repository", default="")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-summary", type=Path)
    args = parser.parse_args(argv)
    report = build_report(
        args.results_root,
        release_inventory=args.release_inventory,
        default_release_repository=args.default_release_repository,
        repository_root=args.repository_root,
    )
    _write(args.output, report)
    _append_summary(args.github_summary, report)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
