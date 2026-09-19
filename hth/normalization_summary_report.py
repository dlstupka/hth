#!/usr/bin/env python3
"""Render an audit-oriented summary of persisted normalization results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hth.canonical_build_evidence import SCOPE_EVIDENCE_PATHS, load_evidence_store
from hth.markdown_links import code_link, github_blob_url, github_commit_url
from hth.report_navigation import add_report_navigation


TRANSFORMATION_MANIFESTS = (
    ("Canonical crop/orientation", "normalization/normalization-manifest.json", "canonical_result_identity"),
    ("Photometric", "normalization/photometric-integration/photometric-normalization-manifest.json", "photometric_result_identity"),
    ("Tonal", "normalization/tonal-integration/tonal-normalization-manifest.json", "tonal_result_identity"),
    ("Chromatic", "normalization/chromatic-integration/chromatic-normalization-manifest.json", "chromatic_result_identity"),
    ("Denoising", "normalization/denoising-integration/denoising-normalization-manifest.json", "denoising_result_identity"),
    ("Sharpening", "normalization/sharpening-integration/sharpening-normalization-manifest.json", "sharpening_result_identity"),
    ("Binarization", "normalization/binarization-integration/binarization-normalization-manifest.json", "binarization_result_identity"),
)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return payload


def _value(payload: dict[str, Any], *paths: tuple[str, ...], default: Any = "unknown") -> Any:
    for path in paths:
        value: Any = payload
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            if value is not None and value != "":
                return value
    return default


def _method(payload: dict[str, Any]) -> str:
    value = payload.get("method", payload.get("policy", "unknown"))
    if isinstance(value, dict):
        return str(value.get("id") or value.get("name") or "none")
    return str(value if value not in (None, "") else "none")


def _link(repository: str, commit: str, relative: str, label: object) -> str:
    return code_link(label, github_blob_url(repository, commit, relative))


def _source_release(record: dict[str, Any]) -> str:
    source = (record.get("effective_inputs") or {}).get("source") or {}
    repository = str(source.get("repository") or "")
    release = str(source.get("release") or "")
    if repository and release:
        return f"{repository}@{release}"
    return release or repository or "unknown"


def generate_full_normalization_summary(
    results_root: Path,
    output: Path,
    *,
    results_repository: str = "",
    results_commit: str = "",
    pipeline_repository: str = "",
    pipeline_commit: str = "",
    run_url: str = "",
) -> Path:
    """Generate a deterministic report without rerunning normalization work."""
    results_root = Path(results_root)
    transformations: list[dict[str, Any]] = []
    missing_manifests: list[str] = []
    invalid_manifests: list[str] = []
    for name, relative, identity_key in TRANSFORMATION_MANIFESTS:
        path = results_root / relative
        if not path.is_file():
            missing_manifests.append(relative)
            transformations.append({"name": name, "path": relative, "missing": True})
            continue
        payload = _read_object(path, f"{name} normalization manifest")
        aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
        transform = payload.get("transform_summary") if isinstance(payload.get("transform_summary"), dict) else {}
        identity = payload.get(identity_key)
        status = payload.get("status", "complete")
        if status != "complete" or not identity:
            invalid_manifests.append(relative)
        transformations.append({
            "name": name,
            "path": relative,
            "missing": False,
            "status": status,
            "method": _method(payload),
            "pages": _value(payload, ("page_count",), ("aggregate", "page_count"), default="unknown"),
            "corrected": _value(payload, ("aggregate", "corrected_pages"), ("transform_summary", "pages_transformed"), default=0),
            "preserved": _value(payload, ("aggregate", "preserved_pages"), ("transform_summary", "pages_preserved"), default="unknown"),
            "identity": identity or "unknown",
        })

    stages: list[dict[str, Any]] = []
    missing_stages: list[str] = []
    for scope, relative in SCOPE_EVIDENCE_PATHS.items():
        if not relative.startswith("normalization/"):
            continue
        store = load_evidence_store(results_root / relative, scope=scope)
        identity = str(store.get("authoritative_identity") or "")
        record = (store.get("records") or {}).get(identity) if identity else None
        if not isinstance(record, dict):
            missing_stages.append(scope)
            stages.append({"scope": scope, "path": relative, "missing": True})
            continue
        result = record.get("canonical_result") or {}
        stages.append({
            "scope": scope,
            "path": relative,
            "missing": False,
            "build": identity,
            "result": result.get("identity", "unknown"),
            "pages": len(result.get("pages") or []),
            "activity": (record.get("execution") or {}).get("activity", "unknown"),
            "source": _source_release(record),
        })

    lifecycle_path = results_root / "metadata/resource-lifecycle.json"
    lifecycle = _read_object(lifecycle_path, "resource lifecycle report") if lifecycle_path.is_file() else None
    lifecycle_valid = bool(
        lifecycle
        and lifecycle.get("report_type") == "canonical-resource-lifecycle"
        and lifecycle.get("resource_state_identity")
        and isinstance(lifecycle.get("summary"), dict)
    )
    complete = not missing_manifests and not invalid_manifests and not missing_stages and lifecycle_valid
    state = "COMPLETE" if complete else "INCOMPLETE"

    commit_label = results_commit or "unavailable"
    lines = [
        "# Full normalization summary",
        "",
        "> Audit report generated exclusively from persisted Canonical Build Evidence, normalization manifests, and the resource-lifecycle ledger. No normalization pixels or domain results were recomputed.",
        "",
        "<details open>",
        "<summary><h2>Audit status</h2></summary>",
        "",
        f"- Status: **{state}**",
        f"- Authoritative normalization CBE stages: `{len(stages) - len(missing_stages)}/{len(stages)}`",
        f"- Durable transformation manifests: `{len(transformations) - len(missing_manifests) - len(invalid_manifests)}/{len(transformations)}` valid",
        f"- Resource lifecycle ledger: `{'valid' if lifecycle_valid else ('invalid' if lifecycle is not None else 'missing')}`",
        f"- Results snapshot: {code_link(commit_label, github_commit_url(results_repository, results_commit))}",
    ]
    if run_url:
        lines.append(f"- Report Writer run: [workflow run]({run_url})")
    if missing_stages or missing_manifests or invalid_manifests or not lifecycle_valid:
        lines.extend(["", "> **Incomplete evidence:** this report identifies absent durable inputs rather than inferring or rebuilding them."])
    lines.extend(["", "</details>"])

    lines.extend([
        "",
        "<details open>",
        "<summary><h2>Normalization result chain</h2></summary>",
        "",
        "| Stage | Status | Method / policy | Pages | Corrected | Preserved | Result identity |",
        "|---|---:|---|---:|---:|---:|---|",
    ])
    for row in transformations:
        if row["missing"]:
            lines.append(f"| {row['name']} | **MISSING** | — | — | — | — | {_link(results_repository, results_commit, row['path'], 'missing manifest')} |")
        else:
            lines.append(
                f"| {row['name']} | `{row['status']}` | `{row['method']}` | `{row['pages']}` | "
                f"`{row['corrected']}` | `{row['preserved']}` | {_link(results_repository, results_commit, row['path'], row['identity'])} |"
            )
    lines.extend(["", "</details>"])

    lines.extend([
        "",
        "<details>",
        "<summary><h2>Canonical Build Evidence ledger</h2></summary>",
        "",
        "| Stage scope | State | Activity | Pages | Effective build | Canonical result | Source release utilized | Evidence |",
        "|---|---:|---|---:|---|---|---|---|",
    ])
    for row in stages:
        evidence = _link(results_repository, results_commit, row["path"], row["path"])
        if row["missing"]:
            lines.append(f"| `{row['scope']}` | **MISSING** | — | — | — | — | — | {evidence} |")
        else:
            lines.append(
                f"| `{row['scope']}` | `authoritative` | `{row['activity']}` | `{row['pages']}` | "
                f"`{row['build']}` | `{row['result']}` | `{row['source']}` | {evidence} |"
            )
    lines.extend(["", "</details>", ""])

    lines.extend([
        "<details open>",
        "<summary><h2>Cache and release lifecycle</h2></summary>",
        "",
    ])
    cleanup: list[dict[str, Any]] = []
    if not lifecycle_valid:
        ledger_state = "invalid" if lifecycle is not None else "missing"
        lines.append(f"- Ledger: {_link(results_repository, results_commit, 'metadata/resource-lifecycle.json', ledger_state)}")
    else:
        summary = lifecycle.get("summary") or {}
        lines.extend([
            f"- Resource state identity: {_link(results_repository, results_commit, 'metadata/resource-lifecycle.json', lifecycle.get('resource_state_identity', 'unknown'))}",
            f"- Build records: `{summary.get('build_records', 0)}`",
            f"- Cache elements: `{summary.get('cache_elements', 0)}` total; `{summary.get('dirty_cache_elements', 0)}` dirty; `{summary.get('cleanup_eligible_cache_elements', 0)}` cleanup-eligible",
            f"- Release elements: `{summary.get('release_elements', 0)}` total; `{summary.get('dirty_release_elements', 0)}` dirty; `{summary.get('cleanup_eligible_release_elements', 0)}` cleanup-eligible",
            "- Cleanup action: `none` — this is a provenance and eligibility report, not a deletion job.",
        ])
        cleanup = [row for row in lifecycle.get("release_elements", []) if row.get("cleanup_eligible")]
        if cleanup:
            lines.extend([
                "",
                "<details>",
                f"<summary><h3>Cleanup-eligible releases ({len(cleanup)})</h3></summary>",
                "",
                "| Release | Kind | Integrity | Lineage | Publication | Reason |",
                "|---|---|---|---|---|---|",
            ])
            for row in cleanup:
                lines.append(
                    f"| `{row.get('repository', 'unknown')}@{row.get('release', 'unknown')}` | `{row.get('release_kind', 'unknown')}` | "
                    f"`{row.get('integrity', 'unknown')}` | `{row.get('lineage', 'unknown')}` | `{row.get('publication', 'unknown')}` | "
                    f"`{row.get('cleanup_reason', 'unknown')}` |"
                )
            lines.extend(["", "</details>"])
    lines.extend(["", "</details>"])

    recommendations: list[str] = []
    if missing_stages or missing_manifests or invalid_manifests or not lifecycle_valid:
        recommendations.append(
            "Restore or regenerate only the missing durable evidence identified above, then rerun Report Writer; do not infer completion from workflow success alone."
        )
    if cleanup:
        recommendations.append(
            f"Review the `{len(cleanup)}` cleanup-eligible release(s) against retention policy before deletion; eligibility is recorded, but this report intentionally performs no cleanup."
        )
    applied = [row for row in transformations if not row["missing"] and int(row.get("corrected") or 0) > 0]
    if complete and not applied:
        recommendations.append(
            "Treat the fully preserved normalization chain as an intentional audited no-op and reuse its authoritative CBE identities on equivalent inputs; rebuild only when an effective input changes or force-verification is explicitly required."
        )
    if recommendations:
        lines.extend([
            "",
            "<details open>",
            "<summary><h2>Engineering recommendations</h2></summary>",
            "",
        ])
        lines.extend(f"- {recommendation}" for recommendation in recommendations)
        lines.extend(["", "</details>"])
    normalization_guide = github_blob_url(
        pipeline_repository,
        pipeline_commit,
        "docs/normalization.md",
    )
    lines.extend([
        "",
        "<details>",
        "<summary><h2>Engineering reference</h2></summary>",
        "",
        f"- [Normalization design, provenance, policy, and operating guidance]({normalization_guide})"
        if normalization_guide
        else "- Normalization design, provenance, policy, and operating guidance: `docs/normalization.md`",
        "",
        "</details>",
    ])

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(add_report_navigation(lines)).rstrip() + "\n", encoding="utf-8")
    return output
