#!/usr/bin/env python3
"""Resolve the current Rank #1 approved detector calibration for document processing."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from hth.markdown_links import code_link, github_blob_url, github_release_url, github_tree_url, link

from hth.calibration_store import resolve_best_parameter_reference, load_index_with_persisted_backfill
from hth.domain.calibration import calibration_search_type, calibration_status
from hth.golden_set_catalog import canonical_release_for_golden_set, resolve_golden_set_for_source


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")


def _score(entry: dict[str, Any]) -> tuple[float, float, float, str]:
    selection = entry.get("selection") if isinstance(entry.get("selection"), dict) else {}
    avg = float(selection.get("best_avg_iou") or -1.0)
    minimum = float(selection.get("minimum_iou") or -1.0)
    stddev = float(selection.get("stddev_iou") or 999.0)
    created = str(entry.get("created_at_utc") or "")
    return (avg, minimum, -stddev, created)


def _approved(entry: dict[str, Any]) -> bool:
    """Apply the same authoritative/approval semantics used by Best Known.

    ``calibration_status=authoritative`` is already the persisted completeness
    contract.  Older index entries need not repeat ``exhaustive_complete=True``,
    and calibration evidence may be persisted either as a rating string or the
    richer evidence object used by calibration intelligence.
    """
    if calibration_status(entry) != "authoritative":
        return False
    if calibration_search_type(entry).replace("_", "-") not in {
        "exhaustive", "exhaustive-with-zombies", "cartesian"
    }:
        return False
    selection = entry.get("selection") if isinstance(entry.get("selection"), dict) else {}
    evidence = selection.get("calibration_evidence")
    if isinstance(evidence, dict):
        evidence = evidence.get("rating")
    return str(evidence or "").strip().lower() == "high"


def resolve_rank_one(index_path: Path, *, golden_set_id: str) -> dict[str, Any]:
    index = load_index_with_persisted_backfill(index_path)
    target = _slug(golden_set_id)
    candidates = [
        entry for entry in index.get("entries", [])
        if isinstance(entry, dict)
        and _slug(str(entry.get("golden_set_id") or "")) == target
        and _approved(entry)
        and int((entry.get("selection") or {}).get("failure_count") or 0) == 0
    ]
    if not candidates:
        raise SystemExit(f"No approved authoritative exhaustive calibration found for Golden Set {golden_set_id}")
    selected = max(candidates, key=_score)
    detector = str(selected.get("detector_id") or "").strip()
    golden_sha = str(selected.get("golden_set_sha256") or "").strip()
    reference = resolve_best_parameter_reference(
        index_path,
        detector=detector,
        golden_set_sha256=golden_sha,
        model_variant=selected.get("model_variant"),
    )
    if not reference:
        raise SystemExit(f"Rank #1 calibration for {detector} has no reconstructable exact parameter set")

    selection = selected.get("selection") if isinstance(selected.get("selection"), dict) else {}
    search = selected.get("search") if isinstance(selected.get("search"), dict) else {}
    build = selected.get("build") if isinstance(selected.get("build"), dict) else {}
    return {
        "rank": 1,
        "selection_policy": "best-approved-authoritative-calibration",
        "approval_level": "Approved",
        "detector": detector,
        "golden_set_id": golden_set_id,
        "golden_set_sha256": golden_sha,
        "parameter_set_id": reference.get("parameter_set_id"),
        "parameter_identity_sha256": reference.get("parameter_identity_sha256"),
        "parameters": reference["parameters"],
        "model_variant": selected.get("model_variant"),
        "best_avg_iou": selection.get("best_avg_iou"),
        "minimum_iou": selection.get("minimum_iou"),
        "stddev_iou": selection.get("stddev_iou"),
        "failure_count": selection.get("failure_count"),
        "calibration_evidence": selection.get("calibration_evidence"),
        "search_strategy": search.get("strategy"),
        "parameter_sets": search.get("parameter_sets"),
        "calibration_id": selected.get("calibration_id"),
        "build_number": build.get("github_run_number"),
        "build_url": build.get("run_url"),
        "created_at_utc": selected.get("created_at_utc"),
        "provenance_source": reference.get("provenance_source"),
        "record_path": reference.get("record_path") or selected.get("record_path"),
        "parameter_provenance_path": (
            reference.get("parameter_provenance_path")
            or selected.get("parameter_provenance_path")
        ),
        "needs_doc_ufcn": detector in {"amsre_doc_ufcn_fusion", "doc_ufcn_page_mask"},
    }


def _display_name(detector: str, catalog: Path | None) -> str:
    if catalog and catalog.is_file():
        payload = _read_json(catalog)
        entry = (payload.get("detectors") or {}).get(detector)
        if isinstance(entry, dict) and entry.get("display_name"):
            return str(entry["display_name"])
    return detector


def render_summary(
    resolved: dict[str, Any],
    *,
    display_name: str,
    pipeline_repository: str = "",
    pipeline_commit: str = "",
    results_repository: str = "",
    results_ref: str = "main",
    github_server_url: str = "https://github.com",
) -> str:
    params = resolved.get("parameters") or {}
    golden_set_url = github_release_url(
        str(resolved.get("golden_set_repository") or ""),
        str(resolved.get("golden_set_release_tag") or ""),
        server_url=github_server_url,
    )
    record_path = str(resolved.get("record_path") or "")
    provenance_path = str(resolved.get("parameter_provenance_path") or "")
    if not provenance_path and record_path and resolved.get("provenance_source") == "parameter-provenance":
        provenance_path = f"{record_path.rstrip('/')}/parameter-provenance.json"
    calibration_url = github_tree_url(
        results_repository,
        results_ref,
        record_path,
        server_url=github_server_url,
    )
    provenance_url = github_blob_url(
        results_repository,
        results_ref,
        provenance_path,
        server_url=github_server_url,
    )
    provenance_label = Path(provenance_path).name if provenance_path else resolved.get("provenance_source")
    detector = str(resolved["detector"])
    detector_doc_path = f"docs/detector-{detector.replace('_', '-')}.md"
    detector_doc_url = github_blob_url(
        pipeline_repository,
        pipeline_commit,
        detector_doc_path,
        server_url=github_server_url,
    ) if Path(detector_doc_path).is_file() else ""
    detector_label = (
        f"{link(display_name, detector_doc_url)} "
        f"({code_link(detector, detector_doc_url)})"
    )
    ranking_url = github_blob_url(
        results_repository,
        results_ref,
        "indexes/calibration-index.json",
        server_url=github_server_url,
    )
    lines = [
        "## Preferred document detector — Rank #1",
        "",
        "Production/test document inference automatically uses the strongest **Approved** authoritative calibration for the requested Golden Set.",
        "",
        f"- **Detector:** {detector_label}",
        f"- **Rank:** {link('#1', ranking_url)}",
        f"- **Approval:** `{resolved['approval_level']}` / evidence `{resolved.get('calibration_evidence')}`",
        f"- **Golden Set:** {code_link(resolved['golden_set_id'], golden_set_url)} (`{resolved['golden_set_sha256']}`)",
        f"- **Parameter Set ID:** `{resolved.get('parameter_set_id')}`",
        f"- **Absolute parameter SHA-256:** `{resolved.get('parameter_identity_sha256')}`",
        f"- **Best Avg IoU:** `{float(resolved['best_avg_iou']):.4f}`",
        f"- **Min IoU:** `{float(resolved['minimum_iou']):.4f}`",
        f"- **StdDev:** `{float(resolved['stddev_iou']):.4f}`",
        f"- **Failures:** `{resolved.get('failure_count')}`",
        f"- **Search:** `{resolved.get('search_strategy')}` / `{resolved.get('parameter_sets')}` parameter sets",
        f"- **Calibration build:** {code_link('#' + str(resolved.get('build_number')), str(resolved.get('build_url') or ''))}",
        f"- **Calibration ID:** {code_link(resolved.get('calibration_id'), calibration_url)}",
        f"- **Parameter provenance:** {code_link(provenance_label, provenance_url)}",
        "",
        "### Winning parameter specification",
        "",
        "| Parameter | Winning value |",
        "|---|---:|",
    ]
    for name, value in sorted(params.items()):
        lines.append(f"| `{name}` | `{json.dumps(value, ensure_ascii=False)}` |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--golden-set-id", default="")
    p.add_argument("--golden-set-freeze-root", type=Path)
    p.add_argument("--source-repository", default="")
    p.add_argument("--source-release-tag", default="")
    p.add_argument("--source-release-manifest-sha256", default="")
    p.add_argument("--catalog", type=Path)
    p.add_argument("--pipeline-repository", default="")
    p.add_argument("--pipeline-commit", default="")
    p.add_argument("--results-repository", default="")
    p.add_argument("--results-ref", default="main")
    p.add_argument("--github-server-url", default="https://github.com")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--github-output", type=Path)
    p.add_argument("--github-summary", type=Path)
    args = p.parse_args(argv)

    golden_set_id = str(args.golden_set_id or "").strip()
    if not golden_set_id:
        if args.golden_set_freeze_root is None:
            p.error(
                "--golden-set-id or --golden-set-freeze-root with immutable "
                "source-release provenance is required"
            )
        golden_set_id = resolve_golden_set_for_source(
            args.golden_set_freeze_root,
            repository_root=Path.cwd(),
            source_repository=args.source_repository,
            source_release_tag=args.source_release_tag,
            source_release_manifest_sha256=args.source_release_manifest_sha256,
        )

    resolved = resolve_rank_one(args.index, golden_set_id=golden_set_id)
    if args.golden_set_freeze_root is not None:
        canonical_release = canonical_release_for_golden_set(
            args.golden_set_freeze_root,
            golden_set_id=golden_set_id,
        )
        resolved["golden_set_repository"] = canonical_release.get("repository")
        resolved["golden_set_release_tag"] = canonical_release.get("tag")
    resolved["display_name"] = _display_name(resolved["detector"], args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = render_summary(
        resolved,
        display_name=resolved["display_name"],
        pipeline_repository=args.pipeline_repository,
        pipeline_commit=args.pipeline_commit,
        results_repository=args.results_repository,
        results_ref=args.results_ref,
        github_server_url=args.github_server_url,
    )
    print(summary, end="")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"detector={resolved['detector']}\n")
            handle.write(f"parameter_set_id={resolved.get('parameter_set_id')}\n")
            handle.write(f"golden_set_id={resolved.get('golden_set_id')}\n")
            handle.write(f"golden_set_repository={resolved.get('golden_set_repository') or ''}\n")
            handle.write(f"golden_set_release_tag={resolved.get('golden_set_release_tag') or ''}\n")
            handle.write(f"needs_doc_ufcn={'true' if resolved.get('needs_doc_ufcn') else 'false'}\n")
    if args.github_summary:
        with args.github_summary.open("a", encoding="utf-8") as handle:
            handle.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
