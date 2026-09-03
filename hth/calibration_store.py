#!/usr/bin/env python3
"""Publish and resolve durable HTH calibration intelligence records."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from hth.persistence import (
    canonical_index_path, readable_index_path, index_results_root, resolve_index_relative_path,
    read_json as _read_json, atomic_write_json as _write_json, load_index, write_index,
)
from typing import Any

from hth.runtime_store import observation_from_run, update_runtime_index
from hth.parallelism_store import observation_from_run as parallelism_observation_from_run, update_parallelism_index
from hth.shape_prediction import record_prediction_observations
from hth.persistence import canonical_index_path
from hth.regression.parameter_provenance import provenance_from_legacy_parameters, resolve_parameter_set
from hth.regression.run_semantics import legacy_run_semantics

from hth.contracts import CALIBRATION_INDEX_SCHEMA_VERSION, adapt_calibration_index
from hth.domain.calibration import authoritative_record

INDEX_SCHEMA_VERSION = CALIBRATION_INDEX_SCHEMA_VERSION
STATUS_PRIORITY = {"provisional": 1, "partial": 2, "authoritative": 3}
PERSISTED_FILES = (
    "manifest.json",
    "parameters.json",
    "parameter-provenance.json",
    "RUN-INFO.json",
    "raw/results.csv",
    "raw/evidence.jsonl",
    "reports/summary.json",
    "reports/winner-pages.json",
    "reports/calibration-intelligence.json",
)
COMPRESSED_RAW_FILES = frozenset({"raw/results.csv", "raw/evidence.jsonl"})
MAX_GIT_BLOB_BYTES = 95 * 1024 * 1024


def _copy_persisted_file(
    source: Path,
    target: Path,
    *,
    compress: bool = False,
    max_bytes: int | None = None,
) -> tuple[Path | None, int]:
    """Copy one record file, using deterministic gzip for large raw evidence."""
    if not compress:
        shutil.copy2(source, target)
        return target, target.stat().st_size
    compressed = target.with_name(f"{target.name}.gz")
    with source.open("rb") as input_handle, compressed.open("wb") as output_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=output_handle, compresslevel=6, mtime=0
        ) as gzip_handle:
            shutil.copyfileobj(input_handle, gzip_handle, length=1024 * 1024)
    compressed_bytes = compressed.stat().st_size
    if max_bytes is not None and compressed_bytes > max_bytes:
        compressed.unlink()
        return None, compressed_bytes
    return compressed, compressed_bytes



def _slug(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or fallback


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_document_id(intelligence: dict[str, Any], fallback: str) -> str:
    identity = intelligence.get("calibration_identity", {})
    source = identity.get("source_document") if isinstance(identity, dict) else None
    if isinstance(source, dict):
        for key in ("id", "collection_id", "source_document_id", "repository", "title"):
            if source.get(key):
                return _slug(source[key], fallback)
    return _slug(fallback, "source-document")


def _golden_set_id(intelligence: dict[str, Any]) -> str:
    identity = intelligence.get("calibration_identity", {})
    golden = identity.get("golden_set") if isinstance(identity, dict) else None
    if isinstance(golden, dict):
        for key in ("id", "collection_id", "name", "configuration"):
            if golden.get(key):
                return _slug(golden[key], "golden-set")
    return "golden-set"


def _compatibility(intelligence: dict[str, Any]) -> dict[str, Any]:
    identity = intelligence.get("calibration_identity", {})
    golden = identity.get("golden_set", {}) if isinstance(identity, dict) else {}
    detector_config = identity.get("detector_configuration", {}) if isinstance(identity, dict) else {}
    parameter = intelligence.get("parameter_intelligence", {})
    model_selection = identity.get("model_selection") if isinstance(identity, dict) else None
    model_variant = model_selection.get("variant") if isinstance(model_selection, dict) else None
    compatibility_model_variant = None if str(model_variant or "").endswith("_current") else model_variant
    return {
        "source_document": identity.get("source_document") if isinstance(identity, dict) else None,
        "golden_set_id": golden.get("collection_id") or golden.get("id") or golden.get("configuration"),
        "golden_set_sha256": golden.get("sha256"),
        "detector_id": intelligence.get("detector"),
        "detector_config_sha256": detector_config.get("sha256"),
        "model_variant": compatibility_model_variant,
        "effect_size_policy": parameter.get("classification_thresholds") if isinstance(parameter, dict) else None,
        "calibration_schema_version": identity.get("calibration_schema_version") if isinstance(identity, dict) else None,
        "intelligence_schema_version": intelligence.get("schema_version"),
    }



def _entry_from_persisted_intelligence(results_root: Path, intelligence_path: Path) -> dict[str, Any] | None:
    """Reconstruct one calibration-index row from durable per-run evidence."""
    try:
        intelligence = _read_json(intelligence_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not intelligence.get("available"):
        return None

    identity = intelligence.get("calibration_identity")
    identity = identity if isinstance(identity, dict) else {}
    calibration_id = str(identity.get("calibration_run_id") or intelligence_path.parent.name)
    detector = str(intelligence.get("detector") or "").strip()
    if not detector:
        return None

    try:
        relative_dir = intelligence_path.parent.relative_to(results_root)
    except ValueError:
        return None

    compatibility = _compatibility(intelligence)
    compatibility_key = _canonical_hash(compatibility)
    golden = identity.get("golden_set") if isinstance(identity.get("golden_set"), dict) else {}
    selection = intelligence.get("detector_selection_intelligence")
    selection = selection if isinstance(selection, dict) else {}
    search = intelligence.get("search")
    search = search if isinstance(search, dict) else {}
    build = identity.get("build") if isinstance(identity.get("build"), dict) else {}
    persistence = intelligence.get("persistence") if isinstance(intelligence.get("persistence"), dict) else {}
    run_mode, evidence_tier = legacy_run_semantics(intelligence)

    provenance = relative_dir / "parameter-provenance.json"
    return {
        "calibration_id": calibration_id,
        "run_mode": run_mode,
        "evidence_tier": evidence_tier,
        "calibration_status": evidence_tier,
        "record_path": relative_dir.as_posix(),
        "intelligence_path": (relative_dir / "calibration-intelligence.json").as_posix(),
        "parameter_provenance_path": provenance.as_posix() if (results_root / provenance).is_file() else None,
        "source_document_id": _source_document_id(intelligence, "source-document"),
        "golden_set_id": _golden_set_id(intelligence),
        "golden_set_sha256": str(golden.get("sha256") or ""),
        "detector_id": detector,
        "detector_config_sha256": compatibility.get("detector_config_sha256"),
        "model_variant": ((identity.get("model_selection") or {}).get("variant") if isinstance(identity.get("model_selection"), dict) else None),
        "compatibility_key": compatibility_key,
        "created_at_utc": identity.get("created_at_utc"),
        "published_at_utc": persistence.get("published_at_utc"),
        "build": dict(build),
        "search": {
            "strategy": search.get("strategy"),
            "parameter_sets": search.get("parameter_sets"),
            "possible_parameter_sets": search.get("possible_parameter_sets"),
            "exhaustive_complete": search.get("exhaustive_complete"),
        },
        "selection": {
            "recommended_parameter_set_id": selection.get("recommended_parameter_set_id"),
            "best_avg_iou": selection.get("best_avg_iou"),
            "minimum_iou": selection.get("minimum_iou"),
            "stddev_iou": selection.get("stddev_iou"),
            "failure_count": selection.get("failure_count"),
            "calibration_evidence": selection.get("calibration_evidence"),
        },
    }


def load_index_with_persisted_backfill(index_path: Path) -> dict[str, Any]:
    """Load calibration index and recover any durable calibrations omitted by it.

    The results index is a cache/discovery structure; per-run calibration evidence
    under source-documents/ is the durable source of truth.  This protects report
    generation and production Rank resolution from an incomplete index migration.
    """
    index_path = Path(index_path)
    results_root = index_results_root(index_path)
    index = load_index(results_root, "calibration-index.json")

    current = index.get("entries") if isinstance(index.get("entries"), list) else []
    def cache_key(item: dict[str, Any]) -> tuple[str, str]:
        compatibility = str(item.get("compatibility_key") or "").strip()
        calibration = str(item.get("calibration_id") or "").strip()
        if compatibility or calibration:
            return compatibility, calibration
        # Preserve older/minimal index rows that predate compatibility identity.
        # record_path is unique per persisted calibration run.
        return "__record_path__", str(item.get("record_path") or id(item))

    by_identity = {
        cache_key(item): item
        for item in current if isinstance(item, dict)
    }

    for intelligence_path in results_root.glob(
        "source-documents/*/golden-sets/*/*/calibrations/*/*/calibration-intelligence.json"
    ):
        recovered = _entry_from_persisted_intelligence(results_root, intelligence_path)
        if not recovered:
            continue
        key = cache_key(recovered)
        existing = by_identity.get(key)
        # Prefer the index row when present because it may carry later schema
        # adaptations; otherwise recover the durable record.
        if existing is None:
            by_identity[key] = recovered

    merged = sorted(
        by_identity.values(),
        key=lambda item: (
            str(item.get("source_document_id") or ""),
            str(item.get("golden_set_id") or ""),
            str(item.get("detector_id") or ""),
            str(item.get("created_at_utc") or ""),
        ),
    )
    index["entries"] = merged
    return index


def publish_run(
    run_dir: Path,
    results_root: Path,
    *,
    mode: str,
    source_fallback: str,
    build: dict[str, Any],
    max_git_blob_bytes: int = MAX_GIT_BLOB_BYTES,
) -> dict[str, Any]:
    intelligence_path = run_dir / "reports" / "calibration-intelligence.json"
    intelligence = _read_json(intelligence_path)
    measurement_state = intelligence.get("measurement_state")
    if isinstance(measurement_state, dict) and measurement_state.get("status") == "no_valid_measurements":
        raise ValueError(f"Calibration has no valid measurements and cannot be persisted: {intelligence_path}")
    if not intelligence.get("available"):
        raise ValueError(f"Calibration intelligence is unavailable in {intelligence_path}")

    identity = intelligence.setdefault("calibration_identity", {})
    calibration_id = str(identity.get("calibration_run_id") or run_dir.name)
    identity.setdefault("calibration_run_id", calibration_id)
    summary_path = run_dir / "reports" / "summary.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    info_path = run_dir / "RUN-INFO.json"
    info = _read_json(info_path) if info_path.is_file() else {}
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    run_mode, evidence_tier = legacy_run_semantics(
        info, summary, intelligence, manifest, fallback_mode=mode
    )
    if run_mode != mode:
        raise ValueError(
            f"Persistence mode {mode!r} does not match run mode {run_mode!r}: {run_dir}"
        )
    persisted_build = dict(build)
    run_time_seconds = info.get("elapsed_seconds", summary.get("elapsed_seconds"))
    if run_time_seconds is not None:
        persisted_build["run_time_seconds"] = run_time_seconds
    identity["build"] = persisted_build
    persisted_build["mode"] = run_mode
    persisted_build["evidence_tier"] = evidence_tier
    intelligence["run_mode"] = run_mode
    intelligence["evidence_tier"] = evidence_tier
    intelligence["calibration_status"] = evidence_tier
    intelligence["persistence"] = {
        "store": "results-repository",
        "index": "indexes/calibration-index.json",
        "published_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "raw_evidence": {
            "encoding": "gzip",
            "results": "raw/results.csv.gz",
            "evidence": "raw/evidence.jsonl.gz",
        },
    }

    source_id = _source_document_id(intelligence, source_fallback)
    golden_id = _golden_set_id(intelligence)
    golden = identity.get("golden_set", {}) if isinstance(identity, dict) else {}
    golden_sha = str(golden.get("sha256") or "unknown")
    detector = str(intelligence.get("detector") or "unknown")
    relative_dir = Path("source-documents") / source_id / "golden-sets" / golden_id / golden_sha[:12] / "calibrations" / detector / calibration_id
    destination = results_root / relative_dir
    destination.mkdir(parents=True, exist_ok=True)

    intelligence["persistence"]["record_path"] = relative_dir.as_posix()
    _write_json(intelligence_path, intelligence)

    raw_omitted: dict[str, dict[str, Any]] = {}
    for relative in PERSISTED_FILES:
        source = run_dir / relative
        if source.is_file():
            target = (
                destination / relative
                if Path(relative).parent != Path(".")
                and not str(relative).startswith("reports/")
                else destination / Path(relative).name
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            is_raw = relative in COMPRESSED_RAW_FILES
            copied, compressed_bytes = _copy_persisted_file(
                source,
                target,
                compress=is_raw,
                max_bytes=max_git_blob_bytes if is_raw else None,
            )
            if is_raw and copied is None:
                key = "results" if relative.endswith("results.csv") else "evidence"
                intelligence["persistence"]["raw_evidence"][key] = None
                raw_omitted[key] = {
                    "reason": "compressed_blob_exceeds_git_limit",
                    "source_bytes": source.stat().st_size,
                    "compressed_bytes": compressed_bytes,
                    "maximum_git_blob_bytes": max_git_blob_bytes,
                    "retained_in": "workflow regression artifact",
                }

    if raw_omitted:
        intelligence["persistence"]["raw_evidence"]["omitted"] = raw_omitted
        _write_json(intelligence_path, intelligence)

    copied_intelligence = destination / "calibration-intelligence.json"
    _write_json(copied_intelligence, intelligence)

    compatibility = _compatibility(intelligence)
    compatibility_key = _canonical_hash(compatibility)
    selection = intelligence.get("detector_selection_intelligence", {})
    search = intelligence.get("search", {})
    entry = {
        "calibration_id": calibration_id,
        "run_mode": run_mode,
        "evidence_tier": evidence_tier,
        "calibration_status": intelligence["calibration_status"],
        "record_path": relative_dir.as_posix(),
        "intelligence_path": (relative_dir / "calibration-intelligence.json").as_posix(),
        "parameter_provenance_path": (relative_dir / "parameter-provenance.json").as_posix()
            if (destination / "parameter-provenance.json").is_file() else None,
        "source_document_id": source_id,
        "golden_set_id": golden_id,
        "golden_set_sha256": golden_sha,
        "detector_id": detector,
        "detector_config_sha256": compatibility.get("detector_config_sha256"),
        "model_variant": ((identity.get("model_selection") or {}).get("variant") if isinstance(identity.get("model_selection"), dict) else None),
        "compatibility_key": compatibility_key,
        "created_at_utc": identity.get("created_at_utc"),
        "published_at_utc": intelligence["persistence"]["published_at_utc"],
        "build": persisted_build,
        "search": {
            "strategy": search.get("strategy") if isinstance(search, dict) else None,
            "parameter_sets": search.get("parameter_sets") if isinstance(search, dict) else None,
            "possible_parameter_sets": search.get("possible_parameter_sets") if isinstance(search, dict) else None,
            "exhaustive_complete": search.get("exhaustive_complete") if isinstance(search, dict) else None,
        },
        "selection": {
            "recommended_parameter_set_id": selection.get("recommended_parameter_set_id") if isinstance(selection, dict) else None,
            "best_avg_iou": selection.get("best_avg_iou") if isinstance(selection, dict) else None,
            "minimum_iou": selection.get("minimum_iou") if isinstance(selection, dict) else None,
            "stddev_iou": selection.get("stddev_iou") if isinstance(selection, dict) else None,
            "failure_count": selection.get("failure_count") if isinstance(selection, dict) else None,
            "calibration_evidence": selection.get("calibration_evidence") if isinstance(selection, dict) else None,
        },
    }
    return entry


def update_index(results_root: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    index_path = canonical_index_path(results_root, "calibration-index.json")
    read_path = readable_index_path(results_root, "calibration-index.json")
    index = load_index_with_persisted_backfill(read_path)
    current = index.get("entries") if isinstance(index.get("entries"), list) else []
    by_identity = {(item.get("compatibility_key"), item.get("calibration_id")): item for item in current if isinstance(item, dict)}
    for entry in entries:
        by_identity[(entry["compatibility_key"], entry["calibration_id"])] = entry
    merged = sorted(by_identity.values(), key=lambda item: (str(item.get("source_document_id")), str(item.get("golden_set_id")), str(item.get("detector_id")), str(item.get("created_at_utc") or "")))

    preferred: dict[str, dict[str, Any]] = {}
    compatibility_groups: dict[str, list[dict[str, Any]]] = {}
    for entry in merged:
        compatibility_groups.setdefault(str(entry.get("compatibility_key")), []).append(entry)
    for key, group in compatibility_groups.items():
        selected = authoritative_record(group)
        if selected:
            preferred[key] = {
                "calibration_id": selected.get("calibration_id"),
                "run_mode": selected.get("run_mode"),
                "evidence_tier": selected.get("evidence_tier"),
                "calibration_status": selected.get("calibration_status"),
                "detector_id": selected.get("detector_id"),
                "intelligence_path": selected.get("intelligence_path"),
                "created_at_utc": selected.get("created_at_utc"),
                "build": selected.get("build"),
                "selection": selected.get("selection"),
            }

    index.update({
        "schema_version": INDEX_SCHEMA_VERSION,
        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "entries": merged,
        "preferred": preferred,
    })
    write_index(results_root, "calibration-index.json", index)

    provenance_entries = []
    for item in merged:
        path = item.get("parameter_provenance_path")
        if not path:
            continue
        provenance_entries.append({
            "detector_id": item.get("detector_id"),
            "calibration_id": item.get("calibration_id"),
            "golden_set_sha256": item.get("golden_set_sha256"),
            "created_at_utc": item.get("created_at_utc"),
            "parameter_provenance_path": path,
            "build": item.get("build"),
        })
    write_index(
        results_root, "parameter-provenance-index.json",
        {
            "schema_version": "1.0",
            "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "registries": provenance_entries,
            "note": "Legacy 12-character IDs are aliases. Resolve against the referenced per-run provenance registry.",
        },
    )
    return index


def resolve(index_path: Path, *, detector: str, golden_set_sha256: str, detector_config_sha256: str | None = None) -> Path | None:
    index = load_index_with_persisted_backfill(index_path)
    candidates = [
        item for item in index.get("entries", [])
        if isinstance(item, dict)
        and item.get("detector_id") == detector
        and item.get("golden_set_sha256") == golden_set_sha256
        and (not detector_config_sha256 or item.get("detector_config_sha256") == detector_config_sha256)
    ]
    if not candidates:
        return None
    selected = authoritative_record(candidates)
    results_root = index_results_root(index_path)
    return results_root / str(selected["intelligence_path"]) if selected else None


def resolve_best_parameter_reference(
    index_path: Path,
    *,
    detector: str,
    golden_set_sha256: str,
    model_variant: str | None = None,
) -> dict[str, Any] | None:
    """Resolve the strongest historic exact parameter set for regression reference.

    Parameter search-grid/config hashes are intentionally not a gate here. The
    detector and Golden Set must match, while absolute parameter provenance lets
    HTH reevaluate the historic best even after the declared search grid changes.
    """
    index = load_index_with_persisted_backfill(index_path)
    requested_variant = str(model_variant or "").strip() or None
    def variant_compatible(item):
        if not requested_variant:
            return True
        recorded = str(item.get("model_variant") or "").strip() or None
        if recorded == requested_variant:
            return True
        # Calibrations created before model selection existed used the current
        # detector model. Preserve that history for an explicitly current variant,
        # but never let it bleed into an experimental/older model variant.
        return recorded is None and requested_variant.endswith("_current")

    candidates = [
        item for item in index.get("entries", [])
        if isinstance(item, dict)
        and item.get("detector_id") == detector
        and item.get("golden_set_sha256") == golden_set_sha256
        and variant_compatible(item)
    ]
    selected = authoritative_record(candidates)
    if not selected:
        return None

    selection = selected.get("selection") if isinstance(selected.get("selection"), dict) else {}
    legacy_id = str(selection.get("recommended_parameter_set_id") or "").strip()
    if not legacy_id:
        return None

    provenance = None
    provenance_source = None
    provenance_rel = selected.get("parameter_provenance_path")
    if provenance_rel:
        provenance_path = resolve_index_relative_path(index_path, str(provenance_rel))
        if provenance_path.is_file():
            provenance = _read_json(provenance_path)
            provenance_source = "parameter-provenance"

    if provenance is None:
        record_dir = resolve_index_relative_path(index_path, str(selected.get("record_path") or ""))
        legacy_parameters = record_dir / "parameters.json"
        if legacy_parameters.is_file():
            provenance = provenance_from_legacy_parameters(legacy_parameters)
            provenance_source = "legacy-parameters"

    if provenance is None:
        return None

    resolved = resolve_parameter_set(provenance, legacy_id)
    if not resolved or not isinstance(resolved.get("parameters"), dict):
        return None

    build = selected.get("build") if isinstance(selected.get("build"), dict) else {}
    return {
        "detector": detector,
        "golden_set_sha256": golden_set_sha256,
        "parameters": resolved["parameters"],
        "parameter_set_id": resolved.get("legacy_parameter_set_id") or legacy_id,
        "parameter_identity_sha256": resolved.get("sha256"),
        "historic_build_number": build.get("github_run_number"),
        "historic_build_url": build.get("run_url"),
        "historic_calibration_id": selected.get("calibration_id"),
        "historic_created_at_utc": selected.get("created_at_utc"),
        "provenance_source": provenance_source,
        "model_variant": selected.get("model_variant"),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--run-dir", type=Path, action="append", required=True)
    publish.add_argument("--results-root", type=Path, required=True)
    publish.add_argument("--mode", choices=("smoke", "full"), required=True)
    publish.add_argument("--source-fallback", required=True)
    publish.add_argument("--workflow", required=True)
    publish.add_argument("--event", required=True)
    publish.add_argument("--repository", required=True)
    publish.add_argument("--run-id", required=True)
    publish.add_argument("--run-number", required=True)
    publish.add_argument("--run-attempt", required=True)
    publish.add_argument("--ref", required=True)
    publish.add_argument("--sha", required=True)
    publish.add_argument("--run-url", required=True)

    resolve_parser = sub.add_parser("resolve")
    resolve_parser.add_argument("--index", type=Path, required=True)
    resolve_parser.add_argument("--detector", required=True)
    resolve_parser.add_argument("--golden-set-sha256", required=True)
    resolve_parser.add_argument("--detector-config-sha256")

    best_parser = sub.add_parser("resolve-best-parameter")
    best_parser.add_argument("--index", type=Path, required=True)
    best_parser.add_argument("--detector", required=True)
    best_parser.add_argument("--golden-set-sha256", required=True)
    best_parser.add_argument("--model-variant")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "resolve":
        path = resolve(args.index, detector=args.detector, golden_set_sha256=args.golden_set_sha256, detector_config_sha256=args.detector_config_sha256)
        if path is None:
            return 1
        print(path)
        return 0

    if args.command == "resolve-best-parameter":
        reference = resolve_best_parameter_reference(
            args.index,
            detector=args.detector,
            golden_set_sha256=args.golden_set_sha256,
            model_variant=args.model_variant,
        )
        if reference is None:
            return 1
        print(json.dumps(reference, sort_keys=True))
        return 0

    build = {
        "workflow": args.workflow,
        "event": args.event,
        "repository": args.repository,
        "github_run_id": args.run_id,
        "github_run_number": args.run_number,
        "github_run_attempt": args.run_attempt,
        "ref": args.ref,
        "pipeline_commit": args.sha,
        "run_url": args.run_url,
        "mode": args.mode,
    }
    entries = [publish_run(run_dir, args.results_root, mode=args.mode, source_fallback=args.source_fallback, build=build) for run_dir in args.run_dir]
    update_index(args.results_root, entries)
    runtime_observations = [observation_from_run(run_dir, build=build) for run_dir in args.run_dir]
    update_runtime_index(args.results_root, runtime_observations)
    parallelism_observations = [parallelism_observation_from_run(run_dir, build=build) for run_dir in args.run_dir]
    update_parallelism_index(args.results_root, parallelism_observations)
    record_prediction_observations(
        canonical_index_path(args.results_root, "optimizer-predictions.json"),
        parallelism_observations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
