#!/usr/bin/env python3
"""Publish and resolve durable HTH calibration intelligence records."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hth.runtime_store import observation_from_run, update_runtime_index
from hth.parallelism_store import observation_from_run as parallelism_observation_from_run, update_parallelism_index
from hth.regression.parameter_provenance import provenance_from_legacy_parameters, resolve_parameter_set

from hth.contracts import CALIBRATION_INDEX_SCHEMA_VERSION, adapt_calibration_index
from hth.domain.calibration import authoritative_record

INDEX_SCHEMA_VERSION = CALIBRATION_INDEX_SCHEMA_VERSION
STATUS_PRIORITY = {"provisional": 1, "partial": 2, "authoritative": 3}
PERSISTED_FILES = (
    "manifest.json",
    "parameters.json",
    "parameter-provenance.json",
    "RUN-INFO.json",
    "reports/summary.json",
    "reports/winner-pages.json",
    "reports/calibration-intelligence.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _status(mode: str, intelligence: dict[str, Any]) -> str:
    if mode == "smoke":
        return "provisional"
    search = intelligence.get("search", {})
    if isinstance(search, dict) and search.get("exhaustive_complete"):
        return "authoritative"
    return "partial"


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


def publish_run(
    run_dir: Path,
    results_root: Path,
    *,
    mode: str,
    source_fallback: str,
    build: dict[str, Any],
) -> dict[str, Any]:
    intelligence_path = run_dir / "reports" / "calibration-intelligence.json"
    intelligence = _read_json(intelligence_path)
    if not intelligence.get("available"):
        raise ValueError(f"Calibration intelligence is unavailable in {intelligence_path}")

    identity = intelligence.setdefault("calibration_identity", {})
    calibration_id = str(identity.get("calibration_run_id") or run_dir.name)
    identity.setdefault("calibration_run_id", calibration_id)
    summary_path = run_dir / "reports" / "summary.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    info_path = run_dir / "RUN-INFO.json"
    info = _read_json(info_path) if info_path.is_file() else {}
    persisted_build = dict(build)
    run_time_seconds = info.get("elapsed_seconds", summary.get("elapsed_seconds"))
    if run_time_seconds is not None:
        persisted_build["run_time_seconds"] = run_time_seconds
    identity["build"] = persisted_build
    intelligence["calibration_status"] = _status(mode, intelligence)
    intelligence["persistence"] = {
        "store": "results-repository",
        "index": "calibration-index.json",
        "published_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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

    for relative in PERSISTED_FILES:
        source = run_dir / relative
        if source.is_file():
            target = destination / Path(relative).name
            shutil.copy2(source, target)

    copied_intelligence = destination / "calibration-intelligence.json"
    _write_json(copied_intelligence, intelligence)

    compatibility = _compatibility(intelligence)
    compatibility_key = _canonical_hash(compatibility)
    selection = intelligence.get("detector_selection_intelligence", {})
    search = intelligence.get("search", {})
    entry = {
        "calibration_id": calibration_id,
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
    index_path = results_root / "calibration-index.json"
    if index_path.is_file():
        index = adapt_calibration_index(_read_json(index_path))
    else:
        index = {"schema_version": INDEX_SCHEMA_VERSION, "entries": [], "preferred": {}}
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
    _write_json(index_path, index)

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
    _write_json(
        results_root / "parameter-provenance-index.json",
        {
            "schema_version": "1.0",
            "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "registries": provenance_entries,
            "note": "Legacy 12-character IDs are aliases. Resolve against the referenced per-run provenance registry.",
        },
    )
    return index


def resolve(index_path: Path, *, detector: str, golden_set_sha256: str, detector_config_sha256: str | None = None) -> Path | None:
    index = _read_json(index_path)
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
    return index_path.parent / str(selected["intelligence_path"]) if selected else None


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
    index = _read_json(index_path)
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
        provenance_path = index_path.parent / str(provenance_rel)
        if provenance_path.is_file():
            provenance = _read_json(provenance_path)
            provenance_source = "parameter-provenance"

    if provenance is None:
        record_dir = index_path.parent / str(selected.get("record_path") or "")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
