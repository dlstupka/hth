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

INDEX_SCHEMA_VERSION = "1.0"
STATUS_PRIORITY = {"provisional": 1, "partial": 2, "authoritative": 3}
PERSISTED_FILES = (
    "manifest.json",
    "parameters.json",
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
    return {
        "source_document": identity.get("source_document") if isinstance(identity, dict) else None,
        "golden_set_id": golden.get("collection_id") or golden.get("id") or golden.get("configuration"),
        "golden_set_sha256": golden.get("sha256"),
        "detector_id": intelligence.get("detector"),
        "detector_config_sha256": detector_config.get("sha256"),
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
    identity["build"] = build
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
        "source_document_id": source_id,
        "golden_set_id": golden_id,
        "golden_set_sha256": golden_sha,
        "detector_id": detector,
        "detector_config_sha256": compatibility.get("detector_config_sha256"),
        "compatibility_key": compatibility_key,
        "created_at_utc": identity.get("created_at_utc"),
        "published_at_utc": intelligence["persistence"]["published_at_utc"],
        "build": build,
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
            "calibration_evidence": selection.get("calibration_evidence") if isinstance(selection, dict) else None,
        },
    }
    return entry


def update_index(results_root: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    index_path = results_root / "calibration-index.json"
    if index_path.is_file():
        index = _read_json(index_path)
    else:
        index = {"schema_version": INDEX_SCHEMA_VERSION, "entries": [], "preferred": {}}
    current = index.get("entries") if isinstance(index.get("entries"), list) else []
    by_identity = {(item.get("compatibility_key"), item.get("calibration_id")): item for item in current if isinstance(item, dict)}
    for entry in entries:
        by_identity[(entry["compatibility_key"], entry["calibration_id"])] = entry
    merged = sorted(by_identity.values(), key=lambda item: (str(item.get("source_document_id")), str(item.get("golden_set_id")), str(item.get("detector_id")), str(item.get("created_at_utc") or "")))

    preferred: dict[str, dict[str, Any]] = {}
    for entry in merged:
        key = str(entry.get("compatibility_key"))
        current_preferred = preferred.get(key)
        candidate_rank = STATUS_PRIORITY.get(str(entry.get("calibration_status")), 0)
        current_rank = STATUS_PRIORITY.get(str(current_preferred.get("calibration_status")), 0) if current_preferred else -1
        if current_preferred is None or candidate_rank > current_rank or (
            candidate_rank == current_rank and str(entry.get("created_at_utc") or "") >= str(current_preferred.get("created_at_utc") or "")
        ):
            preferred[key] = {
                "calibration_id": entry.get("calibration_id"),
                "calibration_status": entry.get("calibration_status"),
                "detector_id": entry.get("detector_id"),
                "intelligence_path": entry.get("intelligence_path"),
                "created_at_utc": entry.get("created_at_utc"),
                "build": entry.get("build"),
            }

    index.update({
        "schema_version": INDEX_SCHEMA_VERSION,
        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "entries": merged,
        "preferred": preferred,
    })
    _write_json(index_path, index)
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
    candidates.sort(key=lambda item: (STATUS_PRIORITY.get(str(item.get("calibration_status")), 0), str(item.get("created_at_utc") or "")), reverse=True)
    return index_path.parent / str(candidates[0]["intelligence_path"])


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
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "resolve":
        path = resolve(args.index, detector=args.detector, golden_set_sha256=args.golden_set_sha256, detector_config_sha256=args.detector_config_sha256)
        if path is None:
            return 1
        print(path)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
