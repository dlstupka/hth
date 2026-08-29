"""Durable per-run execution-optimizer history.

The aggregate optimizer/parallelism indexes are rebuildable planning state.  A
completed optimizer execution is preserved independently beneath
execution-optimizer/<detector>/runs/<run-id>/ so later index regeneration cannot
collapse cross-run evidence.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from hth.optimizer_validity import migrate_optimizer_evidence, migrate_optimizer_run, optimizer_evidence_is_valid


def run_history_dir(results_root: Path, detector: str, run_id: str) -> Path:
    return Path(results_root) / "execution-optimizer" / detector / "runs" / str(run_id)




def completed_optimizer_run_ids(
    optimizer_index: dict[str, Any],
    parallelism_index: dict[str, Any],
    detector: str,
) -> set[str]:
    """Return every valid optimizer run known to have reached durable publication.

    Modern runs prove completion through finalized run metadata.  Legacy results
    predate per-run manifests/stop reasons; under that publication contract,
    run-tagged aggregate ``execution-optimizer`` observations were written to the
    results repository only by successful end-of-run publication.  Those rows
    therefore remain completion evidence even after a newer detector summary
    replaces the legacy summary that originally accompanied them.

    Validity is a hard exclusion at the run boundary: an explicitly invalid run
    record, or any invalid observation belonging to a legacy run, excludes the
    whole run.
    """
    runs = optimizer_index.get("runs") if isinstance(optimizer_index.get("runs"), dict) else {}
    completed: set[str] = set()
    invalid: set[str] = set()

    for run_id, payload in runs.items():
        if not isinstance(payload, dict) or str(payload.get("detector_id") or "") != detector:
            continue
        run_id = str(run_id)
        migrated = migrate_optimizer_run(payload)
        if not optimizer_evidence_is_valid(migrated):
            invalid.add(run_id)
            continue
        metadata = migrated.get("run_metadata") if isinstance(migrated.get("run_metadata"), dict) else {}
        if (
            str(metadata.get("stop_reason") or "").strip()
            or migrated.get("complete") is True
            or str(migrated.get("status") or "").lower() == "completed"
        ):
            completed.add(run_id)

    legacy_rows: dict[str, list[dict[str, Any]]] = {}
    observations = parallelism_index.get("observations")
    if isinstance(observations, list):
        for raw in observations:
            if not isinstance(raw, dict):
                continue
            if str(raw.get("detector_id") or "") != detector or raw.get("source") != "execution-optimizer":
                continue
            run_id = str(raw.get("optimizer_run_id") or "").strip()
            if not run_id:
                continue
            legacy_rows.setdefault(run_id, []).append(migrate_optimizer_evidence(raw))

    for run_id, rows in legacy_rows.items():
        if run_id in invalid or any(not optimizer_evidence_is_valid(row) for row in rows):
            invalid.add(run_id)
            completed.discard(run_id)
            continue
        # Persisted aggregate optimizer observations are legacy completion
        # evidence.  Keep every such completed run, not merely the newest one.
        completed.add(run_id)

    return completed - invalid

def persist_completed_run(*, results_root: Path, detector: str, run_id: str,
                          run_metadata: dict[str, Any], observation_log: Path | None,
                          shard_log: Path | None, runner_metrics_log: Path | None) -> Path | None:
    """Materialize immutable source evidence for one successfully completed run."""
    if not str(run_metadata.get("stop_reason") or "").strip():
        return None
    destination = run_history_dir(results_root, detector, run_id)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "record_type": "execution-optimizer-run",
        "optimizer_run_id": str(run_id),
        "detector_id": detector,
        "complete": True,
        "valid": True,
        "run_metadata": run_metadata,
    }
    (destination / "run.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for source, name in ((observation_log, "observations.jsonl"), (shard_log, "shards.jsonl"),
                         (runner_metrics_log, "runner-metrics.jsonl")):
        if source is not None and source.is_file():
            shutil.copyfile(source, destination / name)
    return destination


def completed_run_records(results_root: Path, detector: str, *, include_invalid: bool = False) -> list[dict[str, Any]]:
    base = Path(results_root) / "execution-optimizer" / detector / "runs"
    records: list[dict[str, Any]] = []
    if not base.is_dir():
        return records
    for run_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        manifest_path = run_dir / "run.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict) or manifest.get("complete") is not True:
            continue
        if str(manifest.get("detector_id") or "") != detector:
            continue
        def read_jsonl(name: str) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            path = run_dir / name
            if not path.is_file():
                return rows
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
            return rows
        observations = read_jsonl("observations.jsonl")
        migrated_manifest = migrate_optimizer_run(manifest, observations)
        if migrated_manifest != manifest:
            manifest_path.write_text(json.dumps(migrated_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = migrated_manifest
        if not include_invalid and not optimizer_evidence_is_valid(manifest):
            continue
        inherited_valid = optimizer_evidence_is_valid(manifest)
        migrated_observations = []
        for row in observations:
            migrated = migrate_optimizer_evidence(row)
            if not inherited_valid:
                migrated["valid"] = False
                migrated["invalid_reason"] = manifest.get("invalid_reason")
            migrated_observations.append(migrated)
        shard_observations = []
        for row in read_jsonl("shards.jsonl"):
            migrated = migrate_optimizer_evidence(row)
            if not inherited_valid:
                migrated["valid"] = False
                migrated["invalid_reason"] = manifest.get("invalid_reason")
            shard_observations.append(migrated)
        records.append({
            "manifest": manifest,
            "observations": migrated_observations,
            "shard_observations": shard_observations,
            "runner_metrics": read_jsonl("runner-metrics.jsonl"),
            "path": run_dir,
        })
    return records
