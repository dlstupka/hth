"""Rebuild HTH derived indexes from authoritative durable per-run evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hth.calibration_store import update_index as update_calibration_index
from hth.multidetector_store import MAX_OBSERVATIONS as MAX_MULTIDETECTOR_OBSERVATIONS
from hth.optimizer_history import completed_run_records
from hth.optimizer_store import update_optimizer_artifacts
from hth.parallelism_store import update_parallelism_index, update_parallelism_shards, observation_from_run as parallelism_observation_from_run
from hth.persistence import INDEX_FILENAMES, canonical_index_path, read_json, write_index
from hth.regression.learned_evidence import rebuild_orli_index
from hth.runtime_store import observation_from_run as runtime_observation_from_run, update_runtime_index


def _calibration_run_dirs(results_root: Path) -> list[Path]:
    return sorted({p.parent for p in Path(results_root).glob(
        "source-documents/*/golden-sets/*/*/calibrations/*/*/calibration-intelligence.json"
    )})


def _calibration_build(run_dir: Path) -> dict[str, Any]:
    intelligence_path = run_dir / "calibration-intelligence.json"
    intelligence = read_json(intelligence_path) if intelligence_path.is_file() else {}
    identity = intelligence.get("calibration_identity") if isinstance(intelligence.get("calibration_identity"), dict) else {}
    build = dict(identity.get("build") or {}) if isinstance(identity, dict) else {}
    status = str(intelligence.get("calibration_status") or "")
    build.setdefault("mode", "smoke" if status == "provisional" else "full")
    build.setdefault("source", "calibration")
    return build


def _optimizer_detectors(results_root: Path) -> list[str]:
    root = Path(results_root) / "execution-optimizer"
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and completed_run_records(results_root, path.name))


def rebuild_all(results_root: Path) -> dict[str, Any]:
    """Delete derived indexes and reconstruct them from durable evidence."""
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    for filename in INDEX_FILENAMES:
        canonical_index_path(results_root, filename).unlink(missing_ok=True)
        (results_root / filename).unlink(missing_ok=True)

    # Calibration records are authoritative for quality, parameter provenance,
    # and ordinary regression runtime/shape observations.
    calibration_runs = _calibration_run_dirs(results_root)
    calibration_index = update_calibration_index(results_root, [])

    runtime_rows: list[dict[str, Any]] = []
    parallel_rows: list[dict[str, Any]] = []
    for run_dir in calibration_runs:
        if not (run_dir / "RUN-INFO.json").is_file() or not (run_dir / "parameters.json").is_file():
            continue
        build = _calibration_build(run_dir)
        try:
            runtime_rows.append(runtime_observation_from_run(run_dir, build=build))
            parallel_rows.append(parallelism_observation_from_run(run_dir, build=build))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    runtime_index = update_runtime_index(results_root, runtime_rows)

    # Optimizer per-run records are authoritative for execution-shape history.
    optimizer_detectors = _optimizer_detectors(results_root)
    shard_rows: list[dict[str, Any]] = []
    for detector in optimizer_detectors:
        for record in completed_run_records(results_root, detector):
            parallel_rows.extend(record.get("observations", []))
            shard_rows.extend(record.get("shard_observations", []))
    parallel_index = update_parallelism_index(results_root, parallel_rows)
    if shard_rows:
        parallel_index = update_parallelism_shards(results_root, shard_rows)

    optimizer_paths = {}
    for detector in optimizer_detectors:
        optimizer_paths[detector] = {
            key: str(value)
            for key, value in update_optimizer_artifacts(results_root, detector).items()
        }

    # Aggregate multi-detector scheduling observations have their own durable log.
    multidetector_rows: list[dict[str, Any]] = []
    md_root = results_root / "execution-history" / "multidetector"
    if md_root.is_dir():
        for path in sorted(md_root.glob("*.json")):
            try:
                row = read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            multidetector_rows.append(row)
    multidetector_rows.sort(key=lambda row: str(row.get("observed_at_utc") or ""), reverse=True)
    write_index(results_root, "multidetector-index.json", {
        "schema_version": 1,
        "observations": multidetector_rows[:MAX_MULTIDETECTOR_OBSERVATIONS],
    })

    orli_path = rebuild_orli_index(results_root=results_root)
    return {
        "calibration_entries": len(calibration_index.get("entries", [])),
        "runtime_observations": len(runtime_index.get("observations", [])),
        "parallelism_observations": len(parallel_index.get("observations", [])),
        "optimizer_detectors": optimizer_detectors,
        "optimizer_paths": optimizer_paths,
        "multidetector_observations": len(multidetector_rows),
        "orli_index": str(orli_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(rebuild_all(args.results_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
