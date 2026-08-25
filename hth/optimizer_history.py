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


def run_history_dir(results_root: Path, detector: str, run_id: str) -> Path:
    return Path(results_root) / "execution-optimizer" / detector / "runs" / str(run_id)


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
        "run_metadata": run_metadata,
    }
    (destination / "run.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for source, name in ((observation_log, "observations.jsonl"), (shard_log, "shards.jsonl"),
                         (runner_metrics_log, "runner-metrics.jsonl")):
        if source is not None and source.is_file():
            shutil.copyfile(source, destination / name)
    return destination


def completed_run_records(results_root: Path, detector: str) -> list[dict[str, Any]]:
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
        records.append({
            "manifest": manifest,
            "observations": read_jsonl("observations.jsonl"),
            "shard_observations": read_jsonl("shards.jsonl"),
            "runner_metrics": read_jsonl("runner-metrics.jsonl"),
            "path": run_dir,
        })
    return records
