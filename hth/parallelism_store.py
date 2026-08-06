#!/usr/bin/env python3
"""Persist execution-shape observations for detector parallelism experiments."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PARALLELISM_INDEX_SCHEMA_VERSION = "1.0"
MAX_OBSERVATIONS_PER_DETECTOR = 200


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def observation_from_run(run_dir: Path, *, build: dict[str, Any]) -> dict[str, Any]:
    info = _read_json(run_dir / "RUN-INFO.json")
    summary = _read_json(run_dir / "reports" / "summary.json")
    pipeline = info.get("detector_pipeline") if isinstance(info.get("detector_pipeline"), dict) else {}
    shard = info.get("shard") if isinstance(info.get("shard"), dict) else {}
    runner = summary.get("runner") if isinstance(summary.get("runner"), dict) else {}
    parameter_space = summary.get("parameter_space") if isinstance(summary.get("parameter_space"), dict) else {}

    threads = max(1, _as_int(info.get("threads") or summary.get("threads")) or 1)
    shards = max(1, _as_int(info.get("shard_count") or shard.get("count")) or 1)
    pipelines = max(1, _as_int(
        pipeline.get("pipeline_count")
        or pipeline.get("count")
        or pipeline.get("detector_pipelines")
    ) or min(shards, 1))
    active_pipelines = min(pipelines, shards)
    wall = _as_float(info.get("wall_elapsed_seconds") or info.get("elapsed_seconds") or summary.get("elapsed_seconds"))
    serial = _as_float(info.get("estimated_serial_runtime_seconds") or summary.get("estimated_serial_runtime_seconds"))
    acceleration = _as_float(info.get("effective_acceleration") or summary.get("effective_acceleration"))
    if acceleration is None and wall and serial and wall > 0:
        acceleration = serial / wall

    return {
        "observation_id": f"{build.get('github_run_id', 'local')}:{info.get('run_id', run_dir.name)}",
        "observed_at_utc": info.get("finished_at_utc") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": info.get("run_id") or run_dir.name,
        "detector_id": info.get("detector") or summary.get("detector") or "unknown",
        "mode": build.get("mode"),
        "strategy": info.get("strategy") or summary.get("strategy"),
        "golden_set_sha256": info.get("golden_set_sha256") or summary.get("golden_set_sha256"),
        "possible_parameter_sets": _as_int(info.get("possible_parameter_sets") or parameter_space.get("possible_parameter_sets")),
        "actual_parameter_sets": _as_int(info.get("actual_parameter_sets") or parameter_space.get("actual_parameter_sets")),
        "shards": shards,
        "active_pipelines": active_pipelines,
        "threads_per_pipeline": threads,
        "allocated_threads": active_pipelines * threads,
        "wall_clock_seconds": wall,
        "estimated_serial_runtime_seconds": serial,
        "effective_acceleration": acceleration,
        "runner": {
            "runner_name": runner.get("runner_name") or info.get("runner_name"),
            "runner_labels": runner.get("github_runner_labels") or info.get("github_runner_labels"),
            "cpu_model": runner.get("cpu_model") or info.get("cpu_model"),
            "logical_cpu_count": _as_int(runner.get("logical_cpu_count") or info.get("logical_cpu_count")),
        },
        "build": build,
    }


def update_parallelism_index(results_root: Path, observations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path = results_root / "parallelism-index.json"
    if path.is_file():
        index = _read_json(path)
    else:
        index = {"schema_version": PARALLELISM_INDEX_SCHEMA_VERSION, "observations": [], "best": {}}

    by_id = {
        str(item.get("observation_id")): item
        for item in index.get("observations", [])
        if isinstance(item, dict) and item.get("observation_id")
    }
    for observation in observations:
        by_id[str(observation["observation_id"])] = observation

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in by_id.values():
        grouped.setdefault(str(item.get("detector_id") or "unknown"), []).append(item)

    trimmed: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {}
    for detector, items in grouped.items():
        items.sort(key=lambda row: str(row.get("observed_at_utc") or ""), reverse=True)
        kept = items[:MAX_OBSERVATIONS_PER_DETECTOR]
        trimmed.extend(kept)
        comparable = [
            row for row in kept
            if row.get("mode") == "full"
            and row.get("strategy") == "exhaustive"
            and _as_int(row.get("actual_parameter_sets")) == _as_int(row.get("possible_parameter_sets"))
            and _as_float(row.get("wall_clock_seconds")) not in (None, 0.0)
        ]
        if comparable:
            winner = min(comparable, key=lambda row: float(row["wall_clock_seconds"]))
            best[detector] = {
                "observation_id": winner.get("observation_id"),
                "shards": winner.get("shards"),
                "active_pipelines": winner.get("active_pipelines"),
                "threads_per_pipeline": winner.get("threads_per_pipeline"),
                "allocated_threads": winner.get("allocated_threads"),
                "wall_clock_seconds": winner.get("wall_clock_seconds"),
                "effective_acceleration": winner.get("effective_acceleration"),
            }

    trimmed.sort(key=lambda row: (str(row.get("detector_id")), str(row.get("observed_at_utc") or "")))
    index.update({
        "schema_version": PARALLELISM_INDEX_SCHEMA_VERSION,
        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "observations": trimmed,
        "best": best,
    })
    _write_json(path, index)
    return index
