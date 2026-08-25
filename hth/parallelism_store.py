#!/usr/bin/env python3
"""Persist execution-shape observations for detector parallelism experiments."""
from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from hth.persistence import canonical_index_path, readable_index_path, read_json as _read_json, atomic_write_json as _write_json, load_index, write_index
from typing import Any, Iterable
import os
import time
from contextlib import contextmanager

from hth.contracts import (
    OPTIMIZER_OBSERVATION_SCHEMA_VERSION,
    PARALLELISM_INDEX_SCHEMA_VERSION,
    adapt_parallelism_index,
)
MAX_OBSERVATIONS_PER_DETECTOR = 500



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


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()



@contextmanager
def _index_lock(path: Path, timeout_seconds: float = 30.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + timeout_seconds
    fd = None
    while fd is None:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for parallelism index lock: {lock}")
            time.sleep(0.05)
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
        os.close(fd)
        fd = None
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def update_parallelism_shards(results_root: Path, shard_observations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path = canonical_index_path(results_root, "parallelism-index.json")
    with _index_lock(path):
        index = load_index(results_root, "parallelism-index.json")
        by_id = {
            str(item.get("observation_id")): item
            for item in index.get("shard_observations", [])
            if isinstance(item, dict) and item.get("observation_id")
        }
        for observation in shard_observations:
            by_id[str(observation["observation_id"])] = observation
        rows = list(by_id.values())
        rows.sort(key=lambda row: (str(row.get("detector_id") or ""), str(row.get("optimizer_run_id") or ""), int(row.get("shape_sequence") or 0), int(row.get("shard_index") or 0)))
        # Optimizer shard observations are durable experiment evidence.  Never
        # age them out: later optimizer runs may deliberately fill missing
        # pipeline shapes and need the earlier shard evidence for audit/recovery.
        optimizer_rows = [row for row in rows if row.get("source") == "execution-optimizer"]
        other_rows = [row for row in rows if row.get("source") != "execution-optimizer"][-5000:]
        index["schema_version"] = PARALLELISM_INDEX_SCHEMA_VERSION
        index["updated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        index["shard_observations"] = optimizer_rows + other_rows
        index.setdefault("observations", [])
        write_index(path.parent.parent, "parallelism-index.json", index)
        return index

def observation_from_run(
    run_dir: Path,
    *,
    build: dict[str, Any],
    wall_clock_seconds: float | None = None,
) -> dict[str, Any]:
    info = _read_json(run_dir / "RUN-INFO.json")
    summary_path = run_dir / "reports" / "summary.json"
    if not summary_path.is_file():
        summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    pipeline = info.get("detector_pipeline") if isinstance(info.get("detector_pipeline"), dict) else {}
    shard = info.get("shard") if isinstance(info.get("shard"), dict) else {}
    runner = summary.get("runner") if isinstance(summary.get("runner"), dict) else {}
    parameter_space = summary.get("parameter_space") if isinstance(summary.get("parameter_space"), dict) else {}
    golden = summary.get("golden_set") if isinstance(summary.get("golden_set"), dict) else {}

    threads = max(1, _as_int(info.get("threads") or summary.get("threads")) or 1)
    shards = max(1, _as_int(info.get("shard_count") or shard.get("count")) or 1)
    pipelines = max(1, _as_int(
        pipeline.get("pipeline_count")
        or pipeline.get("count")
        or pipeline.get("detector_pipelines")
    ) or 1)
    active_pipelines = min(pipelines, shards)
    allocated_threads = active_pipelines * threads
    wall = _as_float(wall_clock_seconds)
    if wall is None:
        wall = _as_float(info.get("wall_elapsed_seconds") or info.get("elapsed_seconds") or summary.get("elapsed_seconds"))
    serial = _as_float(info.get("estimated_serial_runtime_seconds") or summary.get("estimated_serial_runtime_seconds"))
    acceleration = _as_float(info.get("effective_acceleration") or summary.get("effective_acceleration"))
    if wall_clock_seconds is not None and wall and serial and wall > 0:
        acceleration = serial / wall
    elif acceleration is None and wall and serial and wall > 0:
        acceleration = serial / wall

    possible_sets = _as_int(info.get("possible_parameter_sets") or parameter_space.get("possible_parameter_sets"))
    actual_sets = _as_int(info.get("actual_parameter_sets") or parameter_space.get("actual_parameter_sets"))
    page_count = _as_int(info.get("golden_set_pages") or golden.get("pages") or summary.get("golden_set_pages"))
    page_evaluations = _as_int(info.get("planned_page_evaluations") or summary.get("page_evaluations"))
    if page_evaluations is None and actual_sets is not None and page_count is not None:
        page_evaluations = actual_sets * page_count

    runner_label = build.get("runner_label") or info.get("runner_label")
    detector_config_sha256 = info.get("detector_config_sha256") or summary.get("detector_config_sha256")
    golden_sha = info.get("golden_set_sha256") or summary.get("golden_set_sha256")
    workload = {
        "detector_id": info.get("detector") or summary.get("detector") or "unknown",
        "detector_config_sha256": detector_config_sha256,
        "golden_set_sha256": golden_sha,
        "mode": build.get("mode"),
        "strategy": info.get("strategy") or summary.get("strategy"),
        "possible_parameter_sets": possible_sets,
        "actual_parameter_sets": actual_sets,
        "optimizer_benchmark_parameter_sets": _as_int(build.get("optimizer_benchmark_parameter_sets")),
        "max_dimension": _as_int(info.get("max_dimension") or summary.get("max_dimension")),
    }
    runner_identity = {
        "runner_label": runner_label,
        "runner_name": runner.get("runner_name") or info.get("runner_name"),
        "runner_labels": runner.get("github_runner_labels") or info.get("github_runner_labels"),
        "cpu_model": runner.get("cpu_model") or info.get("cpu_model"),
        "logical_cpu_count": _as_int(runner.get("logical_cpu_count") or info.get("logical_cpu_count")),
    }
    compatibility = {**workload, **runner_identity}
    shape = {
        "shards": shards,
        "active_pipelines": active_pipelines,
        "threads_per_pipeline": threads,
        "allocated_threads": allocated_threads,
    }

    observation_run_id = info.get("run_id") or run_dir.name
    # Concurrent detector pipelines can share the same second-resolution run ID.
    # Detector identity is therefore part of the durable observation key.
    observation_id = f"{build.get('github_run_id', 'local')}:{compatibility['detector_id']}:{observation_run_id}"
    return {
        "schema_version": OPTIMIZER_OBSERVATION_SCHEMA_VERSION,
        "observation_id": observation_id,
        "observed_at_utc": info.get("finished_at_utc") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": info.get("run_id") or run_dir.name,
        "detector_id": compatibility["detector_id"],
        "mode": compatibility["mode"],
        "strategy": compatibility["strategy"],
        "golden_set_sha256": golden_sha,
        "detector_config_sha256": detector_config_sha256,
        "possible_parameter_sets": possible_sets,
        "actual_parameter_sets": actual_sets,
        "optimizer_benchmark_parameter_sets": workload.get("optimizer_benchmark_parameter_sets"),
        "golden_set_pages": page_count,
        "page_evaluations": page_evaluations,
        "max_dimension": compatibility["max_dimension"],
        "workload_key": _canonical_hash(workload),
        "runner_key": _canonical_hash(runner_identity),
        "compatibility_key": _canonical_hash(compatibility),
        "execution_shape": f"{active_pipelines}p/{shards}s/{threads}t",
        **shape,
        "wall_clock_seconds": wall,
        "estimated_serial_runtime_seconds": serial,
        "effective_acceleration": acceleration,
        "parallel_efficiency": (acceleration / allocated_threads) if acceleration is not None and allocated_threads else None,
        "allocated_thread_seconds": (wall * allocated_threads) if wall is not None else None,
        "parameter_sets_per_second": (actual_sets / wall) if wall and actual_sets is not None else None,
        "page_evaluations_per_second": (page_evaluations / wall) if wall and page_evaluations is not None else None,
        "runner": {
            "runner_label": runner_label,
            "runner_name": runner.get("runner_name") or info.get("runner_name"),
            "runner_labels": runner.get("github_runner_labels") or info.get("github_runner_labels"),
            "cpu_model": runner.get("cpu_model") or info.get("cpu_model"),
            "physical_core_count": _as_int(runner.get("physical_core_count") or info.get("physical_core_count")),
            "logical_cpu_count": _as_int(runner.get("logical_cpu_count") or info.get("logical_cpu_count")),
            "memory_gib": _as_float(runner.get("memory_gib") or info.get("memory_gib")),
        },
        "build": build,
    }


def _is_comparable(row: dict[str, Any]) -> bool:
    if row.get("mode") != "full" or (_as_float(row.get("wall_clock_seconds")) or 0) <= 0:
        return False
    actual = _as_int(row.get("actual_parameter_sets"))
    possible = _as_int(row.get("possible_parameter_sets"))
    benchmark = _as_int(row.get("optimizer_benchmark_parameter_sets"))
    if benchmark is not None and benchmark > 0 and row.get("source") == "execution-optimizer":
        return actual == min(possible or benchmark, benchmark)
    return row.get("strategy") == "exhaustive" and actual == possible


def update_parallelism_index(results_root: Path, observations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path = canonical_index_path(results_root, "parallelism-index.json")
    with _index_lock(path):
        return _update_parallelism_index_locked(path, observations)


def _update_parallelism_index_locked(path: Path, observations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    index = load_index(path.parent.parent, "parallelism-index.json")

    by_id = {
        str(item.get("observation_id")): item
        for item in index.get("observations", [])
        if isinstance(item, dict) and item.get("observation_id")
    }
    for observation in observations:
        by_id[str(observation["observation_id"])] = observation

    grouped_by_detector: dict[str, list[dict[str, Any]]] = {}
    for item in by_id.values():
        grouped_by_detector.setdefault(str(item.get("detector_id") or "unknown"), []).append(item)

    trimmed: list[dict[str, Any]] = []
    for items in grouped_by_detector.values():
        items.sort(key=lambda row: str(row.get("observed_at_utc") or ""), reverse=True)
        # Execution-optimizer observations are intentionally cumulative.  A
        # later run can fill a sparse 3-7 pipeline interval and must coalesce
        # with compatible shapes measured in earlier completed runs.
        optimizer_items = [row for row in items if row.get("source") == "execution-optimizer"]
        other_items = [row for row in items if row.get("source") != "execution-optimizer"][:MAX_OBSERVATIONS_PER_DETECTOR]
        trimmed.extend(optimizer_items)
        trimmed.extend(other_items)

    comparable = [row for row in trimmed if _is_comparable(row)]
    best_by_compatibility: dict[str, dict[str, Any]] = {}
    shape_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in comparable:
        compatibility_key = str(row.get("compatibility_key") or "legacy")
        shape_key = str(row.get("execution_shape") or "unknown")
        shape_groups.setdefault((compatibility_key, shape_key), []).append(row)
        current = best_by_compatibility.get(compatibility_key)
        if current is None or float(row["wall_clock_seconds"]) < float(current["wall_clock_seconds"]):
            best_by_compatibility[compatibility_key] = row

    shape_summaries: list[dict[str, Any]] = []
    for (compatibility_key, shape_key), rows in shape_groups.items():
        walls = sorted(float(row["wall_clock_seconds"]) for row in rows)
        fastest = min(rows, key=lambda row: float(row["wall_clock_seconds"]))
        latest = max(rows, key=lambda row: str(row.get("observed_at_utc") or ""))
        shape_summaries.append({
            "compatibility_key": compatibility_key,
            "detector_id": fastest.get("detector_id"),
            "execution_shape": shape_key,
            "shards": fastest.get("shards"),
            "active_pipelines": fastest.get("active_pipelines"),
            "threads_per_pipeline": fastest.get("threads_per_pipeline"),
            "allocated_threads": fastest.get("allocated_threads"),
            "observation_count": len(rows),
            "fastest_wall_clock_seconds": walls[0],
            "median_wall_clock_seconds": statistics.median(walls),
            "latest_wall_clock_seconds": latest.get("wall_clock_seconds"),
            "fastest_observation_id": fastest.get("observation_id"),
            "latest_observation_id": latest.get("observation_id"),
        })

    best_compact = {
        key: {
            "observation_id": row.get("observation_id"),
            "detector_id": row.get("detector_id"),
            "execution_shape": row.get("execution_shape"),
            "shards": row.get("shards"),
            "active_pipelines": row.get("active_pipelines"),
            "threads_per_pipeline": row.get("threads_per_pipeline"),
            "allocated_threads": row.get("allocated_threads"),
            "wall_clock_seconds": row.get("wall_clock_seconds"),
            "effective_acceleration": row.get("effective_acceleration"),
            "parallel_efficiency": row.get("parallel_efficiency"),
        }
        for key, row in best_by_compatibility.items()
    }

    # Preserve the convenient detector-level best view for existing readers while
    # the compatibility-scoped optimizer uses best_by_compatibility.
    best_by_detector: dict[str, dict[str, Any]] = {}
    for row in comparable:
        detector = str(row.get("detector_id") or "unknown")
        current = best_by_detector.get(detector)
        if current is None or float(row["wall_clock_seconds"]) < float(current["wall_clock_seconds"]):
            best_by_detector[detector] = {
                "observation_id": row.get("observation_id"),
                "execution_shape": row.get("execution_shape"),
                "shards": row.get("shards"),
                "active_pipelines": row.get("active_pipelines"),
                "threads_per_pipeline": row.get("threads_per_pipeline"),
                "allocated_threads": row.get("allocated_threads"),
                "wall_clock_seconds": row.get("wall_clock_seconds"),
                "effective_acceleration": row.get("effective_acceleration"),
            }

    trimmed.sort(key=lambda row: (str(row.get("detector_id")), str(row.get("observed_at_utc") or "")))
    shape_summaries.sort(key=lambda row: (str(row.get("detector_id")), str(row.get("compatibility_key")), float(row.get("fastest_wall_clock_seconds") or 0)))
    index.update({
        "schema_version": PARALLELISM_INDEX_SCHEMA_VERSION,
        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "observations": trimmed,
        "shape_summaries": shape_summaries,
        "best_by_compatibility": best_compact,
        "best": best_by_detector,
    })
    write_index(path.parent.parent, "parallelism-index.json", index)
    return index
