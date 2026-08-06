"""Plan bounded detector-regression work and manage expiring shard leases."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TARGET_SHARD_SECONDS = 30 * 60
SAFETY_FACTOR = 1.20
RUNNER_MAX_THREADS = {
    "e7k": 64,
    "e9k": 32,
    "github-hosted": 8,
    "hth": 16,
    "rhel8": 16,
    "windows": 8,
}
ALLOWED_THREADS = (1, 2, 4, 8, 16, 32, 48, 64)


@dataclass(frozen=True)
class ShardPlan:
    serial_runtime_seconds: float | None
    threads: int
    shard_count: int
    target_shard_seconds: int
    safety_factor: float
    estimate_source: str


@dataclass(frozen=True)
class ExecutionPlan:
    runner_thread_budget: int
    active_pipelines: int
    requested_threads: str
    threads_per_pipeline: int
    allocated_threads: int
    unused_threads: int


def runner_max_threads(runner_label: str, available_cpus: int | None = None) -> int:
    label = (runner_label or "").strip().lower()
    configured = RUNNER_MAX_THREADS.get(label)
    if configured is None:
        configured = max(1, min(16, int(available_cpus or os.cpu_count() or 1)))
    # Named runner profiles are policy budgets and may intentionally oversubscribe
    # the logical CPUs reported inside a hosted runner.
    return max(1, int(configured))


def plan_execution(
    requested_threads: str | int,
    *,
    runner_label: str,
    active_pipelines: int,
    available_cpus: int | None = None,
) -> ExecutionPlan:
    """Create the one authoritative detector-thread allocation for a build."""
    pipelines = max(1, int(active_pipelines))
    budget = runner_max_threads(runner_label, available_cpus)
    per_pipeline_budget = max(1, budget // pipelines)
    requested = str(requested_threads).strip().lower()
    if requested == "auto":
        threads = per_pipeline_budget
    else:
        threads = min(max(1, int(requested)), per_pipeline_budget)
    allocated = threads * pipelines
    return ExecutionPlan(
        runner_thread_budget=budget,
        active_pipelines=pipelines,
        requested_threads=requested,
        threads_per_pipeline=threads,
        allocated_threads=allocated,
        unused_threads=max(0, budget - allocated),
    )


def budgeted_threads(planned_threads: int, *, runner_label: str, active_pipelines: int) -> int:
    """Compatibility wrapper for explicit per-pipeline thread requests."""
    return plan_execution(
        planned_threads,
        runner_label=runner_label,
        active_pipelines=active_pipelines,
    ).threads_per_pipeline

def automatic_threads(serial_runtime_seconds: float | None, maximum: int) -> int:
    """Use the fewest useful threads for the estimated serial workload."""
    if serial_runtime_seconds is None:
        return min(4, maximum)
    minutes = serial_runtime_seconds / 60.0
    if minutes < 5:
        wanted = 1
    elif minutes < 15:
        wanted = 4
    elif minutes < 30:
        wanted = 8
    else:
        wanted = maximum
    return max(value for value in ALLOWED_THREADS if value <= min(wanted, maximum))


def conservative_speedup(threads: int) -> float:
    """Avoid assuming linear scaling before detector/runner scaling curves exist."""
    return max(1.0, math.sqrt(max(1, threads)))


def plan_shards(
    serial_runtime_seconds: float | None,
    *,
    runner_label: str,
    requested_threads: str | int = "auto",
    available_cpus: int | None = None,
    target_shard_seconds: int = TARGET_SHARD_SECONDS,
    safety_factor: float = SAFETY_FACTOR,
    maximum_shards: int = 96,
    requested_shards: int | None = None,
    possible_parameter_sets: int | None = None,
    estimate_source: str = "runtime-index",
) -> ShardPlan:
    maximum = runner_max_threads(runner_label, available_cpus)
    if str(requested_threads).lower() == "auto":
        threads = automatic_threads(serial_runtime_seconds, maximum)
    else:
        requested = int(requested_threads)
        threads = max(value for value in ALLOWED_THREADS if value <= min(requested, maximum))
    parameter_cap = max(1, int(possible_parameter_sets or maximum_shards))
    shard_cap = min(max(1, int(maximum_shards)), parameter_cap)
    if requested_shards is not None:
        shards = max(1, min(int(requested_shards), parameter_cap))
        source = "explicit-shard-count"
    elif serial_runtime_seconds is None or serial_runtime_seconds <= 0:
        shards = 1
        source = estimate_source
    else:
        predicted = serial_runtime_seconds * safety_factor / conservative_speedup(threads)
        shards = max(1, min(shard_cap, math.ceil(predicted / target_shard_seconds)))
        source = estimate_source
    return ShardPlan(serial_runtime_seconds, threads, shards, target_shard_seconds, safety_factor, source)


def estimate_serial_runtime(observation: dict[str, Any], possible_parameter_sets: int) -> float | None:
    wall = observation.get("wall_clock_seconds")
    actual = observation.get("actual_parameter_sets")
    threads = int(observation.get("configured_threads") or 1)
    try:
        wall_value = float(wall)
        actual_value = int(actual)
    except (TypeError, ValueError):
        return None
    if wall_value <= 0 or actual_value <= 0:
        return None
    # Convert the measured run to a conservative serial-equivalent estimate.
    per_set_serial = wall_value * conservative_speedup(threads) / actual_value
    return per_set_serial * max(1, int(possible_parameter_sets))


def best_smoke_observation(index_path: Path, detector: str) -> dict[str, Any] | None:
    if not index_path.is_file():
        return None
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    rows = [
        row for row in payload.get("observations", [])
        if isinstance(row, dict) and row.get("detector_id") == detector and row.get("mode") == "smoke"
    ]
    rows.sort(key=lambda row: str(row.get("observed_at_utc") or ""), reverse=True)
    return rows[0] if rows else None


def write_lease(path: Path, *, owner: str, ttl_seconds: int, shard: str) -> None:
    now = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"owner": owner, "shard": shard, "renewed_epoch": now, "expires_epoch": now + ttl_seconds}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def lease_expired(path: Path, *, now: float | None = None) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expiry = float(payload["expires_epoch"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return True
    return expiry <= (time.time() if now is None else now)
