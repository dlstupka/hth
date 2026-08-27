from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MIN_THREADS_PER_LPT_WORKER = 48


def plan_lpt_workers(detector_count: int, runner_thread_budget: int) -> int:
    """Choose equal workers for one aggregate LPT detector queue."""
    detectors = max(1, int(detector_count))
    budget = max(1, int(runner_thread_budget))
    queue_target = max(1, round(math.sqrt(detectors)))
    budget_cap = max(1, budget // MIN_THREADS_PER_LPT_WORKER)
    return min(detectors, queue_target, budget_cap)


def workload_class(mode: str, strategy: str, limit: str | None) -> str:
    if str(mode or "").strip().lower() != "full":
        return "short"
    if str(limit or "").strip():
        return "short"
    if str(strategy or "").strip().lower() != "exhaustive":
        return "short"
    return "full-exhaustive"


def _read_index(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"schema_version": 1, "observations": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"schema_version": 1, "observations": []}


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tail_fraction(row: dict[str, Any]) -> float:
    makespan = _as_float(row.get("makespan_seconds")) or 0.0
    tail = _as_float(row.get("final_tail_seconds")) or 0.0
    return 0.0 if makespan <= 0 else max(0.0, min(1.0, tail / makespan))


def _feedback_worker_count(row: dict[str, Any]) -> int:
    workers = max(1, _as_int(row.get("worker_count")) or 1)
    utilization = _as_float(row.get("worker_utilization")) or 0.0
    tail = _tail_fraction(row)
    if utilization >= 0.90 and tail <= 0.15:
        return workers + 1
    if workers > 1 and (utilization < 0.68 or tail >= 0.35):
        return workers - 1
    return workers


def preferred_short_schedule(
    *,
    index_path: Path | None,
    detector_count: int,
    runner_thread_budget: int,
    runner_label: str,
    golden_set_sha256: str | None,
) -> dict[str, Any] | None:
    """Choose short multi-detector concurrency from measured occupation history.

    Same-runner evidence wins. Cross-host evidence is used only as a learned
    threads-per-worker target and is scaled to the current max-thread budget.
    The feedback step changes at most one worker around the best measured run.
    """
    observations = [
        row for row in _read_index(index_path).get("observations", [])
        if isinstance(row, dict)
        and row.get("workload_class") == "short"
        and (_as_int(row.get("worker_count")) or 0) > 0
        and (_as_int(row.get("runner_thread_budget")) or 0) > 0
        and (_as_float(row.get("makespan_seconds")) or 0) > 0
    ]
    if golden_set_sha256:
        exact = [r for r in observations if str(r.get("golden_set_sha256") or "") == str(golden_set_sha256)]
        if exact:
            observations = exact
    if not observations:
        return None

    current_count = max(1, int(detector_count))
    current_budget = max(1, int(runner_thread_budget))
    same_runner = [r for r in observations if str(r.get("runner_label") or "") == str(runner_label or "")]
    pool = same_runner or observations

    def score(row: dict[str, Any]) -> tuple[float, float, float, str]:
        observed_count = max(1, _as_int(row.get("detector_count")) or 1)
        count_distance = abs(math.log(current_count / observed_count))
        makespan = _as_float(row.get("makespan_seconds")) or float("inf")
        utilization = _as_float(row.get("worker_utilization")) or 0.0
        tail = _tail_fraction(row)
        return (count_distance, makespan, -(utilization - 0.35 * tail), str(row.get("observed_at_utc") or ""))

    best = min(pool, key=score)
    observed_budget = max(1, _as_int(best.get("runner_thread_budget")) or current_budget)
    feedback_workers = max(1, _feedback_worker_count(best))
    observed_count = max(1, _as_int(best.get("detector_count")) or current_count)
    target_threads_per_worker = max(MIN_THREADS_PER_LPT_WORKER, observed_budget / feedback_workers)
    scaled_workers = max(1, round(current_budget / target_threads_per_worker))
    scaled_workers = max(1, round(scaled_workers * math.sqrt(current_count / observed_count)))
    budget_cap = max(1, current_budget // MIN_THREADS_PER_LPT_WORKER)
    workers = min(current_count, budget_cap, scaled_workers)
    threads = max(1, current_budget // workers)
    return {
        "pipelines": workers,
        "threads_per_pipeline": threads,
        "allocated_threads": workers * threads,
        "runner_budget": current_budget,
        "source": "multidetector-short-occupancy",
        "evidence_observation_id": best.get("observation_id"),
        "evidence_runner_label": best.get("runner_label"),
        "evidence_worker_count": best.get("worker_count"),
        "evidence_worker_utilization": best.get("worker_utilization"),
        "evidence_final_tail_seconds": best.get("final_tail_seconds"),
        "evidence_makespan_seconds": best.get("makespan_seconds"),
    }

def recommended_schedule(
    *,
    index_path: Path | None,
    detector_count: int,
    runner_thread_budget: int,
    runner_label: str,
    golden_set_sha256: str | None,
    mode: str,
    strategy: str,
    limit: str | None,
) -> dict[str, Any]:
    """Return the canonical multi-detector schedule recommendation.

    Short workloads reuse measured multidetector occupation when compatible
    evidence exists.  Every other case falls back to the same deterministic
    LPT worker planner used by the regression launcher.  Reports and dispatch
    therefore describe one scheduling policy instead of maintaining a static
    recommendation beside the executable planner.
    """
    detectors = max(1, int(detector_count))
    budget = max(1, int(runner_thread_budget))
    if workload_class(mode, strategy, limit) == "short":
        measured = preferred_short_schedule(
            index_path=index_path,
            detector_count=detectors,
            runner_thread_budget=budget,
            runner_label=runner_label,
            golden_set_sha256=golden_set_sha256,
        )
        if measured:
            return measured
    pipelines = plan_lpt_workers(detectors, budget)
    threads = max(1, budget // pipelines)
    return {
        "pipelines": pipelines,
        "threads_per_pipeline": threads,
        "allocated_threads": pipelines * threads,
        "runner_budget": budget,
        "source": "canonical-lpt-planner",
    }

