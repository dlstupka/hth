from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MIN_THREADS_PER_LPT_WORKER = 48
DEFAULT_THREAD_SCALING_EXPONENT = 0.5


def plan_lpt_workers(detector_count: int, runner_thread_budget: int) -> int:
    """Choose equal workers for one aggregate LPT detector queue."""
    detectors = max(1, int(detector_count))
    budget = max(1, int(runner_thread_budget))
    queue_target = max(1, round(math.sqrt(detectors)))
    budget_cap = max(1, budget // MIN_THREADS_PER_LPT_WORKER)
    return min(detectors, queue_target, budget_cap)


def plan_static_lpt_tasks(
    estimates: list[float | int | None],
    pipeline_count: int,
    *,
    estimate_floor_seconds: float = 0.1,
) -> list[dict[str, Any]]:
    """Build one deterministic, fixed LPT schedule before workers start.

    Tasks are sorted by estimated duration descending, then greedily placed on
    the pipeline with the least assigned estimated work.  The returned task
    indexes refer to the caller's original sequence.  Unknown/invalid estimates
    use a small scheduling floor so every task participates deterministically.
    """
    workers = max(1, int(pipeline_count))
    floor = max(0.001, float(estimate_floor_seconds))
    normalized: list[tuple[int, float]] = []
    for index, raw in enumerate(estimates):
        value = _as_float(raw)
        normalized.append((index, max(floor, value if value is not None else floor)))
    normalized.sort(key=lambda item: (-item[1], item[0]))

    schedules = [
        {"pipeline": pipeline + 1, "task_indexes": [], "estimated_seconds": 0.0}
        for pipeline in range(workers)
    ]
    for task_index, seconds in normalized:
        target = min(
            range(workers),
            key=lambda idx: (float(schedules[idx]["estimated_seconds"]), idx),
        )
        schedules[target]["task_indexes"].append(task_index)
        schedules[target]["estimated_seconds"] = float(schedules[target]["estimated_seconds"]) + seconds
    return [row for row in schedules if row["task_indexes"]]


def _runtime_context_score(
    row: dict[str, Any], *, mode: str, strategy: str,
    max_dimension: int, golden_set_sha256: str | None, runner_label: str,
) -> int:
    score = 0
    if str(row.get("mode") or "") == str(mode):
        score += 32
    if str(row.get("resolved_strategy") or row.get("requested_strategy") or "") == str(strategy):
        score += 16
    if _as_int(row.get("max_dimension")) == int(max_dimension):
        score += 8
    if golden_set_sha256 and str(row.get("golden_set_sha256") or "") == str(golden_set_sha256):
        score += 4
    runner = row.get("runner") if isinstance(row.get("runner"), dict) else {}
    labels = runner.get("runner_labels") if isinstance(runner.get("runner_labels"), list) else []
    if runner_label and runner_label in labels:
        score += 2
    return score


def _predicted_runtime(
    rows: list[dict[str, Any]], *, threads: int, mode: str, strategy: str,
    max_dimension: int, golden_set_sha256: str | None, runner_label: str,
) -> float | None:
    candidates: list[tuple[int, float, str, float]] = []
    for row in rows:
        seconds = _as_float(row.get("scheduler_wall_clock_seconds"))
        if seconds is None:
            seconds = _as_float(row.get("wall_clock_seconds"))
        observed_threads = _as_int(row.get("configured_threads"))
        if seconds is None or seconds <= 0 or observed_threads is None or observed_threads <= 0:
            continue
        context = _runtime_context_score(
            row, mode=mode, strategy=strategy, max_dimension=max_dimension,
            golden_set_sha256=golden_set_sha256, runner_label=runner_label,
        )
        distance = abs(math.log(max(1, threads) / observed_threads))
        # Until a detector has a fitted scaling curve, use the same conservative
        # square-root scaling assumption as regression sharding. Exact observed
        # thread counts remain exact predictions.
        predicted = seconds * math.pow(observed_threads / max(1, threads), DEFAULT_THREAD_SCALING_EXPONENT)
        candidates.append((context, -distance, str(row.get("observed_at_utc") or ""), predicted))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def optimize_lpt_schedule(
    *, runtime_index_path: Path | None, detector_ids: list[str],
    runner_thread_budget: int, runner_label: str, golden_set_sha256: str | None,
    mode: str, strategy: str, max_dimension: int,
) -> dict[str, Any] | None:
    """Jointly choose worker count and its deterministic LPT assignment.

    Every feasible worker count is evaluated. Detector runtimes are projected
    from the closest compatible persisted observation, then assigned with the
    executable static LPT planner. Unknown detectors receive the largest known
    estimate, preventing missing history from making a shape look artificially
    cheap.
    """
    payload = _read_index(runtime_index_path)
    observations = [row for row in payload.get("observations", []) if isinstance(row, dict)]
    if not observations or not detector_ids:
        return None
    by_detector: dict[str, list[dict[str, Any]]] = {}
    for row in observations:
        detector = str(row.get("detector_id") or "")
        if detector:
            by_detector.setdefault(detector, []).append(row)

    budget = max(1, int(runner_thread_budget))
    candidates: list[dict[str, Any]] = []
    for pipelines in range(1, min(len(detector_ids), budget) + 1):
        threads = max(1, budget // pipelines)
        estimates = [
            _predicted_runtime(
                by_detector.get(detector, []), threads=threads, mode=mode,
                strategy=strategy, max_dimension=max_dimension,
                golden_set_sha256=golden_set_sha256, runner_label=runner_label,
            )
            for detector in detector_ids
        ]
        known = [value for value in estimates if value is not None]
        if not known:
            continue
        unknown_estimate = max(known)
        complete = [value if value is not None else unknown_estimate for value in estimates]
        schedule = plan_static_lpt_tasks(complete, pipelines)
        makespan = max(float(row["estimated_seconds"]) for row in schedule)
        candidates.append({
            "pipelines": pipelines,
            "threads_per_pipeline": threads,
            "allocated_threads": pipelines * threads,
            "runner_budget": budget,
            "predicted_makespan_seconds": makespan,
            "evidence_detector_count": len(known),
            "detector_count": len(detector_ids),
            "source": "runtime-index-lpt-optimizer",
        })
    if not candidates:
        return None
    # Prefer the simpler shape only when predictions are effectively tied.
    best_time = min(float(row["predicted_makespan_seconds"]) for row in candidates)
    near_best = [row for row in candidates if float(row["predicted_makespan_seconds"]) <= best_time * 1.01]
    selected = min(near_best, key=lambda row: int(row["pipelines"]))
    ranked = sorted(candidates, key=lambda row: (float(row["predicted_makespan_seconds"]), int(row["pipelines"])))
    selected["candidate_count"] = len(candidates)
    selected["leading_candidates"] = [
        {
            "pipelines": int(row["pipelines"]),
            "threads_per_pipeline": int(row["threads_per_pipeline"]),
            "predicted_makespan_seconds": float(row["predicted_makespan_seconds"]),
        }
        for row in ranked[:3]
    ]
    return selected


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
    runtime_index_path: Path | None = None,
    detector_ids: list[str] | None = None,
    max_dimension: int = 0,
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
    optimized = optimize_lpt_schedule(
        runtime_index_path=runtime_index_path,
        detector_ids=list(detector_ids or []), runner_thread_budget=budget,
        runner_label=runner_label, golden_set_sha256=golden_set_sha256,
        mode=mode, strategy=strategy, max_dimension=max_dimension,
    )
    if optimized:
        return optimized
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
