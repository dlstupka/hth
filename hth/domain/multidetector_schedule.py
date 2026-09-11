from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any, Mapping

MIN_MAKESPAN_IMPROVEMENT = 0.20
DEFAULT_SHARD_TARGET_SECONDS = 10 * 60
MAX_MANDATORY_REFERENCE_RUNS = 2


def materially_improves_makespan(
    incumbent_seconds: float | int | None,
    proposed_seconds: float | int | None,
    *,
    high_water_seconds: float | int | None = None,
    minimum_improvement: float = MIN_MAKESPAN_IMPROVEMENT,
) -> bool:
    """Apply the canonical replacement gate for a fixed schedule.

    A replacement must be finite, must not exceed an optional hard high-water
    mark, and must improve the complete incumbent makespan by the configured
    fraction.  The threshold boundary is inclusive.
    """
    incumbent = _as_float(incumbent_seconds)
    proposed = _as_float(proposed_seconds)
    high_water = _as_float(high_water_seconds)
    if incumbent is None or incumbent <= 0 or proposed is None or proposed < 0:
        return False
    if high_water_seconds is not None:
        if high_water is None or high_water < 0 or proposed > high_water + 1e-9:
            return False
    threshold = _as_float(minimum_improvement)
    if threshold is None or not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum_improvement must be finite and between 0 and 1")
    improvement = (incumbent - proposed) / incumbent
    return improvement + 1e-12 >= threshold


def plan_lpt_workers(detector_count: int, runner_thread_budget: int) -> int:
    """Bootstrap at one pipeline per vCPU, bounded by available tasks."""
    detectors = max(1, int(detector_count))
    budget = max(1, int(runner_thread_budget))
    max_pipelines = max(1, budget // 2)
    return min(detectors, max_pipelines)


def normalize_pipeline_assignments(
    detector_ids: list[str],
    assignments: Mapping[str, Any] | None,
    pipeline_count: int,
) -> dict[str, int] | None:
    """Validate a complete unsharded assignment map or request LPT fallback."""
    ceiling = _as_int(pipeline_count)
    if (
        not isinstance(assignments, Mapping)
        or not assignments
        or len(set(detector_ids)) != len(detector_ids)
        or ceiling is None
        or ceiling < 1
    ):
        return None
    normalized: dict[str, int] = {}
    for detector in detector_ids:
        if detector not in assignments:
            return None
        try:
            pipeline = int(assignments[detector])
        except (TypeError, ValueError):
            return None
        if pipeline < 1 or pipeline > ceiling:
            return None
        normalized[detector] = pipeline
    return normalized


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
    workers = max(1, min(int(pipeline_count), len(estimates) or 1))
    floor = _as_float(estimate_floor_seconds)
    if floor is None or floor <= 0:
        raise ValueError("estimate_floor_seconds must be finite and positive")
    normalized: list[tuple[int, float]] = []
    for index, raw in enumerate(estimates):
        value = _as_float(raw)
        normalized.append((index, max(floor, value if value is not None else floor)))
    normalized.sort(key=lambda item: (-item[1], item[0]))

    schedules = [
        {"pipeline": pipeline + 1, "task_indexes": [], "estimated_seconds": 0.0}
        for pipeline in range(workers)
    ]
    available = [(0.0, index) for index in range(workers)]
    heapq.heapify(available)
    for task_index, seconds in normalized:
        load, target = heapq.heappop(available)
        schedules[target]["task_indexes"].append(task_index)
        load += seconds
        schedules[target]["estimated_seconds"] = load
        heapq.heappush(available, (load, target))
    return [row for row in schedules if row["task_indexes"]]


def select_lpt_pipeline_count(
    estimates: list[float | int | None], max_pipelines: int,
) -> int:
    """Choose the smallest LPT shape that does not exceed its longest job."""
    usable = [value for value in (_as_float(raw) for raw in estimates) if value is not None and value > 0]
    if not usable:
        return max(1, min(len(estimates) or 1, int(max_pipelines)))
    unknown = max(usable)
    complete = [
        value if value is not None and value > 0 else unknown
        for value in (_as_float(raw) for raw in estimates)
    ]
    floor_seconds = max(complete)
    ceiling = max(1, min(len(complete), int(max_pipelines)))
    for pipelines in range(1, ceiling + 1):
        schedule = plan_static_lpt_tasks(complete, pipelines)
        makespan = max(float(row["estimated_seconds"]) for row in schedule)
        if makespan <= floor_seconds + 1e-9:
            return pipelines
    return ceiling


def plan_capacity_shards(
    estimates: list[float | int | None],
    max_pipelines: int,
    *,
    target_shard_seconds: float = DEFAULT_SHARD_TARGET_SECONDS,
    minimum_makespan_improvement: float = MIN_MAKESPAN_IMPROVEMENT,
    maximum_shards: list[int | None] | None = None,
) -> dict[str, Any]:
    """Split long LPT jobs only when runner capacity materially lowers makespan."""
    if maximum_shards is not None and len(maximum_shards) != len(estimates):
        raise ValueError("maximum_shards must align with estimates")
    target = _as_float(target_shard_seconds)
    if target is None or target <= 0:
        raise ValueError("target_shard_seconds must be finite and positive")
    capacity = max(1, int(max_pipelines))
    if not estimates:
        return {
            "shard_counts": [], "pipelines": 1, "applied": False,
            "reason": "no-tasks",
        }
    normalized = [_as_float(raw) for raw in estimates]
    known = [value for value in normalized if value is not None and value > 0]
    if not known:
        return {
            "shard_counts": [1] * len(estimates),
            "pipelines": min(len(estimates), capacity),
            "applied": False,
            "reason": "no-measured-runtime",
            "task_count": len(estimates),
        }
    # Unknown tasks participate in the capacity comparison conservatively but
    # are never themselves split from an invented runtime.
    fallback = max(known)
    complete = [value if value is not None and value > 0 else fallback for value in normalized]
    unsharded_pipelines = min(len(complete), capacity)
    unsharded_schedule = plan_static_lpt_tasks(complete, unsharded_pipelines)
    unsharded_makespan = max(float(row["estimated_seconds"]) for row in unsharded_schedule)
    proposed_counts = []
    for index, seconds in enumerate(complete):
        proposed = (
            max(1, math.ceil(seconds / target))
            if normalized[index] is not None and normalized[index] > 0
            else 1
        )
        proposed = min(proposed, capacity)
        if maximum_shards is not None and maximum_shards[index] is not None:
            shard_limit = _as_int(maximum_shards[index])
            if shard_limit is None:
                raise ValueError(f"maximum_shards[{index}] must be an integer or None")
            proposed = min(proposed, max(1, shard_limit))
        proposed_counts.append(proposed)
    proposed_estimates = [
        seconds / count
        for seconds, count in zip(complete, proposed_counts)
        for _ in range(count)
    ]
    proposed_pipelines = min(len(proposed_estimates), capacity)
    proposed_schedule = plan_static_lpt_tasks(proposed_estimates, proposed_pipelines)
    proposed_makespan = max(float(row["estimated_seconds"]) for row in proposed_schedule)
    improvement = (
        (unsharded_makespan - proposed_makespan) / unsharded_makespan
        if unsharded_makespan > 0 else 0.0
    )
    applied = materially_improves_makespan(
        unsharded_makespan, proposed_makespan,
        high_water_seconds=max(complete),
        minimum_improvement=minimum_makespan_improvement,
    )
    return {
        "shard_counts": proposed_counts if applied else [1] * len(complete),
        "pipelines": proposed_pipelines if applied else unsharded_pipelines,
        "applied": applied,
        "reason": "material-makespan-improvement" if applied else "below-makespan-improvement-threshold",
        "unsharded_makespan_seconds": unsharded_makespan,
        "predicted_makespan_seconds": proposed_makespan if applied else unsharded_makespan,
        "makespan_improvement": improvement if applied else 0.0,
        "task_count": len(proposed_estimates) if applied else len(complete),
        "target_shard_seconds": target,
    }


def _scheduler_runtime(row: dict[str, Any]) -> float | None:
    """Return one observation's canonical serial/scheduler/wall cost."""
    for field in (
        "estimated_serial_runtime_seconds",
        "scheduler_wall_clock_seconds",
        "wall_clock_seconds",
    ):
        seconds = _as_float(row.get(field))
        if seconds is not None and seconds > 0:
            return seconds
    return None


def _candidate_shard_limit(row: dict[str, Any]) -> int | None:
    """Bound fan-out by independent candidates, including legacy evidence."""
    exact = _as_int(row.get("full_exhaustive_candidate_count"))
    if exact is not None:
        return max(1, exact)
    actual = _as_int(row.get("actual_parameter_sets"))
    if actual is None:
        return None
    # Old observations did not retain the exhaustive-candidate count. Their
    # actual total can also include baseline and historic-best references.
    return max(1, actual - MAX_MANDATORY_REFERENCE_RUNS)


def optimize_lpt_schedule(
    *, runtime_index_path: Path | None, detector_ids: list[str],
    runner_thread_budget: int, runner_label: str, golden_set_sha256: str | None,
    mode: str, strategy: str, max_dimension: int,
    completion_index_path: Path | None = None,
) -> dict[str, Any] | None:
    """Jointly choose worker count and its deterministic LPT assignment.

    Measured scheduler-facing detector runtimes are assigned by the executable
    static LPT planner.  The longest detector is the unavoidable wall-time
    floor; choose the smallest pipeline count whose LPT makespan does not
    exceed that floor, maximizing sustained pipeline occupation without inventing
    an unmeasured thread-scaling curve.
    """
    payload = _read_index(runtime_index_path)
    observations = [row for row in payload.get("observations", []) if isinstance(row, dict)]
    if not observations or not detector_ids or len(set(detector_ids)) != len(detector_ids):
        return None
    wanted = set(detector_ids)
    completed_build_ids: set[str] | None = None
    if completion_index_path is not None and completion_index_path.is_file():
        completion_rows = [
            row for row in _read_index(completion_index_path).get("observations", [])
            if isinstance(row, dict)
            and str(row.get("mode") or "") == str(mode)
            and str(row.get("strategy") or "") == str(strategy)
            and (_as_int(row.get("detector_count")) or 0) >= len(wanted)
            and (_as_int(row.get("task_count")) or 0) >= len(wanted)
        ]
        completed_build_ids = {
            str(row.get("github_run_id") or "").strip()
            for row in completion_rows if str(row.get("github_run_id") or "").strip()
        }
    builds: dict[str, dict[str, dict[str, Any]]] = {}
    for row in observations:
        detector = str(row.get("detector_id") or "")
        if detector not in wanted or str(row.get("mode") or "") != str(mode):
            continue
        resolved_strategy = str(row.get("resolved_strategy") or row.get("requested_strategy") or "")
        if resolved_strategy != str(strategy):
            continue
        build = row.get("build") if isinstance(row.get("build"), dict) else {}
        build_id = str(build.get("github_run_id") or "").strip()
        if not build_id:
            continue
        if completed_build_ids is not None and build_id not in completed_build_ids:
            continue
        prior = builds.setdefault(build_id, {}).get(detector)
        if prior is None or str(row.get("observed_at_utc") or "") > str(prior.get("observed_at_utc") or ""):
            builds[build_id][detector] = row

    coherent: list[tuple[bool, bool, bool, str, str, list[dict[str, Any]]]] = []
    for build_id, by_id in builds.items():
        if set(by_id) != wanted:
            continue
        rows = list(by_id.values())
        latest = max(str(row.get("observed_at_utc") or "") for row in rows)
        exact_golden = bool(golden_set_sha256) and all(
            str(row.get("golden_set_sha256") or "") == str(golden_set_sha256)
            for row in rows
        )
        exact_dimension = all(_as_int(row.get("max_dimension")) == int(max_dimension) for row in rows)
        exact_runner = bool(runner_label) and all(
            runner_label in (
                row.get("runner", {}).get("runner_labels", [])
                if isinstance(row.get("runner"), dict)
                and isinstance(row.get("runner", {}).get("runner_labels"), list)
                else []
            )
            for row in rows
        )
        coherent.append((exact_golden, exact_dimension, exact_runner, latest, build_id, rows))
    if not coherent:
        return None
    # Prefer the requested Golden Set once it has a valid aggregate completion.
    # Cross-Golden evidence is strictly a bootstrap fallback.
    exact = [item for item in coherent if item[0]]
    selected = max(exact or coherent, key=lambda item: (item[1], item[2], item[3], item[4]))
    selected_exact_golden, selected_exact_dimension, selected_exact_runner, _, evidence_build_id, observations = selected

    selected_by_detector = {
        str(row.get("detector_id") or ""): row for row in observations
    }

    budget = max(1, int(runner_thread_budget))
    candidates: list[dict[str, Any]] = []
    # Sharding may create more runnable jobs than detectors, so preserve the
    # runner's full pipeline capacity here instead of capping it at detector
    # count before the shard plan is known.
    max_pipelines = max(1, budget // 2)
    measured = [
        _scheduler_runtime(selected_by_detector[detector])
        for detector in detector_ids
    ]
    known = [value for value in measured if value is not None and value > 0]
    if len(known) != len(detector_ids):
        return None
    complete = [float(value) for value in measured if value is not None]
    floor_seconds = max(complete)
    incumbent_assignments: dict[str, int] = {}
    incumbent_pipeline_count = 0
    incumbent_loads: dict[int, float] = {}
    for detector, seconds in zip(detector_ids, complete):
        row = selected_by_detector[detector]
        pipeline = _as_int(row.get("detector_pipeline_number"))
        count = _as_int(row.get("detector_pipelines"))
        if pipeline is not None and pipeline > 0:
            incumbent_assignments[detector] = pipeline
            incumbent_loads[pipeline] = incumbent_loads.get(pipeline, 0.0) + seconds
        if count is not None:
            incumbent_pipeline_count = max(incumbent_pipeline_count, count)
    maximum_shards = [
        _candidate_shard_limit(selected_by_detector[detector])
        for detector in detector_ids
    ]
    shard_plan = (
        plan_capacity_shards(complete, max_pipelines, maximum_shards=maximum_shards)
        if max_pipelines > 4 and strategy in {"exhaustive", "exhaustive-with-zombies"}
        else None
    )
    if shard_plan and shard_plan["applied"]:
        shard_counts = list(shard_plan["shard_counts"])
        scheduled_estimates = [
            seconds / count
            for seconds, count in zip(complete, shard_counts)
            for _ in range(count)
        ]
        selected_pipeline_count = int(shard_plan["pipelines"])
        floor_seconds = max(scheduled_estimates)
        max_pipelines = selected_pipeline_count
    else:
        shard_counts = [1] * len(complete)
        scheduled_estimates = complete
        selected_pipeline_count = select_lpt_pipeline_count(complete, max_pipelines)
        max_pipelines = min(len(complete), max_pipelines)
        proposed = plan_static_lpt_tasks(complete, selected_pipeline_count)
        proposed_makespan = max(float(row["estimated_seconds"]) for row in proposed)
        incumbent_makespan = max(incumbent_loads.values()) if incumbent_loads else None
        if (
            incumbent_pipeline_count > 0
            and incumbent_pipeline_count <= max_pipelines
            and len(incumbent_assignments) == len(detector_ids)
            and not materially_improves_makespan(
                incumbent_makespan, proposed_makespan,
                high_water_seconds=floor_seconds,
            )
        ):
            selected_pipeline_count = incumbent_pipeline_count
    for pipelines in range(1, max_pipelines + 1):
        threads = max(1, budget // pipelines)
        schedule = plan_static_lpt_tasks(scheduled_estimates, pipelines)
        makespan = max(float(row["estimated_seconds"]) for row in schedule)
        utilization = sum(scheduled_estimates) / (pipelines * makespan)
        candidates.append({
            "pipelines": pipelines,
            "threads_per_pipeline": threads,
            "allocated_threads": pipelines * threads,
            "runner_budget": budget,
            "predicted_makespan_seconds": makespan,
            "longest_detector_floor_seconds": floor_seconds,
            "predicted_pipeline_utilization": utilization,
            "evidence_detector_count": len(known),
            "detector_count": len(detector_ids),
            "source": "runtime-index-lpt-optimizer",
            "evidence_build_id": evidence_build_id,
            "evidence_golden_set_relation": "exact" if selected_exact_golden else "latest-compatible",
            "evidence_dimension_relation": "exact" if selected_exact_dimension else "fallback",
            "evidence_runner_relation": "exact" if selected_exact_runner else "fallback",
        })
    if not candidates:
        return None
    selected = next(row for row in candidates if int(row["pipelines"]) == selected_pipeline_count)
    if (
        not (shard_plan and shard_plan["applied"])
        and incumbent_pipeline_count == selected_pipeline_count
        and len(incumbent_assignments) == len(detector_ids)
        and incumbent_loads
    ):
        incumbent_makespan = max(incumbent_loads.values())
        selected["predicted_makespan_seconds"] = incumbent_makespan
        selected["predicted_pipeline_utilization"] = sum(complete) / (
            selected_pipeline_count * incumbent_makespan
        )
        selected["detector_pipeline_assignments"] = incumbent_assignments
        selected["schedule_retained"] = True
    ranked = sorted(candidates, key=lambda row: (float(row["predicted_makespan_seconds"]), int(row["pipelines"])))
    selected["candidate_count"] = len(candidates)
    selected["detector_shard_counts"] = {
        detector: count for detector, count in zip(detector_ids, shard_counts)
    }
    selected["sharding_applied"] = bool(shard_plan and shard_plan["applied"])
    selected["shard_target_seconds"] = DEFAULT_SHARD_TARGET_SECONDS
    selected["unsharded_makespan_seconds"] = (
        float(shard_plan["unsharded_makespan_seconds"]) if shard_plan else selected["predicted_makespan_seconds"]
    )
    selected["sharding_makespan_improvement"] = (
        float(shard_plan["makespan_improvement"]) if shard_plan else 0.0
    )
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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {"schema_version": 1, "observations": []}
    if not isinstance(payload, dict) or not isinstance(payload.get("observations", []), list):
        return {"schema_version": 1, "observations": []}
    return payload


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
        completion_index_path=index_path,
        detector_ids=list(detector_ids or []), runner_thread_budget=budget,
        runner_label=runner_label, golden_set_sha256=golden_set_sha256,
        mode=mode, strategy=strategy, max_dimension=max_dimension,
    )
    if optimized:
        return optimized
    pipelines = plan_lpt_workers(detectors, budget)
    threads = max(1, budget // pipelines)
    return {
        "pipelines": pipelines,
        "threads_per_pipeline": threads,
        "allocated_threads": pipelines * threads,
        "runner_budget": budget,
        "source": "canonical-lpt-planner",
        "shard_target_seconds": DEFAULT_SHARD_TARGET_SECONDS,
    }
