from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any, Mapping

from hth.contracts import RUNTIME_INDEX_SCHEMA_VERSION, RUNTIME_OBSERVATION_SCHEMA_VERSION

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
    fixed_preparation_seconds: list[float | int | None] | None = None,
    shardable_work_seconds: list[float | int | None] | None = None,
    pre_fanout_preparation: list[bool] | None = None,
) -> dict[str, Any]:
    """Split long LPT jobs only when runner capacity materially lowers makespan."""
    if maximum_shards is not None and len(maximum_shards) != len(estimates):
        raise ValueError("maximum_shards must align with estimates")
    if fixed_preparation_seconds is not None and len(fixed_preparation_seconds) != len(estimates):
        raise ValueError("fixed_preparation_seconds must align with estimates")
    if shardable_work_seconds is not None and len(shardable_work_seconds) != len(estimates):
        raise ValueError("shardable_work_seconds must align with estimates")
    if pre_fanout_preparation is not None and len(pre_fanout_preparation) != len(estimates):
        raise ValueError("pre_fanout_preparation must align with estimates")
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
    fixed = []
    shardable = []
    for index, total in enumerate(complete):
        raw_fixed = (
            _as_float(fixed_preparation_seconds[index])
            if fixed_preparation_seconds is not None else None
        )
        fixed_cost = min(total, max(0.0, raw_fixed or 0.0))
        raw_shardable = (
            _as_float(shardable_work_seconds[index])
            if shardable_work_seconds is not None else None
        )
        body_cost = (
            max(0.0, raw_shardable)
            if raw_shardable is not None else max(0.0, total - fixed_cost)
        )
        fixed.append(fixed_cost)
        shardable.append(body_cost)
    parent_shared = list(pre_fanout_preparation or [False] * len(complete))
    unsharded_estimates = [
        body if shared else total
        for total, body, shared in zip(complete, shardable, parent_shared)
    ]
    unsharded_shared_preparation = sum(
        fixed_cost for fixed_cost, shared in zip(fixed, parent_shared) if shared
    )
    unsharded_pipelines = min(len(complete), capacity)
    unsharded_schedule = plan_static_lpt_tasks(unsharded_estimates, unsharded_pipelines)
    unsharded_fanout_makespan = max(
        float(row["estimated_seconds"]) for row in unsharded_schedule
    )
    unsharded_makespan = unsharded_shared_preparation + unsharded_fanout_makespan
    proposed_counts = []
    for index, seconds in enumerate(shardable):
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
    proposed_estimates = []
    shared_preparation = 0.0
    for total, fixed_cost, body_cost, count, already_shared in zip(
        complete, fixed, shardable, proposed_counts, parent_shared,
    ):
        if count > 1 or already_shared:
            shared_preparation += fixed_cost
            proposed_estimates.extend([body_cost / count] * count)
        else:
            proposed_estimates.append(total)
    proposed_pipelines = min(len(proposed_estimates), capacity)
    proposed_schedule = plan_static_lpt_tasks(proposed_estimates, proposed_pipelines)
    proposed_fanout_makespan = max(float(row["estimated_seconds"]) for row in proposed_schedule)
    proposed_makespan = shared_preparation + proposed_fanout_makespan
    improvement = (
        (unsharded_makespan - proposed_makespan) / unsharded_makespan
        if unsharded_makespan > 0 else 0.0
    )
    applied = materially_improves_makespan(
        unsharded_makespan, proposed_makespan,
        high_water_seconds=unsharded_makespan,
        minimum_improvement=minimum_makespan_improvement,
    )
    if applied:
        reason = "material-makespan-improvement"
    elif proposed_counts == [1] * len(complete):
        reason = "no-eligible-work-above-target"
    elif proposed_makespan > unsharded_makespan + 1e-9:
        reason = "candidate-increases-makespan"
    else:
        reason = "below-makespan-improvement-threshold"
    return {
        "shard_counts": proposed_counts if applied else [1] * len(complete),
        "pipelines": proposed_pipelines if applied else unsharded_pipelines,
        "applied": applied,
        "reason": reason,
        "unsharded_makespan_seconds": unsharded_makespan,
        "predicted_makespan_seconds": proposed_makespan if applied else unsharded_makespan,
        "makespan_improvement": improvement if applied else 0.0,
        "task_count": len(proposed_estimates) if applied else len(complete),
        "target_shard_seconds": target,
        "shared_preparation_seconds": shared_preparation if applied else 0.0,
        "predicted_fanout_makespan_seconds": proposed_fanout_makespan if applied else unsharded_fanout_makespan,
        "candidate_shard_counts": proposed_counts,
        "candidate_task_count": len(proposed_estimates),
        "candidate_makespan_seconds": proposed_makespan,
        "candidate_makespan_improvement": improvement,
        "candidate_shared_preparation_seconds": shared_preparation,
    }


def plan_golden_set_lanes(
    estimates: list[float | int | None],
    max_lanes: int,
    *,
    maximum_lanes: list[int | None],
    target_lane_seconds: float = DEFAULT_SHARD_TARGET_SECONDS,
    minimum_makespan_improvement: float = MIN_MAKESPAN_IMPROVEMENT,
    fixed_preparation_seconds: list[float | int | None] | None = None,
    lane_work_seconds: list[float | int | None] | None = None,
    pre_fanout_preparation: list[bool] | None = None,
) -> dict[str, Any]:
    """Allocate bounded page lanes without creating independent searches.

    Every detector retains one coordinator. Additional lanes are capacity
    units beneath that coordinator, capped by its measured Golden Set pages.
    Unknown runtimes and missing page counts are never parallelized from
    invented evidence.
    """
    if len(maximum_lanes) != len(estimates):
        raise ValueError("maximum_lanes must align with estimates")
    if fixed_preparation_seconds is not None and len(fixed_preparation_seconds) != len(estimates):
        raise ValueError("fixed_preparation_seconds must align with estimates")
    if lane_work_seconds is not None and len(lane_work_seconds) != len(estimates):
        raise ValueError("lane_work_seconds must align with estimates")
    if pre_fanout_preparation is not None and len(pre_fanout_preparation) != len(estimates):
        raise ValueError("pre_fanout_preparation must align with estimates")
    target = _as_float(target_lane_seconds)
    if target is None or target <= 0:
        raise ValueError("target_lane_seconds must be finite and positive")
    capacity = max(1, int(max_lanes))
    if not estimates:
        return {"lane_counts": [], "pipelines": 1, "applied": False, "reason": "no-tasks"}
    normalized = [_as_float(value) for value in estimates]
    known = [value for value in normalized if value is not None and value > 0]
    if not known or capacity <= len(estimates):
        return {
            "lane_counts": [1] * len(estimates),
            "pipelines": min(len(estimates), capacity), "applied": False,
            "reason": "no-spare-capacity-or-measured-runtime",
        }
    fallback = max(known)
    complete = [value if value is not None and value > 0 else fallback for value in normalized]
    fixed = []
    lane_work = []
    for index, total in enumerate(complete):
        raw_fixed = (
            _as_float(fixed_preparation_seconds[index])
            if fixed_preparation_seconds is not None else None
        )
        fixed_cost = min(total, max(0.0, raw_fixed or 0.0))
        raw_lane_work = (
            _as_float(lane_work_seconds[index]) if lane_work_seconds is not None else None
        )
        fixed.append(fixed_cost)
        lane_work.append(
            max(0.0, raw_lane_work)
            if raw_lane_work is not None else max(0.0, total - fixed_cost)
        )
    parent_shared = list(pre_fanout_preparation or [False] * len(complete))
    global_fixed = sum(cost for cost, shared in zip(fixed, parent_shared) if shared)
    local_fixed = [0.0 if shared else cost for cost, shared in zip(fixed, parent_shared)]
    limits: list[int] = []
    for index, raw in enumerate(maximum_lanes):
        parsed = _as_int(raw)
        if parsed is None or parsed < 1:
            limits.append(1)
            continue
        desired = max(1, math.ceil(lane_work[index] / target))
        limits.append(min(parsed, desired, capacity))

    counts = [1] * len(complete)
    while sum(counts) < capacity:
        eligible = [index for index, count in enumerate(counts) if count < limits[index]]
        if not eligible:
            break
        # Add the lane that lowers the current longest coordinator. Stable
        # index tie-breaking keeps the plan reproducible.
        selected = max(eligible, key=lambda index: (
            local_fixed[index] + lane_work[index] / counts[index], -index,
        ))
        counts[selected] += 1

    incumbent = global_fixed + max(
        fixed_cost + work
        for fixed_cost, work in zip(local_fixed, lane_work)
    )
    proposed = global_fixed + max(
        fixed_cost + work / count
        for fixed_cost, work, count in zip(local_fixed, lane_work, counts)
    )
    improvement = (incumbent - proposed) / incumbent if incumbent > 0 else 0.0
    applied = materially_improves_makespan(
        incumbent, proposed, high_water_seconds=incumbent,
        minimum_improvement=minimum_makespan_improvement,
    )
    if applied:
        reason = "material-makespan-improvement"
    elif counts == [1] * len(complete):
        reason = "no-eligible-work-above-target"
    elif proposed > incumbent + 1e-9:
        reason = "candidate-increases-makespan"
    else:
        reason = "below-makespan-improvement-threshold"
    return {
        "lane_counts": counts if applied else [1] * len(complete),
        "pipelines": sum(counts) if applied else len(complete),
        "applied": applied,
        "reason": reason,
        "uncoordinated_makespan_seconds": incumbent,
        "predicted_makespan_seconds": proposed if applied else incumbent,
        "makespan_improvement": improvement if applied else 0.0,
        "target_lane_seconds": target,
        "shared_preparation_seconds": global_fixed,
        "candidate_lane_counts": counts,
        "candidate_capacity_units": sum(counts),
        "candidate_makespan_seconds": proposed,
        "candidate_makespan_improvement": improvement,
    }


def supports_golden_set_lane_scaling(runner_label: str, max_pipelines: int) -> bool:
    """Return whether a runner exposes flexible capacity beyond hosted defaults."""
    normalized = str(runner_label or "").strip().lower()
    return max_pipelines > 4 and normalized not in {
        "", "unknown", "github-hosted", "ubuntu-latest",
    }


def _scheduler_runtime(row: dict[str, Any]) -> float | None:
    """Return one observation's canonical serial/scheduler/wall cost."""
    for field in (
        "scheduler_end_to_end_serial_seconds",
        "estimated_serial_runtime_seconds",
        "scheduler_wall_clock_seconds",
        "wall_clock_seconds",
    ):
        seconds = _as_float(row.get(field))
        if seconds is not None and seconds > 0:
            return seconds
    return None


def _scheduler_runtime_components(row: dict[str, Any]) -> tuple[float | None, float, float | None, bool]:
    """Return total, fixed preparation, shardable work, and preparation placement."""
    total = _scheduler_runtime(row)
    fixed = _as_float(row.get("scheduler_fixed_preparation_seconds")) or 0.0
    body = _as_float(row.get("scheduler_shardable_work_seconds"))
    if total is None:
        return None, max(0.0, fixed), body, False
    fixed = min(total, max(0.0, fixed))
    if body is None:
        body = max(0.0, total - fixed)
    return (
        total, fixed, max(0.0, body),
        str(row.get("scheduler_preparation_source") or "") == "parent-shared",
    )


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
    index_schema = str(payload.get("schema_version") or "legacy")
    if index_schema not in {"legacy", RUNTIME_INDEX_SCHEMA_VERSION}:
        return None
    observations = [
        row for row in payload.get("observations", [])
        if isinstance(row, dict)
        and (
            index_schema == "legacy"
            or (
                str(row.get("schema_version") or "") == RUNTIME_OBSERVATION_SCHEMA_VERSION
                and _as_float(row.get("scheduler_end_to_end_serial_seconds")) is not None
                and _as_float(row.get("scheduler_shardable_work_seconds")) is not None
            )
        )
    ]
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
    components = [
        _scheduler_runtime_components(selected_by_detector[detector])
        for detector in detector_ids
    ]
    measured = [row[0] for row in components]
    fixed_preparation = [row[1] for row in components]
    shardable_work = [row[2] for row in components]
    parent_shared_preparation = [row[3] for row in components]
    known = [value for value in measured if value is not None and value > 0]
    if len(known) != len(detector_ids):
        return None
    complete = [float(value) for value in measured if value is not None]
    global_parent_preparation = sum(
        fixed for fixed, shared in zip(fixed_preparation, parent_shared_preparation)
        if shared
    )
    base_fanout_estimates = [
        float(body) if shared and body is not None else total
        for total, body, shared in zip(complete, shardable_work, parent_shared_preparation)
    ]
    floor_seconds = global_parent_preparation + max(base_fanout_estimates)
    incumbent_assignments: dict[str, int] = {}
    incumbent_pipeline_count = 0
    incumbent_loads: dict[int, float] = {}
    for detector, seconds in zip(detector_ids, base_fanout_estimates):
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
    retain_incumbent_schedule = False
    lane_plan = (
        plan_golden_set_lanes(
            complete, max_pipelines,
            maximum_lanes=[
                _as_int(selected_by_detector[detector].get("golden_set_pages"))
                for detector in detector_ids
            ],
            fixed_preparation_seconds=fixed_preparation,
            lane_work_seconds=shardable_work,
            pre_fanout_preparation=parent_shared_preparation,
        )
        if supports_golden_set_lane_scaling(runner_label, max_pipelines)
        and strategy in {"adaptive", "binary-refine"}
        else None
    )
    shard_plan = (
        plan_capacity_shards(
            complete, max_pipelines, maximum_shards=maximum_shards,
            fixed_preparation_seconds=fixed_preparation,
            shardable_work_seconds=shardable_work,
            pre_fanout_preparation=parent_shared_preparation,
        )
        if max_pipelines > 4 and strategy in {"exhaustive", "exhaustive-with-zombies"}
        else None
    )
    if lane_plan and lane_plan["applied"]:
        golden_set_lane_counts = list(lane_plan["lane_counts"])
        shard_counts = [1] * len(complete)
        scheduled_estimates = []
        for total, fixed_cost, body_cost, count, parent_shared in zip(
            complete, fixed_preparation, shardable_work, golden_set_lane_counts,
            parent_shared_preparation,
        ):
            body = float(body_cost) if body_cost is not None else max(0.0, total - fixed_cost)
            # Candidate capacity is expressed in lanes. Attribute fixed work
            # once and only the divisible work across lane units.
            scheduled_estimates.append((0.0 if parent_shared else fixed_cost) + body / count)
            scheduled_estimates.extend([body / count] * (count - 1))
        selected_pipeline_count = int(lane_plan["pipelines"])
        floor_seconds = float(lane_plan.get("shared_preparation_seconds") or 0.0) + max(scheduled_estimates)
        max_pipelines = selected_pipeline_count
    elif shard_plan and shard_plan["applied"]:
        golden_set_lane_counts = [1] * len(complete)
        shard_counts = list(shard_plan["shard_counts"])
        scheduled_estimates = []
        for total, fixed_cost, body_cost, count in zip(
            complete, fixed_preparation, shardable_work, shard_counts,
        ):
            body = float(body_cost) if body_cost is not None else max(0.0, total - fixed_cost)
            if count > 1:
                scheduled_estimates.extend([body / count] * count)
            else:
                scheduled_estimates.append(total)
        selected_pipeline_count = int(shard_plan["pipelines"])
        floor_seconds = float(shard_plan.get("shared_preparation_seconds") or 0.0) + max(scheduled_estimates)
        max_pipelines = selected_pipeline_count
    else:
        golden_set_lane_counts = [1] * len(complete)
        shard_counts = [1] * len(complete)
        scheduled_estimates = base_fanout_estimates
        selected_pipeline_count = select_lpt_pipeline_count(base_fanout_estimates, max_pipelines)
        max_pipelines = min(len(complete), max_pipelines)
        # The smallest-count floor is useful for bootstrap, but a feedback
        # reshuffle must not contract an observed topology merely because fewer
        # pipelines tie the same longest-job lower bound.  Contraction requires
        # measured, shape-aware evidence of a material makespan improvement;
        # this fixed-cost LPT model cannot manufacture that evidence.
        if (
            incumbent_pipeline_count > 0
            and incumbent_pipeline_count <= max_pipelines
            and len(incumbent_assignments) == len(detector_ids)
            and selected_pipeline_count < incumbent_pipeline_count
        ):
            contracted = plan_static_lpt_tasks(base_fanout_estimates, selected_pipeline_count)
            retained = plan_static_lpt_tasks(base_fanout_estimates, incumbent_pipeline_count)
            contracted_makespan = global_parent_preparation + max(float(row["estimated_seconds"]) for row in contracted)
            retained_makespan = global_parent_preparation + max(float(row["estimated_seconds"]) for row in retained)
            if not materially_improves_makespan(
                retained_makespan, contracted_makespan,
                high_water_seconds=floor_seconds,
            ):
                selected_pipeline_count = incumbent_pipeline_count
        proposed = plan_static_lpt_tasks(base_fanout_estimates, selected_pipeline_count)
        proposed_makespan = global_parent_preparation + max(float(row["estimated_seconds"]) for row in proposed)
        incumbent_makespan = (
            global_parent_preparation + max(incumbent_loads.values()) if incumbent_loads else None
        )
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
            retain_incumbent_schedule = True
    for pipelines in range(1, max_pipelines + 1):
        threads = max(1, budget // pipelines)
        schedule = plan_static_lpt_tasks(scheduled_estimates, pipelines)
        makespan = max(float(row["estimated_seconds"]) for row in schedule)
        shape_plan = shard_plan if shard_plan and shard_plan["applied"] else lane_plan
        shared_preparation = (
            float(shape_plan.get("shared_preparation_seconds") or 0.0)
            if shape_plan and shape_plan["applied"] else global_parent_preparation
        )
        makespan += shared_preparation
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
        and not (lane_plan and lane_plan["applied"])
        and retain_incumbent_schedule
        and incumbent_pipeline_count == selected_pipeline_count
        and len(incumbent_assignments) == len(detector_ids)
        and incumbent_loads
    ):
        incumbent_makespan = global_parent_preparation + max(incumbent_loads.values())
        selected["predicted_makespan_seconds"] = incumbent_makespan
        selected["predicted_pipeline_utilization"] = sum(base_fanout_estimates) / (
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
    selected["detector_golden_set_lane_counts"] = {
        detector: count for detector, count in zip(detector_ids, golden_set_lane_counts)
    }
    selected["golden_set_lane_scaling_applied"] = bool(lane_plan and lane_plan["applied"])
    selected["golden_set_lane_target_seconds"] = DEFAULT_SHARD_TARGET_SECONDS
    selected["golden_set_lane_makespan_improvement"] = (
        float(lane_plan["makespan_improvement"]) if lane_plan else 0.0
    )
    selected["shard_target_seconds"] = DEFAULT_SHARD_TARGET_SECONDS
    selected["unsharded_makespan_seconds"] = (
        float(shard_plan["unsharded_makespan_seconds"]) if shard_plan else (
            float(lane_plan["uncoordinated_makespan_seconds"])
            if lane_plan else selected["predicted_makespan_seconds"]
        )
    )
    selected["sharding_makespan_improvement"] = (
        float(shard_plan["makespan_improvement"]) if shard_plan else 0.0
    )
    if shard_plan:
        selected["sharding_decision_reason"] = str(shard_plan["reason"])
        for source_key, target_key, converter in (
            ("candidate_task_count", "sharding_candidate_task_count", int),
            ("candidate_makespan_seconds", "sharding_candidate_makespan_seconds", float),
            ("candidate_makespan_improvement", "sharding_candidate_makespan_improvement", float),
            (
                "candidate_shared_preparation_seconds",
                "sharding_candidate_shared_preparation_seconds",
                float,
            ),
        ):
            if source_key in shard_plan:
                selected[target_key] = converter(shard_plan[source_key])
    if lane_plan:
        selected["golden_set_lane_decision_reason"] = str(lane_plan["reason"])
        for source_key, target_key, converter in (
            ("candidate_capacity_units", "golden_set_lane_candidate_capacity_units", int),
            ("candidate_makespan_seconds", "golden_set_lane_candidate_makespan_seconds", float),
            (
                "candidate_makespan_improvement",
                "golden_set_lane_candidate_makespan_improvement",
                float,
            ),
        ):
            if source_key in lane_plan:
                selected[target_key] = converter(lane_plan[source_key])
    shape_plan = shard_plan if shard_plan and shard_plan["applied"] else lane_plan
    selected["shared_preparation_seconds"] = (
        float(shape_plan.get("shared_preparation_seconds") or 0.0)
        if shape_plan and shape_plan.get("applied") else global_parent_preparation
    )
    detector_fanout_estimates: dict[str, float] = {}
    for detector, fixed_cost, body_cost, shards, lanes, parent_shared in zip(
        detector_ids, fixed_preparation, shardable_work,
        shard_counts, golden_set_lane_counts, parent_shared_preparation,
    ):
        body = float(body_cost) if body_cost is not None else 0.0
        detector_fanout_estimates[detector] = (
            body if shards > 1
            else (0.0 if parent_shared else fixed_cost) + body / lanes
        )
    selected["detector_fanout_estimates"] = detector_fanout_estimates
    planned_tasks: list[dict[str, Any]] = []
    for detector, total, fixed_cost, body_cost, shards, lanes, parent_shared in zip(
        detector_ids, complete, fixed_preparation, shardable_work,
        shard_counts, golden_set_lane_counts, parent_shared_preparation,
    ):
        body = float(body_cost) if body_cost is not None else max(0.0, total - fixed_cost)
        if shards > 1:
            planned_tasks.extend({
                "detector": detector,
                "shard_index": index,
                "shard_count": shards,
                "golden_set_lanes": 1,
                "estimate_seconds": body / shards,
            } for index in range(shards))
        else:
            planned_tasks.append({
                "detector": detector,
                "shard_index": 0,
                "shard_count": 1,
                "golden_set_lanes": lanes,
                "estimate_seconds": (0.0 if parent_shared else fixed_cost) + body / lanes,
            })
    selected["planned_tasks"] = planned_tasks
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
