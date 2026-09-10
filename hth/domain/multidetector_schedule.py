from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MAX_FLOOR_OVERRUN = 0.20


def plan_lpt_workers(detector_count: int, runner_thread_budget: int) -> int:
    """Bootstrap at one pipeline per vCPU, bounded by available tasks."""
    detectors = max(1, int(detector_count))
    budget = max(1, int(runner_thread_budget))
    max_pipelines = max(1, budget // 2)
    return min(detectors, max_pipelines)


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


def select_lpt_pipeline_count(
    estimates: list[float | int | None], max_pipelines: int,
) -> int:
    """Choose the smallest LPT shape within 20% of its longest-task floor."""
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
        if makespan <= floor_seconds * (1.0 + MAX_FLOOR_OVERRUN):
            return pipelines
    return ceiling


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
    rows: list[dict[str, Any]], *, mode: str, strategy: str,
    max_dimension: int, golden_set_sha256: str | None, runner_label: str,
) -> float | None:
    candidates: list[tuple[int, float, str, float]] = []
    for row in rows:
        seconds = _as_float(row.get("scheduler_wall_clock_seconds"))
        if seconds is None:
            seconds = _as_float(row.get("wall_clock_seconds"))
        if seconds is None or seconds <= 0:
            continue
        context = _runtime_context_score(
            row, mode=mode, strategy=strategy, max_dimension=max_dimension,
            golden_set_sha256=golden_set_sha256, runner_label=runner_label,
        )
        candidates.append((context, 0.0, str(row.get("observed_at_utc") or ""), seconds))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def optimize_lpt_schedule(
    *, runtime_index_path: Path | None, detector_ids: list[str],
    runner_thread_budget: int, runner_label: str, golden_set_sha256: str | None,
    mode: str, strategy: str, max_dimension: int,
    completion_index_path: Path | None = None,
) -> dict[str, Any] | None:
    """Jointly choose worker count and its deterministic LPT assignment.

    Measured scheduler-facing detector runtimes are assigned by the executable
    static LPT planner.  The longest detector is the unavoidable wall-time
    floor; choose the smallest pipeline count whose LPT makespan is within 20%
    of that floor, maximizing sustained pipeline occupation without inventing
    an unmeasured thread-scaling curve.
    """
    payload = _read_index(runtime_index_path)
    observations = [row for row in payload.get("observations", []) if isinstance(row, dict)]
    if not observations or not detector_ids:
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

    coherent: list[tuple[bool, str, str, list[dict[str, Any]]]] = []
    for build_id, by_id in builds.items():
        if set(by_id) != wanted:
            continue
        rows = list(by_id.values())
        latest = max(str(row.get("observed_at_utc") or "") for row in rows)
        exact_golden = bool(golden_set_sha256) and all(
            str(row.get("golden_set_sha256") or "") == str(golden_set_sha256)
            for row in rows
        )
        coherent.append((exact_golden, latest, build_id, rows))
    if not coherent:
        return None
    # Prefer the requested Golden Set once it has a valid aggregate completion.
    # Cross-Golden evidence is strictly a bootstrap fallback.
    exact = [item for item in coherent if item[0]]
    selected = max(exact or coherent, key=lambda item: (item[1], item[2]))
    selected_exact_golden, _, evidence_build_id, observations = selected

    by_detector: dict[str, list[dict[str, Any]]] = {}
    for row in observations:
        detector = str(row.get("detector_id") or "")
        if detector:
            by_detector.setdefault(detector, []).append(row)

    budget = max(1, int(runner_thread_budget))
    candidates: list[dict[str, Any]] = []
    max_pipelines = plan_lpt_workers(len(detector_ids), budget)
    measured = [
        _predicted_runtime(
            by_detector.get(detector, []), mode=mode, strategy=strategy,
            max_dimension=max_dimension, golden_set_sha256=golden_set_sha256,
            runner_label=runner_label,
        )
        for detector in detector_ids
    ]
    known = [value for value in measured if value is not None]
    if not known:
        return None
    unknown_estimate = max(known)
    complete = [value if value is not None else unknown_estimate for value in measured]
    floor_seconds = max(complete)
    selected_pipeline_count = select_lpt_pipeline_count(complete, max_pipelines)
    for pipelines in range(1, max_pipelines + 1):
        threads = max(1, budget // pipelines)
        schedule = plan_static_lpt_tasks(complete, pipelines)
        makespan = max(float(row["estimated_seconds"]) for row in schedule)
        utilization = sum(complete) / (pipelines * makespan)
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
        })
    if not candidates:
        return None
    selected = next(row for row in candidates if int(row["pipelines"]) == selected_pipeline_count)
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
    }
