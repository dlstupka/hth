"""Canonical fixed task-to-pipeline dispatcher for regression execution."""
from __future__ import annotations

from typing import Any

from hth.domain.multidetector_schedule import plan_static_lpt_tasks


def plan_static_dispatch(
    *,
    task_count: int,
    pipeline_count: int,
    multidetector: bool,
    estimates: list[float | int | None] | None = None,
    estimate_floor_seconds: float = 0.1,
) -> list[dict[str, Any]]:
    """Return the one canonical fixed schedule consumed by workers and reports.

    Multi-detector work uses deterministic LPT balancing. Sharded single-detector
    work is distributed round-robin so every requested active pipeline receives
    work. True single-pipeline execution naturally collapses to pipeline 1.
    """
    tasks = max(0, int(task_count))
    workers = max(1, int(pipeline_count))
    if tasks == 0:
        return []

    if multidetector and workers > 1:
        values = list(estimates or [])
        if len(values) < tasks:
            values.extend([None] * (tasks - len(values)))
        return plan_static_lpt_tasks(
            values[:tasks],
            workers,
            estimate_floor_seconds=estimate_floor_seconds,
        )

    schedules = [
        {"pipeline": pipeline + 1, "task_indexes": [], "estimated_seconds": 0.0}
        for pipeline in range(workers)
    ]
    for task_index in range(tasks):
        schedules[task_index % workers]["task_indexes"].append(task_index)
    return [row for row in schedules if row["task_indexes"]]
