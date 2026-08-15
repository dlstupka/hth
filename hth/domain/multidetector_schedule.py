from __future__ import annotations

import math

MIN_THREADS_PER_LPT_WORKER = 48


def plan_lpt_workers(detector_count: int, runner_thread_budget: int) -> int:
    """Choose equal workers for one aggregate LPT detector queue."""
    detectors = max(1, int(detector_count))
    budget = max(1, int(runner_thread_budget))
    queue_target = max(1, round(math.sqrt(detectors)))
    budget_cap = max(1, budget // MIN_THREADS_PER_LPT_WORKER)
    return min(detectors, queue_target, budget_cap)
