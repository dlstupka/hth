from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_observations(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _pipeline_from_row(row: dict[str, Any]) -> int | None:
    for key in ("active_pipelines", "pipelines", "shards"):
        try:
            value = int(row.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    shape = str(row.get("execution_shape") or "")
    if "p/" in shape:
        try:
            return int(shape.split("p/", 1)[0])
        except ValueError:
            return None
    return None


def _rate_from_row(row: dict[str, Any]) -> float | None:
    try:
        value = float(row.get("parameter_sets_per_second"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def legal_pipelines(low: int, high: int, budget: int, thread_min: int) -> list[int]:
    upper = min(high, budget // thread_min)
    return list(range(low, upper + 1)) if low <= upper else []


def powers_of_two_pipelines(low: int, high: int, budget: int, thread_min: int) -> list[int]:
    legal = legal_pipelines(low, high, budget, thread_min)
    if not legal:
        return []
    upper = legal[-1]
    values = {legal[0], upper}
    power = 1
    while power <= upper:
        if power >= legal[0]:
            values.add(power)
        power *= 2
    return sorted(values)


def _common_shapes(legal: list[int], budget: int) -> list[int]:
    """Prefer shapes that divide the runner budget cleanly.

    These are stable, unsurprising comparison points (no idle thread remainder).
    Bounds are used when no such shape exists in the requested range.
    """
    return [pipeline for pipeline in legal if budget % pipeline == 0]


def _midpoint_candidate(left: int, right: int, untested: set[int]) -> int | None:
    candidates = [value for value in untested if left < value < right]
    if not candidates:
        return None
    # Geometric midpoint narrows wide pipeline ranges quickly while preserving
    # resolution near the thin-pipeline end of the search.
    target = math.sqrt(left * right)
    return min(candidates, key=lambda value: (abs(math.log(value) - math.log(target)), value))


def adaptive_next_pipeline(
    low: int,
    high: int,
    budget: int,
    thread_min: int,
    observations: list[dict[str, Any]],
    *,
    near_best_fraction: float = 0.99,
) -> int | None:
    """Return the next pipeline count for a sparse peak/plateau search.

    The search starts at the lowest and highest cleanly divisible/common shapes
    when available, then repeatedly refines the interval around the best known
    throughput.  It is intentionally heuristic: the optimizer's product is a
    preferred near-best execution region, not a complete curve.
    """
    legal = legal_pipelines(low, high, budget, thread_min)
    if not legal:
        return None

    measured: dict[int, float] = {}
    for row in observations:
        pipeline = _pipeline_from_row(row)
        rate = _rate_from_row(row)
        if pipeline in legal and rate is not None:
            measured[pipeline] = rate

    untested = set(legal) - set(measured)
    if not untested:
        return None

    common = _common_shapes(legal, budget)
    low_probe = common[0] if common else legal[0]
    high_probe = common[-1] if common else legal[-1]
    if low_probe in untested:
        return low_probe
    if high_probe in untested:
        return high_probe

    if not measured:
        return min(untested)

    best_rate = max(measured.values())
    near_best = sorted(
        pipeline for pipeline, rate in measured.items()
        if rate >= best_rate * near_best_fraction
    )
    if not near_best:
        near_best = [max(measured, key=measured.get)]

    tested = sorted(measured)
    near_low, near_high = near_best[0], near_best[-1]

    # First preference: refine immediately beside the near-best region.  This
    # brackets a broad plateau without wasting runs in already-inferior space.
    intervals: list[tuple[int, int]] = []
    lower_tested = [p for p in tested if p < near_low]
    upper_tested = [p for p in tested if p > near_high]
    if lower_tested:
        intervals.append((max(lower_tested), near_low))
    elif legal[0] < near_low:
        intervals.append((legal[0], near_low))
    if upper_tested:
        intervals.append((near_high, min(upper_tested)))
    elif near_high < legal[-1]:
        intervals.append((near_high, legal[-1]))

    candidates: list[tuple[int, int]] = []
    for left, right in intervals:
        candidate = _midpoint_candidate(left, right, untested)
        if candidate is not None:
            candidates.append((right - left, candidate))
    if candidates:
        # Refine the widest unresolved side first.
        return max(candidates, key=lambda item: (item[0], -item[1]))[1]

    # If the near-best interval itself is sparse, fill its largest internal gap.
    boundary = sorted(set(near_best))
    internal: list[tuple[int, int]] = []
    for left, right in zip(boundary, boundary[1:]):
        candidate = _midpoint_candidate(left, right, untested)
        if candidate is not None:
            internal.append((right - left, candidate))
    if internal:
        return max(internal, key=lambda item: (item[0], -item[1]))[1]

    return None


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("exhaustive", "powers-of-2", "adaptive"), required=True)
    parser.add_argument("--pipeline-min", type=int, required=True)
    parser.add_argument("--pipeline-max", type=int, required=True)
    parser.add_argument("--runner-budget", type=int, required=True)
    parser.add_argument("--thread-min", type=int, required=True)
    parser.add_argument("--observation-log", type=Path)
    args = parser.parse_args()

    if args.mode == "exhaustive":
        print(",".join(map(str, legal_pipelines(args.pipeline_min, args.pipeline_max, args.runner_budget, args.thread_min))))
        return 0
    if args.mode == "powers-of-2":
        print(",".join(map(str, powers_of_two_pipelines(args.pipeline_min, args.pipeline_max, args.runner_budget, args.thread_min))))
        return 0

    candidate = adaptive_next_pipeline(
        args.pipeline_min,
        args.pipeline_max,
        args.runner_budget,
        args.thread_min,
        _read_observations(args.observation_log),
    )
    print("" if candidate is None else candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
