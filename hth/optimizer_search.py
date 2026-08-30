from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from hth.optimizer_peak import analyze_peak_bracket


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


def legal_pipelines(low: int, high: int, budget: int, thread_min: int, *, allow_oversubscription: bool = False) -> list[int]:
    upper = high if allow_oversubscription else min(high, budget // thread_min)
    return list(range(low, upper + 1)) if low <= upper else []


def powers_of_two_pipelines(low: int, high: int, budget: int, thread_min: int, *, allow_oversubscription: bool = False) -> list[int]:
    legal = legal_pipelines(low, high, budget, thread_min, allow_oversubscription=allow_oversubscription)
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
    near_best_fraction: float = 0.98,
    allow_oversubscription: bool = False,
    start_pipeline: int | None = None,
) -> int | None:
    """Return the next pipeline count for a sparse peak/plateau search.

    When optimizer intelligence supplies a starting pipeline count, adaptive
    begins there and expands/refines outward across the full legal range.  With
    no starting hint it preserves the historical common-shape probing behavior.
    The product is a preferred near-best execution region, not a complete curve.
    """
    legal = legal_pipelines(low, high, budget, thread_min, allow_oversubscription=allow_oversubscription)
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

    if start_pipeline is not None:
        # A persisted measured/predicted shape is only a seed, never a bound.
        # Clamp it to the legal search domain and measure it first; subsequent
        # adaptive probes are free to move all the way to either legal edge.
        start = min(legal, key=lambda value: (abs(value - int(start_pipeline)), value))
        if start in untested:
            return start
    else:
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
    best_pipeline = min(
        (pipeline for pipeline, rate in measured.items() if rate == best_rate),
        default=max(measured, key=measured.get),
    )

    # Once the perceived peak is geometrically bracketed (or lies on a legal
    # boundary), peak confirmation requires three consecutive measured shapes
    # on every available side to be strictly >2% below that best throughput.
    # Continue with the nearest unmeasured shape on any unresolved side; one
    # isolated degradation witness must never terminate an adaptive run.
    lower_of_best = [p for p in tested if p < best_pipeline]
    upper_of_best = [p for p in tested if p > best_pipeline]
    geometrically_bracketed = (
        (best_pipeline == legal[0] or bool(lower_of_best))
        and (best_pipeline == legal[-1] or bool(upper_of_best))
    )
    if geometrically_bracketed:
        bracket = analyze_peak_bracket(
            [
                {"pipelines": pipeline, "parameter_sets_per_second": rate}
                for pipeline, rate in measured.items()
            ],
            threshold_pct=(1.0 - near_best_fraction) * 100.0,
            pipeline_min=legal[0],
            pipeline_max=legal[-1],
            required_consecutive=3,
        )
        unresolved: list[tuple[int, int]] = []
        if bracket["left_required"] and not bracket["left_confirmed"]:
            left_untested = [p for p in untested if p < best_pipeline]
            if left_untested:
                candidate = max(left_untested)
                unresolved.append((abs(best_pipeline - candidate), candidate))
        if bracket["right_required"] and not bracket["right_confirmed"]:
            right_untested = [p for p in untested if p > best_pipeline]
            if right_untested:
                candidate = min(right_untested)
                unresolved.append((abs(best_pipeline - candidate), candidate))
        if unresolved:
            return min(unresolved, key=lambda item: (item[0], item[1]))[1]
        if bracket["should_stop"]:
            return None

    # Historical/predicted intelligence may choose the first shape to measure,
    # but only measurements from this execution can resolve the search.  If the
    # current near-best region still touches the lowest/highest sampled edge,
    # measure the true legal edge before allowing adaptive completion.  Without
    # this rule a descending search can stop at 2p while 1p is still legal and
    # potentially faster (the ScanTailor regression that exposed this case).
    if near_low == tested[0] and legal[0] < near_low and legal[0] in untested:
        return legal[0]
    if near_high == tested[-1] and near_high < legal[-1] and legal[-1] in untested:
        return legal[-1]

    # Do not declare a tiny measured bracket resolved while an interior integer
    # pipeline count is still unknown.  Sparse endpoint sampling can hide a
    # narrow local peak (for example 1p and 3p can be equal while 2p is better).
    # Exhausting gaps of at most three pipeline counts is cheap and closes that
    # ambiguity before the broader adaptive stopping rules are allowed to fire.
    measured_pipelines = sorted(measured)
    for left, right in zip(measured_pipelines, measured_pipelines[1:]):
        if (
            right - left <= 3
            and measured[left] >= best_rate * near_best_fraction
            and measured[right] >= best_rate * near_best_fraction
        ):
            interior = sorted(p for p in untested if left < p < right)
            if interior:
                target = (left + right) / 2.0
                return min(interior, key=lambda p: (abs(p - target), p))

    # Once the current best is bracketed by completed measurements on both
    # sides, switch from coarse bracketing to exact local refinement.  The
    # preferred-shape report uses a <=2% band, so adaptive must measure the
    # immediate neighbors around the near-best region and continue outward
    # until each side has a completed shape outside that band.  This makes the
    # reported shape range evidence-based instead of stopping at a sparse peak.
    lower_tested = [p for p in tested if p < best_pipeline]
    upper_tested = [p for p in tested if p > best_pipeline]
    if lower_tested and upper_tested:
        legal_set = set(legal)
        threshold = best_rate * near_best_fraction

        # Grow the measured near-best region contiguously from the best shape.
        left = best_pipeline
        while left - 1 in legal_set:
            candidate = left - 1
            if candidate not in measured:
                break
            if measured[candidate] < threshold:
                break
            left = candidate

        right = best_pipeline
        while right + 1 in legal_set:
            candidate = right + 1
            if candidate not in measured:
                break
            if measured[candidate] < threshold:
                break
            right = candidate

        boundary_candidates: list[tuple[int, int]] = []

        left_candidate = left - 1
        if left_candidate in legal_set and left_candidate not in measured:
            boundary_candidates.append((abs(best_pipeline - left_candidate), left_candidate))

        right_candidate = right + 1
        if right_candidate in legal_set and right_candidate not in measured:
            boundary_candidates.append((abs(best_pipeline - right_candidate), right_candidate))

        if boundary_candidates:
            # Always resolve the nearest edge of the <=2% region first.  For
            # an isolated peak at 8p this guarantees 7p and 9p are sampled
            # before searching farther away.
            return min(boundary_candidates, key=lambda item: (item[0], item[1]))[1]

        # Both immediate outer boundaries are already measured (or the legal
        # range itself is the boundary), so the local <=2% shape range is
        # resolved and no additional adaptive samples are required.
        return None

    # The best point is still on an unbracketed edge.  Continue coarse
    # logarithmic narrowing toward it until completed measurements exist on
    # both sides, then the local refinement above takes over.
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
        return max(candidates, key=lambda item: (item[0], -item[1]))[1]

    return None


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("exhaustive", "powers-of-2", "adaptive"), required=True)
    parser.add_argument("--pipeline-min", type=int, required=True)
    parser.add_argument("--pipeline-max", type=int, required=True)
    parser.add_argument("--runner-budget", type=int, required=True)
    parser.add_argument("--thread-min", type=int, required=True)
    parser.add_argument("--observation-log", type=Path)
    parser.add_argument("--start-pipeline", type=int)
    parser.add_argument("--allow-oversubscription", action="store_true")
    args = parser.parse_args()

    if args.mode == "exhaustive":
        print(",".join(map(str, legal_pipelines(args.pipeline_min, args.pipeline_max, args.runner_budget, args.thread_min, allow_oversubscription=args.allow_oversubscription))))
        return 0
    if args.mode == "powers-of-2":
        print(",".join(map(str, powers_of_two_pipelines(args.pipeline_min, args.pipeline_max, args.runner_budget, args.thread_min, allow_oversubscription=args.allow_oversubscription))))
        return 0

    candidate = adaptive_next_pipeline(
        args.pipeline_min,
        args.pipeline_max,
        args.runner_budget,
        args.thread_min,
        _read_observations(args.observation_log),
        allow_oversubscription=args.allow_oversubscription,
        start_pipeline=args.start_pipeline,
    )
    print("" if candidate is None else candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
