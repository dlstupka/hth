from __future__ import annotations

from typing import Any


def analyze_peak_bracket(
    observations: list[dict[str, Any]],
    *,
    threshold_pct: float = 2.0,
    pipeline_min: int,
    pipeline_max: int,
    required_consecutive: int = 3,
) -> dict[str, Any]:
    """Assess whether the best measured throughput is a confirmed peak."""
    if required_consecutive < 1:
        raise ValueError("required_consecutive must be at least 1")
    if pipeline_min > pipeline_max:
        pipeline_min, pipeline_max = pipeline_max, pipeline_min
    if not observations:
        return {
            "best": None,
            "best_rate": None,
            "peak_floor": None,
            "left_required": False,
            "right_required": False,
            "left_confirmed": False,
            "right_confirmed": False,
            "left_streak": 0,
            "right_streak": 0,
            "left_witnesses": [],
            "right_witnesses": [],
            "should_stop": False,
        }

    best = max(
        observations,
        key=lambda row: (float(row["parameter_sets_per_second"]), -int(row["pipelines"])),
    )
    best_rate = float(best["parameter_sets_per_second"])
    best_pipeline = int(best["pipelines"])
    peak_floor = best_rate * (1.0 - threshold_pct / 100.0)

    def side_streak(rows: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
        streak = 0
        witnesses: list[dict[str, Any]] = []
        for row in rows:
            if float(row["parameter_sets_per_second"]) < peak_floor:
                streak += 1
                witnesses.append(row)
                if streak >= required_consecutive:
                    break
            else:
                streak = 0
                witnesses = []
        return streak, witnesses

    left_rows = sorted(
        (row for row in observations if int(row["pipelines"]) < best_pipeline),
        key=lambda row: int(row["pipelines"]),
        reverse=True,
    )
    right_rows = sorted(
        (row for row in observations if int(row["pipelines"]) > best_pipeline),
        key=lambda row: int(row["pipelines"]),
    )
    left_streak, left_witnesses = side_streak(left_rows)
    right_streak, right_witnesses = side_streak(right_rows)

    left_required = best_pipeline > pipeline_min
    right_required = best_pipeline < pipeline_max
    left_confirmed = (not left_required) or left_streak >= required_consecutive
    right_confirmed = (not right_required) or right_streak >= required_consecutive

    return {
        "best": best,
        "best_rate": best_rate,
        "best_pipeline": best_pipeline,
        "peak_floor": peak_floor,
        "left_required": left_required,
        "right_required": right_required,
        "left_confirmed": left_confirmed,
        "right_confirmed": right_confirmed,
        "left_streak": left_streak,
        "right_streak": right_streak,
        "left_witnesses": left_witnesses,
        "right_witnesses": right_witnesses,
        "should_stop": left_confirmed and right_confirmed,
    }
