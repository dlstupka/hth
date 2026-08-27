from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable

PREFERRED_SHAPE_RATE_DECIMALS = 2
DETERMINISTIC_OPTIMIZER_STRATEGIES = frozenset({
    "exhaustive",
    "exhaustive-with-zombies",
    "non-dormant",
    "low+",
    "moderate+",
    "important+",
    "critical",
})


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_cpu_model(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _freshness_epoch(shape: dict[str, Any]) -> float:
    run_id = str(shape.get("optimizer_run_id") or "").strip()
    if run_id.isdigit():
        return float(run_id)
    stamp = str(shape.get("observed_at_utc") or "").strip()
    if stamp:
        try:
            return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
        except ValueError:
            pass
    return 0.0


def select_preferred_shape(shapes: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Canonical optimizer preference: throughput, freshness, then resources.

    Throughput is compared at the report's visible precision.  When compatible
    characterized optimizer runs are indistinguishable at that precision, the
    newest run supersedes stale evidence.  Resource use remains the tie-break
    within one optimizer run (or when freshness provenance is unavailable).
    """
    candidates = [shape for shape in shapes if isinstance(shape, dict)]
    if not candidates:
        return None

    def rank(shape: dict[str, Any]) -> tuple[float, float, int, int, int, float, int]:
        rate = as_float(shape.get("parameter_sets_per_second"))
        displayed_rate = round(rate, PREFERRED_SHAPE_RATE_DECIMALS) if rate is not None else -math.inf
        allocated = as_int(shape.get("allocated_threads"))
        pipelines = as_int(shape.get("pipelines"))
        threads = as_int(shape.get("threads_per_pipeline"))
        wall = as_float(shape.get("fastest_wall_clock_seconds"))
        sequence = as_int(shape.get("optimizer_shape_sequence"))
        freshness = _freshness_epoch(shape)
        return (
            -displayed_rate,
            -freshness,
            allocated if allocated is not None else math.inf,
            pipelines if pipelines is not None else math.inf,
            threads if threads is not None else math.inf,
            wall if wall is not None else math.inf,
            sequence if sequence is not None else math.inf,
        )
    return min(candidates, key=rank)


def optimizer_row_matches_workload(
    row: dict[str, Any], *, detector: str, detector_sha256: str,
    golden_sha256: str, max_dimension: int,
) -> bool:
    if row.get("source") != "execution-optimizer":
        return False
    if str(row.get("detector_id") or "") != detector:
        return False
    if str(row.get("mode") or "") != "full":
        return False
    if str(row.get("strategy") or "") not in DETERMINISTIC_OPTIMIZER_STRATEGIES:
        return False
    row_detector_sha = str(row.get("detector_config_sha256") or "").strip()
    if row_detector_sha and row_detector_sha != detector_sha256:
        return False
    if str(row.get("golden_set_sha256") or "") != golden_sha256:
        return False
    row_dimension = as_int(row.get("max_dimension"))
    if row_dimension is not None and row_dimension != max_dimension:
        return False
    possible = as_int(row.get("possible_parameter_sets"))
    actual = as_int(row.get("actual_parameter_sets"))
    benchmark = as_int(row.get("optimizer_benchmark_parameter_sets"))
    expected = min(possible, benchmark) if possible is not None and benchmark is not None and benchmark > 0 else possible
    return expected is not None and actual == expected and (as_float(row.get("wall_clock_seconds")) or 0.0) > 0.0


def runner_match_tier(row: dict[str, Any], *, name: str, cpu_model: str,
                      physical_cores: int | None, logical_cpus: int) -> int | None:
    runner = row.get("runner") if isinstance(row.get("runner"), dict) else {}
    row_name = str(runner.get("runner_name") or "").strip()
    if row_name and name and row_name == name:
        return 0
    row_model = normalize_cpu_model(runner.get("cpu_model"))
    row_logical = as_int(runner.get("logical_cpu_count"))
    row_physical = as_int(runner.get("physical_core_count"))
    if row_model and row_model == normalize_cpu_model(cpu_model) and row_logical == logical_cpus:
        if physical_cores is None or row_physical is None or row_physical == physical_cores:
            return 1
    return None
