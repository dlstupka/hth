from __future__ import annotations

import json
from pathlib import Path

from hth.parallelism_store import update_parallelism_index


def _observation(identifier: str, *, pipelines: int, shards: int, threads: int, wall: float) -> dict:
    allocated = pipelines * threads
    return {
        "observation_id": identifier,
        "detector_id": "adaptive_radial_edge",
        "mode": "full",
        "strategy": "exhaustive",
        "possible_parameter_sets": 6562,
        "actual_parameter_sets": 6562,
        "compatibility_key": "compatible-workload",
        "execution_shape": f"{pipelines}p/{shards}s/{threads}t",
        "shards": shards,
        "active_pipelines": pipelines,
        "threads_per_pipeline": threads,
        "allocated_threads": allocated,
        "wall_clock_seconds": wall,
        "effective_acceleration": 40,
        "parallel_efficiency": 40 / allocated,
        "observed_at_utc": f"2026-08-06T0{identifier}:00:00Z",
    }


def test_parallelism_index_tracks_fastest_compatible_shape_and_summaries(tmp_path: Path) -> None:
    update_parallelism_index(tmp_path, [
        _observation("1", pipelines=1, shards=1, threads=64, wall=2600),
        _observation("2", pipelines=8, shards=8, threads=8, wall=420),
        _observation("3", pipelines=8, shards=8, threads=8, wall=440),
    ])
    payload = json.loads((tmp_path / "parallelism-index.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0"
    assert payload["best"]["adaptive_radial_edge"]["shards"] == 8
    assert payload["best_by_compatibility"]["compatible-workload"]["threads_per_pipeline"] == 8
    summary = next(row for row in payload["shape_summaries"] if row["execution_shape"] == "8p/8s/8t")
    assert summary["observation_count"] == 2
    assert summary["fastest_wall_clock_seconds"] == 420
    assert summary["median_wall_clock_seconds"] == 430
