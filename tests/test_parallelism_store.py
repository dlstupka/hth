from __future__ import annotations

import json
from pathlib import Path

from hth.parallelism_store import observation_from_run, update_parallelism_index


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
    assert payload["schema_version"] == "2.1"
    assert payload["best"]["adaptive_radial_edge"]["shards"] == 8
    assert payload["best_by_compatibility"]["compatible-workload"]["threads_per_pipeline"] == 8
    summary = next(row for row in payload["shape_summaries"] if row["execution_shape"] == "8p/8s/8t")
    assert summary["observation_count"] == 2
    assert summary["fastest_wall_clock_seconds"] == 420
    assert summary["median_wall_clock_seconds"] == 430


def _parallel_run(path: Path, detector: str, run_id: str) -> Path:
    (path / "reports").mkdir(parents=True)
    (path / "RUN-INFO.json").write_text(json.dumps({
        "run_id": run_id,
        "detector": detector,
        "elapsed_seconds": 5.0,
        "threads": 2,
        "possible_parameter_sets": 10,
        "actual_parameter_sets": 10,
        "golden_set_sha256": "gold",
        "strategy": "exhaustive",
        "detector_pipeline": {"pipeline_count": 4},
    }), encoding="utf-8")
    (path / "reports" / "summary.json").write_text(json.dumps({
        "detector": detector,
        "parameter_space": {"possible_parameter_sets": 10, "actual_parameter_sets": 10},
        "runner": {},
    }), encoding="utf-8")
    return path


def test_parallelism_observation_ids_include_detector_identity(tmp_path: Path) -> None:
    first = _parallel_run(tmp_path / "a", "grabcut", "run-20260807-120000")
    second = _parallel_run(tmp_path / "b", "contour", "run-20260807-120000")
    build = {"github_run_id": "243", "mode": "full", "runner_label": "e7k"}
    a = observation_from_run(first, build=build)
    b = observation_from_run(second, build=build)
    assert a["observation_id"] != b["observation_id"]
