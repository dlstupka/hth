from __future__ import annotations

import json
from pathlib import Path

from hth.parallelism_store import update_parallelism_index


def test_parallelism_index_tracks_fastest_shape(tmp_path: Path) -> None:
    update_parallelism_index(tmp_path, [
        {"observation_id": "1", "detector_id": "adaptive_radial_edge", "mode": "full", "strategy": "exhaustive", "possible_parameter_sets": 6562, "actual_parameter_sets": 6562, "shards": 1, "active_pipelines": 1, "threads_per_pipeline": 64, "allocated_threads": 64, "wall_clock_seconds": 2600, "effective_acceleration": 10, "observed_at_utc": "2026-08-06T00:00:00Z"},
        {"observation_id": "2", "detector_id": "adaptive_radial_edge", "mode": "full", "strategy": "exhaustive", "possible_parameter_sets": 6562, "actual_parameter_sets": 6562, "shards": 8, "active_pipelines": 8, "threads_per_pipeline": 8, "allocated_threads": 64, "wall_clock_seconds": 420, "effective_acceleration": 40, "observed_at_utc": "2026-08-06T01:00:00Z"},
    ])
    payload = json.loads((tmp_path / "parallelism-index.json").read_text(encoding="utf-8"))
    assert payload["best"]["adaptive_radial_edge"]["shards"] == 8
    assert payload["best"]["adaptive_radial_edge"]["threads_per_pipeline"] == 8
