from __future__ import annotations

import json
import tempfile
import unittest
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
        "compatibility_key": "stable-evidence-and-runner",
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


def _parallel_run(path: Path, detector: str, run_id: str, *, strategy: str = "exhaustive", limit: int | None = None) -> Path:
    (path / "reports").mkdir(parents=True)
    (path / "RUN-INFO.json").write_text(json.dumps({
        "run_id": run_id,
        "detector": detector,
        "elapsed_seconds": 5.0,
        "threads": 2,
        "possible_parameter_sets": 10,
        "actual_parameter_sets": 10,
        "golden_set_sha256": "gold",
        "detector_config_sha256": "detector-sha",
        "max_dimension": 1800,
        "strategy": strategy,
        "detector_pipeline": {"pipeline_count": 4},
    }), encoding="utf-8")
    (path / "parameters.json").write_text(json.dumps({
        "strategy": strategy,
        "limit": limit,
        "max_dimension": 1800,
    }), encoding="utf-8")
    (path / "reports" / "summary.json").write_text(json.dumps({
        "detector": detector,
        "parameter_space": {"possible_parameter_sets": 10, "actual_parameter_sets": 10},
        "runner": {"runner_name": "rh8-al321", "logical_cpu_count": 192, "cpu_model": "AMD EPYC"},
    }), encoding="utf-8")
    return path


class ParallelismStoreTests(unittest.TestCase):
    def test_parallelism_index_tracks_fastest_compatible_shape_and_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            update_parallelism_index(root, [
                _observation("1", pipelines=1, shards=1, threads=64, wall=2600),
                _observation("2", pipelines=8, shards=8, threads=8, wall=420),
                _observation("3", pipelines=8, shards=8, threads=8, wall=440),
            ])
            payload = json.loads((root / "indexes" / "parallelism-index.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "2.3")
            self.assertEqual(payload["best"]["adaptive_radial_edge"]["shards"], 8)
            self.assertEqual(payload["best_by_compatibility"]["stable-evidence-and-runner"]["threads_per_pipeline"], 8)
            summary = next(row for row in payload["shape_summaries"] if row["execution_shape"] == "8p/8s/8t")
            self.assertEqual(summary["observation_count"], 2)
            self.assertEqual(summary["fastest_wall_clock_seconds"], 420)
            self.assertEqual(summary["median_wall_clock_seconds"], 430)

    def test_search_scope_is_informational_and_observation_ids_include_detector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = _parallel_run(root / "a", "grabcut", "run-20260807-120000", strategy="critical", limit=10)
            second = _parallel_run(root / "b", "contour", "run-20260807-120000", strategy="exhaustive", limit=None)
            build = {"github_run_id": "243", "mode": "full", "runner_label": "192t"}
            a = observation_from_run(first, build=build)
            b = observation_from_run(second, build=build)
            self.assertNotEqual(a["observation_id"], b["observation_id"])
            self.assertEqual(a["search_scope"]["strategy"], "critical")
            self.assertEqual(a["search_scope"]["limit"], 10)
            self.assertNotIn("workload_key", a)
            self.assertIn("evidence_key", a)

            # For the same detector/evidence/runner, search scope changes do not alter
            # the execution-shape compatibility identity.
            same_detector = _parallel_run(root / "c", "grabcut", "run-20260807-120001", strategy="exhaustive", limit=None)
            c = observation_from_run(same_detector, build=build)
            self.assertEqual(a["compatibility_key"], c["compatibility_key"])
            self.assertEqual(a["evidence_key"], c["evidence_key"])
            self.assertNotEqual(a["search_scope"], c["search_scope"])


if __name__ == "__main__":
    unittest.main()
