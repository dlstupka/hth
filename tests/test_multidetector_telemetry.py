import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from hth.multidetector_store import finalize, publish


class MultiDetectorTelemetryTests(unittest.TestCase):
    def test_finalize_records_worker_task_tail_and_utilization(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            telemetry = root / "telemetry"
            (telemetry / "workers").mkdir(parents=True)
            (telemetry / "tasks").mkdir()
            (telemetry / "batch.tsv").write_text("start\t0\nend\t10\n", encoding="utf-8")
            (telemetry / "workers/1.tsv").write_text("start\t0\nend\t10\n", encoding="utf-8")
            (telemetry / "workers/2.tsv").write_text("start\t0\nend\t10\n", encoding="utf-8")
            (telemetry / "tasks/0.tsv").write_text(
                "claim\t0\t1\ta\t0\t1\t48\nstart\t0\nfinish\t10\tcomplete\n", encoding="utf-8"
            )
            (telemetry / "tasks/1.tsv").write_text(
                "claim\t0\t2\tb\t0\t1\t48\nstart\t0\nfinish\t6\tcomplete\n", encoding="utf-8"
            )
            out = root / "execution.json"
            obs = finalize(Namespace(
                telemetry_root=telemetry, output=out, observation_id="x", github_run_id="1", github_run_number="2",
                mode="smoke", strategy="exhaustive", limit="10", detector_count=2, golden_set_sha256="gold",
                runner_label="192t", runner_name="e9k", runner_thread_budget=96, threads_per_worker=48, allocated_threads=96,
                loading_strategy="lpt", scheduler_source="auto",
            ))
            self.assertAlmostEqual(obs["makespan_seconds"], 10.0)
            self.assertAlmostEqual(obs["worker_utilization"], 0.8)
            self.assertAlmostEqual(obs["final_tail_seconds"], 4.0)
            self.assertAlmostEqual(obs["final_tail_seconds_by_active_workers"]["1"], 4.0)
            self.assertAlmostEqual(obs["workers"][1]["busy_seconds"], 6.0)
            self.assertAlmostEqual(obs["workers"][1]["idle_seconds"], 4.0)
            self.assertEqual(obs["tasks"][0]["allocated_threads"], 48)

            results = root / "results"
            (results / "indexes").mkdir(parents=True)
            (results / "indexes" / "runtime-index.json").write_text(json.dumps({
                "schema_version": 1,
                "observations": [
                    {"observation_id": "ra", "detector_id": "a", "wall_clock_seconds": 9.0, "build": {"github_run_id": "1"}},
                    {"observation_id": "rb", "detector_id": "b", "wall_clock_seconds": 5.0, "build": {"github_run_id": "1"}},
                ],
            }), encoding="utf-8")
            publish(out, results)
            index = json.loads((results / "indexes" / "multidetector-index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["observations"][0]["observation_id"], "x")
            runtime = json.loads((results / "indexes" / "runtime-index.json").read_text(encoding="utf-8"))
            by_detector = {row["detector_id"]: row for row in runtime["observations"]}
            self.assertEqual(by_detector["a"]["scheduler_wall_clock_seconds"], 10.0)
            self.assertEqual(by_detector["b"]["scheduler_wall_clock_seconds"], 10.0)
            self.assertEqual(by_detector["a"]["scheduler_cost_source"], "multidetector-fixed-pipeline-slot")

    def test_scheduler_slots_include_inter_detector_wrapper_overhead(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            telemetry = root / "telemetry"
            (telemetry / "workers").mkdir(parents=True)
            (telemetry / "tasks").mkdir()
            (telemetry / "batch.tsv").write_text("start\t0\nend\t30\n", encoding="utf-8")
            (telemetry / "workers/1.tsv").write_text("start\t0\nend\t30\n", encoding="utf-8")
            (telemetry / "tasks/0.tsv").write_text("claim\t0\t1\ta\t0\t1\t48\nstart\t1\nfinish\t10\tcomplete\n", encoding="utf-8")
            (telemetry / "tasks/1.tsv").write_text("claim\t0\t1\tb\t0\t1\t48\nstart\t15\nfinish\t25\tcomplete\n", encoding="utf-8")
            out = root / "execution.json"
            obs = finalize(Namespace(
                telemetry_root=telemetry, output=out, observation_id="slots", github_run_id="1", github_run_number="2",
                mode="smoke", strategy="exhaustive", limit="10", detector_count=2, golden_set_sha256="gold",
                runner_label="192t", runner_name="e9k", runner_thread_budget=96, threads_per_worker=48, allocated_threads=48,
                loading_strategy="lpt", scheduler_source="auto",
            ))
            self.assertAlmostEqual(obs["tasks"][0]["busy_seconds"], 9.0)
            self.assertAlmostEqual(obs["tasks"][1]["busy_seconds"], 10.0)
            self.assertAlmostEqual(obs["tasks"][0]["scheduler_slot_seconds"], 15.0)
            self.assertAlmostEqual(obs["tasks"][1]["scheduler_slot_seconds"], 15.0)
            self.assertAlmostEqual(sum(t["scheduler_slot_seconds"] for t in obs["tasks"]), 30.0)


if __name__ == "__main__":
    unittest.main()
