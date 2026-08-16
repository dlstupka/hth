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
            publish(out, results)
            index = json.loads((results / "multidetector-index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["observations"][0]["observation_id"], "x")


if __name__ == "__main__":
    unittest.main()
