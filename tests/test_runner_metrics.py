from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hth.runner_metrics import summarize_runner_metrics


class RunnerMetricsTests(unittest.TestCase):
    def test_runner_metrics_summary_is_shape_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "metrics.jsonl"
            rows = [
                {"optimizer_run_id": "1", "shape_sequence": 1, "load1": 10.0, "cpu_pct": 50.0, "iowait_pct": 1.0, "ram_used_bytes": 100.0, "swap_used_bytes": 0.0},
                {"optimizer_run_id": "1", "shape_sequence": 1, "load1": 20.0, "cpu_pct": 70.0, "iowait_pct": 2.0, "ram_used_bytes": 200.0, "swap_used_bytes": 0.0},
                {"optimizer_run_id": "1", "shape_sequence": 2, "load1": 999.0, "cpu_pct": 99.0, "iowait_pct": 9.0, "ram_used_bytes": 999.0, "swap_used_bytes": 0.0},
            ]
            log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            summary = summarize_runner_metrics(log, optimizer_run_id="1", shape_sequence=1)
            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(summary["avg_load1"], 15.0)
            self.assertEqual(summary["peak_load1"], 20.0)
            self.assertEqual(summary["avg_cpu_pct"], 60.0)
            self.assertEqual(summary["peak_ram_used_bytes"], 200.0)


if __name__ == "__main__":
    unittest.main()
