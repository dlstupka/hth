from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from hth.regression.performance import PerformanceSampler
from hth.regression.runner import ALLOWED_THREAD_COUNTS, parse_args, print_parameter_scope


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class RegressionParallelismTests(unittest.TestCase):
    def test_threads_default_and_allowed_values(self) -> None:
        base = [
            "--detector-config", "detector.json",
            "--golden-set", "golden.json",
            "--image-root", "images",
            "--output", "output",
        ]
        self.assertEqual(parse_args(base).threads, 1)
        self.assertEqual(ALLOWED_THREAD_COUNTS, (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024))
        self.assertEqual(parse_args([*base, "--threads", "256"]).threads, 256)

    def test_scope_reports_possible_planned_and_page_evaluations(self) -> None:
        stream = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(stream):
            print_parameter_scope(
                strategy="exhaustive",
                possible_sets=212576,
                planned_sets=10,
                golden_pages=5,
                threads=32,
                limit=10,
            )
        lines = stream.getvalue().splitlines()
        scope_rows = lines[2:9]
        separators = {line.index(":") for line in scope_rows}
        self.assertEqual(separators, {25})
        self.assertIn("Possible Parameter Sets  : 212576", scope_rows)
        self.assertIn("Planned Parameter Sets   : 10", scope_rows)
        self.assertIn("Planned Page Evaluations : 50", scope_rows)
        self.assertIn("Parameter-set Limit      : 10 total (including baseline)", lines)
        self.assertIn("Threads                  : 32", scope_rows)

    def test_performance_sampler_writes_thread_and_throughput_sample(self) -> None:
        wall = FakeClock()
        cpu = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runner-performance.jsonl"
            sampler = PerformanceSampler(
                path,
                snapshot=lambda: {
                    "completed_parameter_sets": 12,
                    "parameter_sets_per_second": 3.0,
                    "active_threads": 4,
                    "configured_threads": 8,
                },
                interval_seconds=0,
                clock=wall,
                process_clock=cpu,
            )
            sampler.start()
            wall.value = 2.0
            cpu.value = 8.0
            samples = sampler.finish()
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(samples), 2)
        self.assertEqual(rows[-1]["configured_threads"], 8)
        self.assertEqual(rows[-1]["active_threads"], 4)
        self.assertEqual(rows[-1]["completed_parameter_sets"], 12)
        self.assertEqual(rows[-1]["process_cpu_percent"], 400.0)


if __name__ == "__main__":
    unittest.main()
