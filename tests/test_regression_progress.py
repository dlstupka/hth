from __future__ import annotations

import io
import unittest

from hth.regression.progress import ProgressReporter


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class RegressionProgressTests(unittest.TestCase):
    def test_header_zero_heartbeat_minute_cadence_and_milestones(self) -> None:
        clock = FakeClock()
        stream = io.StringIO()
        reporter = ProgressReporter(
            total=1000,
            interval_seconds=60,
            stream=stream,
            clock=clock,
            eta_min_completed=2,
            eta_min_elapsed_seconds=60,
        )
        reporter.start()
        initial = stream.getvalue()
        self.assertIn("Best Mean IoU", initial)
        self.assertIn("Worst IoU", initial)
        self.assertIn("Last Improvement", initial)
        self.assertIn("Evaluating", initial)
        self.assertNotIn("Current Profile", initial)
        self.assertIn("0/1000", initial)
        self.assertIn("estimating...", initial)

        first = {
            "parameter_set_id": "abcdef123456",
            "summary": {"mean_iou": 0.8, "minimum_iou": 0.6, "failure_count": 0},
        }
        clock.value = 30
        reporter.begin_evaluation("baseline")
        reporter.observe(first, "baseline")
        self.assertNotIn("00:00:30 >>>", stream.getvalue())

        clock.value = 60
        reporter.begin_evaluation("ps:feedface")
        reporter.emit()
        text = stream.getvalue()
        self.assertIn("1/1000 0.10%", text)
        self.assertIn("estimating...", text)
        self.assertIn("ps:feedface", text)

        clock.value = 61
        better = {
            "parameter_set_id": "1234567890ab",
            "summary": {"mean_iou": 0.9, "minimum_iou": 0.7, "failure_count": 0},
        }
        reporter.observe(better)
        text = stream.getvalue()
        self.assertIn("00:01:01 >>> New best mean IoU ps:12345678", text)
        self.assertIn("00:01:01 >>> New worst-page IoU ps:12345678", text)
        self.assertIn("00:01:01 >>> Baseline surpassed ps:12345678", text)
        milestone_rows = [line for line in text.splitlines() if line.startswith("00:01:01") and ">>>" not in line]
        self.assertEqual(len(milestone_rows), 1)

    def test_worst_page_can_improve_without_mean(self) -> None:
        clock = FakeClock()
        stream = io.StringIO()
        reporter = ProgressReporter(total=10, interval_seconds=60, stream=stream, clock=clock)
        reporter.start()
        reporter.observe({
            "parameter_set_id": "aaaaaaaa",
            "summary": {"mean_iou": 0.9, "minimum_iou": 0.5, "failure_count": 0},
        })
        clock.value = 1
        reporter.observe({
            "parameter_set_id": "bbbbbbbb",
            "summary": {"mean_iou": 0.85, "minimum_iou": 0.6, "failure_count": 0},
        })
        text = stream.getvalue()
        self.assertIn("New worst-page IoU ps:bbbbbbbb", text)
        self.assertNotIn("New best mean IoU ps:bbbbbbbb", text)
        self.assertAlmostEqual(reporter.snapshot().best_mean_iou or 0, 0.9)
        self.assertAlmostEqual(reporter.snapshot().worst_page_iou or 0, 0.6)


if __name__ == "__main__":
    unittest.main()
