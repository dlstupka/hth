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
    def test_eta_starts_after_first_completion_and_columns_align(self) -> None:
        clock = FakeClock()
        stream = io.StringIO()
        reporter = ProgressReporter(total=10, interval_seconds=60, stream=stream, clock=clock)
        reporter.start()

        lines = stream.getvalue().splitlines()
        header_top = next(line for line in lines if line.startswith("Elapsed"))
        header_bottom = lines[lines.index(header_top) + 1]
        initial = lines[-1]
        self.assertIn("TBD", initial)
        self.assertIn("0/10", initial)
        self.assertIn("0.0%", initial)
        self.assertNotIn("estimating", initial)

        clock.value = 60
        reporter.begin_evaluation("baseline")
        reporter.observe_baseline({
            "parameter_set_id": "abcdef123456",
            "summary": {"mean_iou": 0.8, "minimum_iou": 0.6, "stddev_iou": 0.061, "failure_count": 0, "wall_ms": 12.3},
        })
        baseline_row = stream.getvalue().splitlines()[-1]
        self.assertIn("TBD", baseline_row)
        self.assertIn("0/10", baseline_row)
        self.assertIn("0.0%", baseline_row)
        self.assertIn("0.8000", baseline_row)

        clock.value = 120
        reporter.begin_evaluation("ps:12345678")
        reporter.observe({
            "parameter_set_id": "123456789abc",
            "summary": {"mean_iou": 0.79, "minimum_iou": 0.59, "stddev_iou": 0.062, "failure_count": 0, "wall_ms": 18.7},
        })
        reporter.emit(force=True)
        row = stream.getvalue().splitlines()[-1]
        self.assertIn("00:09:00", row)
        self.assertIn("1/10", row)
        self.assertIn("10.0%", row)
        self.assertIn("18.7ms", row)

        self.assertEqual(header_top, ProgressReporter.HEADER_TOP)
        self.assertEqual(header_bottom, ProgressReporter.HEADER_BOTTOM)
        self.assertEqual(
            row,
            "00:02:00  00:09:00  1/10       10.0%  0.017/s   0.7900   0.8000   "
            "0.5900   0.6000   0.0620   0.0610     0     18.7ms  12345678",
        )

    def test_milestones_and_final_row_clear_evaluating(self) -> None:
        clock = FakeClock()
        stream = io.StringIO()
        reporter = ProgressReporter(total=2, interval_seconds=60, stream=stream, clock=clock)
        reporter.start()
        reporter.begin_evaluation("baseline")
        reporter.observe_baseline({
            "parameter_set_id": "aaaaaaaa",
            "summary": {"mean_iou": 0.8, "minimum_iou": 0.5, "stddev_iou": 0.10, "failure_count": 0},
        })
        clock.value = 61
        reporter.begin_evaluation("ps:bbbbbbbb")
        reporter.observe({
            "parameter_set_id": "bbbbbbbb",
            "summary": {"mean_iou": 0.9, "minimum_iou": 0.6, "stddev_iou": 0.08, "failure_count": 0},
        })
        text = stream.getvalue()
        self.assertNotIn("\n\n00:01:01 >>>", text)
        self.assertNotIn("bbbbbbbb\n\n", text)
        self.assertIn("00:01:01 >>> New best average page IoU bbbbbbbb", text)
        self.assertIn("00:01:01 >>> New minimum page IoU bbbbbbbb", text)
        self.assertIn("00:01:01 >>> Baseline surpassed bbbbbbbb", text)

        before_finish = stream.getvalue()
        reporter.finish()
        self.assertEqual(stream.getvalue(), before_finish)

    def test_worst_page_can_improve_without_mean(self) -> None:
        clock = FakeClock()
        stream = io.StringIO()
        reporter = ProgressReporter(total=10, interval_seconds=60, stream=stream, clock=clock)
        reporter.start()
        reporter.observe({
            "parameter_set_id": "aaaaaaaa",
            "summary": {"mean_iou": 0.9, "minimum_iou": 0.5, "stddev_iou": 0.12, "failure_count": 0},
        })
        clock.value = 1
        reporter.observe({
            "parameter_set_id": "bbbbbbbb",
            "summary": {"mean_iou": 0.85, "minimum_iou": 0.6, "stddev_iou": 0.07, "failure_count": 0},
        })
        text = stream.getvalue()
        self.assertIn("New minimum page IoU bbbbbbbb", text)
        self.assertNotIn("New best average page IoU bbbbbbbb", text)
        self.assertAlmostEqual(reporter.snapshot().best_mean_iou or 0, 0.9)
        snapshot = reporter.snapshot()
        self.assertAlmostEqual(snapshot.current_minimum_page_iou or 0, 0.6)
        self.assertAlmostEqual(snapshot.best_minimum_page_iou or 0, 0.6)
        self.assertAlmostEqual(snapshot.current_mean_iou or 0, 0.85)
        self.assertAlmostEqual(snapshot.best_stddev_iou or 0, 0.07)
        self.assertEqual(snapshot.mean_iou_improvements, 1)
        self.assertEqual(snapshot.minimum_iou_improvements, 2)
        self.assertEqual(snapshot.stddev_improvements, 2)
        self.assertEqual(snapshot.total, 10)
        self.assertEqual(snapshot.parameter_sets_with_improvements, 2)


if __name__ == "__main__":
    unittest.main()
