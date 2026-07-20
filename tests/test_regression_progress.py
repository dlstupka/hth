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
        header = next(line for line in lines if line.startswith("Elapsed"))
        initial = lines[-1]
        self.assertIn("TBD", initial)
        self.assertIn("0/10 0.0%", initial)
        self.assertNotIn("estimating", initial)

        clock.value = 60
        reporter.begin_evaluation("baseline")
        reporter.observe({
            "parameter_set_id": "abcdef123456",
            "summary": {"mean_iou": 0.8, "minimum_iou": 0.6, "failure_count": 0},
        }, "baseline")
        reporter.emit(force=True)
        row = stream.getvalue().splitlines()[-1]
        self.assertIn("00:09:00", row)
        self.assertIn("1/10 10.0%", row)
        self.assertIn("00:01:00", row)  # actual elapsed timestamp of last improvement

        self.assertEqual(
            header,
            "Elapsed   ETA       Complete       Eval Rate  Best Mean IoU  Worst IoU  "
            "Failures  Last Improvement  Evaluating",
        )
        self.assertEqual(
            row,
            "00:01:00  00:09:00  1/10 10.0%        0.02/s         0.8000     "
            "0.6000         0  00:01:00          baseline",
        )

    def test_milestones_and_final_row_clear_evaluating(self) -> None:
        clock = FakeClock()
        stream = io.StringIO()
        reporter = ProgressReporter(total=2, interval_seconds=60, stream=stream, clock=clock)
        reporter.start()
        reporter.begin_evaluation("baseline")
        reporter.observe({
            "parameter_set_id": "aaaaaaaa",
            "summary": {"mean_iou": 0.8, "minimum_iou": 0.5, "failure_count": 0},
        }, "baseline")
        clock.value = 61
        reporter.begin_evaluation("ps:bbbbbbbb")
        reporter.observe({
            "parameter_set_id": "bbbbbbbb",
            "summary": {"mean_iou": 0.9, "minimum_iou": 0.6, "failure_count": 0},
        })
        text = stream.getvalue()
        self.assertIn("00:01:01 >>> New best mean IoU ps:bbbbbbbb", text)
        self.assertIn("00:01:01 >>> New worst-page IoU ps:bbbbbbbb", text)
        self.assertIn("00:01:01 >>> Baseline surpassed ps:bbbbbbbb", text)

        reporter.finish()
        final_row = stream.getvalue().splitlines()[-1]
        self.assertTrue(final_row.endswith("--"), final_row)

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
