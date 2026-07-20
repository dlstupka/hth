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
    def test_header_zero_heartbeat_minute_cadence_and_milestone(self) -> None:
        clock = FakeClock()
        stream = io.StringIO()
        reporter = ProgressReporter(total=100, interval_seconds=60, stream=stream, clock=clock)
        reporter.start()
        self.assertIn("Best Mean IoU", stream.getvalue())
        self.assertIn("0/100", stream.getvalue())

        result = {"parameter_set_id": "abcdef123456", "summary": {"mean_iou": .8, "minimum_iou": .6, "failure_count": 0}}
        clock.value = 30
        reporter.observe(result)
        self.assertEqual(stream.getvalue().count("00:00:30"), 0)

        clock.value = 60
        reporter.emit()
        self.assertIn("1/100", stream.getvalue())

        clock.value = 61
        better = {"parameter_set_id": "1234567890ab", "summary": {"mean_iou": .9, "minimum_iou": .7, "failure_count": 0}}
        reporter.observe(better)
        text = stream.getvalue()
        self.assertIn(">>> New best profile ps:12345678", text)
        milestone_lines = [line for line in text.splitlines() if line.startswith("00:01:01")]
        self.assertEqual(len(milestone_lines), 1)


if __name__ == "__main__":
    unittest.main()
