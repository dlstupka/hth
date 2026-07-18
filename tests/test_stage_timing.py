import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hth.stage_timing import display_duration, finish


class StageTimingTests(unittest.TestCase):
    def test_display_duration_preserves_short_stage_precision(self):
        self.assertEqual(display_duration(2.125), "2.1s")
        self.assertEqual(display_duration(65), "1m 5s")

    def test_finish_appends_jsonl_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "timings.jsonl"
            with patch("hth.stage_timing.time.time", return_value=106.25), patch(
                "hth.stage_timing.utc_now", return_value="2026-07-18T20:00:06Z"
            ):
                result = finish(
                    "STAGE_PREPROCESS",
                    "100.0",
                    "2026-07-18T20:00:00Z",
                    "success",
                    str(destination),
                )
            self.assertEqual(result, 0)
            record = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(record["stage"], "STAGE_PREPROCESS")
            self.assertEqual(record["status"], "success")
            self.assertEqual(record["elapsed_seconds"], 6.25)


if __name__ == "__main__":
    unittest.main()
