import argparse
import tempfile
import unittest
from pathlib import Path

from hth.write_run_summary import build_summary


class SummaryTests(unittest.TestCase):
    def test_summary_contains_core_run_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "build"
            output.mkdir()
            args = argparse.Namespace(
                pipeline_name="Preprocess Pipeline",
                status="success",
                collection_id="HTH-0001",
                source_repository="example/source",
                source_commit="1234567890abcdef",
                pipeline_commit="abcdef1234567890",
                workflow_name="preprocess-test",
                run_number="42",
                run_url="https://example.invalid/run/42",
                elapsed_seconds=96,
                docx_count=1,
                page_count=10,
                processed_count=8,
                error_count=2,
                notes="Detector failures were isolated.",
                output=[str(output)],
            )
            text = build_summary(args)
            self.assertIn("HTH Preprocess Pipeline", text)
            self.assertIn("1m 36s", text)
            self.assertIn("`1234567890ab`", text)
            self.assertIn("Pages processed | 8", text)
            self.assertIn("present", text)


if __name__ == "__main__":
    unittest.main()
