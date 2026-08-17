import json
import tempfile
import unittest
from pathlib import Path

from hth.write_regression_summary import build_summary


class ResultReproducibilityIdentityTests(unittest.TestCase):
    def test_result_table_carries_golden_and_detector_config_id_and_tuple_footnote(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = root / "detector" / "run-1"
            (run / "reports").mkdir(parents=True)
            golden = root / "golden.json"
            golden.write_text(json.dumps({
                "collection_id": "HTH-0001",
                "pages": [],
            }), encoding="utf-8")
            (run / "manifest.json").write_text(json.dumps({
                "run_id": "run-1",
                "detector": "contour",
                "strategy": "exhaustive",
                "status": "complete",
                "outputs": [],
            }), encoding="utf-8")
            (run / "RUN-INFO.json").write_text(json.dumps({
                "golden_set": str(golden),
                "golden_set_sha256": "golden-sha",
                "detector_config": "config/detectors/contour.json",
                "detector_config_sha256": "1234567890abcdef",
                "pipeline_commit": "fedcba9876543210",
            }), encoding="utf-8")
            result = {
                "parameter_set_id": "winner",
                "parameter_identity_sha256": "p" * 64,
                "parameters": {"x": 1},
                "profile": None,
                "summary": {
                    "mean_iou": 0.9,
                    "minimum_iou": 0.8,
                    "stddev_iou": 0.01,
                    "mean_iou_success": 0.9,
                    "failure_count": 0,
                    "elapsed_ms_total": 10,
                },
                "pages": [],
            }
            (run / "reports/summary.json").write_text(json.dumps({
                "winner": result,
                "baseline": None,
                "ranked_results": [result],
                "parameter_set_count": 1,
                "parameter_space": {"possible_parameter_sets": 1},
            }), encoding="utf-8")
            (run / "parameters.json").write_text(json.dumps({}), encoding="utf-8")

            text = build_summary(run, "https://example.invalid/run")
            self.assertIn(
                "| Result | Golden Set ID | Detector Config ID* | Parameter Set ID |",
                text,
            )
            self.assertIn("| Winner | `HTH-0001` | `1234567890ab` | `winner` |", text)
            self.assertIn("`fedcba987654`", text)
            self.assertIn("**detector implementation + parameter set + Golden Set**", text)


if __name__ == "__main__":
    unittest.main()
