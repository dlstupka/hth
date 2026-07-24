import json
import tempfile
import unittest
from pathlib import Path

from hth.write_regression_summary import build_combined_summary, build_summary


class RegressionSummaryTests(unittest.TestCase):
    def test_builds_manifest_with_winner_baseline_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run-1"
            (run / "reports").mkdir(parents=True)
            (run / "raw").mkdir()
            (run / "raw" / "results.csv").write_text("x\n", encoding="utf-8")
            (run / "manifest.json").write_text(json.dumps({
                "run_id": "run-1", "detector": "grabcut", "strategy": "binary-refine",
                "status": "complete", "outputs": ["raw/results.csv", "reports/summary.json"]
            }), encoding="utf-8")
            (run / "RUN-INFO.json").write_text(json.dumps({
                "pipeline_commit": "1234567890abcdef", "python_version": "3.12.0",
                "opencv_version": "5.0.0", "started_at_utc": "start", "finished_at_utc": "finish",
                "elapsed_seconds": 61.2, "golden_set": "config/golden_set.json"
            }), encoding="utf-8")
            (run / "parameters.json").write_text(json.dumps({
                "configuration": {"profiles": {"baseline": {}}}
            }), encoding="utf-8")
            winner = {"profile": None, "parameter_set_id": "winner", "summary": {
                "mean_iou": .97, "minimum_iou": .91, "failure_count": 0, "elapsed_ms_total": 12.3, "wall_ms": 18.7
            }}
            baseline = {"profile": "baseline", "parameter_set_id": "base", "summary": {
                "mean_iou": .90, "minimum_iou": .80, "failure_count": 1, "elapsed_ms_total": 15.0, "wall_ms": 21.4
            }}
            (run / "reports" / "summary.json").write_text(json.dumps({
                "page_ordinals": [1, 5, 6, 9, 10], "parameter_set_count": 42,
                "winner": winner, "baseline": baseline,
                "progress": {
                    "mean_iou_improvements": 3,
                    "minimum_iou_improvements": 2,
                    "stddev_improvements": 4,
                    "total_metric_improvements": 9,
                    "parameter_sets_with_improvements": 5,
                    "winner_changes": 2,
                    "baseline_surpassed": True,
                }
            }), encoding="utf-8")

            text = build_summary(run, "https://example.invalid/run")
            self.assertIn("# Regression Manifest", text)
            self.assertIn("`grabcut`", text)
            self.assertIn("`binary-refine`", text)
            self.assertIn("`1234567890ab`", text)
            self.assertIn("| Result | Parameter set | Parameter set ID |", text)
            self.assertIn("| Winner | `winner` | `winner` | 0.9700", text)
            self.assertIn("18.7ms", text)
            self.assertIn("Configured named profiles: `baseline`", text)
            self.assertIn("Evaluation time", text)
            self.assertIn("## Regression statistics", text)
            self.assertIn("| Total metric improvements | 9 |", text)
            self.assertIn("| Winner changes | 2 |", text)
            self.assertIn("| Baseline surpassed | yes |", text)
            self.assertIn("| Baseline | `baseline` | `base` | 0.9000", text)
            self.assertIn("21.4ms", text)
            self.assertIn("`raw/results.csv` — present", text)
            self.assertIn("`reports/summary.json` — present", text)
            self.assertIn("[Open workflow run]", text)

    def test_builds_combined_manifest_for_multiple_detectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dirs = []
            for detector in ("grabcut", "contour"):
                run = root / detector / "run-1"
                (run / "reports").mkdir(parents=True)
                (run / "manifest.json").write_text(json.dumps({
                    "run_id": f"run-{detector}", "detector": detector,
                    "strategy": "exhaustive", "status": "complete", "outputs": []
                }), encoding="utf-8")
                (run / "RUN-INFO.json").write_text(json.dumps({
                    "pipeline_commit": "1234567890abcdef", "python_version": "3.12.0",
                    "opencv_version": "5.0.0", "elapsed_seconds": 1.0,
                    "golden_set": "config/golden_set.json"
                }), encoding="utf-8")
                (run / "parameters.json").write_text(json.dumps({
                    "configuration": {"profiles": {"baseline": {}}}
                }), encoding="utf-8")
                result = {"profile": "baseline", "parameter_set_id": detector, "summary": {
                    "mean_iou": .9, "minimum_iou": .8, "stddev_iou": .01,
                    "failure_count": 0, "elapsed_ms_total": 100
                }}
                (run / "reports" / "summary.json").write_text(json.dumps({
                    "page_ordinals": [1], "parameter_set_count": 1,
                    "winner": result, "baseline": result
                }), encoding="utf-8")
                run_dirs.append(run)

            text = build_combined_summary(run_dirs, "https://example.invalid/run")
            self.assertIn("# Detector Regression Manifest", text)
            self.assertIn("**Detectors evaluated:** 2", text)
            self.assertIn("## grabcut", text)
            self.assertIn("## contour", text)
            self.assertEqual(text.count("[Open workflow run]"), 1)


if __name__ == "__main__":
    unittest.main()
