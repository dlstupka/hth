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
                "elapsed_seconds": 61.2, "golden_set": "config/golden_set.json",
                "golden_set_sha256": "abc123"
            }), encoding="utf-8")
            (run / "parameters.json").write_text(json.dumps({
                "configuration": {"profiles": {"baseline": {}}}
            }), encoding="utf-8")
            winner = {"profile": None, "parameter_short_name": "calibrated-winner", "parameter_set_id": "winner", "summary": {
                "mean_iou": .97, "minimum_iou": .91, "failure_count": 0, "elapsed_ms_total": 12.3, "wall_ms": 18.7
            }}
            baseline = {"profile": "baseline", "parameter_set_id": "base", "summary": {
                "mean_iou": .90, "minimum_iou": .80, "failure_count": 1, "elapsed_ms_total": 15.0, "wall_ms": 21.4
            }}
            (run / "reports" / "summary.json").write_text(json.dumps({
                "page_ordinals": [1, 5, 6, 9, 10], "parameter_set_count": 42,
                "parameter_space": {"possible_parameter_sets": 84},
                "winner": winner, "baseline": baseline,
                "top_parameter_sets": [
                    {**winner, "rank": 1, "search_observation": {"parameter_set_number": 4, "elapsed_seconds": 12.0, "search_fraction": 4/84}},
                    {**baseline, "rank": 2, "search_observation": {"parameter_set_number": 1, "elapsed_seconds": 1.0, "search_fraction": 1/84}},
                ],
                "winner_page_report": {
                    "counts": {
                        "unprocessed_pages": 0,
                        "no_polygon_found": 0,
                        "zero_overlap": 0,
                        "poor_matches": 1,
                        "regressions": 1,
                    },
                    "pages": [
                        {
                            "golden_set_page": 1,
                            "baseline_iou": 0.90,
                            "winner_iou": 0.97,
                            "delta_iou": 0.07,
                            "status": "Improved",
                            "parameter_set": "winner",
                            "problem": False,
                            "problem_reasons": [],
                        },
                        {
                            "golden_set_page": 5,
                            "baseline_iou": 0.60,
                            "winner_iou": 0.40,
                            "delta_iou": -0.20,
                            "status": "Regressed",
                            "parameter_set": "winner",
                            "problem": True,
                            "problem_reasons": ["Poor match", "Regressed"],
                        },
                    ],
                },
                "progress": {
                    "mean_iou_improvements": 3,
                    "minimum_iou_improvements": 2,
                    "stddev_improvements": 4,
                    "total_metric_improvements": 9,
                    "parameter_sets_with_improvements": 5,
                    "winner_changes": 2,
                    "winner_history": [
                        {"change_number": 1, "parameter_set_id": "older", "elapsed_seconds": 4.0, "search_fraction": 2/84},
                        {"change_number": 2, "parameter_set_id": "winner", "elapsed_seconds": 12.0, "search_fraction": 4/84},
                    ],
                    "baseline_surpassed": True,
                }
            }), encoding="utf-8")

            text = build_summary(run, "https://example.invalid/run")
            self.assertIn("# Regression Manifest", text)
            self.assertIn("`grabcut`", text)
            self.assertIn("`binary-refine`", text)
            self.assertIn("`1234567890ab`", text)
            self.assertIn("| Result | Parameter Set ID | Parameter Short Name |", text)
            self.assertIn("| Winner | `winner` | `calibrated-winner` | 0.9700", text)
            self.assertIn("19 ms", text)
            self.assertIn("SHA-256: `abc123`", text)
            self.assertIn("Configured named profiles: `baseline`", text)
            self.assertIn("Evaluation Time", text)
            self.assertIn("### Top Parameter Sets", text)
            self.assertIn("| Rank | Parameter Set ID | Parameter Short Name | Avg IoU | Min IoU | StdDev | Δ Avg IoU | Failures | Discovery Time | Search Space % |", text)
            self.assertIn("| 1 | `winner` | `calibrated-winner` | 0.9700 | 0.9100 | unknown | +0.0000 | 0 | 12s | 4.76% |", text)
            self.assertIn("### Golden Set Winner Summary", text)
            self.assertIn("| Golden Set Page | Parameter Set ID | Baseline | Winner | Δ IoU | Status |", text)
            self.assertIn("| 1 | `winner` | 0.9000 | 0.9700 | +0.0700 | Improved |", text)
            self.assertIn("### Golden Set Page Issues", text)
            self.assertIn("Poor matches (Winner IoU < 0.5000): `1`", text)
            self.assertIn("### Status Definitions", text)
            self.assertIn("Regressed pages (Δ IoU < -0.0010): `1`", text)
            self.assertIn("#### Affected Pages", text)
            self.assertIn("| 5 | 0.4000 | Poor match; Regressed | `winner` |", text)
            self.assertIn("### Regression Statistics for Detector Calibration", text)
            self.assertIn("| Total metric improvements | 9 |", text)
            self.assertIn("| Winner changes | 2 |", text)
            self.assertIn("| Baseline surpassed | yes |", text)
            self.assertIn("| Baseline | `base` | `baseline` | 0.9000", text)
            self.assertIn("21 ms", text)
            self.assertLess(text.index("## Run Information"), text.index("## Results"))
            self.assertLess(text.index("## Results"), text.index("## Page Analysis"))
            self.assertLess(
                text.index("### Regression Statistics for Detector Calibration"),
                text.index("### Top Parameter Sets"),
            )
            self.assertIn("`raw/results.csv` — present", text)
            self.assertIn("`reports/summary.json` — present", text)
            self.assertIn("[Open workflow run]", text)

    def test_builds_combined_manifest_for_multiple_detectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "golden_set.json").write_text(json.dumps({
                "source_document": {
                    "title": "Baptisms: San Antonio. Baptism Records 1788–1824, 1858–1898",
                    "image_count": 929,
                }
            }), encoding="utf-8")
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
                    "golden_set": str(root / "golden_set.json")
                }), encoding="utf-8")
                (run / "parameters.json").write_text(json.dumps({
                    "configuration": {"profiles": {"baseline": {}}}
                }), encoding="utf-8")
                metrics = {
                    "grabcut": (.88, .82, .02, 250.0, 300.0),
                    "contour": (.92, .78, .03, 100.0, 120.0),
                }[detector]
                result = {"profile": "baseline", "parameter_set_id": detector, "summary": {
                    "mean_iou": metrics[0], "minimum_iou": metrics[1], "stddev_iou": metrics[2],
                    "failure_count": 0, "elapsed_ms_total": metrics[3], "wall_ms": metrics[4]
                }}
                (run / "reports" / "summary.json").write_text(json.dumps({
                    "page_ordinals": [1], "parameter_set_count": 1,
                    "winner": result, "baseline": result
                }), encoding="utf-8")
                (run / "reports" / "calibration-intelligence.json").write_text(json.dumps({
                    "available": True,
                    "scope_note": "Scoped to the fixture Golden Set.",
                    "search": {
                        "exhaustive_complete": True,
                        "parameter_sets": 1,
                        "fully_successful_parameter_sets": 1,
                        "fully_successful_rate": 1.0,
                    },
                    "landscape": {
                        "best_mean_iou": metrics[0],
                        "median_mean_iou": metrics[0],
                        "p95_mean_iou": metrics[0],
                        "near_best_tolerance": 0.001,
                        "near_best_count": 1,
                        "near_best_share": 1.0,
                        "equivalent_tolerance": 0.0001,
                        "equivalent_winner_count": 1,
                        "equivalent_winner_share": 1.0,
                    },
                    "parameter_influence": [{
                        "parameter": "threshold",
                        "classification": "Dormant",
                        "eta_squared": 0.0,
                        "mean_iou_range": 0.0,
                        "near_best_value_coverage": 1.0,
                        "best_values": [{"value": "1", "mean_iou": metrics[0], "count": 1}],
                    }],
                    "interactions": [],
                    "page_sensitivity": [{
                        "global_ordinal": 1,
                        "mean_iou": metrics[0],
                        "minimum_iou": metrics[0],
                        "maximum_iou": metrics[0],
                        "stddev_iou": 0.0,
                        "success_rate": 1.0,
                    }],
                    "recommendations": {
                        "dormant_parameters": ["threshold"],
                        "note": "Re-evaluate when the Golden Set changes.",
                    },
                    "calibration_confidence": {
                        "rating": "Medium",
                        "reasons": ["complete exhaustive coverage"],
                    },
                }), encoding="utf-8")
                run_dirs.append(run)

            text = build_combined_summary(run_dirs, "https://example.invalid/run")
            self.assertIn("# Detector Regression Manifest", text)
            self.assertIn("**Detectors evaluated:** 2", text)
            self.assertIn("## Source document", text)
            self.assertIn("**Document:** Baptisms: San Antonio. Baptism Records 1788–1824, 1858–1898", text)
            self.assertIn("**Images:** 929", text)
            self.assertIn("## Ranked Detector Results", text)
            self.assertIn("## Detector Calibration Report", text)
            self.assertIn("## Detector Regression Reports", text)
            self.assertIn("### Regression Completion Summary", text)
            self.assertIn("<summary><strong>Calibration Report Details</strong></summary>", text)
            self.assertIn("<summary><strong>Detector Regression Report Details</strong></summary>", text)
            self.assertIn("Detector short name", text)
            self.assertIn("## Detector Recommendation for this Golden Set", text)
            self.assertIn("### Calibration Report Legend", text)
            self.assertIn("### Calibration Overview", text)
            self.assertLess(text.index("### Calibration Overview"), text.index("### Calibration Report Legend"))
            self.assertIn("| Rank | Detector | Short Name | Detector ID | Role | Coverage |", text)
            self.assertIn("#### Detector Summary", text)
            self.assertIn("#### Evidence of ROI", text)
            self.assertIn("### Detector Evidence", text)
            self.assertIn("#### Parameter Influence", text)
            self.assertIn("#### Dormant Parameters", text)
            self.assertIn("#### Page Sensitivity", text)
            self.assertLess(text.index("### Metric Definitions"), text.index("## Detector Calibration Report"))
            self.assertLess(text.index("## Detector Calibration Report"), text.index("## Detector Regression Reports"))
            self.assertIn("| Rank | Detector | Short Name | Detector ID | Status | Parameter Set ID | Parameter Short Name | Avg IoU |", text)
            self.assertIn("| Eval Rate | Doc Time | Run Elapsed |", text)
            contour_row = "| 1 | Contour Envelope | Contour | `contour` | complete | `contour` | `baseline` | 0.9200 | 0.7800 | 0.0300 | 0 | 1 | 10.00 pg/s | 1m 33s | 1s |"
            grabcut_row = "| 2 | GrabCut Segmentation | GrabCut | `grabcut` | complete | `grabcut` | `baseline` | 0.8800 | 0.8200 | 0.0200 | 0 | 1 | 4.000 pg/s | 3m 52s | 1s |"
            self.assertIn(contour_row, text)
            self.assertIn(grabcut_row, text)
            self.assertLess(text.index(contour_row), text.index(grabcut_row))
            self.assertIn("### GrabCut Segmentation (`grabcut`)", text)
            self.assertIn("### Contour Envelope (`contour`)", text)
            self.assertEqual(text.count("[Open workflow run]"), 1)


if __name__ == "__main__":
    unittest.main()
