import argparse
import json
import tempfile
import unittest
from pathlib import Path

from hth.write_run_summary import _hydrate_from_generated_json, build_summary, parser


class RunSummaryTests(unittest.TestCase):
    def make_args(self, **overrides):
        values = dict(
            pipeline_name="Preprocess Pipeline — Test",
            status="success",
            collection_id="",
            source_repository="dlstupka/source",
            source_commit="1234567890abcdef",
            pipeline_commit="abcdef1234567890",
            workflow_name="HTH preprocess test",
            run_number="32",
            elapsed_seconds=83,
            pipeline_started_at="2026-07-18T19:48:18Z",
            summary_generated_at="2026-07-18T19:50:05Z",
            stage_timings_jsonl="",
            docx_count=None,
            page_count=None,
            processed_count=None,
            error_count=None,
            summary_json="",
            analysis_summary_json="",
            page_analysis_json="",
            notes="",
            output=[],
            run_url="",
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_hydrates_processing_facts_from_generated_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            analysis = root / "analysis-summary.json"
            summary.write_text(json.dumps({
                "collection_id": "HTH-0001",
                "source_docx_count": 1,
                "image_count": 10,
            }), encoding="utf-8")
            analysis.write_text(json.dumps({
                "page_count": 10,
                "quality_status_counts": {"pass": 9, "error": 1},
            }), encoding="utf-8")

            args = self.make_args(
                summary_json=str(summary),
                analysis_summary_json=str(analysis),
            )
            _hydrate_from_generated_json(args)

            self.assertEqual(args.collection_id, "HTH-0001")
            self.assertEqual(args.docx_count, 1)
            self.assertEqual(args.page_count, 10)
            self.assertEqual(args.processed_count, 10)
            self.assertEqual(args.error_count, 1)

    def test_explicit_values_override_generated_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            summary.write_text(json.dumps({
                "collection_id": "HTH-0001",
                "source_docx_count": 1,
                "image_count": 10,
            }), encoding="utf-8")
            args = self.make_args(
                collection_id="HTH-OVERRIDE",
                docx_count=2,
                page_count=20,
                processed_count=19,
                error_count=1,
                summary_json=str(summary),
            )
            _hydrate_from_generated_json(args)
            self.assertEqual(args.collection_id, "HTH-OVERRIDE")
            self.assertEqual(args.docx_count, 2)
            self.assertEqual(args.page_count, 20)

    def test_summary_uses_publication_output_heading(self):
        args = self.make_args(
            collection_id="HTH-0001",
            docx_count=1,
            page_count=10,
            processed_count=10,
            error_count=0,
            output=["missing-publication-path"],
        )
        text = build_summary(args)
        self.assertIn("## Publication outputs", text)
        self.assertIn("HTH-0001", text)
        self.assertIn("| DOCX masters | 1 |", text)
        self.assertIn("| Pages discovered | 10 |", text)

    def test_summary_includes_detector_performance_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            geometry = Path(tmp) / "page-analysis.json"
            geometry.write_text(json.dumps({
                "geometry_candidate_summary": {
                    "detectors": [{
                        "method": "components",
                        "name": "Connected Components",
                        "display_name": "Connected Components (OpenCV)",
                        "origin": "OpenCV",
                        "version": "4.10.0",
                        "repository": "https://github.com/opencv/opencv",
                    }],
                    "method_status_counts": {
                        "components": {"ok": 8, "no_candidate": 2, "error": 0}
                    },
                    "detector_performance": {
                        "components": {
                            "runs": 10,
                            "elapsed_ms_total": 112.0,
                            "elapsed_ms_average": 11.2,
                            "confidence_average": 0.812345,
                        }
                    },
                }
            }), encoding="utf-8")
            args = self.make_args(page_analysis_json=str(geometry))
            text = build_summary(args)
            self.assertIn("## Detector performance", text)
            self.assertIn("Connected Components (OpenCV)", text)
            self.assertIn("11.2 ms", text)
            self.assertIn("0.812", text)

    def test_summary_includes_timestamps_and_stage_performance(self):
        with tempfile.TemporaryDirectory() as tmp:
            timings = Path(tmp) / "stage-timings.jsonl"
            timings.write_text(json.dumps({
                "stage": "STAGE_PREPROCESS",
                "status": "success",
                "started_at_utc": "2026-07-18T19:48:20Z",
                "completed_at_utc": "2026-07-18T19:48:22Z",
                "elapsed_seconds": 2.125,
            }) + "\n", encoding="utf-8")
            args = self.make_args(
                collection_id="HTH-0001",
                docx_count=1,
                page_count=10,
                processed_count=10,
                error_count=0,
                stage_timings_jsonl=str(timings),
            )
            text = build_summary(args)
            self.assertIn("Pipeline started", text)
            self.assertIn("2026-07-18T19:48:18Z", text)
            self.assertIn("## Stage performance", text)
            self.assertIn("STAGE_PREPROCESS", text)
            self.assertIn("2.1s", text)

    def test_geometry_json_remains_compatibility_alias(self):
        args = parser().parse_args(["--geometry-json", "legacy.json"])
        self.assertEqual(args.page_analysis_json, "legacy.json")

    def test_page_analysis_json_is_preferred_argument_name(self):
        args = parser().parse_args(["--page-analysis-json", "page-analysis.json"])
        self.assertEqual(args.page_analysis_json, "page-analysis.json")


if __name__ == "__main__":
    unittest.main()
