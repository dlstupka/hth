import argparse
import json
import tempfile
import unittest
from pathlib import Path

from hth.write_run_summary import _hydrate_from_generated_json, build_summary


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
            docx_count=None,
            page_count=None,
            processed_count=None,
            error_count=None,
            summary_json="",
            analysis_summary_json="",
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


if __name__ == "__main__":
    unittest.main()
