import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "review-document-detector.yml"


class DocumentReviewWorkflowTests(unittest.TestCase):
    def test_review_reuses_complete_production_artifact_without_source_checkout(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("actions/download-artifact@v5", text)
        self.assertIn("run-id: ${{ inputs.source_run_id }}", text)
        self.assertIn("pattern: hth-production-*", text)
        self.assertNotIn("Checkout source repository", text)
        self.assertNotIn("lfs: true", text)

    def test_review_runs_approved_gen3_and_packages_workbench(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("amsre_doc_ufcn_fusion", text)
        self.assertIn("run_document_detector.py", text)
        self.assertIn("reference-collection-editor-multidetector.html", text)
        self.assertIn("hth-document-review-", text)


if __name__ == "__main__":
    unittest.main()
