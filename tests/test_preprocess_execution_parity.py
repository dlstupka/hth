import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / ".github" / "workflows" / "_core-hth.yml"


class PreprocessExecutionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = CORE.read_text(encoding="utf-8")

    def test_results_checkout_is_shallow_sparse_and_includes_reusable_caches(self):
        block = self.text.split("- name: Checkout results repository", 1)[1].split(
            "# SOURCE SELECTION AND VALIDATION", 1
        )[0]
        self.assertIn("fetch-depth: 1", block)
        self.assertNotIn("fetch-depth: 0", block)
        self.assertIn("sparse-checkout:", block)
        self.assertIn("sparse-checkout-cone-mode: false", block)
        self.assertIn("/models/", block)
        self.assertIn("/learned-evidence/", block)
        self.assertIn("/source-documents/", block)
        self.assertIn("/metadata/", block)
        self.assertIn("/reports/", block)
        self.assertIn("/analysis/", block)
        self.assertIn("/test/", block)

    def test_preprocess_uses_managed_runtime_contract(self):
        self.assertIn("uses: ./hth-pipeline/.github/actions/setup-hth-managed-runtime", self.text)
        self.assertIn("need-doc-ufcn: ${{ steps.preferred_document_detector.outputs.needs_doc_ufcn || 'false' }}", self.text)
        action = (ROOT / ".github/actions/setup-hth-managed-runtime/action.yml").read_text(encoding="utf-8")
        self.assertIn("tools/ensure-managed-runtime.sh", action)
        self.assertIn("- name: Verify Doc-UFCN historical page-segmentation runtime", action)
        self.assertNotIn("- name: Install dependencies\n", self.text)
        self.assertNotIn("- name: Install managed detector runtime for document inference", self.text)

    def test_document_inference_reuses_results_model_cache(self):
        block = self.text.split("- name: Run approved detector over production collection", 1)[1].split(
            "# STAGE_DETECT_CANDIDATES", 1
        )[0]
        self.assertIn('lifecycle_root="results-repo"', block)
        self.assertIn('--lifecycle-root "$lifecycle_root"', block)
        self.assertIn("run-local detector/model cache", block)


if __name__ == "__main__":
    unittest.main()
