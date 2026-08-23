import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/_core-hth.yml"


class PreprocessPublicationCollisionTests(unittest.TestCase):
    def _publication_block(self) -> str:
        text = WORKFLOW.read_text(encoding="utf-8")
        start = text.index("- name: Publish production publication")
        end = text.index("\n      - name:", start + 1)
        return text[start:end]

    def test_production_publication_uses_shared_hardened_persistence(self):
        block = self._publication_block()
        self.assertIn("source hth-pipeline/tools/hardened-persistence.sh", block)
        self.assertIn("hth_hardened_persist", block)
        self.assertIn("apply_production_publication", block)
        self.assertIn('tar -C results-repo -xf "$publication_payload"', block)

    def test_retry_preserves_nonproduction_results_repo_content(self):
        block = self._publication_block()
        self.assertNotIn("results-repo/test/", block)
        self.assertNotIn("results-repo/calibration-index.json", block)
        self.assertIn("results-repo/metadata", block)
        self.assertIn("reports/preprocess-summary.json", block)
        self.assertNotIn("results-repo/reports results-repo/analysis", block)
        self.assertIn("calibration intelligence were preserved", block)
        self.assertIn("git -C results-repo add -A --", block)
        self.assertNotIn("git -C results-repo add --all", block)


if __name__ == "__main__":
    unittest.main()
