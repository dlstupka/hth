import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / ".github/workflows/_core-hth.yml"


class PreprocessTestPublicationRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = CORE.read_text(encoding="utf-8")
        start = cls.text.index("      - name: Publish test publication")
        end = cls.text.index("\n\n      - name: Complete STAGE_PUBLISH_TEST", start)
        cls.block = cls.text[start:end]

    def test_test_publication_uses_shared_hardened_persistence(self):
        self.assertIn("source hth-pipeline/tools/hardened-persistence.sh", self.block)
        self.assertIn("hth_hardened_persist", self.block)
        self.assertIn("apply_test_publication", self.block)

    def test_each_retry_reapplies_only_this_test_run(self):
        self.assertIn('test/latest "test/history/run-${GITHUB_RUN_NUMBER}"', self.block)
        self.assertIn('rm -rf "results-repo/test/history/run-${GITHUB_RUN_NUMBER}"', self.block)
        self.assertNotIn("git -C results-repo add --all", self.block)

    def test_older_racing_run_cannot_replace_newer_test_latest(self):
        self.assertIn("workflow_run_number:", self.block)
        self.assertIn("remote_run_number > GITHUB_RUN_NUMBER", self.block)
        self.assertIn("Existing newer", self.block)


if __name__ == "__main__":
    unittest.main()
