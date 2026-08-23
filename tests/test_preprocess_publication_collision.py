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

    def test_production_publication_rebuilds_from_latest_main_on_collision(self):
        block = self._publication_block()
        self.assertIn("max_publish_attempts=5", block)
        self.assertIn("git -C results-repo fetch origin main", block)
        self.assertIn("git -C results-repo reset --hard origin/main", block)
        self.assertIn('tar -C results-repo -xf "$publication_payload"', block)
        self.assertIn("git -C results-repo push origin HEAD:main", block)
        self.assertIn("Production publish collision confirmed;", block)
        self.assertIn("non-fast-forward|fetch first|failed to push some refs", block)

    def test_retry_preserves_nonproduction_results_repo_content(self):
        block = self._publication_block()
        clean = block.index("git -C results-repo clean -fd --")
        restore = block.index('tar -C results-repo -xf "$publication_payload"')
        retry_reset = block[clean:restore]
        self.assertNotIn("test/", retry_reset)
        self.assertNotIn("calibration-index.json", retry_reset)
        self.assertIn("metadata reports analysis", retry_reset)
        self.assertIn("Existing \\`test/\\` and calibration intelligence were preserved", block)


if __name__ == "__main__":
    unittest.main()
