import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/regress-detector.yml"


class CalibrationPersistenceScopeTests(unittest.TestCase):
    def test_persistence_stages_calibration_intelligence_not_models(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        start = text.index("- name: Persist calibration intelligence")
        end = text.index("\n      - name:", start + 1)
        block = text[start:end]

        self.assertIn("calibration-index.json", block)
        self.assertIn("parameter-provenance-index.json", block)
        self.assertIn("runtime-index.json", block)
        self.assertIn("parallelism-index.json", block)
        self.assertIn("multidetector-index.json", block)
        self.assertIn("source-documents/", block)
        self.assertNotIn("git -C results-repo add models/", block)
        self.assertIn("Calibration persistence attempted to stage a model payload.", block)

    def test_only_actual_publish_collisions_are_retried(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        start = text.index("- name: Persist calibration intelligence")
        end = text.index("\n      - name:", start + 1)
        block = text[start:end]
        helper = (ROOT / "tools" / "hardened-persistence.sh").read_text(encoding="utf-8")

        self.assertIn("source hth-pipeline/tools/hardened-persistence.sh", block)
        self.assertIn("hth_hardened_persist", block)
        self.assertIn("non-fast-forward|fetch first|failed to push some refs", helper)
        self.assertIn("concurrent update confirmed", helper)
        self.assertIn("refusing to misclassify and retry it", helper)


if __name__ == "__main__":
    unittest.main()
