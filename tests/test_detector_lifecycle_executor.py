import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class DetectorLifecycleExecutorTests(unittest.TestCase):
    def test_shared_executor_owns_prepare_and_finalize(self):
        text=(ROOT/"tools"/"run-detector-regressions.sh").read_text(encoding="utf-8")
        self.assertIn("hth.detector_lifecycle prepare-config",text)
        self.assertIn("hth.detector_lifecycle finalize-config",text)

    def test_workflows_do_not_reimplement_detector_lifecycle(self):
        for rel in (".github/workflows/regress-detector.yml",".github/workflows/execution-optimizer.yml"):
            text=(ROOT/rel).read_text(encoding="utf-8")
            self.assertNotIn("Prepare detector lifecycle",text)
            self.assertNotIn("Finalize detector lifecycle",text)
            self.assertNotIn("hth.detector_lifecycle prepare",text)
            self.assertNotIn("hth.detector_lifecycle finalize",text)

if __name__=="__main__":
    unittest.main()
