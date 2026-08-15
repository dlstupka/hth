import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class MultiDetectorLptDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = (ROOT / ".github/workflows/regress-detector.yml").read_text(encoding="utf-8")
        cls.driver = (ROOT / "tools/run-detector-regressions.sh").read_text(encoding="utf-8")

    def test_all_is_one_aggregate_job(self):
        self.assertIn("fromJSON(format('[\"{0}\"]'", self.workflow)
        self.assertNotIn('["contour","convex_hull","distance_transform"', self.workflow)

    def test_self_hosted_all_uses_auto_worker_count(self):
        self.assertIn("matrix.detector == 'all'", self.workflow)
        self.assertIn("'auto' || '4'", self.workflow)

    def test_driver_uses_canonical_auto_policy(self):
        self.assertIn("plan_lpt_workers", self.driver)
        self.assertIn('requested_pipelines" != "auto"', self.driver)

    def test_existing_runtime_store_remains_lpt_order_source(self):
        self.assertIn("python -m hth.runtime_store order", self.driver)
        self.assertIn('--loading-strategy "$DETECTOR_LOADING_STRATEGY"', self.driver)

if __name__ == "__main__":
    unittest.main()
