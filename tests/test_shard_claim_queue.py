import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "tools" / "run-detector-regressions.sh").read_text(encoding="utf-8")


class FixedPipelineQueueTests(unittest.TestCase):
    def test_dynamic_claim_locking_is_removed(self):
        self.assertNotIn("acquire_claim_lock", DRIVER)
        self.assertNotIn("release_claim_lock", DRIVER)
        self.assertNotIn('claim_lock_fd', DRIVER)

    def test_failed_task_stops_its_fixed_pipeline(self):
        self.assertIn(': > "$queue_dir/failed/$task_index"', DRIVER)
        self.assertIn('return 1', DRIVER[DRIVER.index("detector_worker() {"):])

    def test_executor_does_not_reassign_expired_work(self):
        worker = DRIVER[DRIVER.index("detector_worker() {"):DRIVER.index("# Learned inference evidence")]
        self.assertNotIn("lease_expired", worker)
        self.assertNotIn("reclaim", worker.lower())


if __name__ == "__main__":
    unittest.main()
