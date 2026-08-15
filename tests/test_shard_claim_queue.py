import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tools" / "run-detector-regressions.sh"


class ShardClaimQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = DRIVER.read_text(encoding="utf-8")

    def test_claim_assignment_has_a_single_short_critical_section(self):
        self.assertIn('exec {claim_lock_fd}>"$queue_dir/claim.lock"', self.text)
        self.assertIn('flock -x "$claim_lock_fd"', self.text)
        self.assertIn('flock -u "$claim_lock_fd"', self.text)
        self.assertIn("acquire_claim_lock", self.text)
        self.assertIn("release_claim_lock", self.text)

    def test_each_worker_opens_its_own_flock_descriptor(self):
        worker = self.text.index("detector_worker() {")
        open_lock = self.text.index('exec {claim_lock_fd}>"$queue_dir/claim.lock"', worker)
        launch = self.text.index("worker_pids=()", worker)
        self.assertLess(worker, open_lock)
        self.assertLess(open_lock, launch)

    def test_normal_claim_pass_does_not_check_active_leases(self):
        first_pass = self.text.index("# First pass: claim an actually unclaimed task.")
        recovery = self.text.index("# Recovery pass: only when no unclaimed work remains")
        lease_check = self.text.index("from hth.regression.sharding import lease_expired", recovery)
        self.assertLess(first_pass, recovery)
        self.assertGreater(lease_check, recovery)
        self.assertNotIn(
            "from hth.regression.sharding import lease_expired",
            self.text[first_pass:recovery],
        )

    def test_expired_lease_recovery_remains_inside_claim_lock(self):
        acquire = self.text.index("    acquire_claim_lock")
        reclaim = self.text.index("Reclaiming expired shard lease", acquire)
        release = self.text.index("    release_claim_lock", acquire)
        self.assertLess(acquire, reclaim)
        self.assertLess(reclaim, release)

    def test_failed_tasks_are_not_reclaimed_by_other_workers(self):
        self.assertGreaterEqual(
            self.text.count('[[ -f "$queue_dir/failed/$task_index" ]] && continue'),
            2,
        )

    def test_non_flock_fallback_is_millisecond_scale_not_pipeline_stagger(self):
        self.assertIn('while ! mkdir "$claim_lock_dir" 2>/dev/null; do', self.text)
        self.assertIn("sleep 0.01", self.text)
        self.assertNotIn("sleep $((RANDOM", self.text)


if __name__ == "__main__":
    unittest.main()
