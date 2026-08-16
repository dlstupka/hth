import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DRIVER=ROOT/"tools"/"run-detector-regressions.sh"

class ShardClaimQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.text=DRIVER.read_text(encoding="utf-8")
    def test_claim_assignment_has_a_single_short_critical_section(self):
        self.assertIn('exec {claim_lock_fd}>"$queue_dir/claim.lock"',self.text)
        self.assertIn('flock -x "$claim_lock_fd"',self.text)
        self.assertIn('flock -u "$claim_lock_fd"',self.text)
        self.assertIn("claim_batch_from_queue",self.text)
    def test_each_worker_opens_its_own_flock_descriptor(self):
        worker=self.text.index("detector_worker() {")
        open_lock=self.text.index('exec {claim_lock_fd}>"$queue_dir/claim.lock"',worker)
        launch=self.text.index("worker_pids=()",worker)
        self.assertLess(worker,open_lock); self.assertLess(open_lock,launch)
    def test_normal_batch_claim_does_not_check_active_leases(self):
        first=self.text.index("claim_batch_from_queue() {")
        recovery=self.text.index("reclaim_expired_task() {",first)
        self.assertNotIn("lease_expired",self.text[first:recovery])
        self.assertIn("lease_expired",self.text[recovery:])
    def test_expired_lease_recovery_remains_inside_claim_lock(self):
        worker=self.text.index("detector_worker() {")
        acquire=self.text.index("      acquire_claim_lock",worker)
        reclaim=self.text.index("reclaim_expired_task",acquire)
        release=self.text.index("      release_claim_lock",acquire)
        self.assertLess(acquire,reclaim); self.assertLess(reclaim,release)
    def test_failed_tasks_are_not_reclaimed(self):
        self.assertGreaterEqual(self.text.count('[[ -f "$queue_dir/failed/$task_index" ]] && continue'),2)
    def test_fallback_retry_is_millisecond_scale(self):
        self.assertIn('while ! mkdir "$claim_lock_dir" 2>/dev/null; do sleep 0.01; done',self.text)
        self.assertNotIn("sleep $((RANDOM",self.text)

if __name__=="__main__": unittest.main()
