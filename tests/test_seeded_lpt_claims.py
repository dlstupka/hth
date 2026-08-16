import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DRIVER=(ROOT/"tools/run-detector-regressions.sh").read_text(encoding="utf-8")
class SeededLptClaimTests(unittest.TestCase):
    def test_short_lpt_eligibility(self):
        self.assertIn('initial_claim_strategy="seeded-first-wave+lpt-refill"',DRIVER)
        self.assertIn('[[ "$REGRESSION_MODE" != "full" ]]',DRIVER)
        self.assertIn('[[ -n "${effective_limit:-}" ]]',DRIVER)
        self.assertIn('[[ "$effective_strategy" != "exhaustive" ]]',DRIVER)
    def test_parent_seeds_one_task_per_pipeline(self):
        self.assertIn('mkdir "$queue_dir/claims/$task_index"',DRIVER)
        self.assertIn('initial_seed_tasks[$pipeline_index]="$task_index"',DRIVER)
    def test_worker_bypasses_lock_for_seed_then_refills(self):
        i=DRIVER.index('if [[ -n "$seeded_task" ]]; then')
        j=DRIVER.index('acquire_claim_lock',i)
        self.assertIn('seeded_task=""',DRIVER[i:j])
    def test_telemetry_gets_claim_strategy(self):
        self.assertIn('--claim-strategy "$initial_claim_strategy"',DRIVER)
if __name__=="__main__": unittest.main()
