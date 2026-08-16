import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DRIVER=(ROOT/"tools/run-detector-regressions.sh").read_text(encoding="utf-8")
class LptClaimBatchTests(unittest.TestCase):
    def test_one_lock_claims_multiple_tasks_until_threshold(self):
        block=DRIVER[DRIVER.index("claim_batch_from_queue()"):DRIVER.index("reclaim_expired_task()")]
        self.assertIn('claimed_batch+=("$task_index")',block)
        self.assertIn("claimed_batch_units >= CLAIM_BATCH_TARGET_DECISECONDS",block)
    def test_worker_executes_whole_batch_before_refill(self):
        self.assertIn('for task_index in "${claimed_batch[@]}"; do',DRIVER)
        self.assertIn('run_detector_config "$task_index" "$pipeline_number"',DRIVER)
    def test_claim_metadata_written_once_per_batch(self):
        self.assertIn('$telemetry_root/claim-batches/$batch_id.tsv',DRIVER)
        self.assertNotIn("printf 'claim\\t",DRIVER)
if __name__=="__main__": unittest.main()
