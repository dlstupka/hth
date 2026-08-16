import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DRIVER=(ROOT/"tools/run-detector-regressions.sh").read_text(encoding="utf-8")
class LptClaimBatchShellContractTests(unittest.TestCase):
    def test_short_lpt_enables_batches(self):
        self.assertIn('initial_claim_strategy="lpt-batches-10s"',DRIVER)
        self.assertIn("CLAIM_BATCH_TARGET_DECISECONDS=100",DRIVER)
        self.assertIn("CLAIM_ESTIMATE_FLOOR_DECISECONDS=1",DRIVER)
    def test_parent_prebatches_initial_wave(self):
        self.assertIn("Initial LPT claim batches",DRIVER)
        self.assertIn('initial_seed_batches[$pipeline_index]',DRIVER)
    def test_telemetry_gets_claim_strategy(self):
        self.assertIn('--claim-strategy "$initial_claim_strategy"',DRIVER)
if __name__=="__main__": unittest.main()
