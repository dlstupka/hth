import unittest
from hth.write_regression_summary import _initial_claim_batches,_next_claim_optimization
class ClaimOptimizationReportingTests(unittest.TestCase):
    def test_lpt_recommends_batches(self):
        r=_next_claim_optimization([{"detector":str(i)} for i in range(10)],{"pipeline_count":6,"loading_strategy":"lpt"})
        self.assertEqual(r["strategy"],"10s LPT claim batches")
        self.assertEqual(r["target_seconds"],10.0)
    def test_batch_accumulates_small_tasks(self):
        rows=[{"detector":"a","estimate_seconds":4.7},{"detector":"b","estimate_seconds":4.3},{"detector":"c","estimate_seconds":1.0}]
        b=_initial_claim_batches(rows,1)[0]
        self.assertEqual(len(b["tasks"]),3)
        self.assertAlmostEqual(b["estimated_seconds"],10.0)
    def test_final_batch_drains_remainder(self):
        b=_initial_claim_batches([{"detector":"a","estimate_seconds":2.0}],1)[0]
        self.assertAlmostEqual(b["estimated_seconds"],2.0)
if __name__=="__main__": unittest.main()
