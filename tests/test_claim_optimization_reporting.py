import unittest
from hth.write_regression_summary import _next_claim_optimization
class ClaimOptimizationReportingTests(unittest.TestCase):
    def test_lpt_seed_recommendation(self):
        r=_next_claim_optimization([{"detector":str(i)} for i in range(10)],{"pipeline_count":6,"loading_strategy":"lpt"})
        self.assertEqual(r["seed_count"],6)
        self.assertEqual(r["initial_lock"],"bypassed")
        self.assertEqual(r["claim_wait"],"0 startup claim wait")
    def test_queue_caps_seed_count(self):
        r=_next_claim_optimization([{"detector":"a"}],{"pipeline_count":6,"loading_strategy":"lpt"})
        self.assertEqual(r["seed_count"],1)
    def test_non_lpt_stays_dynamic(self):
        r=_next_claim_optimization([{"detector":"a"}],{"pipeline_count":4,"loading_strategy":"fifo"})
        self.assertEqual(r["seed_count"],0)
if __name__=="__main__": unittest.main()
