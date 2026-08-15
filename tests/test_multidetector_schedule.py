import unittest
from hth.domain.multidetector_schedule import plan_lpt_workers

class MultiDetectorScheduleTests(unittest.TestCase):
    def test_large_384_thread_host(self):
        self.assertEqual(plan_lpt_workers(37, 384), 6)

    def test_small_host(self):
        self.assertEqual(plan_lpt_workers(37, 8), 1)

    def test_queue_bound(self):
        self.assertEqual(plan_lpt_workers(3, 384), 2)

    def test_thread_floor(self):
        self.assertEqual(plan_lpt_workers(100, 192), 4)

if __name__ == "__main__":
    unittest.main()
