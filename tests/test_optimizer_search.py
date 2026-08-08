from __future__ import annotations

import unittest

from hth.optimizer_search import adaptive_next_pipeline, powers_of_two_pipelines


def row(pipelines: int, rate: float) -> dict:
    return {"active_pipelines": pipelines, "parameter_sets_per_second": rate}


class OptimizerSearchTests(unittest.TestCase):
    def test_powers_of_two_keeps_requested_endpoints(self) -> None:
        self.assertEqual(powers_of_two_pipelines(3, 52, 192, 1), [3, 4, 8, 16, 32, 52])

    def test_adaptive_starts_at_lowest_and_highest_clean_common_shapes(self) -> None:
        self.assertEqual(adaptive_next_pipeline(3, 52, 192, 1, []), 3)
        # 48 is the highest value in 3..52 that divides 192 cleanly.
        self.assertEqual(adaptive_next_pipeline(3, 52, 192, 1, [row(3, 1.0)]), 48)

    def test_adaptive_refines_toward_better_endpoint(self) -> None:
        candidate = adaptive_next_pipeline(2, 192, 192, 1, [row(2, 1.0), row(192, 0.5)])
        self.assertGreater(candidate, 2)
        self.assertLess(candidate, 192)
        self.assertLess(candidate, 96)

    def test_adaptive_refines_both_sides_of_interior_peak(self) -> None:
        observations = [row(2, 1.0), row(192, 0.5), row(20, 5.0)]
        candidate = adaptive_next_pipeline(2, 192, 192, 1, observations)
        self.assertIn(candidate, range(3, 192))
        self.assertNotIn(candidate, {2, 20, 192})

    def test_adaptive_returns_none_when_range_is_exhausted(self) -> None:
        observations = [row(p, float(p)) for p in range(2, 6)]
        self.assertIsNone(adaptive_next_pipeline(2, 5, 192, 1, observations))


if __name__ == "__main__":
    unittest.main()
