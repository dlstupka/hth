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


    def test_adaptive_resolves_immediate_neighbors_of_bracketed_peak(self) -> None:
        observations = [
            row(1, 40.0),
            row(3, 92.0),
            row(8, 122.26),
            row(16, 110.0),
            row(24, 84.0),
            row(64, 70.0),
        ]
        self.assertEqual(adaptive_next_pipeline(1, 64, 64, 1, observations), 7)

        observations.append(row(7, 120.5))
        self.assertEqual(adaptive_next_pipeline(1, 64, 64, 1, observations), 9)

    def test_adaptive_expands_two_percent_region_until_bounded(self) -> None:
        observations = [
            row(1, 40.0),
            row(3, 92.0),
            row(8, 100.0),
            row(16, 90.0),
            row(64, 60.0),
            row(7, 99.0),
            row(9, 99.5),
        ]
        self.assertEqual(adaptive_next_pipeline(1, 64, 64, 1, observations), 6)

        observations.append(row(6, 96.0))
        self.assertEqual(adaptive_next_pipeline(1, 64, 64, 1, observations), 10)

        observations.append(row(10, 97.0))
        self.assertIsNone(adaptive_next_pipeline(1, 64, 64, 1, observations))


    def test_adaptive_uses_optimizer_intelligence_seed_before_exploring_bounds(self) -> None:
        self.assertEqual(
            adaptive_next_pipeline(1, 192, 384, 1, [], start_pipeline=132),
            132,
        )
        # Once the seed is measured, adaptive is free to expand away from it;
        # the seed is not treated as a lower or upper search bound.
        candidate = adaptive_next_pipeline(1, 192, 384, 1, [row(132, 10.0)], start_pipeline=132)
        self.assertIsNotNone(candidate)
        self.assertNotEqual(candidate, 132)
        self.assertGreaterEqual(candidate, 1)
        self.assertLessEqual(candidate, 192)

    def test_adaptive_clamps_seed_to_manual_override_range(self) -> None:
        self.assertEqual(
            adaptive_next_pipeline(50, 100, 384, 1, [], start_pipeline=132),
            100,
        )

    def test_manual_oversubscription_keeps_shapes_beyond_runner_budget(self) -> None:
        self.assertEqual(
            powers_of_two_pipelines(49, 49, 192, 4, allow_oversubscription=True),
            [49],
        )

    def test_adaptive_fills_small_interior_gap_before_stopping(self) -> None:
        observations = [row(1, 10.0), row(3, 10.0)]
        self.assertEqual(adaptive_next_pipeline(1, 3, 192, 1, observations), 2)

    def test_adaptive_fills_small_gap_even_when_endpoint_trend_declines(self) -> None:
        observations = [row(1, 10.0), row(3, 9.8)]
        self.assertEqual(adaptive_next_pipeline(1, 3, 192, 1, observations), 2)

    def test_adaptive_returns_none_when_range_is_exhausted(self) -> None:
        observations = [row(p, float(p)) for p in range(2, 6)]
        self.assertIsNone(adaptive_next_pipeline(2, 5, 192, 1, observations))


if __name__ == "__main__":
    unittest.main()
