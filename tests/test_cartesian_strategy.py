from __future__ import annotations

import unittest

from hth.regression.strategies.cartesian import generate


class CartesianStrategyTests(unittest.TestCase):
    def test_named_profiles_are_first_and_deduplicated_in_limited_run(self) -> None:
        config = {
            "profiles": {"baseline": {"a": 2, "b": "y"}},
            "parameters": {
                "a": {"values": [1, 2]},
                "b": {"values": ["x", "y"]},
            },
        }
        self.assertEqual(generate(config, limit=3), [
            {"a": 2, "b": "y"},
            {"a": 1, "b": "x"},
            {"a": 1, "b": "y"},
        ])

    def test_full_run_preserves_unique_cartesian_space(self) -> None:
        config = {
            "profiles": {"baseline": {"a": 2}},
            "parameters": {"a": {"values": [1, 2, 3]}},
        }
        self.assertEqual(generate(config), [{"a": 2}, {"a": 1}, {"a": 3}])


if __name__ == "__main__":
    unittest.main()
