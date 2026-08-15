import json
import unittest
from pathlib import Path

from hth.regression.strategies.cartesian import generate


class DetectorRefinementSpaceTests(unittest.TestCase):
    def _load(self, detector):
        return json.loads(
            Path(f"config/detectors/{detector}.json").read_text(encoding="utf-8")
        )

    def test_polar_boundary_vote_refinement_is_boundary_focused(self):
        config = self._load("polar_boundary_vote")
        # Cartesian grid is 19,635 combinations plus the explicit baseline.
        self.assertEqual(len(generate(config)), 19636)
        self.assertNotIn("inner_radius_fraction", config["parameters"])
        self.assertNotIn("minimum_support_fraction", config["parameters"])
        self.assertEqual(config["profiles"]["baseline"]["inner_radius_fraction"], 0.06)
        self.assertEqual(config["profiles"]["baseline"]["minimum_support_fraction"], 0.25)

        self.assertIn(0.6, config["parameters"]["outer_radius_fraction"]["values"])
        self.assertLess(min(config["parameters"]["outer_radius_fraction"]["values"]), 0.6)
        self.assertIn(90.0, config["parameters"]["gradient_percentile"]["values"])
        self.assertGreater(max(config["parameters"]["gradient_percentile"]["values"]), 90.0)
        self.assertEqual(min(config["parameters"]["bbox_padding_fraction"]["values"]), 0.0)
        self.assertIn(180, config["parameters"]["ray_count"]["values"])

    def test_gradient_vote_refinement_extends_only_border_search_lower_bound(self):
        config = self._load("gradient_vote")
        # 21 one-dimensional combinations plus the explicit baseline.
        self.assertEqual(len(generate(config)), 22)
        self.assertEqual(set(config["parameters"]), {"border_search_fraction"})

        values = config["parameters"]["border_search_fraction"]["values"]
        self.assertEqual(min(values), 0.05)
        self.assertEqual(max(values), 0.15)
        self.assertIn(0.15, values)
        self.assertEqual(len(values), 21)
        self.assertTrue(
            all(round(values[i + 1] - values[i], 3) == 0.005 for i in range(len(values) - 1))
        )

        baseline = config["profiles"]["baseline"]
        self.assertEqual(baseline["border_search_fraction"], 0.15)
        self.assertEqual(baseline["minimum_span_fraction"], 0.15)
        self.assertEqual(baseline["central_band_fraction"], 1.0)
        self.assertEqual(baseline["gaussian_sigma"], 1.2)
        self.assertEqual(baseline["gradient_percentile"], 70.0)
        self.assertEqual(baseline["minimum_vote_support"], 0.08)
        self.assertEqual(baseline["support_weight"], 0.45)
        self.assertEqual(baseline["vote_smooth_fraction"], 0.012)


if __name__ == "__main__":
    unittest.main()
