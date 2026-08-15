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

    def test_gradient_vote_refinement_is_small_lower_bound_diagnostic(self):
        config = self._load("gradient_vote")
        # 81 two-dimensional combinations plus the explicit baseline.
        self.assertEqual(len(generate(config)), 82)
        self.assertEqual(
            set(config["parameters"]),
            {"border_search_fraction", "minimum_span_fraction"},
        )
        for name in ("border_search_fraction", "minimum_span_fraction"):
            values = config["parameters"][name]["values"]
            self.assertIn(0.35, values)
            self.assertLess(min(values), 0.35)

        baseline = config["profiles"]["baseline"]
        self.assertEqual(baseline["central_band_fraction"], 1.0)
        self.assertEqual(baseline["gaussian_sigma"], 1.2)
        self.assertEqual(baseline["gradient_percentile"], 70.0)
        self.assertEqual(baseline["minimum_vote_support"], 0.08)
        self.assertEqual(baseline["support_weight"], 0.45)
        self.assertEqual(baseline["vote_smooth_fraction"], 0.012)


if __name__ == "__main__":
    unittest.main()
