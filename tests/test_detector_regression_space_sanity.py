import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETECTOR_ROOT = ROOT / "config" / "detectors"
MAX_FULL_PARAMETER_SETS = 50176


class DetectorRegressionSpaceSanityTests(unittest.TestCase):
    def test_declared_full_spaces_are_bounded(self):
        configs = sorted(DETECTOR_ROOT.glob("*.json"))
        self.assertTrue(configs)
        for path in configs:
            config = json.loads(path.read_text(encoding="utf-8"))
            parameters = config.get("parameters", {})
            count = math.prod(
                len(spec.get("values", []))
                for spec in parameters.values()
            ) if parameters else 1
            self.assertLessEqual(
                count,
                MAX_FULL_PARAMETER_SETS,
                f"{path.name} declares {count:,} parameter sets",
            )
            regression = config.get("regression", {})
            self.assertEqual(
                regression.get("historic_best_reference"),
                "mandatory-exact",
                path.name,
            )

    def test_full_regression_runner_always_evaluates_historic_best(self):
        text = (ROOT / "hth" / "regression" / "runner.py").read_text(encoding="utf-8")
        materialization = (ROOT / "hth" / "regression" / "materialization.py").read_text(encoding="utf-8")
        self.assertIn('historic_best_result["reference_roles"] = ["historic_best"]', text)
        self.assertIn('"historic_best": outcome.historic_best', materialization)
        self.assertIn("historic_best_parameters", text)

    def test_oversized_spaces_were_reduced_to_sane_bounds(self):
        expected = {
            "adaptive_radial_edge": 49152,
            "border_fusion_quad": 48020,
            "contour_quad": 41472,
            "grabcut_contour": 46656,
            "multi_scale_radial_edge": 48334,
            "page_background": 48400,
            "radial_edge": 50000,
            "segment_supported_polar_vote": 45360,
            "signed_polar_boundary_vote": 46875,
        }
        for detector, expected_count in expected.items():
            config = json.loads((DETECTOR_ROOT / f"{detector}.json").read_text(encoding="utf-8"))
            count = math.prod(len(spec.get("values", [])) for spec in config["parameters"].values())
            self.assertEqual(count, expected_count, detector)


if __name__ == "__main__":
    unittest.main()
