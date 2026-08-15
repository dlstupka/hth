import json
import unittest
from pathlib import Path
import numpy as np

from hth.geometry import detector_amsre_bfq_spbv_pbg
from hth.regression.parameter_space import parameter_set_id
from hth.regression.strategies.cartesian import generate


class FusionGen2Tests(unittest.TestCase):
    def test_child_anchor_parameter_ids_are_stable(self) -> None:
        for method, item in detector_amsre_bfq_spbv_pbg.CHILD_CALIBRATIONS.items():
            with self.subTest(method=method):
                self.assertEqual(parameter_set_id(item["parameters"]), item["parameter_set_id"])

    def test_amsre_child_uses_calibrated_winner(self) -> None:
        item = detector_amsre_bfq_spbv_pbg.CHILD_CALIBRATIONS["adaptive_multi_scale_radial_edge"]
        self.assertEqual(item["parameter_set_id"], "21ea516c3c5a")
        self.assertEqual(item["parameters"], {
            "coarse_angle_step_degrees": 2.0454545454545454,
            "maximum_refined_sides": 3,
            "refined_angle_step_degrees": 0.35,
            "side_assignment_tolerance_fraction": 0.0075,
            "weak_side_support_fraction": 0.65,
        })

    def test_baseline_is_completed_gen2_full_winner(self) -> None:
        self.assertEqual(detector_amsre_bfq_spbv_pbg.BASELINE_PARAMETERS, {
            "minimum_side_consensus": 0.10,
            "consensus_tolerance_fraction": 0.012664,
            "gradient_weight": 0.25,
            "gradient_percentile": 76.0,
            "consensus_weight": 0.6,
            "source_diversity_weight": 0.15,
            "minimum_side_gradient_support": 0.03,
        })

    def test_config_matches_detector_contract(self) -> None:
        payload = json.loads(
            Path("config/detectors/amsre_bfq_spbv_pbg.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["child_calibrations"], detector_amsre_bfq_spbv_pbg.CHILD_CALIBRATIONS)
        self.assertEqual(payload["profiles"]["baseline"], detector_amsre_bfq_spbv_pbg.BASELINE_PARAMETERS)
        self.assertEqual(payload["parent_fusion_parameter_set_id"], "7b7dbac43ea6")

    def test_search_space_is_50176_active_dimension_refinement(self) -> None:
        payload = json.loads(
            Path("config/detectors/amsre_bfq_spbv_pbg.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(payload["parameters"]),
            {"minimum_side_consensus", "consensus_tolerance_fraction"},
        )
        size = 1
        for spec in payload["parameters"].values():
            size *= len(spec["values"])
        self.assertEqual(size, 50176)
        self.assertEqual(len(generate(payload)), 50177)

        minimum_side = payload["parameters"]["minimum_side_consensus"]["values"]
        tolerance = payload["parameters"]["consensus_tolerance_fraction"]["values"]
        self.assertEqual(len(minimum_side), 224)
        self.assertEqual(len(tolerance), 224)
        self.assertEqual(min(minimum_side), 0.05)
        self.assertEqual(max(minimum_side), 0.55)
        self.assertEqual(min(tolerance), 0.006)
        self.assertEqual(max(tolerance), 0.036)
        self.assertLess(min(minimum_side), 0.10)
        self.assertTrue(any(abs(value - 0.012664) < 0.0001 for value in tolerance))
        self.assertTrue(any(abs(value - 0.0271) < 0.0001 for value in tolerance))

    def test_dormant_fusion_dimensions_are_pinned_not_searched(self) -> None:
        payload = json.loads(
            Path("config/detectors/amsre_bfq_spbv_pbg.json").read_text(encoding="utf-8")
        )
        for name in (
            "gradient_weight",
            "gradient_percentile",
            "consensus_weight",
            "source_diversity_weight",
            "minimum_side_gradient_support",
        ):
            self.assertNotIn(name, payload["parameters"])
            self.assertIn(name, payload["profiles"]["baseline"])

    def test_detector_contract_on_simple_page(self) -> None:
        image = np.full((360, 280, 3), 25, dtype=np.uint8)
        image[35:325, 30:250] = 225
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        candidate = detector_amsre_bfq_spbv_pbg.detect(image_bgr=image, mask=mask)
        self.assertIn(candidate.status, {"ok", "no_candidate"})
        if candidate.status == "ok":
            self.assertEqual(candidate.method, "amsre_bfq_spbv_pbg")
            self.assertEqual(len(candidate.corners), 4)
            self.assertIn("child_calibrations", candidate.diagnostics)


if __name__ == "__main__":
    unittest.main()
