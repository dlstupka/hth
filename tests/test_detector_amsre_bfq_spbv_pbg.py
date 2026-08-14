import json
import unittest
from pathlib import Path
import numpy as np

from hth.geometry import detector_amsre_bfq_spbv_pbg
from hth.regression.parameter_space import parameter_set_id


class FusionGen2Tests(unittest.TestCase):
    def test_child_anchor_parameter_ids_are_stable(self) -> None:
        for method, item in detector_amsre_bfq_spbv_pbg.CHILD_CALIBRATIONS.items():
            with self.subTest(method=method):
                self.assertEqual(parameter_set_id(item["parameters"]), item["parameter_set_id"])

    def test_amsre_child_uses_calibrated_winner(self) -> None:
        item = detector_amsre_bfq_spbv_pbg.CHILD_CALIBRATIONS["adaptive_multi_scale_radial_edge"]
        self.assertEqual(item["parameter_set_id"], "21ea516c3c5a")
        self.assertEqual(item["parameters"], {'coarse_angle_step_degrees': 2.0454545454545454, 'maximum_refined_sides': 3, 'refined_angle_step_degrees': 0.35, 'side_assignment_tolerance_fraction': 0.0075, 'weak_side_support_fraction': 0.65})

    def test_baseline_is_calibrated_gen1_fusion_winner(self) -> None:
        self.assertEqual(detector_amsre_bfq_spbv_pbg.BASELINE_PARAMETERS, {'minimum_side_consensus': 0.867713, 'consensus_tolerance_fraction': 0.031641, 'gradient_weight': 0.25, 'gradient_percentile': 76.0, 'consensus_weight': 0.6, 'source_diversity_weight': 0.15, 'minimum_side_gradient_support': 0.03})

    def test_config_matches_detector_contract(self) -> None:
        payload = json.loads(Path("config/detectors/amsre_bfq_spbv_pbg.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["child_calibrations"], detector_amsre_bfq_spbv_pbg.CHILD_CALIBRATIONS)
        self.assertEqual(payload["profiles"]["baseline"], detector_amsre_bfq_spbv_pbg.BASELINE_PARAMETERS)
        self.assertEqual(payload["parent_fusion_parameter_set_id"], "7b7dbac43ea6")

    def test_search_space_is_50176_sets_and_contains_gen1_winner(self) -> None:
        payload = json.loads(Path("config/detectors/amsre_bfq_spbv_pbg.json").read_text(encoding="utf-8"))
        size = 1
        for spec in payload["parameters"].values():
            size *= len(spec["values"])
        self.assertEqual(size, 50176)
        self.assertIn(0.867713, payload["parameters"]["minimum_side_consensus"]["values"])
        self.assertIn(0.031641, payload["parameters"]["consensus_tolerance_fraction"]["values"])

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
