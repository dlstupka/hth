import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hth.geometry import detector_amsre_doc_ufcn_fusion as detector
from hth.geometry.model import Candidate
from hth.geometry.registry import detector_names
from hth.regression.parameter_space import adaptive_parameter_sets, exhaustive_parameter_sets, parameter_set_id
from hth.regression.runner import PRECOMPUTED_EVIDENCE_LOADERS, PRECOMPUTED_EVIDENCE_PREPARERS


class FusionGen3Tests(unittest.TestCase):
    def test_child_calibrations_are_exact_authoritative_winners(self):
        for method, item in detector.CHILD_CALIBRATIONS.items():
            with self.subTest(method=method):
                self.assertEqual(parameter_set_id(item["parameters"]), item["parameter_set_id"])
        self.assertEqual(detector.CHILD_CALIBRATIONS["adaptive_multi_scale_radial_edge"]["parameter_set_id"], "21ea516c3c5a")
        self.assertEqual(detector.CHILD_CALIBRATIONS["doc_ufcn_page_mask"]["parameter_set_id"], "595002645fcc")

    def test_config_searches_only_small_arbitration_layer(self):
        payload = json.loads(Path("config/detectors/amsre_doc_ufcn_fusion.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["child_calibrations"], detector.CHILD_CALIBRATIONS)
        self.assertEqual(payload["profiles"]["baseline"], detector.BASELINE_PARAMETERS)
        self.assertEqual(len(exhaustive_parameter_sets(payload)), 28)
        self.assertEqual(len(adaptive_parameter_sets(payload)), 2340)
        self.assertEqual(payload["parameters"]["doc_ufcn_minimum_confidence"]["values"], [0.9])
        self.assertEqual(payload["parameters"]["amsre_rescue_score_ceiling"]["values"], [0.95])
        self.assertIn(0.01, payload["parameters"]["minimum_corner_disagreement_fraction"]["values"])
        self.assertIn(0.6, payload["parameters"]["maximum_amsre_refined_support_fraction"]["values"])
        self.assertIn(1.0, payload["parameters"]["maximum_amsre_refined_support_fraction"]["values"])
        self.assertEqual(payload["regression"]["historic_best_reference"], "mandatory-exact")
        self.assertEqual(payload["lifecycle"]["prepare"], "doc_ufcn_page_mask")

    def test_registry_workflows_and_precomputed_doc_ufcn_evidence_are_wired(self):
        self.assertIn(detector.METHOD, detector_names())
        self.assertIn(detector.METHOD, PRECOMPUTED_EVIDENCE_PREPARERS)
        self.assertIn(detector.METHOD, PRECOMPUTED_EVIDENCE_LOADERS)
        for workflow in (".github/workflows/regress-detector.yml", ".github/workflows/execution-optimizer.yml"):
            self.assertIn(f"          - {detector.METHOD}\n", Path(workflow).read_text(encoding="utf-8"))

    def test_doc_ufcn_rescues_only_when_all_gates_pass(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        amsre = Candidate("adaptive_multi_scale_radial_edge", [10,10,90,90], [[10,10],[90,10],[90,90],[10,90]], 0.7, 0.7, {})
        doc = Candidate("doc_ufcn_page_mask", [5,5,95,95], [[5,5],[95,5],[95,95],[5,95]], 0.95, 0.95, {"selected_confidence": 0.95})
        with patch.object(detector.detector_adaptive_multi_scale_radial_edge, "detect", return_value=amsre), patch.object(detector.detector_doc_ufcn_page_mask, "detect", return_value=doc):
            candidate = detector.detect(
                image_bgr=image, mask=mask,
                parameters={"amsre_rescue_score_ceiling":0.8, "doc_ufcn_minimum_confidence":0.9, "minimum_corner_disagreement_fraction":0.01},
            )
        self.assertEqual(candidate.method, detector.METHOD)
        self.assertEqual(candidate.corners, doc.corners)
        self.assertEqual(candidate.diagnostics["decision"], "doc-ufcn-refined-support-gated-rescue")

    def test_refined_support_gate_separates_strong_amsre_from_rescue_case(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        doc = Candidate("doc_ufcn_page_mask", [5,5,95,95], [[5,5],[95,5],[95,95],[5,95]], 0.99, 0.99, {"selected_confidence": 0.99})
        strong = Candidate(
            "adaptive_multi_scale_radial_edge", [10,10,90,90], [[10,10],[90,10],[90,90],[10,90]], 0.87, 0.87,
            {"refinement_triggered": True, "refined_supported_rays": 800, "total_supported_rays": 960},
        )
        weak = Candidate(
            "adaptive_multi_scale_radial_edge", [10,10,90,90], [[10,10],[90,10],[90,90],[10,90]], 0.89, 0.89,
            {"refinement_triggered": True, "refined_supported_rays": 215, "total_supported_rays": 384},
        )
        params = {
            "amsre_rescue_score_ceiling": 0.95,
            "doc_ufcn_minimum_confidence": 0.9,
            "minimum_corner_disagreement_fraction": 0.01,
            "maximum_amsre_refined_support_fraction": 0.6,
        }
        with patch.object(detector.detector_doc_ufcn_page_mask, "detect", return_value=doc):
            with patch.object(detector.detector_adaptive_multi_scale_radial_edge, "detect", return_value=strong):
                candidate = detector.detect(image_bgr=image, mask=mask, parameters=params)
                self.assertEqual(candidate.corners, strong.corners)
                self.assertFalse(candidate.diagnostics["rescue_gates"]["amsre_refined_support_below_ceiling"])
            with patch.object(detector.detector_adaptive_multi_scale_radial_edge, "detect", return_value=weak):
                candidate = detector.detect(image_bgr=image, mask=mask, parameters=params)
                self.assertEqual(candidate.corners, doc.corners)
                self.assertTrue(candidate.diagnostics["rescue_gates"]["amsre_refined_support_below_ceiling"])
                self.assertAlmostEqual(candidate.diagnostics["amsre_refined_support_fraction"], 215 / 384)
                self.assertIn("bbox_iou", candidate.diagnostics["candidate_geometry"])

    def test_amsre_remains_primary_when_rescue_gate_fails(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        amsre = Candidate("adaptive_multi_scale_radial_edge", [10,10,90,90], [[10,10],[90,10],[90,90],[10,90]], 0.95, 0.95, {})
        doc = Candidate("doc_ufcn_page_mask", [5,5,95,95], [[5,5],[95,5],[95,95],[5,95]], 0.99, 0.99, {"selected_confidence": 0.99})
        with patch.object(detector.detector_adaptive_multi_scale_radial_edge, "detect", return_value=amsre), patch.object(detector.detector_doc_ufcn_page_mask, "detect", return_value=doc):
            candidate = detector.detect(image_bgr=image, mask=mask, parameters={"amsre_rescue_score_ceiling":0.8})
        self.assertEqual(candidate.corners, amsre.corners)
        self.assertEqual(candidate.diagnostics["decision"], "amsre-primary")


if __name__ == "__main__":
    unittest.main()
