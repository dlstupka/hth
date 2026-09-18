from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from hth.assess_photometric import estimate_photometric_condition, recommend
from hth.canonical_build_evidence import canonical_hash


class PhotometricAssessmentTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict[str, object]:
        root = Path(__file__).resolve().parents[1]
        return json.loads((root / "config/photometric-assessment.json").read_text(encoding="utf-8"))

    @staticmethod
    def _document(background: np.ndarray | int) -> np.ndarray:
        if isinstance(background, np.ndarray):
            image = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        else:
            image = np.full((800, 600, 3), background, dtype=np.uint8)
        for y in range(80, 760, 50):
            cv2.line(image, (50, y), (550, y), (35, 35, 35), 3)
        return image

    def test_uniform_well_contrasted_page_is_preserved(self) -> None:
        result = estimate_photometric_condition(self._document(220), self._config())
        self.assertEqual(result["decision"], "preserve")
        self.assertLessEqual(result["background_luminance_span"], 0.10)

    def test_strong_illumination_gradient_is_candidate(self) -> None:
        gradient = np.tile(np.linspace(100, 250, 600, dtype=np.uint8), (800, 1))
        result = estimate_photometric_condition(self._document(gradient), self._config())
        self.assertEqual(result["decision"], "correction-candidate")
        self.assertEqual(result["pipeline_action"], "evaluate-correction")
        self.assertTrue(result["correction_eligible"])
        self.assertEqual(result["correction_exclusion_reasons"], [])
        self.assertIn("uneven-background", result["candidate_reasons"])

    def test_noncoherent_background_is_held_out_despite_another_correction_signal(self) -> None:
        config = deepcopy(self._config())
        config["estimator"]["candidate_maximum_tonal_span"] = 1.0
        structure = {
            "complete_grid": True,
            "full_span": 0.20,
            "interior_span": 0.20,
            "gradient_fit_r_squared": 0.10,
            "dominant_profile_step_fraction": 0.20,
        }
        with patch("hth.assess_photometric._background_structure", return_value=structure):
            result = estimate_photometric_condition(self._document(220), config)
        self.assertEqual(result["decision"], "review")
        self.assertEqual(result["pipeline_action"], "preserve-and-continue")
        self.assertFalse(result["correction_eligible"])
        self.assertIn("noncoherent-background-variation", result["correction_exclusion_reasons"])
        self.assertIn("compressed-tonal-range", result["correction_signal_reasons"])
        self.assertEqual(result["candidate_reasons"], [])

    def test_boundary_geometry_is_preserved_instead_of_becoming_a_candidate(self) -> None:
        image = self._document(198)
        height, width = image.shape[:2]
        cv2.rectangle(image, (0, 0), (width - 1, 28), (255, 255, 255), -1)
        cv2.rectangle(image, (0, height - 29), (width - 1, height - 1), (255, 255, 255), -1)
        cv2.rectangle(image, (0, 0), (28, height - 1), (255, 255, 255), -1)
        cv2.rectangle(image, (width - 29, 0), (width - 1, height - 1), (255, 255, 255), -1)
        result = estimate_photometric_condition(image, self._config())
        self.assertEqual(result["decision"], "preserve")
        self.assertEqual(result["pipeline_action"], "preserve-and-continue")
        self.assertFalse(result["correction_eligible"])
        self.assertIn("boundary-dominated-background-geometry", result["correction_exclusion_reasons"])
        self.assertEqual(result["decision_reasons"], ["boundary-dominated-background-geometry"])
        self.assertTrue(result["boundary_geometry_detected"])
        self.assertGreater(result["full_frame_background_luminance_span"], 0.18)
        self.assertLess(result["background_luminance_span"], 0.04)

    def test_abrupt_spread_step_is_preserved_instead_of_becoming_a_candidate(self) -> None:
        background = np.empty((800, 600), dtype=np.uint8)
        background[:, :300] = 155
        background[:, 300:] = 225
        result = estimate_photometric_condition(self._document(background), self._config())
        self.assertEqual(result["decision"], "preserve")
        self.assertEqual(result["pipeline_action"], "preserve-and-continue")
        self.assertFalse(result["correction_eligible"])
        self.assertIn("piecewise-page-background-geometry", result["correction_exclusion_reasons"])
        self.assertEqual(result["decision_reasons"], ["piecewise-page-background-geometry"])
        self.assertTrue(result["piecewise_geometry_detected"])
        self.assertGreater(result["dominant_profile_step_fraction"], 0.55)

    def test_small_background_variation_is_not_a_candidate(self) -> None:
        gradient = np.tile(np.linspace(190, 204, 600, dtype=np.uint8), (800, 1))
        result = estimate_photometric_condition(self._document(gradient), self._config())
        self.assertNotEqual(result["decision"], "correction-candidate")
        self.assertLess(result["background_luminance_span"], 0.04)

    def test_black_ink_clipping_is_not_sufficient_for_correction(self) -> None:
        image = np.full((800, 600, 3), 220, dtype=np.uint8)
        cv2.rectangle(image, (80, 80), (520, 380), (0, 0, 0), -1)
        result = estimate_photometric_condition(image, self._config())
        self.assertNotEqual(result["decision"], "correction-candidate")
        self.assertEqual(result["archetype"], "mixed-polarity-page")
        self.assertGreater(result["shadow_clipping_fraction"], 0.025)

    def test_dark_polarity_frame_is_explicitly_preserved(self) -> None:
        image = np.zeros((800, 600, 3), dtype=np.uint8)
        cv2.putText(image, "END OF ROLL", (40, 420), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (230, 230, 230), 8)
        result = estimate_photometric_condition(image, self._config())
        self.assertEqual(result["archetype"], "dark-polarity-frame")
        self.assertEqual(result["decision"], "preserve")
        self.assertEqual(result["candidate_reasons"], [])
        self.assertEqual(result["decision_reasons"], ["intentional-dark-polarity"])

    def test_recommendation_rejects_tampered_evidence(self) -> None:
        assessment = {
            "schema_version": "1.0",
            "assessment_type": "photometric-uniformity-comparison",
            "status": "diagnostic-only",
            "canonical_normalization_result_identity": "a" * 64,
            "sample_identity": "b" * 64,
            "sample_page_count": 79,
            "config": self._config(),
            "aggregate": {
                "preserve_pages": 70,
                "correction_candidate_pages": 0,
                "review_pages": 9,
                "inconclusive_pages": 0,
            },
            "pages": [],
        }
        fields = ("assessment_type", "canonical_normalization_result_identity", "sample_identity", "config", "pages")
        assessment["assessment_identity"] = canonical_hash({key: assessment[key] for key in fields})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "assessment.json"
            path.write_text(json.dumps(assessment), encoding="utf-8")
            self.assertEqual(recommend(path, root / "photometric-policy.json")["status"], "skip-recommended")
            assessment["pages"] = [{"global_ordinal": 1}]
            path.write_text(json.dumps(assessment), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity"):
                recommend(path, root / "tampered.json")

    def test_workflow_persists_photometric_policy(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/assess-photometric.yml").read_text(encoding="utf-8")
        self.assertIn("python -m hth.assess_photometric evaluate", workflow)
        self.assertIn("python -m hth.assess_photometric recommend", workflow)
        self.assertIn("/normalization/photometric/", workflow)
        self.assertIn("/normalization/photometric-policy.json", workflow)
        self.assertIn('manifest.get("canonical_result_identity")', workflow)
        self.assertNotIn('result_identity = str(record["canonical_result"]["identity"])', workflow)
        self.assertIn("runner_target:", workflow)
        self.assertNotIn("specific_runner:", workflow)
        self.assertNotIn("normalize_document_images", workflow)
        self.assertNotIn("git push", workflow)


if __name__ == "__main__":
    unittest.main()
