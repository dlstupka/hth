from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.assess_photometric import estimate_photometric_condition
from hth.assess_photometric_methods import (
    METHOD_IDENTITY_FIELDS,
    _derive_recommendation,
    apply_method,
    evaluate_variant,
    prepare_sample,
    recommend,
)
from hth.canonical_build_evidence import canonical_hash
from hth.normalize_document_images import _pixel_sha256


class PhotometricMethodAssessmentTests(unittest.TestCase):
    @staticmethod
    def _config(name: str) -> dict[str, object]:
        root = Path(__file__).resolve().parents[1]
        return json.loads((root / "config" / name).read_text(encoding="utf-8"))

    @staticmethod
    def _gradient_page() -> np.ndarray:
        gradient = np.tile(np.linspace(105, 245, 720, dtype=np.uint8), (900, 1))
        page = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
        for y in range(80, 850, 48):
            cv2.line(page, (45, y), (670, y), (35, 35, 35), 3)
        return page

    @staticmethod
    def _photometric_assessment(normalization_identity: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "assessment_type": "photometric-uniformity-comparison",
            "status": "diagnostic-only",
            "canonical_normalization_result_identity": normalization_identity,
            "sample_identity": "b" * 64,
            "sample_page_count": 3,
            "config": {},
            "pages": [
                {"global_ordinal": 8, "estimate": {
                    "archetype": "paper-page", "decision": "correction-candidate",
                    "pipeline_action": "evaluate-correction",
                    "correction_eligible": True, "correction_exclusion_reasons": [],
                    "valid_tile_count": 64, "background_grid_shape": [8, 8], "background_grid": [0.5] * 64,
                    "background_gradient_fit_r_squared": 0.9, "dominant_profile_step_fraction": 0.1,
                    "boundary_geometry_detected": False, "piecewise_geometry_detected": False,
                    "decision_reasons": ["uneven-background"],
                }},
                {"global_ordinal": 9, "estimate": {"archetype": "dark-polarity-frame", "decision": "preserve", "decision_reasons": ["intentional-dark-polarity"]}},
                {"global_ordinal": 10, "estimate": {"archetype": "paper-page", "decision": "review", "decision_reasons": ["threshold-review"]}},
            ],
        }
        fields = ("assessment_type", "canonical_normalization_result_identity", "sample_identity", "config", "pages")
        payload["assessment_identity"] = canonical_hash({key: payload[key] for key in fields})
        return payload

    def test_prepare_selects_only_persisted_paper_candidates(self) -> None:
        identity = "a" * 64
        plan = prepare_sample(
            self._photometric_assessment(identity),
            {"canonical_result_identity": identity, "pages": [{}, {}, {}]},
        )
        self.assertEqual(plan["sample_page_count"], 1)
        self.assertEqual(plan["pages"], [{"global_ordinal": 8, "reasons": ["uneven-background"]}])
        self.assertEqual(len(plan["sample_identity"]), 64)

    def test_prepare_rejects_tampered_upstream_assessment(self) -> None:
        identity = "a" * 64
        assessment = self._photometric_assessment(identity)
        assessment["pages"] = []
        with self.assertRaisesRegex(ValueError, "identity"):
            prepare_sample(assessment, {"canonical_result_identity": identity, "pages": []})

    def test_prepare_rejects_candidate_with_geometry_exclusion(self) -> None:
        identity = "a" * 64
        assessment = self._photometric_assessment(identity)
        estimate = assessment["pages"][0]["estimate"]
        estimate["correction_eligible"] = False
        estimate["correction_exclusion_reasons"] = ["noncoherent-background-variation"]
        fields = ("assessment_type", "canonical_normalization_result_identity", "sample_identity", "config", "pages")
        assessment["assessment_identity"] = canonical_hash({key: assessment[key] for key in fields})
        with self.assertRaisesRegex(ValueError, "not correction eligible"):
            prepare_sample(assessment, {"canonical_result_identity": identity, "pages": [{}, {}, {}]})

    def test_prepare_rejects_candidate_with_inconsistent_geometry_evidence(self) -> None:
        identity = "a" * 64
        assessment = self._photometric_assessment(identity)
        assessment["pages"][0]["estimate"]["background_gradient_fit_r_squared"] = 0.1
        fields = ("assessment_type", "canonical_normalization_result_identity", "sample_identity", "config", "pages")
        assessment["assessment_identity"] = canonical_hash({key: assessment[key] for key in fields})
        with self.assertRaisesRegex(ValueError, "inconsistent geometry eligibility evidence"):
            prepare_sample(assessment, {"canonical_result_identity": identity, "pages": [{}, {}, {}]})

    def test_fixed_method_is_pixel_deterministic(self) -> None:
        config = self._config("photometric-method-assessment.json")
        page = self._gradient_page()
        first = apply_method(page, config["methods"][0], config["background_field"])
        second = apply_method(page, config["methods"][0], config["background_field"])
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(_pixel_sha256(first), _pixel_sha256(second))

    def test_subtraction_reduces_gradient_without_losing_detail(self) -> None:
        photometric_config = self._config("photometric-assessment.json")
        method_config = self._config("photometric-method-assessment.json")
        page = self._gradient_page()
        condition = estimate_photometric_condition(page, photometric_config)
        corrected = apply_method(page, method_config["methods"][0], method_config["background_field"])
        result = evaluate_variant(page, corrected, condition, photometric_config, method_config["safety_gates"])
        self.assertTrue(result["safe"])
        self.assertGreater(result["background_span_reduction_fraction"], 0.25)
        self.assertGreater(result["high_frequency_correlation"], 0.97)

    def test_recommendation_rejects_tampered_method_evidence(self) -> None:
        config = self._config("photometric-method-assessment.json")
        variants = []
        for method in config["methods"]:
            safe = method["mode"] == "subtract"
            variants.append({
                "method_id": method["id"],
                "safe": safe,
                "checks": {
                    "background_span_reduction": safe,
                    "detail_preservation": True,
                    "endpoint_clipping": True,
                    "median_luminance_shift": True,
                },
                "background_span_reduction_fraction": float(method["strength"]),
                "high_frequency_correlation": 0.99,
                "score": 0.80 if method["mode"] == "subtract" else 0.70,
            })
        pages = [{"global_ordinal": 8, "variants": variants}]
        method_summary, globally_safe, recommended = _derive_recommendation(config, pages)
        assessment: dict[str, object] = {
            "schema_version": "1.0",
            "assessment_type": "photometric-method-comparison",
            "status": "diagnostic-only",
            "canonical_normalization_result_identity": "a" * 64,
            "photometric_assessment_identity": "b" * 64,
            "sample_identity": "c" * 64,
            "sample_page_count": 1,
            "config": config,
            "globally_safe_methods": globally_safe,
            "recommended_method_id": recommended["method_id"],
            "method_summary": method_summary,
            "pages": pages,
        }
        assessment["assessment_identity"] = canonical_hash({key: assessment[key] for key in METHOD_IDENTITY_FIELDS})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "assessment.json"
            path.write_text(json.dumps(assessment), encoding="utf-8")
            policy = recommend(path, root / "policy.json")
            self.assertEqual(policy["action"], "validate-method")
            assessment["recommended_method_id"] = "background-subtraction-75"
            path.write_text(json.dumps(assessment), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recommendation fields"):
                recommend(path, root / "tampered-recommendation.json")
            assessment["recommended_method_id"] = recommended["method_id"]
            assessment["pages"] = [{"global_ordinal": 8}]
            path.write_text(json.dumps(assessment), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity"):
                recommend(path, root / "tampered-policy.json")

    def test_workflow_persists_method_policy_without_applying_pixels(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/assess-photometric-methods.yml").read_text(encoding="utf-8")
        self.assertIn("python -m hth.assess_photometric_methods prepare", workflow)
        self.assertIn("python -m hth.assess_photometric_methods evaluate", workflow)
        self.assertIn("python -m hth.assess_photometric_methods recommend", workflow)
        self.assertIn("normalization/photometric-method-policy.json", workflow)
        self.assertIn("specific_runner:", workflow)
        self.assertNotIn("normalize_document_images", workflow)
        self.assertNotIn("git push", workflow)


if __name__ == "__main__":
    unittest.main()
