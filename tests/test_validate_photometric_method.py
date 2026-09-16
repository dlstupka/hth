from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hth.assess_photometric_methods import METHOD_IDENTITY_FIELDS, _derive_recommendation
from hth.canonical_build_evidence import canonical_hash
from hth.validate_photometric_method import VALIDATION_IDENTITY_FIELDS, _derive_result, prepare_sample, recommend


class PhotometricMethodValidationTests(unittest.TestCase):
    @staticmethod
    def _validation_config() -> dict[str, object]:
        root = Path(__file__).resolve().parents[1]
        return json.loads((root / "config/photometric-method-validation.json").read_text(encoding="utf-8"))

    @staticmethod
    def _fixtures() -> tuple[dict, dict, dict, dict]:
        normalization_identity = "a" * 64
        normalization = {
            "canonical_result_identity": normalization_identity,
            "pages": [{"global_ordinal": ordinal} for ordinal in range(1, 6)],
        }
        photometric = {
            "schema_version": "1.0",
            "assessment_type": "photometric-uniformity-comparison",
            "status": "diagnostic-only",
            "canonical_normalization_result_identity": normalization_identity,
            "sample_identity": "b" * 64,
            "config": {},
            "pages": [
                {"global_ordinal": 1, "estimate": {"archetype": "dark-polarity-frame", "decision": "preserve"}},
                {"global_ordinal": 2, "estimate": {"archetype": "paper-page", "decision": "correction-candidate"}},
                {"global_ordinal": 3, "estimate": {"archetype": "paper-page", "decision": "review"}},
                {"global_ordinal": 4, "estimate": {"archetype": "paper-page", "decision": "preserve"}},
            ],
        }
        fields = ("assessment_type", "canonical_normalization_result_identity", "sample_identity", "config", "pages")
        photometric["assessment_identity"] = canonical_hash({key: photometric[key] for key in fields})
        method_config = {
            "schema_version": "1.0",
            "assessment_type": "photometric-method-comparison",
            "background_field": {},
            "methods": [{"id": "background-subtraction-50", "mode": "subtract", "strength": 0.5}],
            "safety_gates": {},
            "recommendation": {
                "policy_id": "test-method",
                "require_one_method_safe_for_every_candidate": True,
                "selection_rule": "minimum-strength-then-highest-score",
            },
        }
        variant = {
            "method_id": "background-subtraction-50",
            "safe": True,
            "checks": {
                "background_span_reduction": True,
                "detail_preservation": True,
                "endpoint_clipping": True,
                "median_luminance_shift": True,
            },
            "background_span_reduction_fraction": 0.5,
            "high_frequency_correlation": 0.99,
            "score": 0.67,
        }
        method_pages = [{"global_ordinal": 2, "variants": [variant]}]
        summaries, globally_safe, recommended = _derive_recommendation(method_config, method_pages)
        method_assessment = {
            "schema_version": "1.0",
            "assessment_type": "photometric-method-comparison",
            "status": "diagnostic-only",
            "canonical_normalization_result_identity": normalization_identity,
            "photometric_assessment_identity": photometric["assessment_identity"],
            "sample_identity": "c" * 64,
            "config": method_config,
            "globally_safe_methods": globally_safe,
            "recommended_method_id": recommended["method_id"],
            "method_summary": summaries,
            "pages": method_pages,
        }
        method_assessment["assessment_identity"] = canonical_hash({key: method_assessment[key] for key in METHOD_IDENTITY_FIELDS})
        policy = {
            "schema_version": "1.0",
            "policy_type": "photometric-method-validation",
            "policy_id": "test-method",
            "status": "validation-candidate",
            "action": "validate-method",
            "recommended_method_id": "background-subtraction-50",
            "compatibility": {
                "canonical_normalization_result_identity": normalization_identity,
                "photometric_assessment_identity": photometric["assessment_identity"],
            },
            "evidence": {
                "assessment_identity": method_assessment["assessment_identity"],
                "globally_safe_methods": globally_safe,
            },
        }
        policy["policy_identity"] = canonical_hash(policy)
        return normalization, photometric, method_assessment, policy

    def test_plan_covers_complete_population_except_development_pages(self) -> None:
        normalization, photometric, assessment, policy = self._fixtures()
        plan = prepare_sample(normalization, photometric, assessment, policy, self._validation_config())
        self.assertEqual(plan["held_out_population_page_count"], 4)
        self.assertEqual(plan["excluded_development_pages"], [2])
        self.assertEqual([page["global_ordinal"] for page in plan["pages"]], [1, 3, 4, 5])
        self.assertIn("known-protected-control", plan["pages"][0]["reasons"])

    def test_complete_population_without_targets_preserves_collection(self) -> None:
        pages = [{
            "route": "preserve", "archetype": "paper-page",
            "source_pixel_sha256": "a", "output_pixel_sha256": "a",
        } for _ in range(4)]
        aggregate, gates, status, action = _derive_result(self._validation_config(), pages, 4)
        self.assertTrue(gates["held_out_population_coverage"])
        self.assertFalse(gates["held_out_candidate_count"])
        self.assertEqual(status, "integration-not-justified")
        self.assertEqual(action, "preserve")
        self.assertEqual(aggregate["held_out_correction_candidates"], 0)

    def test_any_unsafe_candidate_blocks_integration(self) -> None:
        pages = [
            {"route": "apply", "archetype": "paper-page", "source_pixel_sha256": "a", "output_pixel_sha256": "b", "method_result": {"safe": True, "background_span_reduction_fraction": 0.5}},
            {"route": "apply", "archetype": "paper-page", "source_pixel_sha256": "c", "output_pixel_sha256": "d", "method_result": {"safe": False, "background_span_reduction_fraction": 0.1}},
            {"route": "preserve", "archetype": "dark-polarity-frame", "source_pixel_sha256": "e", "output_pixel_sha256": "e"},
        ]
        _, gates, status, action = _derive_result(self._validation_config(), pages, 3)
        self.assertFalse(gates["every_candidate_safe"])
        self.assertEqual(status, "validation-failed")
        self.assertEqual(action, "preserve")

    def test_safe_targets_and_protected_controls_allow_integration_candidate(self) -> None:
        pages = [
            {"route": "apply", "archetype": "paper-page", "source_pixel_sha256": "a", "output_pixel_sha256": "b", "method_result": {"safe": True, "background_span_reduction_fraction": 0.5}},
            {"route": "apply", "archetype": "paper-page", "source_pixel_sha256": "c", "output_pixel_sha256": "d", "method_result": {"safe": True, "background_span_reduction_fraction": 0.4}},
            {"route": "preserve", "archetype": "dark-polarity-frame", "source_pixel_sha256": "e", "output_pixel_sha256": "e"},
        ]
        _, gates, status, action = _derive_result(self._validation_config(), pages, 3)
        self.assertTrue(all(gates.values()))
        self.assertEqual(status, "integration-candidate")
        self.assertEqual(action, "prepare-integration")

    def test_recommendation_rejects_derived_field_tampering(self) -> None:
        config = self._validation_config()
        pages = [
            {"route": "apply", "archetype": "paper-page", "source_pixel_sha256": "a", "output_pixel_sha256": "b", "method_result": {"safe": True, "background_span_reduction_fraction": 0.5}},
            {"route": "apply", "archetype": "paper-page", "source_pixel_sha256": "c", "output_pixel_sha256": "d", "method_result": {"safe": True, "background_span_reduction_fraction": 0.4}},
        ]
        aggregate, gates, status, action = _derive_result(config, pages, 2)
        validation = {
            "schema_version": "1.0",
            "validation_type": "photometric-method-held-out-validation",
            "status": "diagnostic-only",
            "canonical_normalization_result_identity": "a" * 64,
            "photometric_assessment_identity": "b" * 64,
            "method_assessment_identity": "c" * 64,
            "method_policy_identity": "d" * 64,
            "sample_identity": "e" * 64,
            "held_out_population_page_count": 2,
            "config": config,
            "method": {"id": "background-subtraction-50", "mode": "subtract", "strength": 0.5},
            "aggregate": aggregate,
            "gates": gates,
            "recommendation_status": status,
            "recommendation_action": action,
            "pages": pages,
        }
        validation["validation_identity"] = canonical_hash({key: validation[key] for key in VALIDATION_IDENTITY_FIELDS})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "validation.json"
            path.write_text(json.dumps(validation), encoding="utf-8")
            self.assertEqual(recommend(path, root / "policy.json")["action"], "prepare-integration")
            validation["recommendation_action"] = "preserve"
            path.write_text(json.dumps(validation), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recommendation"):
                recommend(path, root / "tampered-policy.json")

    def test_workflow_is_diagnostic_and_persists_policy(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/validate-photometric-method.yml").read_text(encoding="utf-8")
        self.assertIn("python -m hth.validate_photometric_method prepare", workflow)
        self.assertIn("python -m hth.validate_photometric_method evaluate", workflow)
        self.assertIn("python -m hth.validate_photometric_method recommend", workflow)
        self.assertIn("normalization/photometric-integration-policy.json", workflow)
        self.assertIn("specific_runner:", workflow)
        self.assertNotIn("normalize_document_images", workflow)
        self.assertNotIn("git push", workflow)


if __name__ == "__main__":
    unittest.main()
