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
)
from hth.canonical_build_evidence import canonical_hash
from hth.integrate_photometric import RESULT_IDENTITY_FIELDS, integrate, package_release, prepare_plan
from hth.normalize_document_images import _pixel_sha256
from hth.validate_photometric_method import VALIDATION_IDENTITY_FIELDS, _derive_result


class PhotometricIntegrationTests(unittest.TestCase):
    @staticmethod
    def _config(name: str) -> dict:
        root = Path(__file__).resolve().parents[1]
        return json.loads((root / "config" / name).read_text(encoding="utf-8"))

    @staticmethod
    def _gradient_page(reverse: bool = False) -> np.ndarray:
        values = np.linspace(105, 245, 720, dtype=np.uint8)
        if reverse:
            values = values[::-1]
        page = cv2.cvtColor(np.tile(values, (900, 1)), cv2.COLOR_GRAY2BGR)
        for y in range(80, 850, 48):
            cv2.line(page, (45, y), (670, y), (35, 35, 35), 3)
        return page

    @staticmethod
    def _uniform_page() -> np.ndarray:
        page = np.full((900, 720, 3), 220, dtype=np.uint8)
        for y in range(80, 850, 48):
            cv2.line(page, (45, y), (670, y), (35, 35, 35), 3)
        return page

    def _fixtures(self, image_root: Path) -> tuple[dict, dict, dict, dict, dict, dict]:
        images = {1: self._gradient_page(), 2: self._gradient_page(True), 3: self._uniform_page()}
        for ordinal, image in images.items():
            self.assertTrue(cv2.imwrite(str(image_root / f"fs_{ordinal:04d}.png"), image))

        normalization_identity = "a" * 64
        normalization = {
            "canonical_result_identity": normalization_identity,
            "pages": [
                {"global_ordinal": ordinal, "output_pixel_sha256": _pixel_sha256(image)}
                for ordinal, image in sorted(images.items())
            ],
        }
        photometric_config = self._config("photometric-assessment.json")
        conditions = {ordinal: estimate_photometric_condition(image, photometric_config) for ordinal, image in images.items()}
        self.assertEqual(conditions[1]["decision"], "correction-candidate")
        self.assertEqual(conditions[2]["decision"], "correction-candidate")
        self.assertNotEqual(conditions[3]["decision"], "correction-candidate")
        photometric = {
            "schema_version": "1.0",
            "assessment_type": "photometric-uniformity-comparison",
            "status": "diagnostic-only",
            "canonical_normalization_result_identity": normalization_identity,
            "sample_identity": "b" * 64,
            "sample_page_count": 1,
            "config": photometric_config,
            "pages": [{"global_ordinal": 1, "source_pixel_sha256": _pixel_sha256(images[1]), "estimate": conditions[1]}],
        }
        photometric_fields = ("assessment_type", "canonical_normalization_result_identity", "sample_identity", "config", "pages")
        photometric["assessment_identity"] = canonical_hash({key: photometric[key] for key in photometric_fields})

        method_config = self._config("photometric-method-assessment.json")
        variants = []
        for method in method_config["methods"]:
            output = apply_method(images[1], method, method_config["background_field"])
            result = evaluate_variant(images[1], output, conditions[1], photometric_config, method_config["safety_gates"])
            result.update({"method_id": method["id"], "output_pixel_sha256": _pixel_sha256(output)})
            variants.append(result)
        method_pages = [{"global_ordinal": 1, "source_pixel_sha256": _pixel_sha256(images[1]), "variants": variants}]
        method_summary, globally_safe, recommended = _derive_recommendation(method_config, method_pages)
        self.assertIsNotNone(recommended)
        method_assessment = {
            "schema_version": "1.0",
            "assessment_type": "photometric-method-comparison",
            "status": "diagnostic-only",
            "canonical_normalization_result_identity": normalization_identity,
            "photometric_assessment_identity": photometric["assessment_identity"],
            "sample_identity": "c" * 64,
            "sample_page_count": 1,
            "config": method_config,
            "globally_safe_methods": globally_safe,
            "recommended_method_id": recommended["method_id"],
            "method_summary": method_summary,
            "pages": method_pages,
        }
        method_assessment["assessment_identity"] = canonical_hash({key: method_assessment[key] for key in METHOD_IDENTITY_FIELDS})
        method_policy = {
            "schema_version": "1.0",
            "policy_type": "photometric-method-validation",
            "policy_id": method_config["recommendation"]["policy_id"],
            "status": "validation-candidate",
            "action": "validate-method",
            "recommended_method_id": recommended["method_id"],
            "compatibility": {
                "canonical_normalization_result_identity": normalization_identity,
                "photometric_assessment_identity": photometric["assessment_identity"],
            },
            "evidence": {"assessment_identity": method_assessment["assessment_identity"], "globally_safe_methods": globally_safe},
        }
        method_policy["policy_identity"] = canonical_hash(method_policy)

        selected_method = next(method for method in method_config["methods"] if method["id"] == recommended["method_id"])
        page2_output = apply_method(images[2], selected_method, method_config["background_field"])
        page2_result = evaluate_variant(images[2], page2_output, conditions[2], photometric_config, method_config["safety_gates"])
        self.assertTrue(page2_result["safe"])
        page2_result["output_pixel_sha256"] = _pixel_sha256(page2_output)
        validation_pages = [
            {
                "global_ordinal": 2, "route": "apply", "archetype": conditions[2]["archetype"],
                "source_pixel_sha256": _pixel_sha256(images[2]), "output_pixel_sha256": _pixel_sha256(page2_output),
                "method_result": page2_result,
            },
            {
                "global_ordinal": 3, "route": "preserve", "archetype": conditions[3]["archetype"],
                "source_pixel_sha256": _pixel_sha256(images[3]), "output_pixel_sha256": _pixel_sha256(images[3]),
                "method_result": None,
            },
        ]
        validation_config = self._config("photometric-method-validation.json")
        validation_config["gates"]["minimum_held_out_correction_candidates"] = 1
        aggregate, gates, status, action = _derive_result(validation_config, validation_pages, 2)
        self.assertTrue(all(gates.values()))
        validation = {
            "schema_version": "1.0",
            "validation_type": "photometric-method-held-out-validation",
            "status": "diagnostic-only",
            "canonical_normalization_result_identity": normalization_identity,
            "photometric_assessment_identity": photometric["assessment_identity"],
            "method_assessment_identity": method_assessment["assessment_identity"],
            "method_policy_identity": method_policy["policy_identity"],
            "sample_identity": "d" * 64,
            "held_out_population_page_count": 2,
            "config": validation_config,
            "method": selected_method,
            "aggregate": aggregate,
            "gates": gates,
            "recommendation_status": status,
            "recommendation_action": action,
            "pages": validation_pages,
        }
        validation["validation_identity"] = canonical_hash({key: validation[key] for key in VALIDATION_IDENTITY_FIELDS})
        integration_policy = {
            "schema_version": "1.0",
            "policy_type": "photometric-method-integration",
            "policy_id": validation_config["recommendation"]["policy_id"],
            "status": "integration-candidate",
            "action": "prepare-integration",
            "method": selected_method,
            "compatibility": {
                "canonical_normalization_result_identity": normalization_identity,
                "photometric_assessment_identity": photometric["assessment_identity"],
                "method_assessment_identity": method_assessment["assessment_identity"],
                "method_policy_identity": method_policy["policy_identity"],
            },
            "evidence": {"validation_identity": validation["validation_identity"], "gates": gates, "aggregate": aggregate},
        }
        integration_policy["policy_identity"] = canonical_hash(integration_policy)
        return normalization, photometric, method_assessment, method_policy, validation, integration_policy

    def test_integration_reproduces_development_and_held_out_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            fixtures = self._fixtures(images)
            plan = prepare_plan(*fixtures)
            output = root / "output"
            result = integrate(images, *fixtures, plan, output, review_every=0)
            self.assertEqual(result["aggregate"]["page_count"], 3)
            self.assertEqual(result["aggregate"]["corrected_pages"], 2)
            self.assertEqual(result["aggregate"]["preserved_pages"], 1)
            self.assertEqual(result["aggregate"]["safe_corrected_pages"], 2)
            self.assertEqual(result["photometric_result_identity"], canonical_hash({key: result[key] for key in RESULT_IDENTITY_FIELDS}))
            self.assertEqual([page["pipeline_action"] for page in result["pages"]], [
                "corrected-and-continue", "corrected-and-continue", "preserve-and-continue",
            ])
            self.assertTrue((output / "photometric-normalized/fs_0003.png").is_file())

            first = root / "first.zip"
            second = root / "second.zip"
            tag = f"HTH-PHOTOMETRIC-{result['photometric_result_identity']}"
            first_record = package_release(output, first, tag)
            (output / "release.json").write_text(json.dumps(first_record), encoding="utf-8")
            second_record = package_release(output, second, tag)
            self.assertEqual(first_record["asset_sha256"], second_record["asset_sha256"])
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_plan_rejects_incomplete_population_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            images = Path(temporary)
            fixtures = list(self._fixtures(images))
            fixtures[4]["pages"] = fixtures[4]["pages"][:1]
            fixtures[4]["held_out_population_page_count"] = 1
            aggregate, gates, status, action = _derive_result(fixtures[4]["config"], fixtures[4]["pages"], 1)
            fixtures[4].update({"aggregate": aggregate, "gates": gates, "recommendation_status": status, "recommendation_action": action})
            fixtures[4]["validation_identity"] = canonical_hash({key: fixtures[4][key] for key in VALIDATION_IDENTITY_FIELDS})
            fixtures[5]["compatibility"] = {
                **fixtures[5]["compatibility"],
            }
            fixtures[5]["evidence"] = {"validation_identity": fixtures[4]["validation_identity"], "gates": gates, "aggregate": aggregate}
            fixtures[5]["policy_identity"] = canonical_hash({key: value for key, value in fixtures[5].items() if key != "policy_identity"})
            with self.assertRaisesRegex(ValueError, "does not partition"):
                prepare_plan(*fixtures)

    def test_workflow_exists_and_publishes_durable_release(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/integrate-photometric.yml").read_text(encoding="utf-8")
        self.assertIn("name: HTH normalize photometric integration", workflow)
        self.assertIn("python -m hth.integrate_photometric prepare", workflow)
        self.assertIn("python -m hth.integrate_photometric integrate", workflow)
        self.assertIn("python -m hth.integrate_photometric package", workflow)
        self.assertIn("gh release create", workflow)
        self.assertIn("photometric-normalization-manifest.json", workflow)
        self.assertIn("hth_hardened_persist", workflow)


if __name__ == "__main__":
    unittest.main()
