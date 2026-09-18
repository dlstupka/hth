import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.canonical_build_evidence import (
    DENOISING_ASSESSMENT_SCOPE,
    DENOISING_INTEGRATION_SCOPE,
    DENOISING_METHOD_ASSESSMENT_SCOPE,
    DENOISING_VALIDATION_SCOPE,
    SHARPENING_ASSESSMENT_SCOPE,
    SHARPENING_INTEGRATION_SCOPE,
    SHARPENING_METHOD_ASSESSMENT_SCOPE,
    SHARPENING_VALIDATION_SCOPE,
    SCOPE_ARTIFACT_PROFILES,
    SCOPE_EVIDENCE_PATHS,
)
from hth.normalize_document_images import _pixel_sha256
from hth.normalize_restoration import assess, compare, integrate, package_release, validate
from hth.restoration_summary import summary_lines


class RestorationNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.chromatic = self.root / "chromatic"
        images = self.chromatic / "chromatic-normalized"
        images.mkdir(parents=True)
        pages = []
        rng = np.random.default_rng(17)
        for ordinal in range(1, 11):
            base = np.full((48, 64, 3), 185, np.uint8)
            cv2.rectangle(base, (10, 8), (54, 40), (35, 35, 35), 2)
            cv2.putText(base, str(ordinal), (18, 31), cv2.FONT_HERSHEY_SIMPLEX, .6, (80, 80, 80), 1)
            noise = rng.normal(0, 8, base.shape).astype(np.int16)
            image = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            target = images / f"fs_{ordinal:04d}.png"
            self.assertTrue(cv2.imwrite(str(target), image))
            pages.append({"global_ordinal": ordinal, "output_pixel_sha256": _pixel_sha256(image)})
        self.chromatic_manifest = {"chromatic_result_identity": "c" * 64, "pages": pages}

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def assessment_config(domain):
        gates = ({"minimum_noise_sigma": 0.0, "minimum_impulse_fraction": 0.0} if domain == "denoising"
                 else {"maximum_candidate_detail_energy": 999.0, "maximum_safe_noise_sigma": 999.0})
        return {"domain": domain, "assessment_type": f"{domain}-assessment", "candidate_gates": gates,
                "partition": {"development_modulus": 2, "development_residue": 0}}

    @staticmethod
    def method_config(domain):
        if domain == "denoising":
            methods = [{"id": "bilateral-test", "mode": "bilateral", "sigma": 12.0}]
            gates = {"minimum_noise_reduction_fraction": -1.0, "minimum_detail_correlation": -1.0,
                     "minimum_edge_retention_fraction": 0.0, "maximum_new_clipping_fraction": 1.0,
                     "maximum_absolute_median_luminance_shift": 1.0}
        else:
            methods = [{"id": "unsharp-test", "mode": "unsharp-mask", "strength": 0.25, "radius": 1.0}]
            gates = {"minimum_detail_gain_fraction": -1.0, "maximum_detail_gain_fraction": 100.0,
                     "minimum_detail_correlation": -1.0, "maximum_noise_amplification_fraction": 100.0,
                     "maximum_new_clipping_fraction": 1.0, "maximum_absolute_median_luminance_shift": 1.0}
        return {"domain": domain, "assessment_type": f"{domain}-methods", "methods": methods,
                "safety_gates": gates}

    def run_family(self, domain, collection, upstream):
        assessment = assess(domain, collection, upstream, self.assessment_config(domain))
        comparison = compare(domain, collection, upstream, assessment, self.method_config(domain))
        validation = validate(domain, collection, upstream, assessment, comparison,
                              {"domain": domain, "validation_type": f"{domain}-validation"})
        output = self.root / f"{domain}-output"
        manifest = integrate(domain, collection, upstream, assessment, comparison, validation, output)
        return assessment, comparison, validation, output, manifest

    def test_four_stage_families_chain_and_preserve_provenance(self):
        denoise = self.run_family("denoising", self.chromatic, self.chromatic_manifest)
        self.assertEqual(denoise[2]["decision"], "apply")
        self.assertEqual(denoise[4]["aggregate"]["page_count"], 10)
        self.assertEqual(denoise[4]["aggregate"]["corrected_pages"], 10)
        sharpen = self.run_family("sharpening", denoise[3], denoise[4])
        self.assertEqual(sharpen[4]["upstream_result_identity"], denoise[4]["denoising_result_identity"])
        self.assertEqual(sharpen[4]["aggregate"]["page_count"], 10)
        self.assertTrue((sharpen[3] / "sharpening-normalized/fs_0001.png").is_file())

    def test_preserve_decision_is_pixel_identical(self):
        assessment = assess("denoising", self.chromatic, self.chromatic_manifest, self.assessment_config("denoising"))
        comparison = compare("denoising", self.chromatic, self.chromatic_manifest, assessment, self.method_config("denoising"))
        validation = validate("denoising", self.chromatic, self.chromatic_manifest, assessment, comparison, {"domain": "denoising", "validation_type": "test"})
        validation["decision"] = "preserve"
        output = self.root / "preserved"
        manifest = integrate("denoising", self.chromatic, self.chromatic_manifest, assessment, comparison, validation, output)
        self.assertEqual(manifest["aggregate"]["preserved_pages"], 10)
        for page in manifest["pages"]:
            self.assertEqual(page["input_pixel_sha256"], page["output_pixel_sha256"])

    def test_release_package_is_deterministic(self):
        *_, output, manifest = self.run_family("denoising", self.chromatic, self.chromatic_manifest)
        tag = f"HTH-DENOISING-{manifest['denoising_result_identity']}"
        first = package_release("denoising", output, self.root / "first.zip", tag)
        second = package_release("denoising", output, self.root / "second.zip", tag)
        self.assertEqual(first["asset_sha256"], second["asset_sha256"])

    def test_cbe_registry_and_human_summary_cover_all_eight_stages(self):
        scopes = {DENOISING_ASSESSMENT_SCOPE, DENOISING_METHOD_ASSESSMENT_SCOPE,
                  DENOISING_VALIDATION_SCOPE, DENOISING_INTEGRATION_SCOPE,
                  SHARPENING_ASSESSMENT_SCOPE, SHARPENING_METHOD_ASSESSMENT_SCOPE,
                  SHARPENING_VALIDATION_SCOPE, SHARPENING_INTEGRATION_SCOPE}
        self.assertTrue(scopes <= set(SCOPE_ARTIFACT_PROFILES))
        self.assertTrue(scopes <= set(SCOPE_EVIDENCE_PATHS))
        comparison = {"domain": "denoising", "candidate_count": 0, "globally_safe_methods": [],
                      "recommended_method_id": None, "config": {"methods": []}, "pages": []}
        rendered = "\n".join(summary_lines("compare", comparison))
        self.assertIn("Bounded denoising method comparison", rendered)
        self.assertIn("Gate failures", rendered)

    def test_workflows_expose_both_four_stage_chains_and_cache_restore(self):
        root = Path(__file__).resolve().parents[1]
        orchestrator = (root / ".github/workflows/normalize.yml").read_text(encoding="utf-8")
        core = (root / ".github/workflows/_core-restoration-evidence.yml").read_text(encoding="utf-8")
        integration = (root / ".github/workflows/integrate-restoration.yml").read_text(encoding="utf-8")
        for stage in ("denoising-assessment", "denoising-method-assessment", "denoising-validation",
                      "denoising-integration", "sharpening-assessment", "sharpening-method-assessment",
                      "sharpening-validation", "sharpening-integration"):
            self.assertIn(stage, orchestrator)
        self.assertIn("restore-immutable-release", core)
        self.assertIn("restore-immutable-release", integration)
        self.assertIn("release_provenance_summary", integration)


if __name__ == "__main__":
    unittest.main()
