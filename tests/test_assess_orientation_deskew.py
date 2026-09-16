from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hth.canonical_build_evidence import canonical_hash
from hth.assess_orientation_deskew import (
    assess,
    estimate_hough_lines,
    estimate_projection_profile,
    materialize_sample,
    orientation_axis_scores,
    recommend_policy,
    rotate_expand,
    select_sample,
)
from hth.normalize_document_images import _pixel_sha256


class OrientationDeskewAssessmentTests(unittest.TestCase):
    @staticmethod
    def _manifest(page_count: int = 40) -> dict[str, object]:
        pages = []
        for ordinal in range(1, page_count + 1):
            source_width = 1000
            source_height = 800
            inset = 20 + ordinal
            pages.append({
                "global_ordinal": ordinal,
                "source_width": source_width,
                "source_height": source_height,
                "output_width": source_width - inset * 2,
                "output_height": source_height - 80,
                "crop_left": inset,
                "crop_top": 40,
                "crop_right_exclusive": source_width - inset,
                "crop_bottom_exclusive": source_height - 40,
                "detector_confidence": 1.0 - ordinal / 1000.0,
            })
        return {
            "collection": {"id": "HTH-TEST"},
            "normalization_identity": "a" * 64,
            "canonical_result_identity": "b" * 64,
            "pages": pages,
        }

    def test_sample_combines_golden_cadence_and_outliers(self) -> None:
        config = {
            "sampling": {
                "cadence": 10,
                "lowest_detector_confidence": 2,
                "lowest_area_retention": 2,
                "largest_crop_asymmetry": 2,
            }
        }
        sample = select_sample(self._manifest(), config, [3, 17])
        selected = {page["global_ordinal"]: set(page["reasons"]) for page in sample["pages"]}

        self.assertIn(1, selected)
        self.assertIn(40, selected)
        self.assertIn(3, selected)
        self.assertIn("golden-set", selected[17])
        self.assertIn("cadence-10", selected[11])
        self.assertIn("low-detector-confidence", selected[40])
        self.assertEqual(sample["population_page_count"], 40)
        self.assertEqual(len(sample["sample_identity"]), 64)

    def test_estimators_recover_small_synthetic_skew(self) -> None:
        image = np.full((500, 700, 3), 255, dtype=np.uint8)
        for y in range(80, 450, 35):
            cv2.line(image, (70, y), (630, y), (0, 0, 0), 3)
        skewed = rotate_expand(image, 2.0)

        projection = estimate_projection_profile(skewed, 3.0, 0.15, 0.25, 0.05)
        hough = estimate_hough_lines(skewed, 3.0, 0.15)
        orientation = orientation_axis_scores(skewed)

        self.assertAlmostEqual(projection["applied_correction_degrees"], -2.0, delta=0.2)
        self.assertAlmostEqual(hough["applied_correction_degrees"], -2.0, delta=0.2)
        self.assertGreater(projection["confidence"], 0.25)
        self.assertGreater(hough["confidence"], 0.5)
        self.assertEqual(orientation["preferred_axis"], "horizontal-text-axis")
        self.assertIn("indeterminate", orientation["upright_vs_upside_down"])

    def test_assessment_writes_bounded_diagnostic_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            image = np.full((300, 500, 3), 245, dtype=np.uint8)
            for y in range(50, 270, 28):
                cv2.line(image, (40, y), (460, y), (20, 20, 20), 2)
            image_path = images / "fs_0003.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            manifest = root / "normalization.json"
            manifest.write_text(json.dumps({
                "collection": None,
                "golden_set": {"id": "HTH-GOLDEN-TEST"},
                "canonical_result_identity": "c" * 64,
                "pages": [{"global_ordinal": 3}],
            }), encoding="utf-8")
            sample = root / "sample.json"
            sample.write_text(json.dumps({
                "canonical_normalization_result_identity": "c" * 64,
                "population_page_count": 1,
                "sample_identity": "d" * 64,
                "pages": [{"global_ordinal": 3, "reasons": ["golden-set"]}],
            }), encoding="utf-8")
            config = root / "config.json"
            config.write_text(json.dumps({
                "deskew": {
                    "maximum_correction_degrees": 3.0,
                    "deadband_degrees": 0.15,
                    "projection_coarse_step_degrees": 0.25,
                    "projection_fine_step_degrees": 0.05,
                    "agreement_tolerance_degrees": 0.25,
                    "conflict_threshold_degrees": 0.75,
                },
                "review": {"panel_width": 240, "panel_height": 200, "jpeg_quality": 85},
            }), encoding="utf-8")

            output = root / "assessment"
            payload = assess(images, manifest, sample, config, output)

            self.assertEqual(payload["status"], "diagnostic-only")
            self.assertEqual(payload["decision"], "manual-review-required")
            self.assertEqual(payload["collection"]["id"], "HTH-GOLDEN-TEST")
            self.assertEqual(payload["sample_page_count"], 1)
            for relative in (
                "assessment.json", "assessment.csv", "summary.md", "index.html",
                "contact-sheets/fs_0003.jpg",
            ):
                self.assertTrue((output / relative).is_file(), relative)
            self.assertFalse((output / "variants").exists())
            self.assertIn("No production normalization policy", (output / "summary.md").read_text(encoding="utf-8"))

    def test_recommendation_automates_safe_policy_but_requires_explicit_activation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assessment = root / "assessment.json"
            assessment_payload = {
                "schema_version": "2.0",
                "assessment_type": "orientation-deskew-comparison",
                "status": "diagnostic-only",
                "sample_identity": "a" * 64,
                "sample_page_count": 1,
                "population_page_count": 929,
                "canonical_normalization_result_identity": "b" * 64,
                "canonical_preprocess": {"canonical_result_identity": "c" * 64},
                "detector_selection": {
                    "detector": "doc_ufcn_page_mask",
                    "parameter_identity_sha256": "d" * 64,
                },
                "base_normalization_policy": {"id": "axis-aligned-detector-envelope-v1"},
                "config": {
                    "recommendation": {
                        "policy_id": "hough-lines-conservative-v1",
                        "minimum_sample_pages": 1,
                        "minimum_mean_hough_confidence": 0.75,
                        "minimum_hough_over_projection_confidence_gap": 0.4,
                        "maximum_hough_boundary_limited_pages": 0,
                        "gross_orientation_action": "preserve",
                        "deskew": {
                            "minimum_absolute_correction_degrees": 0.5,
                            "maximum_absolute_correction_degrees": 1.5,
                            "minimum_confidence": 0.7,
                            "minimum_line_count": 20,
                            "maximum_weighted_mad_degrees": 1.0,
                            "canvas": "expanded-white",
                            "interpolation": "linear",
                        },
                    }
                },
                "aggregate": {
                    "hough-lines": {
                        "mean_confidence": 0.84,
                        "maximum_absolute_correction_degrees": 1.38,
                        "boundary_limited_pages": 0,
                    },
                    "projection-profile": {"mean_confidence": 0.17},
                },
                "pages": [{
                    "global_ordinal": 26,
                    "estimators": {"hough-lines": {
                        "estimated_correction_degrees": -1.38,
                        "confidence": 0.72,
                        "line_count": 532,
                        "weighted_mad_degrees": 0.84,
                        "boundary_limited": False,
                    }},
                }],
            }
            assessment_payload["assessment_identity"] = canonical_hash({
                "assessment_type": assessment_payload["assessment_type"],
                "canonical_normalization_result_identity": assessment_payload["canonical_normalization_result_identity"],
                "sample_identity": assessment_payload["sample_identity"],
                "config": assessment_payload["config"],
                "pages": assessment_payload["pages"],
            })
            assessment.write_text(json.dumps(assessment_payload), encoding="utf-8")

            policy = recommend_policy(assessment, root / "normalization-policy.json")

            self.assertEqual(policy["status"], "recommended-for-validation")
            self.assertEqual(policy["activation"], "explicit-normalization-workflow-selection-required")
            self.assertEqual(policy["gross_orientation"]["action"], "preserve")
            self.assertEqual(policy["evidence"]["sample_pages_passing_transform_gates"][0]["global_ordinal"], 26)
            self.assertEqual(len(policy["policy_identity"]), 64)
            self.assertTrue((root / "recommendation.md").is_file())

            stale = dict(assessment_payload)
            stale["schema_version"] = "1.0"
            assessment.write_text(json.dumps(stale), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be regenerated"):
                recommend_policy(assessment, root / "stale-policy.json")

    def test_materialization_reconstructs_and_proves_exact_canonical_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_root.mkdir()
            image = Image.new("RGB", (12, 8), "white")
            for x in range(2, 10):
                image.putpixel((x, 3), (0, 0, 0))
            encoded = BytesIO()
            image.save(encoded, format="PNG")
            embedded = encoded.getvalue()
            docx = source_root / "collection.docx"
            with zipfile.ZipFile(docx, "w") as archive:
                archive.writestr("word/media/image1.png", embedded)
                archive.writestr("word/_rels/document.xml.rels", """\
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Target="media/image1.png"/>
</Relationships>""")
                archive.writestr("word/document.xml", """\
<document xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
  <pic:pic><pic:blipFill><a:blip r:embed="rId1"/></pic:blipFill></pic:pic>
</document>""")
            canonical_manifest = root / "canonical.json"
            canonical_manifest.write_text(json.dumps({
                "collection_id": "HTH-TEST",
                "records": [{
                    "global_ordinal": 1,
                    "source_docx": docx.name,
                    "source_ordinal": 1,
                    "relationship_id": "rId1",
                    "media_path": "word/media/image1.png",
                    "embedded_bytes": len(embedded),
                    "embedded_sha256": hashlib.sha256(embedded).hexdigest(),
                    "word_crop_left": 0,
                    "word_crop_top": 0,
                    "word_crop_right": 0,
                    "word_crop_bottom": 0,
                    "sha256": hashlib.sha256(embedded).hexdigest(),
                    "width_px": 12,
                    "height_px": 8,
                    "mode": "RGB",
                    "detected_format": "PNG",
                }],
            }), encoding="utf-8")
            decoded = cv2.imdecode(np.frombuffer(embedded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            crop = decoded[1:7, 2:10].copy()
            normalization_manifest = root / "normalization.json"
            normalized = rotate_expand(crop, 1.0)
            normalization_manifest.write_text(json.dumps({
                "canonical_result_identity": "c" * 64,
                "pages": [{
                    "global_ordinal": 1,
                    "source_sha256": hashlib.sha256(embedded).hexdigest(),
                    "source_width": 12,
                    "source_height": 8,
                    "crop_left": 2,
                    "crop_top": 1,
                    "crop_right_exclusive": 10,
                    "crop_bottom_exclusive": 7,
                    "output_width": normalized.shape[1],
                    "output_height": normalized.shape[0],
                    "output_pixel_sha256": _pixel_sha256(normalized),
                    "transform_decision": "apply",
                    "deskew_correction_degrees": 1.0,
                }],
            }), encoding="utf-8")
            sample = root / "sample.json"
            sample.write_text(json.dumps({
                "sample_identity": "d" * 64,
                "pages": [{"global_ordinal": 1, "reasons": ["golden-set"]}],
            }), encoding="utf-8")

            output = root / "materialized"
            payload = materialize_sample(
                source_root,
                canonical_manifest,
                normalization_manifest,
                sample,
                output,
            )

            self.assertEqual(payload["status"], "verified")
            self.assertEqual(payload["page_count"], 1)
            actual = cv2.imread(str(output / "normalized/fs_0001.png"), cv2.IMREAD_UNCHANGED)
            self.assertTrue(np.array_equal(actual, normalized))
            self.assertTrue((output / "materialization-evidence.json").is_file())

    def test_workflow_prepares_and_persists_a_reusable_recommendation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/assess-orientation-deskew.yml").read_text(encoding="utf-8")

        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("normalization/normalization-manifest.json", workflow)
        self.assertIn("normalization/canonical-build-evidence.json", workflow)
        self.assertIn("python -m hth.assess_orientation_deskew prepare", workflow)
        self.assertIn("python -m hth.assess_orientation_deskew materialize", workflow)
        self.assertIn("python -m hth.assess_orientation_deskew evaluate", workflow)
        self.assertIn("python -m hth.assess_orientation_deskew recommend", workflow)
        self.assertIn("normalization/orientation-deskew-policy.json", workflow)
        self.assertIn("hth_hardened_persist", workflow)
        self.assertIn("specific_runner:", workflow)
        self.assertIn("custom_runner_label:", workflow)
        self.assertIn("fromJSON(format('[\"self-hosted\",\"{0}\"]', inputs.custom_runner_label))", workflow)
        self.assertNotIn("run_document_detector", workflow)
        self.assertNotIn("git push", workflow)
        pipeline_checkout = workflow.split("- name: Checkout HTH pipeline", 1)[1].split("- name:", 1)[0]
        results_checkout = workflow.split("- name: Checkout canonical normalization evidence", 1)[1].split("- name:", 1)[0]
        self.assertIn("persist-credentials: false", pipeline_checkout)
        self.assertIn("persist-credentials: true", results_checkout)


if __name__ == "__main__":
    unittest.main()
