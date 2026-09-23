from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from hth.normalize_document_images import (
    POLICY_ID,
    _load_transform_policy,
    _pixel_sha256,
    axis_aligned_bounds,
    axis_aligned_crop,
    materialize_canonical_images,
    normalize,
)
from hth.canonical_build_evidence import canonical_hash
from hth.orientation_deskew import rotate_expand
from hth.normalization_report import should_render_review


class NormalizeDocumentImagesTests(unittest.TestCase):
    def _stale_policy_fixture(self, root: Path):
        policy = {
            "policy_type": "orientation-deskew-normalization",
            "policy_id": "hough-lines-conservative-v1",
            "status": "recommended-for-validation",
            "gross_orientation": {"action": "preserve"},
            "deskew": {
                "estimator": "hough-lines", "canvas": "expanded-white", "interpolation": "linear",
                "minimum_absolute_correction_degrees": 0.5,
                "maximum_absolute_correction_degrees": 1.5,
                "minimum_confidence": 0.7,
                "minimum_line_count": 20,
                "maximum_weighted_mad_degrees": 1.0,
            },
            "compatibility": {
                "canonical_preprocess_result_identity": "a" * 64,
                "detector": "test_detector",
                "parameter_identity_sha256": "p" * 64,
                "base_normalization_policy_id": POLICY_ID,
            },
            "evidence": {"canonical_normalization_result_identity": "n" * 64},
        }
        policy["policy_identity"] = canonical_hash(policy)
        policy_path = root / "policy.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        prior_path = root / "normalization/normalization-manifest.json"
        prior_path.parent.mkdir()
        corners = [[2, 2], [17, 2], [17, 17], [2, 17]]
        prior = {
            "canonical_preprocess": {"canonical_result_identity": "a" * 64},
            "canonical_result_identity": "n" * 64,
            "detector_selection": {"detector": "test_detector", "parameter_identity_sha256": "p" * 64},
            "policy": {"base_policy_id": POLICY_ID},
            "pages": [{
                "global_ordinal": 1,
                "source_sha256": "s" * 64,
                "detector_corners": corners,
                "base_crop_pixel_sha256": "c" * 64,
            }],
        }
        prior_path.write_text(json.dumps(prior), encoding="utf-8")
        evidence = {"canonical_result": {"identity": "b" * 64}}
        selection = {"detector": "test_detector", "parameter_identity_sha256": "p" * 64}
        manifest = {"records": [{"global_ordinal": 1, "sha256": "s" * 64}]}
        analysis = {"records": [{
            "global_ordinal": 1,
            "geometry_candidates": [{"method": "test_detector", "status": "ok", "corners": corners}],
        }]}
        return policy_path, prior_path, evidence, selection, manifest, analysis

    def test_stale_policy_requires_proven_identical_crop_basis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            policy_path, prior_path, evidence, selection, manifest, analysis = self._stale_policy_fixture(Path(temporary))
            with patch("hth.normalize_document_images.load_evidence_store", return_value={
                "authoritative_identity": "verified", "records": {"verified": {}}
            }), patch("hth.normalize_document_images.validate_published_results"):
                policy, crop_hashes = _load_transform_policy(
                    policy_path, evidence, selection, manifest, analysis, prior_path
                )
                self.assertEqual(policy["policy_id"], "hough-lines-conservative-v1")
                self.assertEqual(crop_hashes, {1: "c" * 64})
                manifest["records"][0]["sha256"] = "different"
                with self.assertRaisesRegex(ValueError, "source or crop"):
                    _load_transform_policy(policy_path, evidence, selection, manifest, analysis, prior_path)
                manifest["records"][0]["sha256"] = "s" * 64
                analysis["records"][0]["geometry_candidates"][0]["corners"] = [[3, 2], [17, 2], [17, 17], [2, 17]]
                with self.assertRaisesRegex(ValueError, "source or crop"):
                    _load_transform_policy(policy_path, evidence, selection, manifest, analysis, prior_path)

    def test_stale_policy_fails_closed_without_valid_published_basis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            policy_path, prior_path, evidence, selection, manifest, analysis = self._stale_policy_fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "no prior normalization basis"):
                _load_transform_policy(policy_path, evidence, selection, manifest, analysis, None)
            with patch("hth.normalize_document_images.load_evidence_store", return_value={
                "authoritative_identity": "verified", "records": {"verified": {}}
            }), patch("hth.normalize_document_images.validate_published_results", side_effect=ValueError("tampered manifest")):
                with self.assertRaisesRegex(ValueError, "tampered manifest"):
                    _load_transform_policy(policy_path, evidence, selection, manifest, analysis, prior_path)

    def test_stale_policy_rejects_different_cropped_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_path, prior_path, evidence, selection, manifest, analysis = self._stale_policy_fixture(root)
            images = root / "images"
            images.mkdir()
            image = np.full((20, 20, 3), 127, dtype=np.uint8)
            source = images / "fs_0001.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest["records"][0]["sha256"] = source_sha
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            prior["pages"][0]["source_sha256"] = source_sha
            prior_path.write_text(json.dumps(prior), encoding="utf-8")
            manifest_path = root / "image-manifest.json"
            analysis_path = root / "page-analysis.json"
            analysis["document_detector"] = selection
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
            evidence["effective_build_identity"] = "e" * 64
            evidence["canonical_result"]["pages"] = [{"global_ordinal": 1, "canonical_image_sha256": source_sha}]
            with patch("hth.normalize_document_images.load_evidence_store", return_value={
                "authoritative_identity": "verified", "records": {"verified": {}}
            }), patch("hth.normalize_document_images.validate_published_results"):
                with self.assertRaisesRegex(ValueError, "different cropped pixels"):
                    normalize(
                        None, images, manifest_path, analysis_path, evidence, root / "output",
                        transform_policy_path=policy_path,
                        prior_normalization_manifest_path=prior_path,
                    )

            prior["pages"][0]["base_crop_pixel_sha256"] = _pixel_sha256(image[2:17, 2:17])
            prior_path.write_text(json.dumps(prior), encoding="utf-8")
            with patch("hth.normalize_document_images.load_evidence_store", return_value={
                "authoritative_identity": "verified", "records": {"verified": {}}
            }), patch("hth.normalize_document_images.validate_published_results"):
                payload = normalize(
                    None, images, manifest_path, analysis_path, evidence, root / "verified-output",
                    contact_sheet_every=0,
                    transform_policy_path=policy_path,
                    prior_normalization_manifest_path=prior_path,
                )
            self.assertEqual(
                payload["transform_policy_compatibility"]["mode"],
                "verified-identical-base-crops",
            )
            self.assertEqual(payload["transform_policy_compatibility"]["pages_verified"], 1)

    def test_review_selection_is_noncanonical_and_always_includes_transforms(self) -> None:
        self.assertFalse(should_render_review(4, 10, 0, "preserve"))
        self.assertTrue(should_render_review(4, 10, 0, "apply"))
        self.assertTrue(should_render_review(0, 10, 25, "preserve"))
        self.assertTrue(should_render_review(9, 10, 25, "preserve"))

    def test_axis_aligned_crop_is_an_exact_pixel_slice(self) -> None:
        image = np.arange(40 * 60 * 3, dtype=np.uint8).reshape(40, 60, 3)
        corners = [[10.8, 5.2], [49.1, 6.0], [48.6, 34.2], [11.1, 33.8]]

        cropped, bounds = axis_aligned_crop(image, corners)

        self.assertEqual(bounds, (10, 5, 50, 35))
        self.assertTrue(np.array_equal(cropped, image[5:35, 10:50]))

    def test_bounds_clip_to_source_image(self) -> None:
        self.assertEqual(
            axis_aligned_bounds([[-3, -2], [25, -1], [26, 22], [-4, 23]], (20, 24, 3)),
            (0, 0, 24, 20),
        )

    def test_normalization_writes_lossless_review_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            image = np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3)
            source = images / "fs_0003.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()

            golden = root / "golden.json"
            golden.write_text(json.dumps({
                "collection_id": "HTH-GOLDEN-TEST",
                "pages": [{
                    "global_ordinal": 3,
                    "review_status": "approved",
                    "image_sha256": source_sha,
                }],
            }), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "records": [{"global_ordinal": 3, "sha256": source_sha}],
            }), encoding="utf-8")
            analysis = root / "analysis.json"
            analysis.write_text(json.dumps({
                "document_detector": {
                    "detector": "test_detector",
                    "parameter_set_id": "params-1",
                    "parameter_identity_sha256": "a" * 64,
                },
                "records": [{
                    "global_ordinal": 3,
                    "geometry_candidates": [{
                        "method": "test_detector",
                        "status": "ok",
                        "confidence": 0.99,
                        "corners": [[10.8, 5.2], [89.1, 6.0], [88.6, 74.2], [11.1, 73.8]],
                    }],
                }],
            }), encoding="utf-8")
            evidence = {
                "effective_build_identity": "b" * 64,
                "canonical_result": {
                    "identity": "c" * 64,
                    "pages": [{
                        "global_ordinal": 3,
                        "canonical_image_sha256": source_sha,
                        "canonical_page_result_sha256": "d" * 64,
                    }],
                },
            }

            output = root / "output"
            payload = normalize(golden, images, manifest, analysis, evidence, output)

            self.assertEqual(payload["policy"]["id"], POLICY_ID)
            self.assertEqual(payload["canonical_preprocess"]["activity"], "REUSED")
            self.assertEqual(payload["page_count"], 1)
            actual = cv2.imread(str(output / "normalized/fs_0003.png"), cv2.IMREAD_UNCHANGED)
            self.assertTrue(np.array_equal(actual, image[5:75, 10:90]))
            for relative in (
                "normalization-manifest.json", "normalization-manifest.csv",
                "preprocess-evidence.json", "geometry-evidence.json",
                "review-manifest.json", "contact-sheets/fs_0003.jpg", "index.html",
            ):
                self.assertTrue((output / relative).is_file(), relative)

    def test_collection_normalization_uses_every_canonical_manifest_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            image = np.full((30, 40, 3), 127, dtype=np.uint8)
            source = images / "fs_0001.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "collection_id": "HTH-TEST",
                "records": [{"global_ordinal": 1, "sha256": source_sha}],
            }), encoding="utf-8")
            analysis = root / "analysis.json"
            analysis.write_text(json.dumps({
                "document_detector": {"detector": "test", "parameter_identity_sha256": "a" * 64},
                "records": [{
                    "global_ordinal": 1,
                    "geometry_candidates": [{
                        "method": "test", "status": "ok", "confidence": 1.0,
                        "corners": [[2, 3], [38, 3], [38, 27], [2, 27]],
                    }],
                }],
            }), encoding="utf-8")
            evidence = {
                "effective_build_identity": "b" * 64,
                "canonical_result": {
                    "identity": "c" * 64,
                    "pages": [{
                        "global_ordinal": 1,
                        "canonical_image_sha256": source_sha,
                        "canonical_page_result_sha256": "d" * 64,
                    }],
                },
            }

            payload = normalize(None, images, manifest, analysis, evidence, root / "output", status="complete")

            self.assertEqual(payload["target"]["type"], "collection")
            self.assertEqual(payload["collection"]["id"], "HTH-TEST")
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["ordinals"], [1])

    def test_recommended_hough_policy_is_compatibility_checked_and_selective(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            level = np.full((700, 900, 3), 255, dtype=np.uint8)
            for y in range(60, 660, 24):
                cv2.line(level, (70, y), (830, y), (0, 0, 0), 2)
            image = rotate_expand(level, 1.0)
            source = images / "fs_0001.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "collection_id": "HTH-TEST",
                "records": [{"global_ordinal": 1, "sha256": source_sha}],
            }), encoding="utf-8")
            analysis = root / "analysis.json"
            analysis.write_text(json.dumps({
                "document_detector": {
                    "detector": "test",
                    "parameter_identity_sha256": "a" * 64,
                },
                "records": [{
                    "global_ordinal": 1,
                    "geometry_candidates": [{
                        "method": "test", "status": "ok", "confidence": 1.0,
                        "corners": [[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]],
                    }],
                }],
            }), encoding="utf-8")
            evidence = {
                "effective_build_identity": "b" * 64,
                "canonical_result": {
                    "identity": "c" * 64,
                    "pages": [{
                        "global_ordinal": 1,
                        "canonical_image_sha256": source_sha,
                        "canonical_page_result_sha256": "d" * 64,
                    }],
                },
            }
            policy = root / "policy.json"
            policy_payload = {
                "policy_type": "orientation-deskew-normalization",
                "policy_id": "hough-lines-conservative-v1",
                "status": "recommended-for-validation",
                "gross_orientation": {"action": "preserve", "degrees": 0},
                "deskew": {
                    "estimator": "hough-lines",
                    "minimum_absolute_correction_degrees": 0.5,
                    "maximum_absolute_correction_degrees": 1.5,
                    "minimum_confidence": 0.7,
                    "minimum_line_count": 20,
                    "maximum_weighted_mad_degrees": 1.0,
                    "canvas": "expanded-white",
                    "interpolation": "linear",
                },
                "compatibility": {
                    "canonical_preprocess_result_identity": "c" * 64,
                    "detector": "test",
                    "parameter_identity_sha256": "a" * 64,
                    "base_normalization_policy_id": POLICY_ID,
                },
            }
            from hth.canonical_build_evidence import canonical_hash
            policy_payload["policy_identity"] = canonical_hash(policy_payload)
            policy.write_text(json.dumps(policy_payload), encoding="utf-8")

            payload = normalize(
                None, images, manifest, analysis, evidence, root / "output",
                status="complete", contact_sheet_every=0, transform_policy_path=policy,
            )

            page = payload["pages"][0]
            self.assertEqual(page["transform_decision"], "apply")
            self.assertGreater(abs(page["deskew_correction_degrees"]), 0.5)
            self.assertEqual(payload["transform_summary"]["pages_transformed"], 1)
            self.assertEqual(payload["policy"]["transform_policy_id"], "hough-lines-conservative-v1")
            self.assertTrue((root / "output/normalization-policy.json").is_file())
            self.assertTrue((root / "output/contact-sheets/fs_0001.jpg").is_file())
            review = json.loads((root / "output/review-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(review["contact_sheet_ordinals"], [1])
            canonical = json.loads((root / "output/normalization-manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("contact_sheet_ordinals", canonical)
            self.assertNotIn("review_contact_sheet_ordinals", canonical)

            policy_payload["deskew"]["minimum_confidence"] = 0.1
            policy.write_text(json.dumps(policy_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity does not match"):
                normalize(
                    None, images, manifest, analysis, evidence, root / "tampered-output",
                    status="complete", transform_policy_path=policy,
                )

    def test_materialization_reconstructs_manifest_image_from_docx(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            image = Image.new("RGB", (12, 8), "white")
            encoded = BytesIO()
            image.save(encoded, format="PNG")
            data = encoded.getvalue()
            docx = source / "collection.docx"
            with zipfile.ZipFile(docx, "w") as archive:
                archive.writestr("word/media/image1.png", data)
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
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "collection_id": "HTH-TEST",
                "records": [{
                    "global_ordinal": 1,
                    "source_docx": docx.name,
                    "source_ordinal": 1,
                    "relationship_id": "rId1",
                    "media_path": "word/media/image1.png",
                    "embedded_bytes": len(data),
                    "embedded_sha256": hashlib.sha256(data).hexdigest(),
                    "word_crop_left": 0,
                    "word_crop_top": 0,
                    "word_crop_right": 0,
                    "word_crop_bottom": 0,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "width_px": 12,
                    "height_px": 8,
                    "mode": "RGB",
                    "detected_format": "PNG",
                }],
            }), encoding="utf-8")

            result = materialize_canonical_images(source, manifest, root / "materialized")

            self.assertEqual(result["page_count"], 1)
            self.assertEqual((root / "materialized/fs_0001.png").read_bytes(), data)

    def test_workflow_reuses_canonical_geometry_without_detector_inference(self) -> None:
        root = Path(__file__).resolve().parents[1]
        collection = (root / ".github/workflows/normalize.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", collection)
        self.assertIn("metadata/canonical-build-evidence.json", collection)
        self.assertIn("analysis/page-analysis.json", collection)
        self.assertIn("python -m hth.normalize_document_images", collection)
        self.assertIn("normalization_recipe:", collection)
        self.assertIn("prepared-recommendation", collection)
        self.assertIn("--transform-policy", collection)
        self.assertIn("runner_target:", collection)
        self.assertNotIn("specific_runner:", collection)
        self.assertNotIn("custom_runner_label:", collection)
        self.assertNotIn("run_document_detector", collection)
        self.assertNotIn("setup-hth-managed-runtime", collection)
        self.assertNotIn("git push", collection)
        self.assertIn("--source-root source-repo/images", collection)
        self.assertIn("--scope hth-normalization", collection)
        self.assertIn("canonical_normalization_policy:", collection)
        self.assertIn("upload_full_artifact:", collection)
        self.assertIn("--implementation hth-pipeline/hth/normalize_document_images.py", collection)
        self.assertNotIn("--implementation hth-pipeline/hth/normalization_report.py", collection)
        self.assertNotIn("--implementation hth-pipeline/hth/canonical_build_evidence.py", collection)
        cleanup = 'rm -rf normalized-collection "$RUNNER_TEMP/canonical-collection-images"'
        self.assertIn(cleanup, collection)
        self.assertLess(collection.index(cleanup), collection.index("python -m hth.normalize_document_images"))


if __name__ == "__main__":
    unittest.main()
