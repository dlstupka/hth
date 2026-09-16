from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hth.normalize_document_images import (
    POLICY_ID,
    axis_aligned_bounds,
    axis_aligned_crop,
    materialize_canonical_images,
    normalize,
)


class NormalizeDocumentImagesTests(unittest.TestCase):
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
                "contact-sheets/fs_0003.jpg", "index.html",
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
        for name in ("normalize-gs0002.yml", "normalize.yml"):
            workflow = (root / ".github/workflows" / name).read_text(encoding="utf-8")
            self.assertIn("workflow_dispatch:", workflow)
            self.assertIn("metadata/canonical-build-evidence.json", workflow)
            self.assertIn("analysis/page-analysis.json", workflow)
            self.assertIn("python -m hth.normalize_document_images", workflow)
            self.assertIn("specific_runner:", workflow)
            self.assertIn("custom_runner_label:", workflow)
            self.assertNotIn("run_document_detector", workflow)
            self.assertNotIn("setup-hth-managed-runtime", workflow)
            self.assertNotIn("git push", workflow)

        collection = (root / ".github/workflows/normalize.yml").read_text(encoding="utf-8")
        self.assertIn("--source-root source-repo/images", collection)
        self.assertIn("--scope hth-normalization", collection)
        self.assertIn("canonical_normalization_policy:", collection)
        self.assertIn("upload_full_artifact:", collection)


if __name__ == "__main__":
    unittest.main()
