from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.normalize_document_images import (
    POLICY_ID,
    axis_aligned_bounds,
    axis_aligned_crop,
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

    def test_workflow_reuses_canonical_geometry_without_detector_inference(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/normalize-gs0002.yml").read_text(encoding="utf-8")

        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("metadata/canonical-build-evidence.json", workflow)
        self.assertIn("analysis/page-analysis.json", workflow)
        self.assertIn("python -m hth.normalize_document_images", workflow)
        self.assertIn("specific_runner:", workflow)
        self.assertIn("custom_runner_label:", workflow)
        self.assertNotIn("run_document_detector", workflow)
        self.assertNotIn("setup-hth-managed-runtime", workflow)
        self.assertNotIn("git push", workflow)


if __name__ == "__main__":
    unittest.main()
