from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.assess_crop_framing import (
    ALGORITHMS,
    apply_framing,
    assess,
    order_corners,
    prepare_inputs,
    reprojection_error,
)


class CropFramingAssessmentTests(unittest.TestCase):
    def test_all_transforms_are_invertible_and_produce_images(self) -> None:
        image = np.full((180, 240, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (35, 25), (205, 155), (20, 20, 20), 3)
        corners = [[35, 25], [205, 35], [195, 155], [45, 150]]

        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                framed, matrix, source = apply_framing(image, corners, algorithm)
                self.assertGreater(framed.shape[0], 100)
                self.assertGreater(framed.shape[1], 150)
                self.assertLess(reprojection_error(source, matrix), 0.001)

    def test_corner_order_is_stable_for_shuffled_points(self) -> None:
        ordered = order_corners([[90, 80], [10, 10], [15, 85], [100, 20]])
        self.assertEqual(ordered[0].tolist(), [10.0, 10.0])
        self.assertGreater(float(cv2.contourArea(ordered)), 0.0)

    def test_assessment_writes_reviewable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            image_root.mkdir()
            image = np.full((180, 240, 3), 245, dtype=np.uint8)
            cv2.rectangle(image, (35, 25), (205, 155), (25, 25, 25), 4)
            self.assertTrue(cv2.imwrite(str(image_root / "fs_0003.png"), image))
            golden_set = root / "golden.json"
            golden_set.write_text(json.dumps({
                "schema_version": "1.0",
                "collection_id": "HTH-GOLDEN-TEST",
                "pages": [{
                    "global_ordinal": 3,
                    "review_status": "approved",
                    "physical_document_bbox": [30, 20, 210, 160],
                }],
            }), encoding="utf-8")
            analysis = root / "analysis.json"
            analysis.write_text(json.dumps({
                "document_detector": {
                    "detector": "test_detector",
                    "display_name": "Test Detector",
                    "parameter_set_id": "params-1",
                    "calibration_id": "cal-1",
                },
                "records": [{
                    "global_ordinal": 3,
                    "geometry_candidates": [{
                        "method": "test_detector",
                        "status": "ok",
                        "corners": [[35, 25], [205, 35], [195, 155], [45, 150]],
                    }],
                }],
            }), encoding="utf-8")

            output = root / "assessment"
            payload = assess(golden_set, image_root, analysis, output)

            self.assertEqual(payload["page_count"], 1)
            self.assertEqual(set(payload["algorithms"]), set(ALGORITHMS))
            self.assertTrue((output / "assessment.json").is_file())
            self.assertTrue((output / "assessment.csv").is_file())
            self.assertTrue((output / "index.html").is_file())
            self.assertTrue((output / "contact-sheets/fs_0003.jpg").is_file())
            for algorithm in ALGORITHMS:
                self.assertTrue((output / f"variants/{algorithm}/fs_0003.png").is_file())
            summary = (output / "summary.md").read_text(encoding="utf-8")
            self.assertIn("diagnostic evidence", summary)
            self.assertIn("axis-aligned boxes", summary)

    def test_prepare_inputs_uses_materialized_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            (image_root / "raw").mkdir(parents=True)
            (image_root / "raw/fs_0003.png").write_bytes(b"image")
            golden_set = root / "golden.json"
            golden_set.write_text(json.dumps({"pages": [{"global_ordinal": 3}]}), encoding="utf-8")
            manifest = root / "manifest.json"
            analysis = root / "analysis.json"

            prepare_inputs(golden_set, image_root, manifest, analysis)

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["records"][0]["raw_file"], "raw/fs_0003.png")
            self.assertEqual(json.loads(analysis.read_text(encoding="utf-8"))["records"], [{"global_ordinal": 3}])

    def test_workflow_is_gs0002_diagnostic_only(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/assess-crop-framing.yml").read_text(encoding="utf-8")

        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("HTH-GOLDEN-0002", workflow)
        self.assertIn("Resolve Rank #1 approved GS0002 detector", workflow)
        self.assertIn("python -m hth.assess_crop_framing evaluate", workflow)
        self.assertIn("runner:", workflow)
        self.assertIn("specific_runner:", workflow)
        self.assertIn("custom_runner_label:", workflow)
        self.assertIn("- self-hosted-e7k", workflow)
        self.assertIn("- self-hosted-e9k", workflow)
        self.assertIn("inputs.specific_runner == 'custom'", workflow)
        self.assertIn(
            "fromJSON(format('[\"self-hosted\",\"{0}\"]', inputs.custom_runner_label))",
            workflow,
        )
        self.assertIn("runner-label: ${{ env.HTH_SELECTED_RUNNER_LABEL }}", workflow)
        self.assertNotIn("git push", workflow)
        self.assertNotIn("publish_results", workflow)


if __name__ == "__main__":
    unittest.main()
