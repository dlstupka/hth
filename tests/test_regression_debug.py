from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.regression.adapters.components import detect as components_detect
from hth.regression.adapters.contour import detect as contour_detect
from hth.regression.adapters.ransac import detect as ransac_detect
from hth.regression.runner import write_debug_artifacts


class RegressionDebugTests(unittest.TestCase):
    def test_regression_adapter_populates_registry_provenance(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (280, 180), 255, -1)

        candidate = contour_detect(image_bgr=image, mask=mask)

        self.assertEqual(candidate.detector_name, "Contour")
        self.assertEqual(candidate.origin, "HTH")
        self.assertTrue(candidate.foundation)
        self.assertTrue(candidate.authors)
        self.assertTrue(candidate.version)
        self.assertTrue(candidate.repository)

    def test_failure_debug_directory_is_obvious_and_self_describing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.jpg"
            image = np.zeros((120, 180, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)
            page = {
                "global_ordinal": 6,
                "label": "title_or_index_sheet",
                "layout_type": "single_page",
                "image_path": str(image_path),
                "image": image,
                "mask": np.zeros((120, 180), dtype=np.uint8),
                "approved_bbox": [10, 10, 170, 110],
            }
            result = {
                "parameter_set_id": "baseline123",
                "pages": [{
                    "global_ordinal": 6,
                    "label": page["label"],
                    "layout_type": page["layout_type"],
                    "status": "no_candidate",
                    "iou": 0.0,
                    "candidate": {"diagnostics": {"reason": "no_plausible_contour"}},
                }],
            }

            outputs = write_debug_artifacts(
                root,
                "contour",
                "run-test",
                policy="failures",
                ranked=[result],
                pages=[page],
            )

            debug_root = root / "debug" / "contour" / "run-test"
            debug_page = debug_root / "baseline123" / "page-0006"
            self.assertTrue((debug_root / "README.txt").is_file())
            self.assertTrue((debug_page / "01-original.jpg").is_file())
            self.assertTrue((debug_page / "02-input-mask.png").is_file())
            self.assertTrue((debug_page / "03-overlay.jpg").is_file())
            diagnostics = json.loads((debug_page / "04-diagnostics.json").read_text())
            self.assertEqual(diagnostics["result"]["status"], "no_candidate")
            self.assertIn("debug/contour/run-test/README.txt", outputs)

    def test_components_debug_writes_detector_intermediate_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.jpg"
            image = np.zeros((120, 180, 3), dtype=np.uint8)
            mask = np.zeros((120, 180), dtype=np.uint8)
            mask[25:95, 30:150] = 255
            cv2.imwrite(str(image_path), image)
            candidate = components_detect(image_bgr=image, mask=mask)
            page = {
                "global_ordinal": 6,
                "label": "title_or_index_sheet",
                "layout_type": "single_page",
                "image_path": str(image_path),
                "image": image,
                "mask": mask,
                "approved_bbox": [10, 10, 170, 110],
            }
            result = {
                "parameter_set_id": "baseline123",
                "pages": [{
                    "global_ordinal": 6,
                    "label": page["label"],
                    "layout_type": page["layout_type"],
                    "status": "ok",
                    "iou": 0.5,
                    "candidate": candidate.__dict__,
                }],
            }

            write_debug_artifacts(
                root, "components", "run-test", policy="winner",
                ranked=[result], pages=[page],
            )

            debug_page = root / "debug" / "components" / "run-test" / "baseline123" / "page-0006"
            self.assertTrue((debug_page / "01-original.jpg").is_file())
            self.assertTrue((debug_page / "02-input-mask.png").is_file())
            self.assertTrue((debug_page / "03-after-morphology.png").is_file())
            self.assertTrue((debug_page / "04-component-labels.png").is_file())
            self.assertTrue((debug_page / "05-significant-components.png").is_file())
            self.assertTrue((debug_page / "06-selected-components.png").is_file())
            self.assertTrue((debug_page / "07-candidate-envelope.png").is_file())
            self.assertTrue((debug_page / "08-overlay.jpg").is_file())
            self.assertTrue((debug_page / "09-diagnostics.json").is_file())

    def test_ransac_debug_writes_detector_intermediate_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.jpg"
            image = np.zeros((240, 360, 3), dtype=np.uint8)
            mask = np.zeros((240, 360), dtype=np.uint8)
            cv2.rectangle(mask, (40, 30), (320, 210), 255, -1)
            cv2.imwrite(str(image_path), image)
            candidate = ransac_detect(image_bgr=image, mask=mask)
            page = {
                "global_ordinal": 6,
                "label": "title_or_index_sheet",
                "layout_type": "single_page",
                "image_path": str(image_path),
                "image": image,
                "mask": mask,
                "approved_bbox": [35, 25, 325, 215],
            }
            result = {
                "parameter_set_id": "baseline123",
                "pages": [{
                    "global_ordinal": 6,
                    "label": page["label"],
                    "layout_type": page["layout_type"],
                    "status": "ok",
                    "iou": 0.8,
                    "candidate": candidate.__dict__,
                }],
            }

            write_debug_artifacts(
                root, "ransac", "run-test", policy="winner",
                ranked=[result], pages=[page],
            )

            debug_page = root / "debug" / "ransac" / "run-test" / "baseline123" / "page-0006"
            for filename in (
                "01-original.jpg",
                "02-input-mask.png",
                "03-boundary-samples.png",
                "04-fitted-edge-models.png",
                "05-ransac-inliers.png",
                "06-candidate-quadrilateral.png",
                "07-overlay.jpg",
                "08-diagnostics.json",
            ):
                self.assertTrue((debug_page / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
