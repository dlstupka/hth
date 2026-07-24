from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.regression.adapters.contour import detect as contour_detect
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
            self.assertTrue((debug_page / "original.jpg").is_file())
            self.assertTrue((debug_page / "input-mask.png").is_file())
            self.assertTrue((debug_page / "overlay.jpg").is_file())
            diagnostics = json.loads((debug_page / "diagnostics.json").read_text())
            self.assertEqual(diagnostics["result"]["status"], "no_candidate")
            self.assertIn("debug/contour/run-test/README.txt", outputs)


if __name__ == "__main__":
    unittest.main()
