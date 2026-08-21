import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.geometry import detector_scantailor_page_frame as detector
from hth.geometry.registry import detector_spec


class ScanTailorPageFrameTests(unittest.TestCase):
    def _synthetic_page(self):
        image = np.full((500, 700, 3), 45, dtype=np.uint8)
        cv2.rectangle(image, (90, 55), (620, 450), (225, 225, 225), -1)
        for y in range(120, 400, 36):
            cv2.line(image, (150, y), (550, y), (70, 70, 70), 4)
        return image

    def test_detector_is_registered_with_scan_processing_provenance(self):
        spec = detector_spec("scantailor_page_frame")
        self.assertEqual(spec.name, "ScanTailor Page Frame")
        self.assertIn("ScanTailor-style scan processing", spec.foundation)
        self.assertIn("scantailor", spec.repository)

    def test_declared_grid_is_focused_two_dimensional_refinement(self):
        payload = json.loads(Path("config/detectors/scantailor_page_frame.json").read_text(encoding="utf-8"))
        count = 1
        for item in payload["parameters"].values():
            count *= len(item["values"])
        self.assertEqual(count, 54)
        self.assertEqual(payload["profiles"]["baseline"], detector.BASELINE_PARAMETERS)

        self.assertEqual(payload["parameters"]["illumination_sigma_fraction"]["values"], [0.09])
        self.assertEqual(payload["parameters"]["ink_quantile"]["values"], [0.76])
        self.assertEqual(payload["parameters"]["content_close_fraction"]["values"], [0.012])
        self.assertEqual(payload["parameters"]["minimum_page_area_fraction"]["values"], [0.12])

        smoothing = payload["parameters"]["projection_smooth_fraction"]["values"]
        self.assertEqual(smoothing, [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045])
        self.assertIn(0.003, smoothing)

        margins = payload["parameters"]["boundary_search_margin_fraction"]["values"]
        self.assertEqual(
            margins,
            [0.065, 0.0675, 0.07, 0.0725, 0.075, 0.0775, 0.08, 0.0825, 0.085],
        )
        self.assertIn(0.075, margins)

    def test_synthetic_page_produces_plausible_frame(self):
        image = self._synthetic_page()
        candidate = detector.detect(image_bgr=image, mask=np.zeros(image.shape[:2], np.uint8))
        self.assertEqual(candidate.status, "ok")
        self.assertIsNotNone(candidate.corners)
        corners = np.asarray(candidate.corners)
        self.assertEqual(corners.shape, (4, 2))
        x1, y1, x2, y2 = candidate.bbox
        self.assertLess(x1, 150)
        self.assertLess(y1, 120)
        self.assertGreater(x2, 550)
        self.assertGreater(y2, 400)
        self.assertEqual(candidate.diagnostics["evidence"], "scantailor_style_content_guided_page_frame")

    def test_unknown_parameter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown ScanTailor Page Frame parameters"):
            detector.detect(image_bgr=self._synthetic_page(), mask=np.zeros((500, 700), np.uint8), parameters={"bogus": 1})


if __name__ == "__main__":
    unittest.main()
