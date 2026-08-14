import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from hth.geometry import detector_dhsegment_page_mask as detector
from hth.regression.parameter_space import exhaustive_parameter_sets


class FakeModel:
    def predict(self, image_bgr):
        probability = np.zeros((320, 256), np.float32)
        probability[35:285, 28:228] = 0.95
        return probability, image_bgr.shape[:2]


class DhSegmentPageMaskTests(unittest.TestCase):
    def test_initial_exhaustive_space_has_10000_sets_and_retains_baseline(self):
        path = Path(__file__).resolve().parents[1] / "config/detectors/dhsegment_page_mask.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        parameter_sets = exhaustive_parameter_sets(config)
        self.assertEqual(len(parameter_sets), 10000)
        self.assertIn(config["profiles"]["baseline"], parameter_sets)
        self.assertIn(-1.0, config["parameters"]["probability_threshold"]["values"])
        self.assertEqual(config["parameters"]["fill_holes"]["values"], [0, 1])

    def test_detector_uses_lifecycle_assets_and_returns_page(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            model.mkdir()
            (model / "saved_model.pb").write_bytes(b"saved-model")
            provenance = root / "provenance.json"
            provenance.write_text(
                json.dumps(
                    {
                        "model_id": "dhsegment-page-v0.2",
                        "archive_sha256": "abc",
                        "model_url": "https://example.invalid/model.zip",
                        "upstream_repository": "https://github.com/dhlab-epfl/dhSegment",
                        "license": "GPL-3.0",
                    }
                ),
                encoding="utf-8",
            )
            env = {
                detector.MODEL_DIR_ENV: str(model),
                detector.PROVENANCE_ENV: str(provenance),
            }
            with patch.dict(os.environ, env, clear=False), patch.object(detector, "_model", return_value=FakeModel()):
                candidate = detector.detect(
                    image_bgr=np.full((500, 400, 3), 255, np.uint8),
                    mask=np.zeros((500, 400), np.uint8),
                )
        self.assertEqual(candidate.status, "ok")
        self.assertEqual(candidate.method, "dhsegment_page_mask")
        self.assertEqual(candidate.diagnostics["model_id"], "dhsegment-page-v0.2")

    def test_missing_assets_raise_configuration_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "lifecycle did not set"):
                detector.detect(
                    image_bgr=np.zeros((50, 50, 3), np.uint8),
                    mask=np.zeros((50, 50), np.uint8),
                )

    def test_otsu_and_fixed_threshold_paths_produce_masks(self):
        probability = np.zeros((100, 120), np.float32)
        probability[10:90, 15:105] = 0.9
        for threshold in (-1.0, 0.5):
            values = detector._parameters({"probability_threshold": threshold})
            binary, contour = detector._postprocess(probability, values)
            self.assertEqual(binary.shape, probability.shape)
            self.assertIsNotNone(contour)
            self.assertGreater(cv2.contourArea(contour), 0)


if __name__ == "__main__":
    unittest.main()
