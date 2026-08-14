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


class _FakeShape:
    def __init__(self, rank):
        self.rank = rank


class _FakeTensor:
    def __init__(self, name, rank):
        self.name = name
        self.shape = _FakeShape(rank)


class _FakeOperation:
    def __init__(self, name, *outputs):
        self.name = name
        self.outputs = list(outputs)


class _FakeGraph:
    def __init__(self, tensors, operations):
        self._tensors = tensors
        self._operations = operations

    def get_tensor_by_name(self, name):
        if name not in self._tensors:
            raise KeyError(name)
        return self._tensors[name]

    def get_operations(self):
        return list(self._operations)


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


    def test_legacy_adapter_selects_filename_and_probability_signature_entries(self):
        class Info:
            def __init__(self, name):
                self.name = name
        input_key, input_info = detector._SavedModel._select_input(
            {"images": Info("images:0"), "filename": Info("filename:0")}
        )
        output_key, output_info = detector._SavedModel._select_probability_output(
            {"original_shape": Info("shape:0"), "probs": Info("probs:0")}
        )
        self.assertEqual((input_key, input_info.name), ("filename", "filename:0"))
        self.assertEqual((output_key, output_info.name), ("probs", "probs:0"))

    def test_legacy_adapter_recovers_from_stale_softmax_signature_name(self):
        probability = _FakeTensor("network/probs:0", 4)
        adapter = object.__new__(detector._SavedModel)
        adapter.graph = _FakeGraph(
            tensors={},
            operations=[_FakeOperation("network/probs", probability)],
        )
        resolved = adapter._resolve_tensor(
            "softmax:0",
            role="probability output",
            fallback_tokens=("prob", "softmax", "prediction"),
            allow_stale_signature=True,
        )
        self.assertIs(resolved, probability)

    def test_page_probability_matches_upstream_class_one_normalization(self):
        raw = np.array(
            [[[[0.9, 0.2], [0.1, 0.8]], [[0.7, 0.4], [0.3, 0.6]]]],
            dtype=np.float32,
        )
        probability = detector._SavedModel._page_probability(raw)
        expected = np.array([[0.25, 1.0], [0.5, 0.75]], dtype=np.float32)
        np.testing.assert_allclose(probability, expected, rtol=0, atol=1e-6)

    def test_probability_postprocessing_respects_fixed_threshold(self):
        probability = np.array([[0.1, 0.2], [0.7, 0.9]], dtype=np.float32)
        values = detector._parameters({
            "probability_threshold": 0.5,
            "close_kernel_fraction": 0.0,
            "open_kernel_fraction": 0.0,
            "fill_holes": 0,
        })
        binary, contour = detector._postprocess(probability, values)
        self.assertEqual(int(binary[0, 0]), 0)
        self.assertEqual(int(binary[1, 1]), 255)


if __name__ == "__main__":
    unittest.main()
