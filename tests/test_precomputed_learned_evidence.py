import sys
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

from hth.geometry import detector_dhsegment_page_mask as dh
from hth.geometry import detector_kraken_page_mask as kraken
from hth.regression.runner import logical_golden_set, PRECOMPUTED_EVIDENCE_PREPARERS


class _FakeKrakenModel:
    def __init__(self):
        self.calls = 0

    def predict(self, *, im, config):
        self.calls += 1
        region = types.SimpleNamespace(boundary=[(1,1),(8,1),(8,8),(1,8),(1,1)])
        return types.SimpleNamespace(regions={"text":[region]}, lines=[], text_direction="horizontal-lr")


class _FakeDhModel:
    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return np.ones((4,4), dtype=np.float32), image.shape[:2]


class PrecomputedLearnedEvidenceTests(unittest.TestCase):
    def setUp(self):
        kraken._EVIDENCE_CACHE.clear()
        kraken._RUNTIME_DIAGNOSTICS.clear()
        dh._EVIDENCE_CACHE.clear()

    def test_runner_registers_both_learned_evidence_preparers(self):
        self.assertIn("kraken_page_mask", PRECOMPUTED_EVIDENCE_PREPARERS)
        self.assertIn("dhsegment_page_mask", PRECOMPUTED_EVIDENCE_PREPARERS)

    def test_each_parameter_evaluation_gets_private_logical_page_metadata(self):
        image = np.zeros((5,5,3), dtype=np.uint8)
        pages = [{"global_ordinal":1, "image":image, "label":"one"}]
        left = logical_golden_set(pages)
        right = logical_golden_set(pages)
        self.assertIsNot(left, right)
        self.assertIsNot(left[0], right[0])
        self.assertIs(left[0]["image"], right[0]["image"])
        left[0]["label"] = "changed"
        self.assertEqual(right[0]["label"], "one")

    def test_dhsegment_probability_map_is_single_flight_and_read_only(self):
        image = np.zeros((8,8,3), dtype=np.uint8)
        model = _FakeDhModel()
        with patch.object(dh, "_model", return_value=model):
            first = dh._infer_evidence(image)
            second = dh._infer_evidence(image)
        self.assertEqual(model.calls, 1)
        self.assertIs(first[0], second[0])
        self.assertFalse(first[0].flags.writeable)

    def test_kraken_evidence_is_single_flight_and_frozen(self):
        image = np.zeros((10,10,3), dtype=np.uint8)
        model = _FakeKrakenModel()
        fake_configs = types.ModuleType("kraken.configs")
        fake_configs.SegmentationInferenceConfig = type("SegmentationInferenceConfig", (), {})
        fake_kraken = types.ModuleType("kraken")
        fake_kraken.configs = fake_configs
        with patch.dict(sys.modules, {"kraken":fake_kraken, "kraken.configs":fake_configs}), \
             patch.object(kraken, "_load_model", return_value=model), \
             patch.object(kraken, "_capture_kraken_runtime_chatter") as capture:
            capture.return_value.__enter__.return_value = {
                "lightning_srun_advisories":0,
                "kraken_polygonizer_warnings":0,
                "filtered_messages":[],
            }
            first = kraken._infer_evidence(image)
            second = kraken._infer_evidence(image)
        self.assertEqual(model.calls, 1)
        self.assertIs(first, second)
        self.assertIsInstance(first["regions"], tuple)
        self.assertIsInstance(first["regions"][0], tuple)

    def test_fd2_capture_paths_have_dedicated_locks(self):
        self.assertIsInstance(kraken._STDERR_CAPTURE_LOCK, type(threading.Lock()))
        self.assertIsInstance(dh._STDERR_CAPTURE_LOCK, type(threading.Lock()))


if __name__ == "__main__":
    unittest.main()
