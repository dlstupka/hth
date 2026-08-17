import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hth.geometry import detector_dhsegment_page_mask as dh
from hth.geometry import detector_kraken_page_mask as kraken


class SharedLearnedEvidenceParentTests(unittest.TestCase):
    def setUp(self):
        kraken._EVIDENCE_CACHE.clear()
        kraken._RUNTIME_DIAGNOSTICS.clear()
        dh._EVIDENCE_CACHE.clear()

    def test_kraken_artifact_round_trip_avoids_inference(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        key = kraken._image_key(image)
        frozen = kraken._freeze_evidence({
            "regions": [[(1, 1), (6, 1), (6, 6), (1, 6)]],
            "lines": [],
            "baselines": [],
            "text_direction": "horizontal-lr",
        })
        with kraken._EVIDENCE_CACHE_LOCK:
            kraken._EVIDENCE_CACHE[key] = frozen
        with tempfile.TemporaryDirectory() as td:
            kraken.export_precomputed_golden_set_evidence([image], Path(td))
            kraken._EVIDENCE_CACHE.clear()
            with patch.object(kraken, "_load_model", side_effect=AssertionError("model must not load")):
                keys = kraken.load_precomputed_golden_set_evidence(Path(td), [image])
                evidence = kraken._infer_evidence(image)
        self.assertEqual(keys, (key,))
        self.assertEqual(evidence, frozen)

    def test_dhsegment_artifact_round_trip_uses_readonly_mmap(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        key = dh._image_key(image)
        probability = np.arange(16, dtype=np.float32).reshape(4, 4)
        probability.setflags(write=False)
        with dh._EVIDENCE_CACHE_LOCK:
            dh._EVIDENCE_CACHE[key] = (probability, (8, 8))
        with tempfile.TemporaryDirectory() as td:
            dh.export_precomputed_golden_set_evidence([image], Path(td))
            dh._EVIDENCE_CACHE.clear()
            with patch.object(dh, "_model", side_effect=AssertionError("model must not load")):
                keys = dh.load_precomputed_golden_set_evidence(Path(td), [image])
                loaded, shape = dh._infer_evidence(image)
        self.assertEqual(keys, (key,))
        self.assertEqual(shape, (8, 8))
        self.assertFalse(loaded.flags.writeable)
        np.testing.assert_array_equal(loaded, probability)
        if os.name == "nt":
            self.assertFalse(isinstance(loaded, np.memmap))

    def test_shell_prepares_shared_evidence_before_worker_fanout(self):
        text = Path("tools/run-detector-regressions.sh").read_text(encoding="utf-8")
        prepare = text.index("python -m hth.regression.learned_evidence prepare")
        fanout = text.index('detector_worker "$pipeline_index"', prepare)
        self.assertLess(prepare, fanout)
        self.assertIn('if (( learned_count > 1 )); then', text)
        self.assertIn('--precomputed-evidence "$shared_evidence_dir"', text)

    def test_page_level_telemetry_is_visible(self):
        text = Path("hth/regression/learned_evidence.py").read_text(encoding="utf-8")
        self.assertIn("page {index}/{total} START", text)
        self.assertIn("page {index}/{total} READY", text)
        self.assertIn("SHARED EVIDENCE READY", text)

    def test_stderr_capture_remains_serialized(self):
        ktext = Path("hth/geometry/detector_kraken_page_mask.py").read_text(encoding="utf-8")
        dtext = Path("hth/geometry/detector_dhsegment_page_mask.py").read_text(encoding="utf-8")
        self.assertIn("with _STDERR_CAPTURE_LOCK:", ktext)
        self.assertIn("with _STDERR_CAPTURE_LOCK:", dtext)


if __name__ == "__main__":
    unittest.main()
