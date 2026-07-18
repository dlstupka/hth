from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from hth.geometry.model import Candidate
from hth.geometry import registry


class RegistryIsolationTests(unittest.TestCase):
    def test_detector_exception_becomes_failed_candidate_and_next_detector_runs(self) -> None:
        calls: list[str] = []

        def broken(*, image_bgr: np.ndarray, mask: np.ndarray) -> Candidate:
            del image_bgr, mask
            calls.append("broken")
            raise RuntimeError("deliberate test failure")

        def healthy(*, image_bgr: np.ndarray, mask: np.ndarray) -> Candidate:
            del image_bgr, mask
            calls.append("healthy")
            return Candidate("healthy", [1, 1, 9, 9], None, 0.8, 0.8, {})

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        with patch.object(registry, "_REGISTRY", (("broken", broken), ("healthy", healthy))):
            candidates = registry.run_registered_detectors(image_bgr=image, mask=mask)

        self.assertEqual(calls, ["broken", "healthy"])
        self.assertEqual(candidates[0].status, "error")
        self.assertEqual(candidates[0].diagnostics["reason"], "detector_exception")
        self.assertEqual(candidates[1].status, "ok")

    def test_normal_empty_result_is_no_candidate_not_error(self) -> None:
        def empty(*, image_bgr: np.ndarray, mask: np.ndarray) -> Candidate:
            del image_bgr, mask
            return Candidate("empty", None, None, 0.0, 0.0, {"reason": "none_found"})

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        with patch.object(registry, "_REGISTRY", (("empty", empty),)):
            candidates = registry.run_registered_detectors(image_bgr=image, mask=mask)

        self.assertEqual(candidates[0].status, "no_candidate")
        self.assertEqual(candidates[0].diagnostics["reason"], "none_found")


if __name__ == "__main__":
    unittest.main()
