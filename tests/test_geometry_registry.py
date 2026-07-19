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

        specs = (
            registry.DetectorSpec("broken", "Broken", "Test", broken, authors=("Test Author",)),
            registry.DetectorSpec("healthy", "Healthy", "Test", healthy, foundation=("Test Library",), authors=("Test Author",)),
        )
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        with patch.object(registry, "_REGISTRY", specs):
            candidates = registry.run_registered_detectors(image_bgr=image, mask=mask)

        self.assertEqual(calls, ["broken", "healthy"])
        self.assertEqual(candidates[0].status, "error")
        self.assertEqual(candidates[0].diagnostics["reason"], "detector_exception")
        self.assertEqual(candidates[0].detector_name, "Broken")
        self.assertEqual(candidates[1].status, "ok")
        self.assertEqual(candidates[1].origin, "Test")
        self.assertEqual(candidates[1].foundation, ["Test Library"])
        self.assertEqual(candidates[1].authors, ["Test Author"])
        self.assertIn("elapsed_ms", candidates[1].diagnostics)

    def test_normal_empty_result_is_no_candidate_not_error(self) -> None:
        def empty(*, image_bgr: np.ndarray, mask: np.ndarray) -> Candidate:
            del image_bgr, mask
            return Candidate("empty", None, None, 0.0, 0.0, {"reason": "none_found"})

        spec = registry.DetectorSpec("empty", "Empty", "Test", empty)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        with patch.object(registry, "_REGISTRY", (spec,)):
            candidates = registry.run_registered_detectors(image_bgr=image, mask=mask)

        self.assertEqual(candidates[0].status, "no_candidate")
        self.assertEqual(candidates[0].diagnostics["reason"], "none_found")

    def test_catalog_exposes_connected_components_provenance(self) -> None:
        item = next(x for x in registry.detector_catalog() if x["method"] == "components")
        self.assertEqual(item["name"], "Connected Components")
        self.assertEqual(item["origin"], "OpenCV")
        self.assertEqual(item["foundation"], ["OpenCV"])
        self.assertEqual(item["authors"], ["OpenCV contributors"])
        self.assertTrue(item["version"])
        self.assertIn("opencv", item["repository"])

    def test_hth_detectors_preserve_source_authorship(self) -> None:
        catalog = {item["method"]: item for item in registry.detector_catalog()}
        self.assertEqual(catalog["contour"]["origin"], "HTH")
        self.assertEqual(catalog["contour"]["authors"], ["OpenAI ChatGPT"])
        self.assertIn("OpenCV", catalog["contour"]["foundation"])
        self.assertEqual(catalog["ransac"]["origin"], "HTH")
        self.assertEqual(catalog["ransac"]["authors"], ["OpenAI ChatGPT"])
        self.assertIn("RANSAC", catalog["ransac"]["foundation"])


if __name__ == "__main__":
    unittest.main()
