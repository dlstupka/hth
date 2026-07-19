from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from hth.geometry.model import Candidate
from hth.geometry import registry
from hth.version import HTH_REPOSITORY, HTH_VERSION


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
        self.assertEqual(catalog["contour"]["version"], HTH_VERSION)
        self.assertEqual(catalog["contour"]["repository"], HTH_REPOSITORY)
        self.assertEqual(catalog["ransac"]["origin"], "HTH")
        self.assertEqual(catalog["ransac"]["authors"], ["OpenAI ChatGPT"])
        self.assertIn("RANSAC", catalog["ransac"]["foundation"])
        self.assertEqual(catalog["ransac"]["version"], HTH_VERSION)
        self.assertEqual(catalog["ransac"]["repository"], HTH_REPOSITORY)


    def test_lsd_and_grabcut_are_registered_with_opencv_provenance(self) -> None:
        catalog = {item["method"]: item for item in registry.detector_catalog()}
        for method in ("lsd", "grabcut"):
            self.assertIn(method, catalog)
            self.assertEqual(catalog[method]["origin"], "OpenCV")
            self.assertIn("OpenCV", catalog[method]["foundation"])
            self.assertEqual(catalog[method]["authors"], ["OpenCV contributors"])
            self.assertTrue(catalog[method]["version"])
            self.assertIn("opencv", catalog[method]["repository"])

    def test_registry_is_authoritative_for_serialized_metadata(self) -> None:
        def misleading(*, image_bgr: np.ndarray, mask: np.ndarray) -> Candidate:
            del image_bgr, mask
            return Candidate(
                "test", [1, 1, 9, 9], None, 0.8, 0.8, {},
                detector_name="Wrong",
                origin="Wrong",
                foundation=["Wrong"],
                authors=["Wrong"],
                version="wrong",
                repository="wrong",
            )

        spec = registry.DetectorSpec(
            method="test",
            name="Registry Name",
            origin="Registry Origin",
            entrypoint=misleading,
            foundation=("Registry Foundation",),
            authors=("Registry Author",),
            version="1.2.3",
            repository="https://example.test/repository",
        )
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        with patch.object(registry, "_REGISTRY", (spec,)):
            candidate = registry.run_registered_detectors(
                image_bgr=image, mask=mask
            )[0]

        self.assertEqual(candidate.detector_name, "Registry Name")
        self.assertEqual(candidate.origin, "Registry Origin")
        self.assertEqual(candidate.foundation, ["Registry Foundation"])
        self.assertEqual(candidate.authors, ["Registry Author"])
        self.assertEqual(candidate.version, "1.2.3")
        self.assertEqual(candidate.repository, "https://example.test/repository")


if __name__ == "__main__":
    unittest.main()


def test_registry_imports_when_hth_directory_is_python_path(tmp_path):
    """Match GitHub Actions script-mode imports used by detect_geometry_candidates.py."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    hth_dir = Path(__file__).resolve().parents[1] / "hth"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(hth_dir)
    completed = subprocess.run(
        [sys.executable, "-c", "import geometry.registry as r; assert r.detector_names()"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
