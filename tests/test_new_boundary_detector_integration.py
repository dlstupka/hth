import unittest
from pathlib import Path

from hth.detector_catalog import configured_detectors
from hth.geometry.registry import detector_catalog
from hth.regression.calibration_intelligence import detector_characterization
from hth.regression.runner import DETECTORS


class NewBoundaryDetectorIntegrationTests(unittest.TestCase):
    METHODS = ("multi_scale_radial_edge", "projective_gradient_vote", "border_fusion_quad")

    def test_detectors_are_configured_registered_and_regressible(self) -> None:
        configured = set(configured_detectors(Path("config/detectors")))
        registered = {item["method"] for item in detector_catalog()}
        for method in self.METHODS:
            self.assertIn(method, configured)
            self.assertIn(method, registered)
            self.assertIn(method, DETECTORS)

    def test_manual_workflows_expose_new_detector_choices(self) -> None:
        regression = Path(".github/workflows/regress-detector.yml").read_text(encoding="utf-8")
        optimizer = Path(".github/workflows/execution-optimizer.yml").read_text(encoding="utf-8")
        for method in self.METHODS:
            self.assertIn(f"          - {method}\n", regression)
            self.assertIn(f"          - {method}\n", optimizer)
            self.assertIn(f'"{method}"', regression)

    def test_calibration_intelligence_has_explicit_characterization(self) -> None:
        for method in self.METHODS:
            item = detector_characterization(method)
            self.assertNotEqual(item["role"], "Unknown")
            self.assertTrue(item["evidence"])
            self.assertNotIn("not yet been registered", " ".join(row[2] for row in item["evidence"]))


if __name__ == "__main__":
    unittest.main()
