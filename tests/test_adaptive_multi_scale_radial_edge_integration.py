import unittest
from pathlib import Path

from hth.detector_catalog import configured_detectors
from hth.geometry.registry import detector_catalog
from hth.regression.calibration_intelligence import detector_characterization
from hth.regression.runner import DETECTORS


class AdaptiveMultiScaleRadialEdgeIntegrationTests(unittest.TestCase):
    METHOD = "adaptive_multi_scale_radial_edge"

    def test_detector_is_configured_registered_and_regressible(self) -> None:
        configured = set(configured_detectors(Path("config/detectors")))
        registered = {row["method"] for row in detector_catalog()}
        self.assertIn(self.METHOD, configured)
        self.assertIn(self.METHOD, registered)
        self.assertIn(self.METHOD, DETECTORS)

    def test_manual_regression_and_optimizer_menus_expose_detector(self) -> None:
        for path in (
            Path(".github/workflows/regress-detector.yml"),
            Path(".github/workflows/execution-optimizer.yml"),
        ):
            self.assertIn(f"          - {self.METHOD}\n", path.read_text(encoding="utf-8"))

    def test_calibration_intelligence_describes_controlled_adaptation(self) -> None:
        item = detector_characterization(self.METHOD)
        self.assertIn("Generator", item["role"])
        text = " ".join(str(value) for row in item["evidence"] for value in row)
        self.assertIn("adaptive", text.lower())
        self.assertIn("multiscale", text.lower())


if __name__ == "__main__":
    unittest.main()
