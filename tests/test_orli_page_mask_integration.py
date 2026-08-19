import json
import unittest
from pathlib import Path

from hth.geometry.registry import detector_names
from hth.regression.strategies.cartesian import generate


ROOT = Path(__file__).resolve().parents[1]


class OrliPageMaskIntegrationTests(unittest.TestCase):
    def test_detector_is_registered_and_has_declared_calibration_space(self):
        self.assertIn("orli_page_mask", detector_names())
        config = json.loads((ROOT / "config/detectors/orli_page_mask.json").read_text(encoding="utf-8"))
        self.assertEqual(config["detector"], "orli_page_mask")
        self.assertEqual(len(generate(config)), 1680)
        self.assertEqual(len(generate(config, include_zombies=True)), 16800)
        self.assertEqual(config["lifecycle"]["prepare"], "orli_page_mask")

    def test_regression_and_optimizer_expose_orli(self):
        for workflow in ("regress-detector.yml", "execution-optimizer.yml"):
            text = (ROOT / ".github/workflows" / workflow).read_text(encoding="utf-8")
            self.assertIn("          - orli_page_mask\n", text)
            self.assertIn("HTH_NEED_ORLI:", text)
            self.assertIn("orli==", (ROOT / "tools/ensure-managed-runtime.sh").read_text(encoding="utf-8"))

    def test_shared_learned_evidence_supports_orli(self):
        text = (ROOT / "tools/run-detector-regressions.sh").read_text(encoding="utf-8")
        self.assertIn("kraken_page_mask|orli_page_mask|dhsegment_page_mask", text)
        self.assertIn("for learned_detector in kraken_page_mask orli_page_mask dhsegment_page_mask", text)


if __name__ == "__main__":
    unittest.main()
