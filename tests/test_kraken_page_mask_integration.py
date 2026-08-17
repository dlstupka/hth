import unittest
from pathlib import Path


class KrakenPageMaskIntegrationTests(unittest.TestCase):
    def test_registry_and_lifecycle_are_registered(self):
        registry=Path("hth/geometry/registry.py").read_text(encoding="utf-8")
        lifecycle=Path("hth/detector_lifecycle.py").read_text(encoding="utf-8")
        self.assertIn("detector_kraken_page_mask",registry)
        self.assertIn('"kraken_page_mask":_prepare_kraken_page_mask_hook',lifecycle)
        self.assertIn('"kraken_page_mask":_finalize_kraken_page_mask_hook',lifecycle)

    def test_workflows_pin_kraken_702_and_offer_detector(self):
        manager = Path("tools/ensure-managed-runtime.sh").read_text(encoding="utf-8")
        self.assertIn('python -m pip install "kraken==7.0.2"', manager)
        for rel in (
            ".github/workflows/regress-detector.yml",
            ".github/workflows/execution-optimizer.yml",
        ):
            text=Path(rel).read_text(encoding="utf-8")
            self.assertIn("kraken_page_mask",text)
            self.assertIn("hth-pipeline/tools/ensure-managed-runtime.sh", text)


if __name__ == "__main__":
    unittest.main()
