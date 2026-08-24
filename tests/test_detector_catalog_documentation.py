import unittest
from pathlib import Path

from hth.geometry.registry import detector_specs

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs" / "detector-catalog.md"
CONFIG = ROOT / "config" / "detectors"


class DetectorCatalogDocumentationTests(unittest.TestCase):
    def test_catalog_contains_every_registered_detector(self):
        text = CATALOG.read_text(encoding="utf-8")
        specs = detector_specs()
        self.assertGreaterEqual(len(specs), 1)
        for spec in specs:
            with self.subTest(detector=spec.method):
                self.assertIn(f"`{spec.method}`", text)
                self.assertIn(spec.name, text)

    def test_catalog_links_every_registered_detector_to_existing_documentation(self):
        text = CATALOG.read_text(encoding="utf-8")
        for spec in detector_specs():
            slug = spec.method.replace("_", "-")
            relative = f"detector-{slug}.md"
            with self.subTest(detector=spec.method):
                self.assertIn(f"]({relative})", text)
                self.assertTrue((ROOT / "docs" / relative).is_file())

    def test_registered_detectors_match_calibration_configs(self):
        registered = {spec.method for spec in detector_specs()}
        configured = {path.stem for path in CONFIG.glob("*.json")}
        self.assertEqual(registered, configured)

    def test_catalog_declares_current_registry_count(self):
        text = CATALOG.read_text(encoding="utf-8")
        count = len(detector_specs())
        self.assertIn(f"**{count} detectors**", text)


if __name__ == "__main__":
    unittest.main()
