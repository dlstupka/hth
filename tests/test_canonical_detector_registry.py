import json
import unittest
from pathlib import Path

from hth.geometry.registry import detector_entrypoint, detector_names, detector_spec
from hth.regression.runner import DETECTORS


class CanonicalDetectorRegistryTests(unittest.TestCase):
    def test_every_detector_config_resolves_through_authoritative_registry(self):
        registered = set(detector_names())
        configured = set()
        for path in Path("config/detectors").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            configured.add(str(payload["detector"]))
        self.assertTrue(configured <= registered, sorted(configured - registered))

    def test_kraken_resolves_through_canonical_registry(self):
        spec = detector_spec("kraken_page_mask")
        self.assertEqual(spec.method, "kraken_page_mask")
        self.assertIs(detector_entrypoint("kraken_page_mask"), spec.entrypoint)
        self.assertTrue(callable(spec.entrypoint))

    def test_regression_compatibility_map_is_generated_from_registry(self):
        self.assertEqual(set(DETECTORS), set(detector_names()))
        for name in detector_names():
            self.assertIs(DETECTORS[name], detector_entrypoint(name))

    def test_runner_dispatches_through_registry_not_compatibility_map(self):
        text = Path("hth/regression/runner.py").read_text(encoding="utf-8")
        self.assertIn("detector=detector_entrypoint(name)", text)
        self.assertIn("name not in detector_names()", text)
        self.assertIn(
            "DETECTORS={name: detector_entrypoint(name) for name in detector_names()}",
            text,
        )

    def test_unknown_detector_fails_at_registry_boundary(self):
        with self.assertRaisesRegex(KeyError, "Unknown detector"):
            detector_spec("__not_a_detector__")


if __name__ == "__main__":
    unittest.main()
