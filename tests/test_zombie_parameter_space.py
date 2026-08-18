import json
import unittest
from pathlib import Path

from hth.parameter_liveness import audit_detector_directory
from hth.regression.parameter_space import exhaustive_parameter_sets, canonical_search_space
from hth.regression.strategies.cartesian import generate


class ZombieParameterSpaceTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("config/detectors/orli_page_mask.json").read_text(encoding="utf-8"))

    def test_orli_default_exhaustive_pins_zombies(self):
        sets = exhaustive_parameter_sets(self.config)
        self.assertEqual(len(sets), 1000)
        self.assertTrue(all(row["close_kernel_fraction"] == 0.006 for row in sets))
        self.assertTrue(all(row["fill_holes"] == 1 for row in sets))

    def test_orli_exhaustive_with_zombies_restores_legacy_domain(self):
        sets = exhaustive_parameter_sets(self.config, include_zombies=True)
        self.assertEqual(len(sets), 10000)
        self.assertEqual({row["close_kernel_fraction"] for row in sets}, {0.0, 0.003, 0.006, 0.012, 0.024})
        self.assertEqual({row["fill_holes"] for row in sets}, {0, 1})
        self.assertEqual(len(generate(self.config, include_zombies=True)), 10000)

    def test_normal_cartesian_keeps_mandatory_baseline_reference(self):
        sets = generate(self.config)
        self.assertEqual(len(sets), 1000)
        self.assertEqual(sets[0], self.config["profiles"]["baseline"])

    def test_framework_liveness_metadata_audits_cleanly(self):
        report = audit_detector_directory(Path("config/detectors"))
        self.assertEqual(report["error_count"], 0)
        zombie = [item for item in report["detectors"] if item["zombie_parameters"]]
        self.assertEqual([(item["detector"], item["zombie_parameters"]) for item in zombie], [
            ("orli_page_mask", ["close_kernel_fraction", "fill_holes"])
        ])

    def test_workflows_expose_zombie_strategy(self):
        for workflow in ("regress-detector.yml", "execution-optimizer.yml"):
            text = Path(".github/workflows", workflow).read_text(encoding="utf-8")
            self.assertIn("- exhaustive-with-zombies", text)


    def test_canonical_search_space_counts_are_strategy_independent_metadata(self):
        normal = canonical_search_space(self.config, "exhaustive")
        zombie = canonical_search_space(self.config, "exhaustive-with-zombies")
        self.assertEqual(normal["live_exhaustive_parameter_sets"], 1000)
        self.assertEqual(normal["exhaustive_with_zombies_parameter_sets"], 10000)
        self.assertEqual(normal["effective_parameter_sets"], 1000)
        self.assertEqual(zombie["effective_parameter_sets"], 10000)
        self.assertEqual(normal["configured_zombie_parameters"], ["close_kernel_fraction", "fill_holes"])


if __name__ == "__main__":
    unittest.main()
