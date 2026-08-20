import json
import unittest
from pathlib import Path

from hth.parameter_liveness import audit_detector_directory
from hth.regression.parameter_space import exhaustive_parameter_sets, canonical_search_space, parameter_set_id
from hth.regression.runner import _filter_parameter_sets
from hth.regression.strategies.cartesian import generate


class ZombieParameterSpaceTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("config/detectors/orli_page_mask.json").read_text(encoding="utf-8"))

    def test_orli_default_exhaustive_pins_zombies(self):
        sets = exhaustive_parameter_sets(self.config)
        self.assertEqual(len(sets), 1680)
        self.assertTrue(all(row["close_kernel_fraction"] == 0.006 for row in sets))
        self.assertTrue(all(row["fill_holes"] == 1 for row in sets))

    def test_orli_exhaustive_with_zombies_restores_legacy_domain(self):
        sets = exhaustive_parameter_sets(self.config, include_zombies=True)
        self.assertEqual(len(sets), 16800)
        self.assertEqual({row["close_kernel_fraction"] for row in sets}, {0.0, 0.003, 0.006, 0.012, 0.024})
        self.assertEqual({row["fill_holes"] for row in sets}, {0, 1})
        self.assertEqual(len(generate(self.config, include_zombies=True)), 16800)

    def test_normal_cartesian_keeps_mandatory_baseline_reference(self):
        sets = generate(self.config)
        self.assertEqual(len(sets), 1680)
        self.assertEqual(sets[0], self.config["profiles"]["baseline"])

    def test_framework_liveness_metadata_audits_cleanly(self):
        report = audit_detector_directory(Path("config/detectors"))
        self.assertEqual(report["error_count"], 0)
        zombie = [item for item in report["detectors"] if item["zombie_parameters"]]
        self.assertEqual([(item["detector"], item["zombie_parameters"]) for item in zombie], [
            ("eynollah_page_mask", ["close_kernel_fraction", "minimum_page_area_fraction"]),
            ("orli_page_mask", ["close_kernel_fraction", "fill_holes"]),
            ("pagenet_page_mask", ["minimum_mask_area_fraction"]),
        ])

    def test_contracted_search_pins_excluded_dimensions_to_baseline(self):
        baseline = dict(self.config["profiles"]["baseline"])
        all_sets = exhaustive_parameter_sets(self.config)
        domain = {
            "included_parameters": ["include_lines", "dilation_fraction"],
            # Retained intelligence from an older run may contain winner-relative
            # fixed values. Execution must ignore them and use the baseline.
            "fixed_parameters": {"page_padding_fraction": 0.12, "minimum_page_area_fraction": 0.08},
        }
        filtered = _filter_parameter_sets(all_sets, domain, baseline)
        self.assertTrue(filtered)
        for row in filtered:
            self.assertEqual(row["page_padding_fraction"], baseline["page_padding_fraction"])
            self.assertEqual(row["minimum_page_area_fraction"], baseline["minimum_page_area_fraction"])
            self.assertEqual(row["close_kernel_fraction"], baseline["close_kernel_fraction"])
            self.assertEqual(row["fill_holes"], baseline["fill_holes"])

    def test_contracted_parameter_ids_are_invariant_to_historic_fixed_values(self):
        baseline = dict(self.config["profiles"]["baseline"])
        all_sets = exhaustive_parameter_sets(self.config)
        domain_a = {"included_parameters": ["include_lines"], "fixed_parameters": {"dilation_fraction": 0.0}}
        domain_b = {"included_parameters": ["include_lines"], "fixed_parameters": {"dilation_fraction": 0.06}}
        ids_a = [parameter_set_id(row) for row in _filter_parameter_sets(all_sets, domain_a, baseline)]
        ids_b = [parameter_set_id(row) for row in _filter_parameter_sets(all_sets, domain_b, baseline)]
        self.assertEqual(ids_a, ids_b)

    def test_workflows_expose_zombie_strategy(self):
        for workflow in ("regress-detector.yml", "execution-optimizer.yml"):
            text = Path(".github/workflows", workflow).read_text(encoding="utf-8")
            self.assertIn("- exhaustive-with-zombies", text)


    def test_canonical_search_space_counts_are_strategy_independent_metadata(self):
        normal = canonical_search_space(self.config, "exhaustive")
        zombie = canonical_search_space(self.config, "exhaustive-with-zombies")
        self.assertEqual(normal["live_exhaustive_parameter_sets"], 1680)
        self.assertEqual(normal["exhaustive_with_zombies_parameter_sets"], 16800)
        self.assertEqual(normal["effective_parameter_sets"], 1680)
        self.assertEqual(zombie["effective_parameter_sets"], 16800)
        self.assertEqual(normal["configured_zombie_parameters"], ["close_kernel_fraction", "fill_holes"])


if __name__ == "__main__":
    unittest.main()
