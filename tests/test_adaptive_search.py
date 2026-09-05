import json
import unittest
from pathlib import Path

from hth.regression.parameter_space import adaptive_parameter_sets, exhaustive_parameter_sets
from hth.regression.reports import ranking_key
from hth.regression.strategies.adaptive import search


def result(parameters, score):
    return {
        "parameters": dict(parameters),
        "summary": {
            "mean_iou": score,
            "minimum_iou": score,
            "failure_count": 0,
            "mean_edge_error_px": 1.0 - score,
        },
    }


class AdaptiveSearchTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "parameters": {
                "dominant": {"values": [0, 2], "adaptive_values": [0, 1, 2, 3, 4]},
                "secondary": {"values": [0, 2], "adaptive_values": [0, 1, 2]},
            },
            "profiles": {"baseline": {"dominant": 0, "secondary": 0}},
            "adaptive_search": {"initial_parameter_sets": 5, "batch_size": 2, "max_parameter_sets": 9},
        }

    def run_search(self):
        batches = []

        def evaluate_batch(parameters):
            batches.append([dict(row) for row in parameters])
            return [result(row, 1.0 - abs(row["dominant"] - 3) * 0.1 - abs(row["secondary"] - 1) * 0.01) for row in parameters]

        outcome = search(
            self.config,
            adaptive_parameter_sets(self.config),
            evaluate_batch,
            ranking_key,
            seed_results=[result(self.config["profiles"]["baseline"], 0.6)],
        )
        return outcome, batches

    def test_search_is_budgeted_deterministic_and_records_eta_telemetry(self):
        first, first_batches = self.run_search()
        second, second_batches = self.run_search()
        self.assertEqual(first_batches, second_batches)
        self.assertEqual(len(first.results), 9)
        self.assertEqual(first.telemetry["candidate_parameter_sets"], 15)
        self.assertEqual(first.telemetry["evaluated_parameter_sets"], 9)
        self.assertGreater(first.telemetry["eta_squared"]["dominant"], first.telemetry["eta_squared"]["secondary"])
        self.assertGreaterEqual(len(first.telemetry["rounds"]), 3)

    def test_adaptive_values_do_not_change_exhaustive_grid(self):
        self.assertEqual(len(exhaustive_parameter_sets(self.config)), 4)
        self.assertEqual(len(adaptive_parameter_sets(self.config)), 15)

    def test_dynamic_refinement_generates_bounded_midpoints_near_influential_incumbent(self):
        config = {
            "parameters": {
                "threshold": {
                    "type": "float",
                    "values": [0.0, 10.0],
                    "adaptive_values": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                },
            },
            "profiles": {"baseline": {"threshold": 0.0}},
            "adaptive_search": {
                "initial_parameter_sets": 3,
                "batch_size": 2,
                "max_parameter_sets": 6,
                "dynamic_refinement": {
                    "enabled": True,
                    "minimum_eta_squared": 0.0,
                    "maximum_dimensions": 1,
                    "batch_share": 1.0,
                },
            },
        }

        def evaluate_batch(parameters):
            return [result(row, 1.0 - abs(row["threshold"] - 3.0) * 0.1) for row in parameters]

        outcome = search(
            config,
            adaptive_parameter_sets(config),
            evaluate_batch,
            ranking_key,
            seed_results=[result(config["profiles"]["baseline"], 0.5)],
        )
        evaluated_values = {row["parameters"]["threshold"] for row in outcome.results}
        self.assertIn(3.0, evaluated_values)
        self.assertIn(3.0, outcome.telemetry["generated_values"]["threshold"])
        self.assertGreater(outcome.telemetry["generated_refinement_parameter_sets"], 0)
        self.assertTrue(all(0.0 <= value <= 10.0 for value in evaluated_values))


class Gen3AdaptiveConfigurationTests(unittest.TestCase):
    def test_gen3_retains_small_exhaustive_oracle_and_declares_dense_adaptive_space(self):
        config = json.loads(Path("config/detectors/amsre_doc_ufcn_fusion.json").read_text(encoding="utf-8"))
        self.assertEqual(len(exhaustive_parameter_sets(config)), 28)
        self.assertEqual(len(adaptive_parameter_sets(config)), 2340)
        self.assertEqual(config["adaptive_search"]["max_parameter_sets"], 48)
        self.assertIn(0.8, config["parameters"]["maximum_amsre_refined_support_fraction"]["adaptive_values"])


if __name__ == "__main__":
    unittest.main()
