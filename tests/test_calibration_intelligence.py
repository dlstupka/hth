import unittest

from hth.regression.calibration_intelligence import build_calibration_intelligence, detector_characterization


class CalibrationIntelligenceTests(unittest.TestCase):
    def test_characterizes_parameter_influence_basin_and_pages(self):
        ranked = []
        for index, (alpha, beta, score) in enumerate([
            (1, "x", 0.90),
            (1, "y", 0.89),
            (2, "x", 0.70),
            (2, "y", 0.69),
        ]):
            ranked.append({
                "parameter_set_id": str(index),
                "parameters": {"alpha": alpha, "beta": beta},
                "summary": {"mean_iou": score, "failure_count": 0},
                "pages": [
                    {"global_ordinal": 1, "status": "ok", "iou": score},
                    {"global_ordinal": 5, "status": "ok", "iou": score - 0.1},
                ],
            })

        report = build_calibration_intelligence(
            ranked,
            detector="example",
            strategy="exhaustive",
            possible_parameter_sets=4,
        )

        self.assertTrue(report["available"])
        self.assertTrue(report["search"]["exhaustive_complete"])
        self.assertEqual(report["landscape"]["equivalent_winner_count"], 1)
        influence = {item["parameter"]: item for item in report["parameter_influence"]}
        self.assertGreater(influence["alpha"]["eta_squared"], influence["beta"]["eta_squared"])
        self.assertEqual(influence["alpha"]["classification"], "Critical")
        self.assertEqual(len(report["page_sensitivity"]), 2)
        self.assertEqual(report["page_sensitivity"][0]["global_ordinal"], 1)
        self.assertIn(report["calibration_confidence"]["rating"], {"Medium", "High"})
        self.assertEqual(report["schema_version"], "1.1")
        self.assertIn("calibration_identity", report)
        self.assertIn("regression_metadata", report)
        self.assertEqual(report["detector_evidence"]["detector_id"], "example")
        self.assertIn("parameters", report["parameter_intelligence"])
        self.assertIn("domains", report["domain_space_intelligence"])
        self.assertEqual(report["detector_selection_intelligence"]["recommended_detector_id"], "example")

    def test_kraken_characterization_is_registered_as_generator(self):
        characterization = detector_characterization("kraken_page_mask")
        self.assertEqual(characterization["friendly_name"], "Kraken Page Mask")
        self.assertEqual(characterization["role"], "Generator")
        evidence = {row[0]: row for row in characterization["evidence"]}
        self.assertIn("Kraken BLLA segmentation", evidence)
        self.assertIn("Sparse multi-region envelope", evidence)
        self.assertIn("Model identity", evidence)


    def test_marks_flat_parameter_as_zombie(self):
        ranked = [
            {
                "parameters": {"unused": value},
                "summary": {"mean_iou": 0.8, "failure_count": 0},
                "pages": [],
            }
            for value in (1, 2, 3)
        ]
        report = build_calibration_intelligence(
            ranked,
            detector="flat",
            strategy="exhaustive",
            possible_parameter_sets=3,
        )
        self.assertEqual(report["parameter_influence"][0]["classification"], "Zombie")
        self.assertEqual(report["recommendations"]["zombie_parameters"], ["unused"])


    def test_preserves_calibration_and_regression_context(self):
        ranked = [{
            "parameter_set_id": "winner-id",
            "parameters": {"alpha": 1},
            "summary": {"mean_iou": 0.9, "minimum_iou": 0.8, "stddev_iou": 0.05, "failure_count": 0},
            "pages": [],
        }]
        calibration_context = {
            "calibration_run_id": "run-1",
            "source_document": {"title": "Example"},
            "golden_set": {"collection_id": "GS-1", "sha256": "abc"},
            "detector_configuration": {"sha256": "def"},
        }
        regression_context = {"requested_strategy": "critical", "resolved_strategy": "exhaustive"}
        report = build_calibration_intelligence(
            ranked, detector="example", strategy="exhaustive", possible_parameter_sets=1,
            calibration_context=calibration_context, regression_context=regression_context,
        )
        self.assertEqual(report["calibration_identity"], calibration_context)
        self.assertEqual(report["regression_metadata"], regression_context)
        selection = report["detector_selection_intelligence"]
        self.assertEqual(selection["recommended_parameter_set_id"], "winner-id")
        self.assertEqual(selection["applicability"]["golden_set"]["collection_id"], "GS-1")
        self.assertEqual(report["domain_space_intelligence"]["default_strategy"], "exhaustive")
        self.assertEqual(report["domain_space_intelligence"]["fallback_order"][-1], "exhaustive")


    def test_withholds_parameter_influence_when_calibration_has_no_valid_measurements(self):
        ranked = [
            {
                "parameter_set_id": str(value),
                "parameters": {"threshold": value},
                "summary": {"mean_iou": 0.0, "failure_count": 2},
                "pages": [
                    {"global_ordinal": 1, "status": "no_candidate", "iou": 0.0},
                    {"global_ordinal": 2, "status": "error", "iou": 0.0},
                ],
            }
            for value in (0.3, 0.5, 0.7)
        ]
        report = build_calibration_intelligence(
            ranked, detector="example", strategy="exhaustive", possible_parameter_sets=3,
        )
        self.assertFalse(report["measurement_state"]["informative"])
        self.assertEqual(report["measurement_state"]["status"], "no_valid_measurements")
        self.assertEqual(report["parameter_influence"], [])
        self.assertEqual(report["domain_space"], {})
        self.assertEqual(report["recommendations"]["dormant_parameters"], [])

    def test_withholds_parameter_influence_when_candidates_have_zero_overlap(self):
        ranked = [{
            "parameter_set_id": "zero",
            "parameters": {"threshold": 0.5},
            "summary": {"mean_iou": 0.0, "failure_count": 0},
            "pages": [{"global_ordinal": 1, "status": "ok", "iou": 0.0}],
        }]
        report = build_calibration_intelligence(
            ranked, detector="example", strategy="exhaustive", possible_parameter_sets=1,
        )
        self.assertFalse(report["measurement_state"]["informative"])
        self.assertEqual(report["measurement_state"]["status"], "no_overlap_signal")
        self.assertEqual(report["parameter_influence"], [])

    def test_domain_space_reports_zombie_universe_separately_from_live_exhaustive(self):
        ranked = [
            {
                "parameter_set_id": str(index),
                "parameters": {"live": live, "zombie": zombie},
                "summary": {"mean_iou": score, "failure_count": 0},
                "pages": [{"global_ordinal": 1, "status": "ok", "iou": score}],
            }
            for index, (live, zombie, score) in enumerate([
                (0, 0, 0.80),
                (1, 0, 0.90),
                (0, 1, 0.80),
                (1, 1, 0.90),
            ])
        ]
        report = build_calibration_intelligence(
            ranked,
            detector="example",
            strategy="exhaustive-with-zombies",
            possible_parameter_sets=4,
            regression_context={
                "zombie_parameters": ["zombie"],
                "live_possible_parameter_sets": 2,
                "zombie_possible_parameter_sets": 4,
            },
        )
        domains = report["domain_space"]
        self.assertEqual(domains["exhaustive_with_zombies"]["parameter_set_count"], 4)
        self.assertEqual(domains["exhaustive"]["parameter_set_count"], 2)
        self.assertEqual(domains["non_dormant"]["parameter_set_count"], 2)

    def test_domain_space_reports_equal_exhaustive_rows_without_zombies(self):
        ranked = [{
            "parameter_set_id": "one",
            "parameters": {"live": 1},
            "summary": {"mean_iou": 0.9, "failure_count": 0},
            "pages": [{"global_ordinal": 1, "status": "ok", "iou": 0.9}],
        }]
        report = build_calibration_intelligence(
            ranked,
            detector="example",
            strategy="exhaustive",
            possible_parameter_sets=1,
            regression_context={
                "live_possible_parameter_sets": 1,
                "zombie_possible_parameter_sets": 1,
            },
        )
        domains = report["domain_space"]
        self.assertEqual(domains["exhaustive_with_zombies"]["parameter_set_count"], 1)
        self.assertEqual(domains["exhaustive"]["parameter_set_count"], 1)


    def test_out_of_space_reference_cannot_inflate_live_domain(self):
        ranked = [
            {
                "parameter_set_id": "live1",
                "search_space_member": True,
                "parameters": {"live": 1, "zombie": 1},
                "summary": {"mean_iou": 0.90, "failure_count": 0},
                "pages": [{"global_ordinal": 1, "status": "ok", "iou": 0.90}],
            },
            {
                "parameter_set_id": "live0",
                "search_space_member": True,
                "parameters": {"live": 0, "zombie": 1},
                "summary": {"mean_iou": 0.80, "failure_count": 0},
                "pages": [{"global_ordinal": 1, "status": "ok", "iou": 0.80}],
            },
            {
                "parameter_set_id": "historic",
                "search_space_member": False,
                "reference_roles": ["historic_best"],
                "parameters": {"live": 1, "zombie": 0},
                "summary": {"mean_iou": 0.95, "failure_count": 0},
                "pages": [{"global_ordinal": 1, "status": "ok", "iou": 0.95}],
            },
        ]
        report = build_calibration_intelligence(
            ranked, detector="example", strategy="exhaustive", possible_parameter_sets=2,
            regression_context={
                "zombie_parameters": ["zombie"],
                "live_possible_parameter_sets": 2,
                "zombie_possible_parameter_sets": 4,
            },
        )
        self.assertEqual(report["search"]["parameter_sets"], 2)
        self.assertEqual(report["landscape"]["best_mean_iou"], 0.90)
        self.assertNotIn("zombie", [item["parameter"] for item in report["parameter_influence"]])
        self.assertEqual(report["domain_space"]["exhaustive_with_zombies"]["parameter_set_count"], 4)
        self.assertEqual(report["domain_space"]["exhaustive"]["parameter_set_count"], 2)
        self.assertLessEqual(report["domain_space"]["non_dormant"]["parameter_set_count"], 2)
        self.assertEqual(report["canonical_search_space"]["evaluated_search_space_parameter_sets"], 2)


if __name__ == "__main__":
    unittest.main()

class CanonicalEffectSizeClassificationTests(unittest.TestCase):
    def test_canonical_threshold_spec_is_persisted(self):
        ranked = [
            {"parameters": {"p": v}, "summary": {"mean_iou": score, "failure_count": 0}, "pages": []}
            for v, score in [(0, 0.80), (1, 0.81), (2, 0.79)]
        ]
        report = build_calibration_intelligence(ranked, detector="thresholds", strategy="exhaustive", possible_parameter_sets=3)
        thresholds = report["parameter_intelligence"]["classification_thresholds"]
        self.assertEqual(thresholds["zombie"]["eta_squared_below"], 0.0005)
        self.assertEqual(thresholds["dormant"]["eta_squared_below"], 0.005)
        self.assertEqual(thresholds["low"]["eta_squared_minimum"], 0.005)
        self.assertEqual(thresholds["moderate"]["eta_squared_minimum"], 0.02)
        self.assertEqual(thresholds["important"]["eta_squared_minimum"], 0.06)
        self.assertEqual(thresholds["critical"]["eta_squared_minimum"], 0.14)

    def test_retained_zombie_evidence_is_visible_but_not_active(self):
        ranked = [
            {"parameters": {"live": v, "dead": 1}, "summary": {"mean_iou": score, "failure_count": 0}, "pages": []}
            for v, score in [(0, 0.70), (1, 0.90)]
        ]
        report = build_calibration_intelligence(
            ranked, detector="retained", strategy="exhaustive", possible_parameter_sets=2,
            regression_context={
                "zombie_parameters": ["dead"],
                "live_possible_parameter_sets": 2,
                "zombie_possible_parameter_sets": 4,
                "zombie_parameter_evidence": {
                    "dead": {"classification": "Zombie", "eta_squared": 0.0, "mean_iou_range": 0.0,
                             "near_best_value_coverage": 1.0, "value_count": 2, "source": "retained test evidence"}
                },
            },
        )
        influence = {item["parameter"]: item for item in report["parameter_influence"]}
        self.assertTrue(influence["dead"]["retained"])
        self.assertEqual(influence["dead"]["evidence_source"], "retained test evidence")
        self.assertNotIn("dead", report["parameter_intelligence"]["active_parameters"])
        self.assertEqual(report["domain_space"]["exhaustive"]["parameter_set_count"], 2)
        self.assertEqual(report["domain_space"]["exhaustive_with_zombies"]["parameter_set_count"], 4)
