from __future__ import annotations

import copy
import unittest

from hth.regression.materialization import (
    CANONICAL_REPORT_OUTPUTS,
    build_calibration_identity,
    build_canonical_manifest,
    build_canonical_summary,
    canonical_outcome_summary_fields,
    derive_canonical_outcome,
)
from hth.regression.reports import ranking_key


def _result(parameter_set_id: str, iou: float, *, profile=None, roles=(), requested=False):
    return {
        "parameter_set_id": parameter_set_id,
        "profile": profile,
        "reference_roles": list(roles),
        "requested_search_member": requested,
        "parameters": {"threshold": iou},
        "summary": {
            "mean_iou": iou,
            "minimum_iou": iou,
            "stddev_iou": 0.0,
            "mean_edge_error_px": 1.0,
            "failure_count": 0,
            "success_count": 1,
            "elapsed_ms_total": 1.0,
        },
        "pages": [
            {
                "global_ordinal": 1,
                "label": "page-1",
                "layout_type": "single",
                "status": "ok",
                "iou": iou,
                "edge_error_mean_px": 1.0,
                "edge_error_maximum_px": 1,
                "elapsed_ms": 1.0,
                "approved_bbox": [0, 0, 10, 10],
                "predicted_bbox": [0, 0, 10, 10],
            }
        ],
    }


def _winner_pages(winner, baseline):
    return {
        "available": True,
        "winner": winner["parameter_set_id"],
        "baseline": baseline["parameter_set_id"] if baseline else None,
    }


class RegressionMaterializationTests(unittest.TestCase):
    def _canonical_outcome(self):
        evidence = [
            _result("baseline", 0.7, profile="baseline", roles=("baseline",)),
            _result("historic", 0.8, roles=("historic_best",)),
            _result("winner", 0.9, requested=True),
        ]
        return derive_canonical_outcome(
            copy.deepcopy(evidence),
            ranking_key=ranking_key,
            winner_page_builder=_winner_pages,
        )

    def test_outcome_reduction_owns_reference_and_search_ranking(self):
        outcome = self._canonical_outcome()
        self.assertEqual(outcome.winner["parameter_set_id"], "winner")
        self.assertEqual(outcome.baseline["parameter_set_id"], "baseline")
        self.assertEqual(outcome.historic_best["parameter_set_id"], "historic")
        self.assertEqual([row["parameter_set_id"] for row in outcome.search_ranked], ["winner"])
        self.assertEqual(outcome.search_ranked[0]["search_rank"], 1)

    def test_all_paths_share_the_same_outcome_summary_fields(self):
        outcome = self._canonical_outcome()
        fields = canonical_outcome_summary_fields(outcome)
        summary = build_canonical_summary(
            outcome,
            run_id="run-1",
            detector="detector",
            strategy="exhaustive",
            requested_strategy="exhaustive",
            strategy_fallback_reason=None,
            threads=4,
            shard={"count": 2},
            detector_pipeline={"pipeline_count": 1},
            parameter_space={"possible_parameter_sets": 3},
            page_ordinals=[1],
            golden_set_sha256="gold",
            detector_config_sha256="config",
            model_selection={"variant": "current"},
            max_dimension=1800,
            runner={"python_version": "3.12"},
            source_commit="source",
            progress={"failures": 0},
            performance={"sample_count": 1},
        )
        for key, value in fields.items():
            self.assertEqual(summary[key], value, key)
        self.assertEqual(summary["detector_config_sha256"], "config")
        self.assertEqual(summary["model_selection"], {"variant": "current"})
        self.assertEqual(summary["detector_pipeline"], {"pipeline_count": 1})

    def test_calibration_identity_preserves_model_and_pipeline_identity(self):
        identity = build_calibration_identity(
            run_id="run-1",
            created_at_utc="2026-09-01T00:00:00Z",
            source_document={"id": "source"},
            golden_set={"sha256": "gold"},
            detector="detector",
            detector_configuration="detector.json",
            detector_config_sha256="config",
            model_selection={"variant": "current"},
            pipeline_commit="pipeline",
            source_commit="source-commit",
            python_version="3.12",
            opencv_version="4.14",
        )
        self.assertEqual(identity["model_selection"]["variant"], "current")
        self.assertEqual(identity["pipeline"]["commit"], "pipeline")
        self.assertEqual(identity["detector_configuration"]["sha256"], "config")

    def test_manifest_has_one_canonical_output_contract(self):
        manifest = build_canonical_manifest(
            self._canonical_outcome(),
            run_id="run-1",
            detector="detector",
            strategy="exhaustive",
            started_at_utc="start",
            finished_at_utc="finish",
            additional_outputs=("logs/runner-performance.jsonl",),
        )
        self.assertEqual(tuple(manifest["outputs"][: len(CANONICAL_REPORT_OUTPUTS)]), CANONICAL_REPORT_OUTPUTS)
        self.assertEqual(manifest["outputs"].count("RUN-INFO.json"), 1)
        self.assertIn("logs/runner-performance.jsonl", manifest["outputs"])


if __name__ == "__main__":
    unittest.main()
