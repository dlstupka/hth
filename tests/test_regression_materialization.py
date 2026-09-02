from __future__ import annotations

import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from hth.regression.materialization import (
    CANONICAL_REPORT_OUTPUTS,
    build_calibration_identity,
    build_canonical_calibration_from_summary,
    build_canonical_manifest,
    build_canonical_summary,
    canonical_outcome_summary_fields,
    derive_canonical_outcome,
)
from hth.regression.reports import ranking_key, write_raw_evidence, write_raw_results
from hth.regression.merge_shards import _merged_pipeline_context, _require_common, _results_from_raw, merge
from hth.regression.runner import parse_args, run
from hth.historical_rerank import rerank_run


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

    def test_raw_evidence_round_trip_preserves_extension_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.csv"
            result = _result("candidate", 0.91, requested=True)
            result.update({
                "metadata": {"model": {"sha256": "abc"}},
                "warnings": [{"code": "soft-limit", "detail": {"value": 3}}],
                "error": {"type": "aggregate", "context": {"retry": False}},
                "future_result_field": {"nested": [1, "two", None]},
            })
            result["pages"][0].update({
                "metadata": {"candidate_count": 4},
                "warnings": [{"code": "clipped"}],
                "error": {"type": "page-note", "context": {"source": "detector"}},
                "future_page_field": {"confidence": 0.87654321},
            })
            empty_result = _result("no-pages", 0.0)
            empty_result["pages"] = []
            empty_result["future_result_field"] = {"survives_without_pages": True}

            write_raw_results(path, [result, empty_result])
            write_raw_evidence(path.with_name("evidence.jsonl"), [result, empty_result])
            restored_results = _results_from_raw(path)
            restored = restored_results[0]

            for key in ("metadata", "warnings", "error", "future_result_field"):
                self.assertEqual(restored[key], result[key], key)
            for key in ("metadata", "warnings", "error", "future_page_field"):
                self.assertEqual(restored["pages"][0][key], result["pages"][0][key], key)
            self.assertEqual(restored["parameters"], result["parameters"])
            self.assertEqual(len(restored_results), 2)
            self.assertEqual(restored_results[1]["pages"], [])
            self.assertEqual(
                restored_results[1]["future_result_field"],
                {"survives_without_pages": True},
            )

    def test_all_paths_share_the_same_outcome_summary_fields(self):
        outcome = self._canonical_outcome()
        fields = canonical_outcome_summary_fields(outcome)
        summary = build_canonical_summary(
            outcome,
            run_id="run-1",
            detector="detector",
            run_mode="full",
            evidence_tier="authoritative",
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
        self.assertEqual(summary["run_mode"], "full")
        self.assertEqual(summary["evidence_tier"], "authoritative")

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

    def test_calibration_context_is_derived_from_the_canonical_summary(self):
        outcome = self._canonical_outcome()
        summary = build_canonical_summary(
            outcome,
            run_id="run-1",
            detector="detector",
            run_mode="full",
            evidence_tier="authoritative",
            strategy="exhaustive",
            requested_strategy="exhaustive",
            strategy_fallback_reason=None,
            threads=4,
            shard={"count": 1},
            detector_pipeline={"pipeline_count": 2, "pipeline_number": 1},
            parameter_space={
                "possible_parameter_sets": 3,
                "planned_parameter_sets": 3,
                "actual_page_evaluations": 3,
                "live_possible_parameter_sets": 3,
                "zombie_possible_parameter_sets": 6,
                "canonical_search_space": {"configured_zombie_parameters": ["threshold"]},
            },
            page_ordinals=[1],
            golden_set_sha256="gold",
            detector_config_sha256="config",
            model_selection={"variant": "current"},
            max_dimension=1800,
            runner={
                "pipeline_commit": "pipeline",
                "python_version": "3.12",
                "opencv_version": "4.14",
            },
            source_commit="source",
            progress={"failures": 0, "average_eval_rate": 2.5},
        )
        calibration = build_canonical_calibration_from_summary(
            outcome,
            summary=summary,
            created_at_utc="2026-09-02T00:00:00Z",
            golden_set_configuration="golden.json",
            golden_set_payload={
                "collection_id": "HTH-0001",
                "schema_version": "1",
                "source_document": {"id": "source-document"},
            },
            detector_configuration="detector.json",
            detector_config={
                "profiles": {"baseline": {"threshold": 1}},
                "zombie_parameters": {"threshold": {"last_measured": {"build": 1}}},
            },
        )
        identity = calibration["calibration_identity"]
        metadata = calibration["regression_metadata"]
        self.assertEqual(identity["golden_set"]["page_ordinals"], [1])
        self.assertEqual(identity["pipeline"]["commit"], "pipeline")
        self.assertEqual(metadata["configured_threads"], 4)
        self.assertEqual(metadata["evaluated_parameter_sets"], 3)
        self.assertEqual(metadata["baseline_parameters"], {"threshold": 1})
        self.assertEqual(metadata["zombie_parameter_evidence"]["threshold"], {"build": 1})

    def test_manifest_has_one_canonical_output_contract(self):
        manifest = build_canonical_manifest(
            self._canonical_outcome(),
            run_id="run-1",
            detector="detector",
            run_mode="full",
            evidence_tier="authoritative",
            strategy="exhaustive",
            started_at_utc="start",
            finished_at_utc="finish",
            additional_outputs=("logs/runner-performance.jsonl",),
        )
        self.assertEqual(tuple(manifest["outputs"][: len(CANONICAL_REPORT_OUTPUTS)]), CANONICAL_REPORT_OUTPUTS)
        self.assertEqual(manifest["outputs"].count("RUN-INFO.json"), 1)
        self.assertIn("logs/runner-performance.jsonl", manifest["outputs"])

    def test_shard_merge_rejects_identity_drift_and_aggregates_pipeline_provenance(self):
        self.assertEqual(_require_common("identity", ["same", "same"]), "same")
        with self.assertRaisesRegex(ValueError, "Shard identity mismatch"):
            _require_common("identity", ["first", "second"])
        pipeline = _merged_pipeline_context(
            [
                {"detector_pipeline": {"pipeline_count": 2, "pipeline_number": 1, "loading_strategy": "lpt"}},
                {"detector_pipeline": {"pipeline_count": 2, "pipeline_number": 2, "loading_strategy": "lpt"}},
            ],
            [{}, {}],
        )
        self.assertEqual(pipeline["pipeline_count"], 2)
        self.assertIsNone(pipeline["pipeline_number"])
        self.assertEqual(pipeline["pipeline_numbers"], [1, 2])

    def test_direct_merged_and_reranked_runs_share_canonical_semantics(self):
        self.maxDiff = None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            image_root.mkdir()
            image = np.zeros((150, 200, 3), dtype=np.uint8)
            cv2.rectangle(image, (20, 15), (180, 135), (245, 245, 245), -1)
            cv2.imwrite(str(image_root / "fs_0001.png"), image)

            golden_set = root / "golden.json"
            golden_set.write_text(json.dumps({
                "schema_version": "1",
                "collection_id": "TEST",
                "source_document": {"id": "source"},
                "pages": [{
                    "global_ordinal": 1,
                    "label": "page-1",
                    "layout_type": "single",
                    "physical_document_bbox": [20, 15, 181, 136],
                }],
            }), encoding="utf-8")
            baseline = {
                "minimum_contour_area_fraction": 0.12,
                "polygon_epsilon_fraction": 0.018,
                "close_kernel_fraction": 0.0,
                "close_iterations": 0,
                "rectangularity_weight": 0.25,
                "bbox_padding_fraction": 0.0,
                "merge_fragmented_contours": False,
            }
            config = root / "contour.json"
            config.write_text(json.dumps({
                "schema_version": "0.1",
                "detector": "contour",
                "profiles": {"baseline": baseline},
                "parameters": {
                    name: {"values": ([0.12, 0.2] if name == "minimum_contour_area_fraction" else [value])}
                    for name, value in baseline.items()
                },
                "regression": {"debug_artifacts": "none"},
            }), encoding="utf-8")

            common = [
                "--detector-config", str(config),
                "--golden-set", str(golden_set),
                "--image-root", str(image_root),
                "--max-dimension", "200",
                "--debug-level", "none",
                "--threads", "1",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                direct = run(parse_args([*common, "--output", str(root / "direct")]))
                shared_baseline = root / "shared-baseline.json"
                shard_zero = run(parse_args([
                    *common, "--output", str(root / "shard-zero"),
                    "--shard-index", "0", "--shard-count", "2",
                    "--shared-baseline", str(shared_baseline),
                ]))
                shard_one = run(parse_args([
                    *common, "--output", str(root / "shard-one"),
                    "--shard-index", "1", "--shard-count", "2",
                    "--shared-baseline", str(shared_baseline),
                ]))
                merged = merge(
                    [shard_zero, shard_one], root / "merged", config,
                    expected_shard_count=2,
                    golden_set=golden_set,
                    image_root=image_root,
                    max_dimension=200,
                    debug_level="none",
                )

            direct_summary = json.loads((direct / "reports" / "summary.json").read_text(encoding="utf-8"))
            merged_summary = json.loads((merged / "reports" / "summary.json").read_text(encoding="utf-8"))
            for key in (
                "schema_version", "detector", "strategy", "requested_strategy",
                "strategy_fallback_reason", "golden_set_sha256",
                "detector_config_sha256", "model_selection", "max_dimension",
                "measurement_state", "run_mode", "evidence_tier",
            ):
                self.assertEqual(direct_summary[key], merged_summary[key], key)
            self.assertEqual(
                direct_summary["winner"]["parameter_identity_sha256"],
                merged_summary["winner"]["parameter_identity_sha256"],
            )
            direct_calibration = json.loads((direct / "reports" / "calibration-intelligence.json").read_text(encoding="utf-8"))
            merged_calibration = json.loads((merged / "reports" / "calibration-intelligence.json").read_text(encoding="utf-8"))
            for key in ("detector_configuration", "model_selection"):
                self.assertEqual(
                    direct_calibration["calibration_identity"][key],
                    merged_calibration["calibration_identity"][key],
                    key,
                )
            for key in (
                "requested_strategy", "resolved_strategy", "possible_parameter_sets",
                "evaluated_parameter_sets", "golden_set_pages", "page_evaluations",
                "baseline_parameters", "canonical_search_space",
            ):
                self.assertEqual(
                    direct_calibration["regression_metadata"][key],
                    merged_calibration["regression_metadata"][key],
                    key,
                )

            results_root = root / "results"
            results_root.mkdir()
            (results_root / "calibration-index.json").write_text(json.dumps({
                "entries": [{
                    "calibration_id": merged_summary["run_id"],
                    "run_mode": merged_summary["run_mode"],
                    "evidence_tier": merged_summary["evidence_tier"],
                    "calibration_status": merged_summary["evidence_tier"],
                    "source_document_id": "source",
                    "build": {"github_run_number": "1", "mode": "full"},
                }],
            }), encoding="utf-8")
            fake_entry = {
                "calibration_id": merged_summary["run_id"],
                "compatibility_key": "key",
                "run_mode": "full",
                "evidence_tier": "authoritative",
                "calibration_status": "authoritative",
            }
            with patch("hth.historical_rerank.publish_run", return_value=fake_entry), patch(
                "hth.historical_rerank.update_index"
            ):
                rerank_run(merged, results_root)

            reranked_summary = json.loads((merged / "reports" / "summary.json").read_text(encoding="utf-8"))
            reranked_calibration = json.loads((merged / "reports" / "calibration-intelligence.json").read_text(encoding="utf-8"))

            def semantic_view(summary, calibration):
                def result_view(result):
                    if result is None:
                        return None
                    view = {
                        key: result.get(key)
                        for key in (
                            "parameter_set_id", "parameter_identity_sha256",
                            "parameter_set_equivalence_family_id", "profile",
                            "reference_roles", "parameters",
                        )
                    }
                    view["quality"] = {
                        key: (result.get("summary") or {}).get(key)
                        for key in (
                            "mean_iou", "mean_iou_success", "minimum_iou",
                            "stddev_iou", "mean_edge_error_px", "failure_count",
                            "success_count", "page_count",
                        )
                    }
                    return view
                return {
                    "run_mode": summary["run_mode"],
                    "evidence_tier": summary["evidence_tier"],
                    "detector": summary["detector"],
                    "strategy": summary["strategy"],
                    "measurement_state": summary["measurement_state"],
                    "parameter_set_count": summary["parameter_set_count"],
                    "winner": result_view(summary["winner"]),
                    "baseline": result_view(summary["baseline"]),
                    "calibration_run_mode": calibration["run_mode"],
                    "calibration_evidence_tier": calibration["evidence_tier"],
                    "selection": calibration["detector_selection_intelligence"],
                    "search": calibration["search"],
                }

            self.assertEqual(
                semantic_view(direct_summary, direct_calibration),
                semantic_view(merged_summary, merged_calibration),
            )
            self.assertEqual(
                semantic_view(merged_summary, merged_calibration),
                semantic_view(reranked_summary, reranked_calibration),
            )


if __name__ == "__main__":
    unittest.main()
