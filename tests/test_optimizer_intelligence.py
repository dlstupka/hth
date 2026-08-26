from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from hth.optimizer_intelligence import (
    compatible_optimizer_rows,
    resolve_optimizer_intelligence,
    resolve_selector_intelligence,
)
from hth.regression_shape import RunnerProfile, resolve_preferred_dispatch, resolve_workflow_shape


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _row(*, detector: str, detector_sha: str, golden_sha: str, runner_name: str,
         runner_label: str, logical: int, pipelines: int, threads: int, rate: float,
         strategy: str = "exhaustive") -> dict:
    return {
        "source": "execution-optimizer",
        "detector_id": detector,
        "mode": "full",
        "strategy": strategy,
        "detector_config_sha256": detector_sha,
        "golden_set_sha256": golden_sha,
        "possible_parameter_sets": 100,
        "actual_parameter_sets": 100,
        "max_dimension": 1800,
        "wall_clock_seconds": 10.0,
        "parameter_sets_per_second": rate,
        "active_pipelines": pipelines,
        "shards": pipelines,
        "threads_per_pipeline": threads,
        "allocated_threads": pipelines * threads,
        "runner": {
            "runner_label": runner_label,
            "runner_labels": ["self-hosted", "Linux", "X64", runner_label],
            "runner_name": runner_name,
            "cpu_model": "AMD EPYC",
            "physical_core_count": logical,
            "logical_cpu_count": logical,
        },
    }


class OptimizerIntelligenceTests(unittest.TestCase):
    def test_linear_cross_vcpu_projection_uses_pipeline_fraction(self) -> None:
        rows = [_row(
            detector="adaptive_radial_edge", detector_sha="sha", golden_sha="gold",
            runner_name="rh8-s32", runner_label="32t", logical=32,
            pipelines=22, threads=2, rate=74.57,
        )]
        result = resolve_optimizer_intelligence(
            detector="adaptive_radial_edge", rows=rows,
            target_runner_name="rh8-new", target_runner_label="192t",
            target_cpu_model="different", target_physical_cores=192,
            target_logical_cpus=192,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["provenance"], "predicted")
        self.assertEqual(result["relation"], "scaled-vcpu")
        self.assertEqual(result["anchor_logical_cpus"], 32)
        self.assertEqual(result["predicted_shape"]["pipelines"], 132)
        self.assertEqual(result["predicted_shape"]["threads_per_pipeline"], 2)
        self.assertEqual(result["predicted_shape"]["allocated_threads"], 264)

    def test_requested_runner_measured_evidence_beats_cross_vcpu_prediction(self) -> None:
        rows = [
            _row(detector="example", detector_sha="sha", golden_sha="gold",
                 runner_name="source32", runner_label="32t", logical=32,
                 pipelines=22, threads=2, rate=75.0),
            _row(detector="example", detector_sha="sha", golden_sha="gold",
                 runner_name="source192", runner_label="192t", logical=192,
                 pipelines=48, threads=8, rate=80.0),
        ]
        result = resolve_selector_intelligence(
            detector="example", rows=rows,
            required_labels=["self-hosted", "192t"],
            target_runner_label="192t", target_logical_cpus=192,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["provenance"], "measured")
        self.assertEqual(result["predicted_shape"]["pipelines"], 48)
        self.assertEqual(result["predicted_shape"]["threads_per_pipeline"], 8)

    def test_are_dispatch_projects_32_vcpu_anchor_to_192_vcpu_requested_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_root = root / "detectors"
            detector = detector_root / "adaptive_radial_edge.json"
            golden = root / "golden.json"
            _write_json(detector, {"detector": "adaptive_radial_edge"})
            _write_json(golden, {"pages": []})
            row = _row(
                detector="adaptive_radial_edge", detector_sha=_sha(detector), golden_sha=_sha(golden),
                runner_name="rh8-s32", runner_label="32t", logical=32,
                pipelines=22, threads=2, rate=74.57,
            )
            index = root / "parallelism-index.json"
            _write_json(index, {"observations": [row]})

            result = resolve_preferred_dispatch(
                shape_mode="preferred", regression_mode="full", strategy="exhaustive", limit="",
                detector="adaptive_radial_edge", parallelism_index=index,
                detector_config_root=detector_root, golden_set=golden, max_dimension=1800,
                requested_runner="github-hosted", specific_runner="custom", custom_runner_label="192t",
            )
            self.assertTrue(result["exact"])
            self.assertEqual(result["runs_on"], ["self-hosted", "192t"])
            self.assertEqual((result["pipelines"], result["threads_per_pipeline"]), (132, 2))
            self.assertEqual(result["runner_budget"], 384)
            self.assertEqual(result["source"], "predicted-low-linear-vcpu-dispatch")


    def test_dispatch_accepts_completed_deterministic_optimizer_strategy_for_exact_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_root = root / "detectors"
            detector = detector_root / "adaptive_multi_scale_radial_edge.json"
            golden = root / "golden.json"
            _write_json(detector, {
                "detector": "adaptive_multi_scale_radial_edge",
                "optimizer_shape_compatibility": "detector-implementation",
            })
            _write_json(golden, {"pages": []})
            row = _row(
                detector="adaptive_multi_scale_radial_edge", detector_sha="older-grid",
                golden_sha=_sha(golden), runner_name="rh8-al319", runner_label="192t",
                logical=192, pipelines=48, threads=8, rate=24.94, strategy="critical",
            )
            index = root / "parallelism-index.json"
            _write_json(index, {"observations": [row]})

            result = resolve_preferred_dispatch(
                shape_mode="preferred", regression_mode="full", strategy="exhaustive", limit="",
                detector="adaptive_multi_scale_radial_edge", parallelism_index=index,
                detector_config_root=detector_root, golden_set=golden, max_dimension=1800,
                requested_runner="github-hosted", specific_runner="custom", custom_runner_label="192t",
            )
            self.assertTrue(result["exact"])
            self.assertEqual((result["pipelines"], result["threads_per_pipeline"]), (48, 8))
            self.assertEqual(result["provenance"], "measured")

    def test_dispatch_projects_completed_deterministic_strategy_across_vcpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_root = root / "detectors"
            detector = detector_root / "adaptive_radial_edge.json"
            golden = root / "golden.json"
            _write_json(detector, {
                "detector": "adaptive_radial_edge",
                "optimizer_shape_compatibility": "detector-implementation",
            })
            _write_json(golden, {"pages": []})
            row = _row(
                detector="adaptive_radial_edge", detector_sha="older-grid",
                golden_sha=_sha(golden), runner_name="rh8-s32", runner_label="32t", logical=32,
                pipelines=22, threads=2, rate=74.57, strategy="important+",
            )
            index = root / "parallelism-index.json"
            _write_json(index, {"observations": [row]})

            result = resolve_preferred_dispatch(
                shape_mode="preferred", regression_mode="full", strategy="exhaustive", limit="",
                detector="adaptive_radial_edge", parallelism_index=index,
                detector_config_root=detector_root, golden_set=golden, max_dimension=1800,
                requested_runner="github-hosted", specific_runner="custom", custom_runner_label="192t",
            )
            self.assertTrue(result["exact"])
            self.assertEqual((result["pipelines"], result["threads_per_pipeline"]), (132, 2))
            self.assertEqual(result["provenance"], "predicted")

    def test_predicted_dispatch_is_re_resolved_on_concrete_runner_and_writes_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_root = root / "detectors"
            detector = detector_root / "adaptive_radial_edge.json"
            golden = root / "golden.json"
            prediction_out = root / "prediction.json"
            _write_json(detector, {"detector": "adaptive_radial_edge"})
            _write_json(golden, {"pages": []})
            index = root / "parallelism-index.json"
            _write_json(index, {"observations": [_row(
                detector="adaptive_radial_edge", detector_sha=_sha(detector), golden_sha=_sha(golden),
                runner_name="rh8-s32", runner_label="32t", logical=32,
                pipelines=22, threads=2, rate=74.57,
            )]})

            result = resolve_workflow_shape(
                shape_mode="preferred", regression_mode="full", strategy="exhaustive", limit="",
                detector="adaptive_radial_edge", manual_shape="", parallelism_index=index,
                predictions_index=root / "optimizer-predictions.json", detector_config_root=detector_root,
                golden_set=golden, max_dimension=1800,
                profile=RunnerProfile("rh8-al330", "192t", "AMD EPYC", 192, 192),
                prediction_out=prediction_out, runner_budget=384,
                pre_resolved_pipelines=132, pre_resolved_threads=2,
                pre_resolved_source="predicted-low-linear-vcpu-dispatch",
            )
            self.assertTrue(result["exact"])
            self.assertEqual((result["pipelines"], result["threads_per_pipeline"]), (132, 2))
            self.assertEqual(result["source"], "predicted-low-linear-vcpu")
            self.assertTrue(prediction_out.is_file())
            payload = json.loads(prediction_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["relation"], "scaled-vcpu")
            self.assertEqual(payload["anchor_logical_cpus"], 32)


    def test_dispatch_recovers_legacy_published_summary_when_index_has_no_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_root = root / "detectors"
            detector = detector_root / "adaptive_radial_edge.json"
            golden = root / "golden.json"
            _write_json(detector, {"detector": "adaptive_radial_edge", "optimizer_shape_compatibility": "detector-implementation"})
            _write_json(golden, {"pages": []})
            index = root / "indexes" / "parallelism-index.json"
            _write_json(index, {"observations": []})
            summary = root / "execution-optimizer" / "adaptive_radial_edge" / "summary.md"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                "| Runner | Pipelines | Threads / pipeline | Allocated | Sets/s | Shape time |\n"
                "|---|---:|---:|---:|---:|---:|\n"
                "| 32t — rh8-s32 (32 vCPU) | 22 | 2 | 44 | 74.57 | 1m 28s |\n",
                encoding="utf-8",
            )
            result = resolve_preferred_dispatch(
                shape_mode="preferred", regression_mode="full", strategy="exhaustive", limit="",
                detector="adaptive_radial_edge", parallelism_index=index,
                detector_config_root=detector_root, golden_set=golden, max_dimension=1800,
                requested_runner="github-hosted", specific_runner="custom", custom_runner_label="192t",
            )
            self.assertTrue(result["exact"])
            self.assertEqual((result["pipelines"], result["threads_per_pipeline"]), (132, 2))
            self.assertEqual(result["source"], "predicted-low-linear-vcpu-dispatch")

    def test_dispatch_uses_same_capacity_legacy_published_summary_as_measured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_root = root / "detectors"
            detector = detector_root / "adaptive_multi_scale_radial_edge.json"
            golden = root / "golden.json"
            _write_json(detector, {"detector": "adaptive_multi_scale_radial_edge", "optimizer_shape_compatibility": "detector-implementation"})
            _write_json(golden, {"pages": []})
            index = root / "indexes" / "parallelism-index.json"
            _write_json(index, {"observations": []})
            summary = root / "execution-optimizer" / "adaptive_multi_scale_radial_edge" / "summary.md"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                "| Runner | Pipelines | Threads / pipeline | Allocated | Sets/s | Shape time |\n"
                "|---|---:|---:|---:|---:|---:|\n"
                "| 192t — rh8-al319 (192 vCPU) | 48 | 8 | 384 | 24.94 | 6m 41s |\n",
                encoding="utf-8",
            )
            result = resolve_preferred_dispatch(
                shape_mode="preferred", regression_mode="full", strategy="exhaustive", limit="",
                detector="adaptive_multi_scale_radial_edge", parallelism_index=index,
                detector_config_root=detector_root, golden_set=golden, max_dimension=1800,
                requested_runner="github-hosted", specific_runner="custom", custom_runner_label="192t",
            )
            self.assertTrue(result["exact"])
            self.assertEqual((result["pipelines"], result["threads_per_pipeline"]), (48, 8))
            self.assertEqual(result["source"], "preferred-dispatch-optimizer")


if __name__ == "__main__":
    unittest.main()
