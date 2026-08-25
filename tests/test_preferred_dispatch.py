from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from hth.regression_shape import resolve_preferred_dispatch, resolve_workflow_shape, RunnerProfile


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class PreferredDispatchTests(unittest.TestCase):
    def test_preferred_dispatch_routes_to_optimizer_runner_and_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_root = root / "detectors"
            detector = detector_root / "learned_page_mask.json"
            golden = root / "golden.json"
            _write_json(detector, {
                "detector": "learned_page_mask",
                "optimizer_shape_compatibility": "detector-implementation",
            })
            _write_json(golden, {"pages": []})
            row = {
                "source": "execution-optimizer",
                "detector_id": "learned_page_mask",
                "mode": "full",
                "strategy": "exhaustive",
                "detector_config_sha256": _sha(detector),
                "golden_set_sha256": _sha(golden),
                "possible_parameter_sets": 10000,
                "actual_parameter_sets": 10000,
                "max_dimension": 1800,
                "wall_clock_seconds": 31.0,
                "parameter_sets_per_second": 7.84,
                "active_pipelines": 4,
                "shards": 4,
                "threads_per_pipeline": 96,
                "allocated_threads": 384,
                "runner": {
                    "runner_label": "192t",
                    "runner_name": "rh8-al319",
                    "runner_labels": ["self-hosted", "Linux", "X64", "192t"],
                    "logical_cpu_count": 192,
                },
            }
            index = root / "parallelism-index.json"
            _write_json(index, {"observations": [row]})

            result = resolve_preferred_dispatch(
                shape_mode="preferred",
                regression_mode="full",
                strategy="exhaustive",
                limit="",
                detector="learned_page_mask",
                parallelism_index=index,
                detector_config_root=detector_root,
                golden_set=golden,
                max_dimension=1800,
                requested_runner="github-hosted",
                specific_runner="any",
                custom_runner_label="",
            )

            self.assertFalse(result["exact"])
            self.assertEqual(result["runs_on"], ["ubuntu-latest"])
            self.assertEqual(result["runner_label"], "github-hosted")
            self.assertEqual(result["source"], "requested-runner-no-compatible-preferred-history")


    def test_capacity_runner_budget_preserves_free_threads_for_preferred_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_root = root / "detectors"
            detector = detector_root / "page_background.json"
            golden = root / "golden.json"
            _write_json(detector, {"detector": "page_background", "optimizer_shape_compatibility": "detector-implementation"})
            _write_json(golden, {"pages": []})
            row = {
                "source": "execution-optimizer",
                "detector_id": "page_background",
                "mode": "full",
                "strategy": "exhaustive",
                "detector_config_sha256": _sha(detector),
                "golden_set_sha256": _sha(golden),
                "possible_parameter_sets": 2187,
                "actual_parameter_sets": 2187,
                "max_dimension": 1800,
                "wall_clock_seconds": 16.0,
                "parameter_sets_per_second": 136.75,
                "active_pipelines": 5,
                "shards": 5,
                "threads_per_pipeline": 76,
                "allocated_threads": 380,
                "runner": {
                    "runner_label": "192t",
                    "runner_name": "rh8-al307",
                    "runner_labels": ["self-hosted", "Linux", "X64", "192t"],
                    "logical_cpu_count": 192,
                },
            }
            index = root / "parallelism-index.json"
            _write_json(index, {"observations": [row]})
            result = resolve_preferred_dispatch(
                shape_mode="preferred", regression_mode="full", strategy="exhaustive", limit="",
                detector="page_background", parallelism_index=index, detector_config_root=detector_root,
                golden_set=golden, max_dimension=1800, requested_runner="github-hosted",
                specific_runner="any", custom_runner_label="",
            )
            self.assertFalse(result["exact"])
            self.assertEqual(result["runs_on"], ["ubuntu-latest"])
            self.assertEqual(result["runner_label"], "github-hosted")

    def test_pre_resolved_shape_is_validated_against_dispatch_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_root = root / "detectors"
            _write_json(detector_root / "learned_page_mask.json", {"detector": "learned_page_mask"})
            golden = root / "golden.json"
            _write_json(golden, {"pages": []})
            index = root / "parallelism-index.json"
            _write_json(index, {"observations": []})

            result = resolve_workflow_shape(
                shape_mode="preferred",
                regression_mode="full",
                strategy="exhaustive",
                limit="",
                detector="learned_page_mask",
                manual_shape="",
                parallelism_index=index,
                predictions_index=None,
                detector_config_root=detector_root,
                golden_set=golden,
                max_dimension=1800,
                profile=RunnerProfile("rh8-al319", "192t", "AMD", 192, 192),
                prediction_out=None,
                runner_budget=384,
                pre_resolved_pipelines=4,
                pre_resolved_threads=96,
                pre_resolved_source="preferred-dispatch-optimizer",
            )
            self.assertTrue(result["exact"])
            self.assertEqual(result["allocated_threads"], 384)
            self.assertEqual(result["runner_budget"], 384)
            self.assertEqual(result["source"], "preferred-dispatch-optimizer")


    def test_preferred_dispatch_reuses_shape_after_calibration_grid_change_when_declared_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_root = root / "detectors"
            detector = detector_root / "multi_scale_radial_edge.json"
            golden = root / "golden.json"
            _write_json(detector, {
                "detector": "multi_scale_radial_edge",
                "optimizer_shape_compatibility": "detector-implementation",
                "parameters": {"generation": 1},
            })
            _write_json(golden, {"pages": []})
            old_detector_sha = _sha(detector)
            row = {
                "source": "execution-optimizer",
                "detector_id": "multi_scale_radial_edge",
                "mode": "full",
                "strategy": "exhaustive",
                "detector_config_sha256": old_detector_sha,
                "golden_set_sha256": _sha(golden),
                "possible_parameter_sets": 100001,
                "actual_parameter_sets": 100001,
                "max_dimension": 1800,
                "wall_clock_seconds": 12.0,
                "parameter_sets_per_second": 60.83,
                "active_pipelines": 9,
                "shards": 9,
                "threads_per_pipeline": 42,
                "allocated_threads": 378,
                "runner": {
                    "runner_label": "192t",
                    "runner_name": "rh8-al318",
                    "runner_labels": ["self-hosted", "Linux", "X64", "192t"],
                    "logical_cpu_count": 192,
                },
            }
            index = root / "parallelism-index.json"
            _write_json(index, {"observations": [row]})

            # Simulate a later calibration-grid expansion. The config SHA changes,
            # but the detector implementation and execution characteristics do not.
            _write_json(detector, {
                "detector": "multi_scale_radial_edge",
                "optimizer_shape_compatibility": "detector-implementation",
                "parameters": {"generation": 2, "expanded": True},
            })
            self.assertNotEqual(_sha(detector), old_detector_sha)

            result = resolve_preferred_dispatch(
                shape_mode="preferred",
                regression_mode="full",
                strategy="exhaustive",
                limit="",
                detector="multi_scale_radial_edge",
                parallelism_index=index,
                detector_config_root=detector_root,
                golden_set=golden,
                max_dimension=1800,
                requested_runner="github-hosted",
                specific_runner="any",
                custom_runner_label="",
            )

            self.assertFalse(result["exact"])
            self.assertEqual(result["runs_on"], ["ubuntu-latest"])
            self.assertEqual(result["source"], "requested-runner-no-compatible-preferred-history")

    def test_nonpreferred_dispatch_preserves_requested_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_root = root / "detectors"
            _write_json(detector_root / "learned_page_mask.json", {"detector": "learned_page_mask"})
            golden = root / "golden.json"
            _write_json(golden, {"pages": []})
            index = root / "parallelism-index.json"
            _write_json(index, {"observations": []})
            result = resolve_preferred_dispatch(
                shape_mode="auto",
                regression_mode="full",
                strategy="exhaustive",
                limit="",
                detector="learned_page_mask",
                parallelism_index=index,
                detector_config_root=detector_root,
                golden_set=golden,
                max_dimension=1800,
                requested_runner="self-hosted-e7k",
                specific_runner="any",
                custom_runner_label="",
            )
            self.assertFalse(result["exact"])
            self.assertEqual(result["runs_on"], ["self-hosted", "Linux", "X64", "e7k"])
            self.assertEqual(result["runner_label"], "e7k")


class PreferredDispatchConfigurationContractTests(unittest.TestCase):
    def test_refined_detector_grids_preserve_optimizer_shape_compatibility(self) -> None:
        root = Path(__file__).parents[1]
        detector_names = (
            "radial_edge",
            "adaptive_multi_scale_radial_edge",
            "amsre_bfq_spbv_pbg",
            "amsre_doc_ufcn_fusion",
            "adaptive_radial_edge",
            "msre_bfq_spbv_pbg",
            "multi_scale_radial_edge",
            "border_fusion_quad",
            "signed_polar_boundary_vote",
            "segment_supported_polar_vote",
        )
        for detector_name in detector_names:
            with self.subTest(detector=detector_name):
                payload = json.loads(
                    (root / "config" / "detectors" / f"{detector_name}.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    payload.get("optimizer_shape_compatibility"),
                    "detector-implementation",
                )


class PreferredDispatchWorkflowContractTests(unittest.TestCase):
    def test_workflow_resolves_runner_before_regression_job_dispatch(self) -> None:
        root = Path(__file__).parents[1]
        text = (root / ".github" / "workflows" / "regress-detector.yml").read_text(encoding="utf-8")
        self.assertIn("resolve-execution-dispatch:", text)
        self.assertIn("python -m hth.regression_shape dispatch-resolve", text)
        self.assertIn("needs: resolve-execution-dispatch", text)
        self.assertIn("runs-on: ${{ fromJSON(needs.resolve-execution-dispatch.outputs.runs_on) }}", text)
        self.assertIn("HTH_PRE_RESOLVED_PIPELINES", text)
        self.assertIn("--runner-budget", text)


if __name__ == "__main__":
    unittest.main()
