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

            self.assertTrue(result["exact"])
            self.assertEqual(result["runs_on"], ["self-hosted", "Linux", "X64", "192t"])
            self.assertEqual(result["runner_label"], "192t")
            self.assertEqual(result["runner_name"], "rh8-al319")
            self.assertEqual((result["pipelines"], result["threads_per_pipeline"]), (4, 96))
            self.assertEqual(result["runner_budget"], 384)

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
