from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hth.optimizer_resume import prepare_resume, shape_completed


class OptimizerResumeTests(unittest.TestCase):
    def _checkpoint(self, root: Path, *, run_id: str = "100") -> Path:
        work = root / "optimizer-work"
        work.mkdir()
        (work / "run-metadata.json").write_text(json.dumps({
            "optimizer_run_id": run_id,
            "detector_id": "grabcut",
            "runner_label": "e7k",
            "runner_thread_budget": 192,
            "thread_min": 192,
            "thread_max": 192,
            "pipeline_enumeration": "exhaustive",
        }), encoding="utf-8")
        (work / "observations.jsonl").write_text(json.dumps({
            "observation_id": f"optimizer:{run_id}:1:run-a",
            "optimizer_run_id": run_id,
            "optimizer_shape_sequence": 1,
            "active_pipelines": 1,
            "threads_per_pipeline": 192,
            "parameter_sets_per_second": 1.9184,
        }) + "\n", encoding="utf-8")
        (work / "shards.jsonl").write_text(json.dumps({
            "observation_id": f"optimizer-shard:{run_id}:1:0:run-a",
            "optimizer_run_id": run_id,
            "shape_sequence": 1,
            "shard_index": 0,
        }) + "\n", encoding="utf-8")
        return work

    def test_prepare_resume_rewrites_completed_shapes_to_current_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._checkpoint(root)
            results = root / "results"
            results.mkdir()
            destination = root / "staged"
            result = prepare_resume(
                source_dir=source, destination_dir=destination, results_root=results,
                mode="auto", current_run_id="200", detector="grabcut", runner_label="e7k",
                runner_budget=192, thread_min=192, thread_max=192, enumeration="exhaustive",
                pipeline_min=1, pipeline_max=1,
            )
            self.assertTrue(result["resumed"])
            self.assertEqual(result["resumed_from_optimizer_run_id"], "100")
            row = json.loads((destination / "observations.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["optimizer_run_id"], "200")
            self.assertEqual(row["resumed_from_optimizer_run_id"], "100")
            self.assertIn("optimizer:200:", row["observation_id"])
            self.assertTrue(shape_completed(destination / "observations.jsonl", pipelines=1, threads=192))

    def test_auto_does_not_resume_already_published_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._checkpoint(root)
            results = root / "results"
            results.mkdir()
            (results / "optimizer-index.json").write_text(json.dumps({"runs": {"100": {}}}), encoding="utf-8")
            result = prepare_resume(
                source_dir=source, destination_dir=root / "staged", results_root=results,
                mode="auto", current_run_id="200", detector="grabcut", runner_label="e7k",
                runner_budget=192, thread_min=192, thread_max=192, enumeration="exhaustive",
                pipeline_min=1, pipeline_max=1,
            )
            self.assertFalse(result["resumed"])
            self.assertEqual(result["reason"], "checkpoint-already-published")

    def test_incompatible_checkpoint_is_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._checkpoint(root)
            results = root / "results"
            results.mkdir()
            result = prepare_resume(
                source_dir=source, destination_dir=root / "staged", results_root=results,
                mode="auto", current_run_id="200", detector="adaptive_radial_edge", runner_label="e7k",
                runner_budget=192, thread_min=192, thread_max=192, enumeration="exhaustive",
                pipeline_min=1, pipeline_max=1,
            )
            self.assertFalse(result["resumed"])
            self.assertEqual(result["reason"], "checkpoint-incompatible")


if __name__ == "__main__":
    unittest.main()
