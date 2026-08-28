from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hth.optimizer_history import completed_run_records, persist_completed_run


class OptimizerHistoryTests(unittest.TestCase):
    def test_completed_run_is_preserved_as_independent_durable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observations = root / "work" / "observations.jsonl"
            observations.parent.mkdir()
            observations.write_text(json.dumps({"observation_id": "o1", "optimizer_run_id": "42", "detector_id": "det"}) + "\n", encoding="utf-8")
            metadata = {"stop_reason": "range_complete", "optimization_wall_seconds": 12}
            destination = persist_completed_run(results_root=root, detector="det", run_id="42", run_metadata=metadata,
                                                observation_log=observations, shard_log=None, runner_metrics_log=None)
            self.assertEqual(destination, root / "execution-optimizer" / "det" / "runs" / "42")
            records = completed_run_records(root, "det")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["manifest"]["optimizer_run_id"], "42")
            self.assertTrue(records[0]["manifest"]["valid"])
            self.assertEqual(records[0]["observations"][0]["observation_id"], "o1")

            bad = destination.parent / "43"
            bad.mkdir()
            (bad / "run.json").write_text(json.dumps({
                "schema_version": 1, "record_type": "execution-optimizer-run",
                "optimizer_run_id": "43", "detector_id": "det", "complete": True,
                "run_metadata": {"stop_reason": "range_complete", "optimization_started_epoch": 1787947200},
            }), encoding="utf-8")
            (bad / "observations.jsonl").write_text(json.dumps({
                "observation_id": "bad", "source": "execution-optimizer",
                "optimizer_run_id": "43", "detector_id": "det",
                "observed_at_utc": "2026-08-28T18:00:00Z",
            }) + "\n", encoding="utf-8")
            self.assertEqual(len(completed_run_records(root, "det")), 1)
            invalid = completed_run_records(root, "det", include_invalid=True)
            migrated = next(row for row in invalid if row["manifest"]["optimizer_run_id"] == "43")
            self.assertFalse(migrated["manifest"]["valid"])
            self.assertEqual(migrated["manifest"]["invalid_reason"], "single-detector pipeline fan-out bug")

    def test_incomplete_run_is_not_materialized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = persist_completed_run(results_root=root, detector="det", run_id="42", run_metadata={},
                                           observation_log=None, shard_log=None, runner_metrics_log=None)
            self.assertIsNone(result)
            self.assertFalse((root / "execution-optimizer" / "det" / "runs" / "42").exists())
