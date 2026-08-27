import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "tools" / "run-detector-regressions.sh").read_text(encoding="utf-8")


class StaticPipelineScheduleTests(unittest.TestCase):
    def test_worker_executes_only_its_preassigned_tasks(self):
        self.assertIn('for task_index in "${scheduled_tasks[@]}"; do', DRIVER)
        self.assertIn('run_detector_config "$task_index" "$pipeline_number"', DRIVER)

    def test_one_schedule_telemetry_record_is_written_per_pipeline(self):
        self.assertIn('schedule_batch_id="$(printf \'p%03d-static\' "$pipeline_number")"', DRIVER)
        self.assertIn('$telemetry_root/claim-batches/$schedule_batch_id.tsv', DRIVER)

    def test_no_dynamic_refill_path_remains(self):
        self.assertNotIn("claim_batch_from_queue()", DRIVER)
        self.assertNotIn("reclaim_expired_task()", DRIVER)
        self.assertNotIn("Seeded initial LPT batch", DRIVER)


if __name__ == "__main__":
    unittest.main()
