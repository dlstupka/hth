import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MultiDetectorExecutionMetadataContractTests(unittest.TestCase):
    def test_executor_records_claim_start_finish_worker_and_batch_events(self):
        text = (ROOT / "tools/run-detector-regressions.sh").read_text(encoding="utf-8")
        for token in (
            'telemetry/workers', 'telemetry/tasks', 'telemetry/claim-batches', r"printf 'claim_batch\t",
            r"printf 'start\t", r"printf 'finish\t", 'multidetector-execution.json',
            'hth.multidetector_store finalize',
        ):
            self.assertIn(token, text)

    def test_workflow_persists_multidetector_index(self):
        text = (ROOT / ".github/workflows/regress-detector.yml").read_text(encoding="utf-8")
        self.assertIn("--multidetector-index results-repo/indexes/multidetector-index.json", text)
        self.assertIn("hth.multidetector_store publish", text)
        self.assertIn("multidetector-index.json", text)


if __name__ == "__main__":
    unittest.main()
