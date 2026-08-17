import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github/workflows/regress-detector.yml",
    ROOT / ".github/workflows/execution-optimizer.yml",
)


class AtomicRuntimeRepairWorkflowTests(unittest.TestCase):
    def test_each_repair_path_uses_atomic_rebuild(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("atomic_rebuild_runtime base", text, workflow.name)
            self.assertIn("atomic_rebuild_runtime dhsegment", text, workflow.name)
            self.assertIn("atomic_rebuild_runtime kraken", text, workflow.name)

    def test_rebuild_starts_from_fresh_venv_not_in_place_mutation(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn('mv "$HTH_VENV" "$backup"', text, workflow.name)
            self.assertIn('"$HTH_BOOTSTRAP_PYTHON" -m venv "$HTH_VENV"', text, workflow.name)
            self.assertIn('rm -rf "$backup"', text, workflow.name)

    def test_all_detector_kraken_rebuild_preserves_tensorflow(self):
        regress = WORKFLOWS[0].read_text(encoding="utf-8")
        optimizer = WORKFLOWS[1].read_text(encoding="utf-8")
        self.assertIn("HTH_INCLUDE_DHSEGMENT: ${{ matrix.detector == 'all' && 'true' || 'false' }}", regress)
        self.assertIn("HTH_INCLUDE_DHSEGMENT: ${{ inputs.algorithm == 'all' && 'true' || 'false' }}", optimizer)
        for text in (regress, optimizer):
            self.assertIn('if [[ "${HTH_INCLUDE_DHSEGMENT:-false}" == "true" ]]; then', text)
            self.assertIn('python -m pip install "tensorflow-cpu>=2.18,<2.21"', text)


if __name__ == "__main__":
    unittest.main()
