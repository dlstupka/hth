import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANAGER = (ROOT / "tools" / "ensure-managed-runtime.sh").read_text(encoding="utf-8")


class AtomicRuntimeRepairWorkflowTests(unittest.TestCase):
    def test_one_complete_repair_path_is_atomic(self):
        self.assertIn('backup="${HTH_VENV}.backup-', MANAGER)
        self.assertIn('mv "$HTH_VENV" "$backup"', MANAGER)
        self.assertIn('"$HTH_BOOTSTRAP_PYTHON" -m venv "$HTH_VENV"', MANAGER)
        self.assertIn("Managed runtime rebuild failed; restoring previous reusable runtime.", MANAGER)
        self.assertIn('mv "$backup" "$HTH_VENV"', MANAGER)
        self.assertIn("trap restore_previous_runtime EXIT", MANAGER)
        self.assertIn("trap - EXIT", MANAGER)

    def test_complete_rebuild_installs_all_requested_families_in_one_pass(self):
        self.assertIn('python -m pip install --requirement hth-pipeline/requirements.txt', MANAGER)
        self.assertIn('if [[ "$need_dhsegment" == "true" ]]; then', MANAGER)
        self.assertIn('python -m pip install "tensorflow-cpu>=2.18,<2.21"', MANAGER)
        self.assertIn('if [[ "$need_kraken" == "true" ]]; then', MANAGER)
        self.assertIn('python -m pip install "kraken==7.0.2"', MANAGER)


if __name__ == "__main__":
    unittest.main()
