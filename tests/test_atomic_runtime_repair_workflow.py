import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANAGER = (ROOT / "tools" / "ensure-managed-runtime.sh").read_text(encoding="utf-8")


class AtomicRuntimeRepairWorkflowTests(unittest.TestCase):
    def test_full_base_rebuild_is_atomic(self):
        self.assertIn('runtime_backup="${HTH_VENV}.backup-', MANAGER)
        self.assertIn('mv "$HTH_VENV" "$runtime_backup"', MANAGER)
        self.assertIn('"$HTH_BOOTSTRAP_PYTHON" -m venv "$HTH_VENV"', MANAGER)
        self.assertIn("Managed runtime update failed; restoring previous reusable runtime.", MANAGER)
        self.assertIn('mv "$runtime_backup" "$HTH_VENV"', MANAGER)
        self.assertIn("trap restore_runtime_backup EXIT", MANAGER)
        self.assertIn("trap - EXIT", MANAGER)

    def test_incremental_augmentation_snapshots_verified_runtime_for_rollback(self):
        self.assertIn("begin_incremental_update()", MANAGER)
        self.assertIn("Snapshotting verified managed runtime before incremental augmentation.", MANAGER)
        self.assertIn('cp -a --reflink=auto "$HTH_VENV" "$runtime_backup"', MANAGER)
        self.assertIn('cp -a "$HTH_VENV" "$runtime_backup"', MANAGER)
        self.assertIn("trap restore_runtime_backup EXIT", MANAGER)

    def test_full_rebuild_can_install_all_requested_families(self):
        self.assertIn('python -m pip install --requirement hth-pipeline/requirements.txt', MANAGER)
        self.assertIn('install_dhsegment_layer', MANAGER)
        self.assertIn('install_kraken_layer', MANAGER)


if __name__ == "__main__":
    unittest.main()
