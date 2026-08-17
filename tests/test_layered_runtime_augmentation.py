import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANAGER = (ROOT / "tools" / "ensure-managed-runtime.sh").read_text(encoding="utf-8")


class LayeredRuntimeAugmentationTests(unittest.TestCase):
    def test_base_failure_is_the_only_path_that_reinstalls_requirements(self):
        requirements_install = 'python -m pip install --requirement hth-pipeline/requirements.txt'
        self.assertEqual(MANAGER.count(requirements_install), 1)
        self.assertIn("if ! verify_pip || ! verify_base; then", MANAGER)
        self.assertIn("Managed base runtime is invalid; rebuilding the base environment once.", MANAGER)

    def test_valid_base_with_missing_kraken_augments_only_kraken_layer(self):
        self.assertIn('missing_kraken=false', MANAGER)
        self.assertIn('if [[ "$need_kraken" == "true" ]] && ! verify_kraken; then', MANAGER)
        self.assertIn('missing_kraken=true', MANAGER)
        self.assertIn('if [[ "$missing_kraken" == "true" ]]; then\n  install_kraken_layer\nfi', MANAGER)
        self.assertIn("Installing missing matched CPU-only PyTorch/Torchvision layer for Kraken.", MANAGER)
        self.assertIn("Installing missing Kraken 7.0.2 layer only.", MANAGER)

    def test_valid_base_with_missing_dhsegment_augments_only_dhsegment_layer(self):
        self.assertIn('missing_dhsegment=false', MANAGER)
        self.assertIn('if [[ "$need_dhsegment" == "true" ]] && ! verify_dhsegment; then', MANAGER)
        self.assertIn('missing_dhsegment=true', MANAGER)
        self.assertIn('if [[ "$missing_dhsegment" == "true" ]]; then\n  install_dhsegment_layer\nfi', MANAGER)
        self.assertIn("Installing missing dhSegment TensorFlow layer only.", MANAGER)

    def test_incremental_update_reverifies_complete_runtime_before_commit(self):
        augment = MANAGER.index("begin_incremental_update")
        verify = MANAGER.rindex("verify_complete_runtime")
        commit = MANAGER.rindex("commit_runtime_update")
        self.assertLess(augment, verify)
        self.assertLess(verify, commit)


if __name__ == "__main__":
    unittest.main()
