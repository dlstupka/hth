import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github/workflows/regress-detector.yml",
    ROOT / ".github/workflows/execution-optimizer.yml",
)
PYTHON_ACTION = ROOT / ".github/actions/setup-hth-python/action.yml"
MANAGED_ACTION = ROOT / ".github/actions/setup-hth-managed-runtime/action.yml"
RUNTIME_MANAGER = ROOT / "tools" / "ensure-managed-runtime.sh"


class RuntimeVerifyInstallWorkflowTests(unittest.TestCase):
    def test_self_hosted_runtime_is_reusable_and_manually_wipeable(self):
        python_action = PYTHON_ACTION.read_text(encoding="utf-8")
        self.assertIn('runtime_root="/tmp/.ar/.hth-runtime"', python_action)
        self.assertIn('Reusing verified Python environment', python_action)
        self.assertIn('HTH_VENV_REUSED', python_action)
        self.assertIn('PIP_DISABLE_PIP_VERSION_CHECK=1', python_action)
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("clean_runner:", text, workflow.name)
            self.assertIn('description: "Advanced — Wipe runner workspace and rebuild from scratch"', text, workflow.name)
            self.assertIn("- name: Wipe runner workspace", text, workflow.name)
            self.assertIn('find "$GITHUB_WORKSPACE" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +', text, workflow.name)
            wipe = text.index("- name: Wipe runner workspace")
            checkout = text.index("- name: Checkout HTH pipeline", wipe)
            self.assertLess(wipe, checkout, workflow.name)
            self.assertIn('rm -rf "/tmp/.ar/.hth-runtime"', text, workflow.name)
            self.assertIn("uses: ./hth-pipeline/.github/actions/setup-hth-python", text, workflow.name)

    def test_regression_cleans_reusable_results_checkout_before_checkout_action(self):
        text = (ROOT / ".github/workflows/regress-detector.yml").read_text(encoding="utf-8")
        self.assertEqual(text.count("- name: Clean reusable results checkout"), 3)
        self.assertEqual(text.count("git -C results-repo reset --hard"), 3)
        self.assertEqual(text.count("git -C results-repo clean -ffd"), 3)
        for match in [
            pos for pos in range(len(text))
            if text.startswith("- name: Clean reusable results checkout", pos)
        ]:
            checkout = text.index("- name: Checkout results repository", match)
            self.assertLess(match, checkout)

    def test_runtime_is_built_once_then_specialized_steps_only_verify(self):
        manager = RUNTIME_MANAGER.read_text(encoding="utf-8")
        action = MANAGED_ACTION.read_text(encoding="utf-8")
        self.assertIn("- name: Verify / Install complete managed runtime", action)
        self.assertIn("tools/ensure-managed-runtime.sh", action)
        self.assertIn("- name: Verify dhSegment TensorFlow runtime", action)
        self.assertIn("- name: Verify Kraken historical-document segmentation runtime", action)
        self.assertNotIn("- name: Verify / Install dhSegment TensorFlow runtime", action)
        self.assertNotIn("- name: Verify / Install Kraken historical-document segmentation runtime", action)
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("uses: ./hth-pipeline/.github/actions/setup-hth-managed-runtime", text, workflow.name)
        self.assertIn("Managed runtime verified — using previous install; no install required.", manager)
        self.assertIn("Managed base runtime verified; augmenting only missing optional runtime layer(s).", manager)

    def test_runtime_verification_checks_target_versions_and_imports(self):
        manager = RUNTIME_MANAGER.read_text(encoding="utf-8")
        self.assertIn('Requirement(line)', manager)
        self.assertIn('SpecifierSet(">=2.18,<2.21")', manager)
        self.assertIn('import tensorflow as tf', manager)
        self.assertIn('version != "7.0.2"', manager)
        self.assertIn('from kraken.tasks.segmentation import SegmentationTaskModel', manager)
        self.assertIn('except metadata.PackageNotFoundError:', manager)
        self.assertIn("python -m pip check", manager)

    def test_github_hosted_stays_run_local(self):
        action = PYTHON_ACTION.read_text(encoding="utf-8")
        self.assertIn('venv_dir="$RUNNER_TEMP/hth-python-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"', action)
        self.assertIn('if [[ "${{ runner.environment }}" == "self-hosted" ]]; then', action)


if __name__ == "__main__":
    unittest.main()
