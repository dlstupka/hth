import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github/workflows/regress-detector.yml",
    ROOT / ".github/workflows/execution-optimizer.yml",
)
RUNTIME_MANAGER = ROOT / "tools" / "ensure-managed-runtime.sh"


class RuntimeVerifyInstallWorkflowTests(unittest.TestCase):
    def test_self_hosted_runtime_is_reusable_and_manually_wipeable(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("clean_runner:", text, workflow.name)
            self.assertIn('description: "Advanced — Wipe runner workspace and rebuild from scratch"', text, workflow.name)
            self.assertIn("- name: Wipe runner workspace", text, workflow.name)
            self.assertIn('find "$GITHUB_WORKSPACE" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +', text, workflow.name)
            wipe = text.index("- name: Wipe runner workspace")
            checkout = text.index("- name: Checkout HTH pipeline", wipe)
            self.assertLess(wipe, checkout, workflow.name)
            self.assertIn('runtime_root="/tmp/.ar/.hth-runtime"', text, workflow.name)
            self.assertIn('Reusing verified Python environment', text, workflow.name)
            self.assertIn('HTH_VENV_REUSED', text, workflow.name)
            self.assertIn('rm -rf "/tmp/.ar/.hth-runtime"', text, workflow.name)
            self.assertIn('PIP_DISABLE_PIP_VERSION_CHECK=1', text, workflow.name)

    def test_runtime_is_built_once_then_specialized_steps_only_verify(self):
        manager = RUNTIME_MANAGER.read_text(encoding="utf-8")
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("- name: Verify / Install complete managed runtime", text, workflow.name)
            self.assertIn("bash hth-pipeline/tools/ensure-managed-runtime.sh", text, workflow.name)
            self.assertIn("- name: Verify dhSegment TensorFlow runtime", text, workflow.name)
            self.assertIn("- name: Verify Kraken historical-document segmentation runtime", text, workflow.name)
            self.assertNotIn("- name: Verify / Install dhSegment TensorFlow runtime", text, workflow.name)
            self.assertNotIn("- name: Verify / Install Kraken historical-document segmentation runtime", text, workflow.name)
        self.assertIn("Managed runtime verified — using previous install; no install required.", manager)
        self.assertIn("rebuilding once with the complete required dependency set", manager)

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
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn('venv_dir="$RUNNER_TEMP/hth-python-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"', text)
            self.assertIn('if [[ "${{ runner.environment }}" == "self-hosted" ]]; then', text)


if __name__ == "__main__":
    unittest.main()
