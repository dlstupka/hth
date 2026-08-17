import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github/workflows/regress-detector.yml",
    ROOT / ".github/workflows/execution-optimizer.yml",
)


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
            self.assertIn('runtime_root="$GITHUB_WORKSPACE/.hth-runtime"', text, workflow.name)
            self.assertIn('Reusing verified Python environment', text, workflow.name)
            self.assertIn('HTH_VENV_REUSED', text, workflow.name)

    def test_install_steps_verify_before_installing(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("- name: Verify / Install dependencies", text, workflow.name)
            self.assertIn("- name: Verify / Install dhSegment TensorFlow runtime", text, workflow.name)
            self.assertIn("- name: Verify / Install Kraken historical-document segmentation runtime", text, workflow.name)
            self.assertIn("if verify_requirements; then", text, workflow.name)
            self.assertIn("if verify_dhsegment_runtime; then", text, workflow.name)
            self.assertIn("if verify_kraken_runtime; then", text, workflow.name)
            self.assertIn("install skipped", text, workflow.name)

    def test_runtime_verification_checks_target_versions_and_imports(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn('Requirement(line)', text, workflow.name)
            self.assertIn('SpecifierSet(">=2.18,<2.21")', text, workflow.name)
            self.assertIn('import tensorflow as tf', text, workflow.name)
            self.assertIn('version != "7.0.2"', text, workflow.name)
            self.assertIn('from kraken.tasks.segmentation import SegmentationTaskModel', text, workflow.name)
            self.assertIn("python -m pip check", text, workflow.name)

    def test_github_hosted_stays_run_local(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn('venv_dir="$RUNNER_TEMP/hth-python-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"', text)
            self.assertIn('if [[ "${{ runner.environment }}" == "self-hosted" ]]; then', text)


if __name__ == "__main__":
    unittest.main()
