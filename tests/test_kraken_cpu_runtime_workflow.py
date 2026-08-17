import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github/workflows/regress-detector.yml",
    ROOT / ".github/workflows/execution-optimizer.yml",
)


class KrakenCpuRuntimeWorkflowTests(unittest.TestCase):
    def test_self_hosted_runtime_uses_requested_persistent_path(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn('runtime_root="/tmp/.ar/.hth-runtime"', text, workflow.name)
            self.assertIn('rm -rf "/tmp/.ar/.hth-runtime"', text, workflow.name)

    def test_kraken_installs_matched_cpu_torchvision_pair_before_kraken(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            torch = text.index('"torch==2.10.0"')
            vision = text.index('"torchvision==0.25.0"', torch)
            cpu = text.index("--index-url https://download.pytorch.org/whl/cpu", vision)
            kraken = text.index('python -m pip install "kraken==7.0.2"', cpu)
            self.assertLess(torch, vision, workflow.name)
            self.assertLess(vision, cpu, workflow.name)
            self.assertLess(cpu, kraken, workflow.name)

    def test_kraken_cpu_backend_is_verified(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("import torchvision", text, workflow.name)
            self.assertIn("if torch.cuda.is_available():", text, workflow.name)
            self.assertIn("PyTorch/Torchvision backend: CPU-only", text, workflow.name)

    def test_failed_repair_restores_previous_runtime(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn('local backup="${HTH_VENV}.backup-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"', text)
            self.assertIn("Runtime rebuild failed; restoring previous reusable runtime.", text)
            self.assertIn('mv "$HTH_VENV" "$backup"', text)
            self.assertIn('mv "$backup" "$HTH_VENV"', text)
            self.assertIn("trap restore_previous_runtime EXIT", text)
            self.assertIn("trap - EXIT", text)


if __name__ == "__main__":
    unittest.main()
