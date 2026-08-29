import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github/workflows/regress-detector.yml",
    ROOT / ".github/workflows/execution-optimizer.yml",
)
MANAGER = (ROOT / "tools" / "ensure-managed-runtime.sh").read_text(encoding="utf-8")
PYTHON_ACTION = (ROOT / ".github/actions/setup-hth-python/action.yml").read_text(encoding="utf-8")


class KrakenCpuRuntimeWorkflowTests(unittest.TestCase):
    def test_self_hosted_runtime_uses_requested_persistent_path(self):
        self.assertIn('runtime_root="/tmp/.ar/.hth-runtime"', PYTHON_ACTION)
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn('rm -rf "/tmp/.ar/.hth-runtime"', text, workflow.name)
            self.assertIn("uses: ./hth-pipeline/.github/actions/setup-hth-python", text, workflow.name)

    def test_kraken_installs_matched_cpu_torchvision_pair_before_kraken(self):
        torch = MANAGER.index('"torch==2.10.0"')
        vision = MANAGER.index('"torchvision==0.25.0"', torch)
        cpu = MANAGER.index("--index-url https://download.pytorch.org/whl/cpu", vision)
        kraken = MANAGER.index('python -m pip install "kraken==7.0.2"', cpu)
        self.assertLess(torch, vision)
        self.assertLess(vision, cpu)
        self.assertLess(cpu, kraken)

    def test_kraken_cpu_backend_is_verified(self):
        self.assertIn("import torchvision", MANAGER)
        self.assertIn("if torch.cuda.is_available():", MANAGER)
        self.assertIn("Kraken PyTorch/Torchvision backend verified: CPU-only", MANAGER)


if __name__ == "__main__":
    unittest.main()
