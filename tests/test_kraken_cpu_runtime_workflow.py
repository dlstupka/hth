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

    def test_kraken_installs_cpu_torch_before_kraken(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            cpu = text.index("--index-url https://download.pytorch.org/whl/cpu")
            torch = text.index('"torch==2.10.0"', cpu)
            kraken = text.index('python -m pip install "kraken==7.0.2"', torch)
            self.assertLess(cpu, kraken, workflow.name)
            self.assertLess(torch, kraken, workflow.name)

    def test_kraken_cpu_backend_is_verified(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("if torch.cuda.is_available():", text, workflow.name)
            self.assertIn("PyTorch backend: CPU-only", text, workflow.name)


if __name__ == "__main__":
    unittest.main()
