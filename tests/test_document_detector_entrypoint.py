import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DocumentDetectorEntrypointTests(unittest.TestCase):
    def test_run_document_detector_imports_in_direct_script_mode(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "hth" / "run_document_detector.py"), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--selection", proc.stdout)
        self.assertIn("--lifecycle-root", proc.stdout)

    def test_preprocess_and_test_both_delegate_preferred_detector_to_core(self):
        production = (ROOT / ".github" / "workflows" / "preprocess.yml").read_text(encoding="utf-8")
        test = (ROOT / ".github" / "workflows" / "preprocess-test.yml").read_text(encoding="utf-8")
        core = (ROOT / ".github" / "workflows" / "_core-hth.yml").read_text(encoding="utf-8")

        self.assertIn("uses: ./.github/workflows/_core-hth.yml", production)
        self.assertIn("uses: ./.github/workflows/_core-hth.yml", test)
        self.assertIn("document_detector: preferred", production)
        self.assertIn("document_detector: preferred", test)
        self.assertIn("Run approved detector over production collection", core)
        self.assertIn("hth-pipeline/hth/run_document_detector.py", core)


if __name__ == "__main__":
    unittest.main()
