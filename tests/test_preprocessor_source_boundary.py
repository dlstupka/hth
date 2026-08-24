import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PreprocessorSourceBoundaryTests(unittest.TestCase):
    def test_framework_has_no_local_source_placeholder(self):
        self.assertFalse((ROOT / "data" / "source").exists())

    def test_preprocessor_requires_explicit_input(self):
        text = (ROOT / "hth" / "preprocess.py").read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r'p\.add_argument\("--input",\s*type=Path,\s*required=True',
        )
        self.assertNotIn('default=Path("data/source")', text)

    def test_local_launcher_requires_external_source_path(self):
        text = (ROOT / "tools" / "run-preprocess.ps1").read_text(encoding="utf-8")
        self.assertIn("[Parameter(Mandatory = $true", text)
        self.assertIn("[string]$SourcePath", text)
        self.assertIn("--input $SourcePath", text)
        self.assertNotRegex(text, r"data[\\/]+source")


if __name__ == "__main__":
    unittest.main()
