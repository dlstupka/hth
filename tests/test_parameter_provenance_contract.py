import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ParameterProvenanceContractTests(unittest.TestCase):
    def test_runner_persists_provenance(self):
        text = (ROOT / "hth/regression/runner.py").read_text(encoding="utf-8")
        self.assertIn('parameter-provenance.json', text)
        self.assertIn("attach_identity", text)
        self.assertIn("build_provenance", text)

    def test_csv_carries_full_identity(self):
        text = (ROOT / "hth/regression/reports.py").read_text(encoding="utf-8")
        self.assertIn("parameter_identity_sha256", text)
        self.assertIn("parameter_grid_ordinal", text)

    def test_calibration_store_persists_registry(self):
        text = (ROOT / "hth/calibration_store.py").read_text(encoding="utf-8")
        self.assertIn('"parameter-provenance.json"', text)
        self.assertIn('"parameter-provenance-index.json"', text)

    def test_workflow_commits_global_registry_index(self):
        text = (ROOT / ".github/workflows/regress-detector.yml").read_text(encoding="utf-8")
        self.assertIn("parameter-provenance-index.json", text)


if __name__ == "__main__":
    unittest.main()
