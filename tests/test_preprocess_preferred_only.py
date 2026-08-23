import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class PreprocessPreferredOnlyTests(unittest.TestCase):
    def test_test_mode_does_not_run_detector_zoo(self):
        core = (ROOT / ".github/workflows/_core-hth.yml").read_text(encoding="utf-8")
        self.assertIn("id: generate_geometry_candidates\n        if: ${{ inputs.mode == 'calibration' }}", core)
        self.assertIn("name: STAGE_DETECT_CANDIDATES\n        id: stage_detect_candidates_start\n        if: ${{ inputs.mode == 'calibration' }}", core)
        self.assertIn("Run approved detector over production collection", core)

if __name__ == "__main__":
    unittest.main()
