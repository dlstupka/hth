import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HTH = ROOT / "hth"
MODULE_PATH = HTH / "detect_geometry_candidates.py"

def load_module():
    sys.path.insert(0, str(HTH))
    try:
        spec = importlib.util.spec_from_file_location(
            "detect_geometry_candidates_cli_test", MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)

class DetectGeometryCandidatesCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_detector_and_parameters_json_are_declared(self):
        argv = [
            str(MODULE_PATH),
            "--manifest", "manifest.json",
            "--analysis", "analysis.json",
            "--image-root", "images",
            "--output", "output.json",
            "--detector", "amsre_doc_ufcn_fusion",
            "--parameters-json", "parameters.json",
        ]
        with patch.object(sys, "argv", argv):
            args = self.module.parse_args()
        self.assertEqual(args.detector, "amsre_doc_ufcn_fusion")
        self.assertEqual(args.parameters_json, Path("parameters.json"))

    def test_default_candidate_generation_has_optional_detector_fields(self):
        argv = [
            str(MODULE_PATH),
            "--manifest", "manifest.json",
            "--analysis", "analysis.json",
            "--image-root", "images",
            "--output", "output.json",
        ]
        with patch.object(sys, "argv", argv):
            args = self.module.parse_args()
        self.assertIsNone(args.detector)
        self.assertIsNone(args.parameters_json)

if __name__ == "__main__":
    unittest.main()
