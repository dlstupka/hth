import unittest
from pathlib import Path


class DebugArtifactDispatchTests(unittest.TestCase):
    def test_generic_detector_debug_filename_order_is_optional(self):
        text = Path("hth/regression/runner.py").read_text(encoding="utf-8")
        self.assertIn("basic_names = basic_names_by_method.get(method, [])", text)
        self.assertNotIn("}[method]\n        images = module.debug_images(", text)


if __name__ == "__main__":
    unittest.main()
