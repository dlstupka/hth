import unittest
from pathlib import Path


class DhSegmentTensorFlowImportOrderTests(unittest.TestCase):
    def test_runtime_environment_is_configured_before_tensorflow_import(self):
        text = Path("hth/geometry/detector_dhsegment_page_mask.py").read_text(encoding="utf-8")
        configure = text.index("_configure_tensorflow_runtime_environment()\n        try:\n            import tensorflow as tf")
        self.assertGreaterEqual(configure, 0)


if __name__ == "__main__":
    unittest.main()
