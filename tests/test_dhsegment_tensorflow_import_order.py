import unittest
from pathlib import Path


class DhSegmentTensorFlowImportOrderTests(unittest.TestCase):
    def test_runtime_environment_and_native_guard_precede_tensorflow_import(self):
        text = Path("hth/geometry/detector_dhsegment_page_mask.py").read_text(encoding="utf-8")
        configure = text.index("_configure_tensorflow_runtime_environment()")
        guard = text.index("with _suppress_native_stderr_during_tensorflow_startup():", configure)
        tf_import = text.index("import tensorflow as tf", guard)
        self.assertLess(configure, guard)
        self.assertLess(guard, tf_import)


if __name__ == "__main__":
    unittest.main()
