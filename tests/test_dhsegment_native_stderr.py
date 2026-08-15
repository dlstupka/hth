import os
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from hth.geometry import detector_dhsegment_page_mask as detector


class DhSegmentNativeStderrTests(unittest.TestCase):
    def test_native_stderr_context_restores_fd2(self):
        with detector._suppress_native_stderr_during_tensorflow_startup():
            os.write(2, b"hidden startup noise\n")
        os.write(2, b"")

    def test_python_stderr_remains_visible_after_startup(self):
        buf = StringIO()
        with redirect_stderr(buf):
            print("visible runtime error", file=sys.stderr)
        self.assertIn("visible runtime error", buf.getvalue())

    def test_inference_session_run_remains_outside_native_stderr_guard(self):
        text = Path("hth/geometry/detector_dhsegment_page_mask.py").read_text(encoding="utf-8")
        guard = text.index("with _suppress_native_stderr_during_tensorflow_startup():")
        predict = text.index("    def predict(", guard)
        session_run = text.index("outputs = self.session.run(", predict)
        self.assertGreater(session_run, predict)


if __name__ == "__main__":
    unittest.main()
