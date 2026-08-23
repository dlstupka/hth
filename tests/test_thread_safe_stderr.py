import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

from hth.thread_safe_stderr import capture_native_stderr, suppress_native_stderr

ROOT = Path(__file__).resolve().parents[1]


class ThreadSafeStderrTests(unittest.TestCase):
    def test_all_production_fd2_redirection_is_centralized(self):
        offenders = []
        shared = ROOT / "hth" / "thread_safe_stderr.py"
        for path in (ROOT / "hth").rglob("*.py"):
            if path == shared:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "os.dup2(" in text or "contextlib.redirect_stderr" in text or "redirect_stderr(" in text:
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_kraken_and_dhsegment_use_same_shared_wrapper(self):
        kraken = (ROOT / "hth/geometry/detector_kraken_page_mask.py").read_text(encoding="utf-8")
        dhsegment = (ROOT / "hth/geometry/detector_dhsegment_page_mask.py").read_text(encoding="utf-8")
        self.assertIn("from hth.thread_safe_stderr import capture_native_stderr", kraken)
        self.assertIn("with capture_native_stderr() as captured:", kraken)
        self.assertIn("from hth.thread_safe_stderr import suppress_native_stderr", dhsegment)
        self.assertIn("with suppress_native_stderr():", dhsegment)

    def test_geometry_candidate_cli_imports_in_direct_script_mode(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "hth/detect_geometry_candidates.py"), "--help"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--manifest", result.stdout)

    def test_capture_contexts_serialize_process_global_fd2(self):
        entered = []
        first_inside = threading.Event()
        release_first = threading.Event()

        def first():
            with capture_native_stderr():
                entered.append("first")
                first_inside.set()
                release_first.wait(timeout=2)

        def second():
            first_inside.wait(timeout=2)
            with suppress_native_stderr():
                entered.append("second")

        a = threading.Thread(target=first)
        b = threading.Thread(target=second)
        a.start()
        b.start()
        first_inside.wait(timeout=2)
        time.sleep(0.05)
        self.assertEqual(entered, ["first"])
        release_first.set()
        a.join(timeout=2)
        b.join(timeout=2)
        self.assertEqual(entered, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
