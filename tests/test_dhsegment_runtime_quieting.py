import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from hth import detector_lifecycle as lifecycle


class DhSegmentRuntimeQuietingTests(unittest.TestCase):
    def test_prepare_exports_quiet_cpu_tensorflow_environment(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            def fake_download(url, target):
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(target, "w") as zf:
                    zf.writestr("export/123/saved_model.pb", b"saved")

            old_env = {
                key: os.environ.get(key)
                for key in (
                    "TF_CPP_MIN_LOG_LEVEL",
                    "ABSL_MIN_LOG_LEVEL",
                    "GLOG_minloglevel",
                    "CUDA_VISIBLE_DEVICES",
                )
            }
            try:
                with patch.object(lifecycle, "_download", side_effect=fake_download), \
                     patch.object(lifecycle.importlib.util, "find_spec", return_value=object()):
                    lifecycle._prepare_dhsegment_page_mask_hook(
                        results_root=root,
                        policy="reuse",
                        env_file=None,
                    )
                self.assertEqual(os.environ["TF_CPP_MIN_LOG_LEVEL"], "3")
                self.assertEqual(os.environ["ABSL_MIN_LOG_LEVEL"], "3")
                self.assertEqual(os.environ["GLOG_minloglevel"], "3")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_regression_and_optimizer_install_cpu_only_tensorflow(self):
        manager = Path("tools/ensure-managed-runtime.sh").read_text(encoding="utf-8")
        self.assertIn('tensorflow-cpu>=2.18,<2.21', manager)
        self.assertNotIn('pip install "tensorflow>=2.18,<2.21"', manager)
        for rel in (
            ".github/workflows/regress-detector.yml",
            ".github/workflows/execution-optimizer.yml",
        ):
            text = Path(rel).read_text(encoding="utf-8")
            self.assertIn("hth-pipeline/tools/ensure-managed-runtime.sh", text, rel)

    def test_legacy_loader_suppresses_python_warning_chatter_locally(self):
        text = Path("hth/geometry/detector_dhsegment_page_mask.py").read_text(encoding="utf-8")
        self.assertIn("tf_logger = tf.get_logger()", text)
        self.assertIn("tf_logger.setLevel(logging.ERROR)", text)
        self.assertIn("tf.compat.v1.saved_model.loader.load(", text)
        self.assertIn("tf_logger.setLevel(previous_level)", text)


if __name__ == "__main__":
    unittest.main()
