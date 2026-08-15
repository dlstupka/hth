import os
import unittest
from unittest.mock import patch

from hth.geometry import detector_dhsegment_page_mask as detector


class DhSegmentRuntimeEnvironmentTests(unittest.TestCase):
    def test_runtime_environment_is_set_before_tensorflow_import(self):
        keys = (
            "TF_CPP_MIN_LOG_LEVEL",
            "ABSL_MIN_LOG_LEVEL",
            "GLOG_minloglevel",
            "CUDA_VISIBLE_DEVICES",
        )
        with patch.dict(os.environ, {}, clear=False):
            for key in keys:
                os.environ.pop(key, None)
            detector._configure_tensorflow_runtime_environment()
            self.assertEqual(os.environ["TF_CPP_MIN_LOG_LEVEL"], "3")
            self.assertEqual(os.environ["ABSL_MIN_LOG_LEVEL"], "3")
            self.assertEqual(os.environ["GLOG_minloglevel"], "3")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")

    def test_runtime_environment_overrides_noisy_preexisting_values(self):
        with patch.dict(
            os.environ,
            {
                "TF_CPP_MIN_LOG_LEVEL": "0",
                "ABSL_MIN_LOG_LEVEL": "0",
                "GLOG_minloglevel": "0",
                "CUDA_VISIBLE_DEVICES": "0",
            },
            clear=False,
        ):
            detector._configure_tensorflow_runtime_environment()
            self.assertEqual(os.environ["TF_CPP_MIN_LOG_LEVEL"], "3")
            self.assertEqual(os.environ["ABSL_MIN_LOG_LEVEL"], "3")
            self.assertEqual(os.environ["GLOG_minloglevel"], "3")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")


if __name__ == "__main__":
    unittest.main()
