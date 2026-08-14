import json
import unittest
from pathlib import Path

import numpy as np

from hth.geometry import detector_msre_bfq_spbv_pbg
from hth.regression.parameter_space import parameter_set_id


class FusionGen1Tests(unittest.TestCase):
    def test_child_anchor_parameter_ids_are_stable(self) -> None:
        for method, item in detector_msre_bfq_spbv_pbg.CHILD_CALIBRATIONS.items():
            with self.subTest(method=method):
                self.assertEqual(parameter_set_id(item["parameters"]), item["parameter_set_id"])

    def test_config_child_calibrations_match_detector_contract(self) -> None:
        payload = json.loads(Path("config/detectors/msre_bfq_spbv_pbg.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["child_calibrations"], detector_msre_bfq_spbv_pbg.CHILD_CALIBRATIONS)

    def test_initial_search_space_is_2187_sets(self) -> None:
        payload = json.loads(Path("config/detectors/msre_bfq_spbv_pbg.json").read_text(encoding="utf-8"))
        size = 1
        for spec in payload["parameters"].values():
            size *= len(spec["values"])
        self.assertEqual(size, 2187)

    def test_detector_contract_on_simple_page(self) -> None:
        image = np.full((360, 280, 3), 25, dtype=np.uint8)
        image[35:325, 30:250] = 225
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        candidate = detector_msre_bfq_spbv_pbg.detect(image_bgr=image, mask=mask)
        self.assertIn(candidate.status, {"ok", "no_candidate"})
        if candidate.status == "ok":
            self.assertEqual(candidate.method, "msre_bfq_spbv_pbg")
            self.assertEqual(len(candidate.corners), 4)
            self.assertIn("child_calibrations", candidate.diagnostics)


if __name__ == "__main__":
    unittest.main()
