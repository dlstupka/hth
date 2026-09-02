import unittest

from hth.geometry import detector_amsre_bfq_spbv_pbg, detector_msre_bfq_spbv_pbg
from hth.geometry.detector_side_consensus_fusion import SideConsensusFusion


class SideConsensusFusionFamilyTests(unittest.TestCase):
    def test_gen1_and_gen2_are_configurations_of_one_family(self):
        gen1 = detector_msre_bfq_spbv_pbg._FAMILY
        gen2 = detector_amsre_bfq_spbv_pbg._FAMILY
        self.assertIsInstance(gen1, SideConsensusFusion)
        self.assertIsInstance(gen2, SideConsensusFusion)
        self.assertEqual(gen1.method, detector_msre_bfq_spbv_pbg.METHOD)
        self.assertEqual(gen2.method, detector_amsre_bfq_spbv_pbg.METHOD)
        self.assertEqual(gen1.baseline_parameters, detector_msre_bfq_spbv_pbg.BASELINE_PARAMETERS)
        self.assertEqual(gen2.baseline_parameters, detector_amsre_bfq_spbv_pbg.BASELINE_PARAMETERS)
        self.assertEqual(gen1.debug_prefix, "fusion-gen1")
        self.assertEqual(gen2.debug_prefix, "fusion-gen2")

    def test_shared_child_calibrations_remain_identical(self):
        gen1 = detector_msre_bfq_spbv_pbg.CHILD_CALIBRATIONS
        gen2 = detector_amsre_bfq_spbv_pbg.CHILD_CALIBRATIONS
        for detector in ("border_fusion_quad", "signed_polar_boundary_vote", "page_background"):
            self.assertEqual(gen1[detector], gen2[detector])
        self.assertIn("multi_scale_radial_edge", gen1)
        self.assertIn("adaptive_multi_scale_radial_edge", gen2)


if __name__ == "__main__":
    unittest.main()
