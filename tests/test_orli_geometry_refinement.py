import unittest

import numpy as np

from hth.geometry import detector_orli_page_mask as orli


class OrliGeometryRefinementTests(unittest.TestCase):
    def test_learned_geometry_envelope_spans_disconnected_baseline_columns(self):
        evidence = {
            "regions": (),
            "lines": (),
            "baselines": (
                ((120, 150), (420, 155)),
                ((125, 300), (425, 305)),
                ((620, 160), (920, 165)),
                ((615, 310), (915, 315)),
                ((130, 700), (430, 705)),
                ((610, 710), (910, 715)),
            ),
            "text_direction": "horizontal-lr",
        }
        corners, diagnostics = orli._learned_geometry_envelope(
            evidence,
            image_shape=(1000, 1100, 3),
        )
        self.assertIsNotNone(corners)
        self.assertTrue(diagnostics["available"])
        xs = np.asarray(corners)[:, 0]
        ys = np.asarray(corners)[:, 1]
        self.assertLess(float(xs.min()), 150.0)
        self.assertGreater(float(xs.max()), 880.0)
        self.assertLess(float(ys.min()), 190.0)
        self.assertGreater(float(ys.max()), 680.0)

    def test_learned_geometry_envelope_rejects_tiny_baseline_fragment(self):
        evidence = {
            "regions": (),
            "lines": (),
            "baselines": (
                ((100, 100), (500, 100)),
                ((100, 300), (500, 300)),
                ((100, 500), (500, 500)),
                ((990, 990), (991, 990)),
            ),
            "text_direction": "horizontal-lr",
        }
        corners, diagnostics = orli._learned_geometry_envelope(
            evidence,
            image_shape=(1000, 1000, 3),
        )
        self.assertIsNotNone(corners)
        self.assertGreater(diagnostics["baseline_length_floor"], 1.0)
        self.assertLess(float(np.asarray(corners)[:, 0].max()), 600.0)

    def test_proposal_prefers_larger_learned_geometry_over_connected_component(self):
        image = np.zeros((800, 1200, 3), dtype=np.uint8)
        evidence = orli._freeze_evidence({
            "regions": [],
            "lines": [],
            "baselines": [
                [(120, 100), (500, 100)],
                [(120, 240), (500, 240)],
                [(700, 110), (1080, 110)],
                [(700, 250), (1080, 250)],
                [(130, 650), (490, 650)],
                [(710, 660), (1070, 660)],
            ],
            "text_direction": "horizontal-lr",
        })
        original = orli._infer_evidence
        try:
            orli._infer_evidence = lambda _image: evidence
            values = dict(orli.BASELINE_PARAMETERS)
            values.update({
                "dilation_fraction": 0.0,
                "close_kernel_fraction": 0.0,
                "fill_holes": 0,
                "page_padding_fraction": 0.0,
            })
            _, _, _, corners, _, diagnostics = orli._proposal(image, values)
        finally:
            orli._infer_evidence = original

        self.assertIsNotNone(corners)
        xs = np.asarray(corners)[:, 0]
        ys = np.asarray(corners)[:, 1]
        self.assertLess(float(xs.min()), 150.0)
        self.assertGreater(float(xs.max()), 1050.0)
        self.assertGreater(float(ys.max()), 620.0)
        self.assertIn(diagnostics["mode"], {"learned-geometry-consensus", "multi-region-envelope"})


if __name__ == "__main__":
    unittest.main()
