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

    def test_arbitration_keeps_taller_learned_geometry_even_when_contour_area_is_larger(self):
        contour = np.asarray([[50, 200], [1050, 200], [1050, 600], [50, 600]], dtype=np.float32)
        learned = np.asarray([[250, 80], [850, 80], [850, 720], [250, 720]], dtype=np.float32)
        chosen, source, diagnostics = orli._arbitrate_envelopes(
            contour, learned, image_shape=(800, 1100, 3)
        )
        self.assertEqual(source, "learned")
        self.assertEqual(diagnostics["reason"], "material-vertical-extent")
        self.assertTrue(np.array_equal(chosen, learned))
        self.assertLess(diagnostics["learned_area"], diagnostics["contour_area"])

    def test_arbitration_rejects_axis_stretch_that_loses_too_much_orthogonal_span(self):
        contour = np.asarray([[50, 200], [1050, 200], [1050, 600], [50, 600]], dtype=np.float32)
        learned = np.asarray([[480, 40], [620, 40], [620, 760], [480, 760]], dtype=np.float32)
        chosen, source, diagnostics = orli._arbitrate_envelopes(
            contour, learned, image_shape=(800, 1100, 3)
        )
        self.assertEqual(source, "contour")
        self.assertEqual(diagnostics["reason"], "contour-retains-document-extent")
        self.assertTrue(np.array_equal(chosen, contour))


    def test_learned_document_frame_extrapolates_cross_axis_from_broad_baselines(self):
        evidence = {
            "regions": (),
            "lines": (),
            "baselines": (
                ((110, 180), (990, 185)),
                ((115, 260), (985, 265)),
                ((120, 340), (980, 345)),
                ((125, 420), (975, 425)),
                ((130, 500), (970, 505)),
            ),
            "text_direction": "horizontal-lr",
        }
        corners, diagnostics = orli._learned_document_frame(
            evidence,
            image_shape=(1000, 1100, 3),
        )
        self.assertIsNotNone(corners)
        self.assertTrue(diagnostics["available"])
        self.assertGreater(diagnostics["main_axis_coverage"], 0.55)
        ys = np.asarray(corners)[:, 1]
        self.assertLess(float(ys.min()), 140.0)
        self.assertGreater(float(ys.max()), 850.0)
        self.assertGreater(diagnostics["inferred_cross_margin"], 20.0)

    def test_learned_document_frame_rejects_localized_text_block(self):
        evidence = {
            "regions": (),
            "lines": (),
            "baselines": (
                ((400, 200), (620, 200)),
                ((405, 300), (625, 300)),
                ((410, 400), (630, 400)),
                ((415, 500), (635, 500)),
            ),
            "text_direction": "horizontal-lr",
        }
        corners, diagnostics = orli._learned_document_frame(
            evidence,
            image_shape=(1000, 1100, 3),
        )
        self.assertIsNone(corners)
        self.assertEqual(diagnostics["reason"], "insufficient-main-axis-coverage")

    def test_arbitration_prefers_extrapolated_document_frame_over_short_contour(self):
        contour = np.asarray([[70, 180], [1030, 180], [1030, 560], [70, 560]], dtype=np.float32)
        learned = np.asarray([[120, 190], [980, 190], [980, 520], [120, 520]], dtype=np.float32)
        frame = np.asarray([[120, 80], [980, 80], [980, 920], [120, 920]], dtype=np.float32)
        chosen, source, diagnostics = orli._arbitrate_envelopes(
            contour, learned, image_shape=(1000, 1100, 3), frame_corners=frame
        )
        self.assertEqual(source, "frame")
        self.assertEqual(diagnostics["reason"], "extrapolated-vertical-document-extent")
        self.assertTrue(np.array_equal(chosen, frame))

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
