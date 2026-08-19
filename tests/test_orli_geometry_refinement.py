import unittest

import cv2

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

    def test_directional_completion_extends_missing_bottom_from_top_and_side_anchors(self):
        corners = np.asarray([[90, 60], [1040, 60], [1040, 470], [90, 470]], dtype=np.float32)
        completed, diagnostics = orli._directional_page_completion(
            corners, image_shape=(1000, 1100, 3)
        )
        self.assertIn(diagnostics["reason"], {"directional-edge-anchor-completion", "corner-anchored-two-axis-completion"})
        bounds = orli._quad_axis_bounds(completed)
        self.assertLess(bounds["top"], 80.0)
        self.assertGreater(bounds["bottom"], 900.0)
        self.assertGreaterEqual(diagnostics["anchor_count"], 2)

    def test_directional_completion_recovers_opposite_sides_from_top_right_fragment(self):
        corners = np.asarray([[600, 60], [1070, 60], [1070, 250], [600, 250]], dtype=np.float32)
        completed, diagnostics = orli._directional_page_completion(
            corners, image_shape=(1000, 1200, 3)
        )
        bounds = orli._quad_axis_bounds(completed)
        self.assertIn(diagnostics["reason"], {"directional-edge-anchor-completion", "corner-anchored-two-axis-completion"})
        self.assertLess(bounds["left"], 180.0)
        self.assertGreater(bounds["bottom"], 900.0)
        self.assertGreaterEqual(len(diagnostics["changed_axes"]), 2)

    def test_directional_completion_recovers_physical_top_right_corner_when_rotated_basis_is_ambiguous(self):
        # Regression shape modeled on Golden Set page 10: a compact upper-right
        # Orli fragment with trustworthy top/right image anchors and both
        # opposite page dimensions absent.
        corners = np.asarray([[492, 31], [1178, 31], [1178, 282], [492, 282]], dtype=np.float32)
        completed, diagnostics = orli._directional_page_completion(
            corners, image_shape=(1000, 1200, 3)
        )
        bounds = orli._quad_axis_bounds(completed)
        self.assertEqual(diagnostics["reason"], "corner-anchored-two-axis-completion")
        self.assertEqual(diagnostics["corner_anchor"], "top-right")
        self.assertLess(bounds["left"], 80.0)
        self.assertGreater(bounds["bottom"], 930.0)
        self.assertEqual(set(diagnostics["changed_axes"]), {"left", "bottom"})

    def test_directional_completion_physical_corner_bypasses_rotated_anchor_count(self):
        # A rotated upper-right fragment can expose only one anchor in the
        # proposal basis even though its physical bounds clearly touch the top
        # and right source-image sides.  Physical-corner completion must be
        # evaluated before the rotated-basis anchor-count rejection.
        rect = ((850.0, 180.0), (650.0, 220.0), -20.0)
        corners = orli._canonical_quad(
            cv2.boxPoints(rect), width=1200, height=1000
        )
        completed, diagnostics = orli._directional_page_completion(
            corners, image_shape=(1000, 1200, 3)
        )
        bounds = orli._quad_axis_bounds(completed)
        self.assertEqual(diagnostics["anchor_count"], 1)
        self.assertEqual(diagnostics["reason"], "corner-anchored-two-axis-completion")
        self.assertEqual(diagnostics["corner_anchor"], "top-right")
        self.assertLess(bounds["left"], 80.0)
        self.assertGreater(bounds["bottom"], 930.0)
        self.assertEqual(set(diagnostics["changed_axes"]), {"left", "bottom"})

    def test_directional_completion_rejects_unanchored_local_text_block(self):
        corners = np.asarray([[380, 250], [700, 250], [700, 520], [380, 520]], dtype=np.float32)
        completed, diagnostics = orli._directional_page_completion(
            corners, image_shape=(1000, 1100, 3)
        )
        self.assertTrue(np.array_equal(completed, corners))
        self.assertEqual(diagnostics["reason"], "insufficient-edge-anchors")
        self.assertEqual(diagnostics["changed_axes"], [])

    def test_directional_completion_leaves_broad_page_proposal_unchanged(self):
        corners = np.asarray([[80, 70], [1020, 70], [1020, 930], [80, 930]], dtype=np.float32)
        completed, diagnostics = orli._directional_page_completion(
            corners, image_shape=(1000, 1100, 3)
        )
        self.assertTrue(np.array_equal(completed, corners))
        self.assertEqual(diagnostics["reason"], "anchored-envelope-not-materially-truncated")

    def test_image_supported_boundary_recovery_extends_localized_fragment_to_proved_page_edges(self):
        image = np.full((1000, 1200, 3), 24, dtype=np.uint8)
        cv2.rectangle(image, (90, 40), (1150, 960), (220, 220, 220), thickness=-1)
        # Local upper-right Orli fragment: top/right are near the physical page,
        # while left and bottom are badly truncated.
        seed = np.asarray([[500, 55], [1135, 55], [1135, 285], [500, 285]], dtype=np.float32)
        recovered, diagnostics = orli._image_supported_boundary_recovery(image, seed)
        bounds = orli._quad_axis_bounds(recovered)
        self.assertEqual(diagnostics["reason"], "image-supported-boundary-recovery")
        self.assertEqual(set(diagnostics["recovered_sides"]), {"left", "bottom"})
        self.assertLess(abs(bounds["left"] - 90.0), 20.0)
        self.assertLess(abs(bounds["bottom"] - 960.0), 20.0)
        self.assertLess(bounds["top"], 80.0)
        self.assertGreater(bounds["right"], 1100.0)

    def test_image_supported_boundary_recovery_rejects_local_fragment_without_page_edges(self):
        image = np.full((1000, 1200, 3), 128, dtype=np.uint8)
        seed = np.asarray([[500, 55], [1135, 55], [1135, 285], [500, 285]], dtype=np.float32)
        recovered, diagnostics = orli._image_supported_boundary_recovery(image, seed)
        self.assertTrue(np.array_equal(recovered, seed))
        self.assertEqual(diagnostics["reason"], "insufficient-image-boundary-evidence")
        self.assertEqual(diagnostics["recovered_sides"], [])

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
