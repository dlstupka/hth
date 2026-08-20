import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hth.geometry import detector_doc_ufcn_page_mask as detector
from hth.geometry.registry import detector_names
from hth.regression.runner import PRECOMPUTED_EVIDENCE_LOADERS, PRECOMPUTED_EVIDENCE_PREPARERS

ROOT = Path(__file__).resolve().parents[1]


class DocUFCNPageMaskIntegrationTests(unittest.TestCase):
    def test_registry_config_lifecycle_and_workflows(self):
        self.assertIn("doc_ufcn_page_mask", detector_names())
        config = json.loads((ROOT / "config/detectors/doc_ufcn_page_mask.json").read_text(encoding="utf-8"))
        self.assertEqual(config["detector"], "doc_ufcn_page_mask")
        self.assertEqual(config["lifecycle"]["prepare"], "doc_ufcn_page_mask")
        self.assertEqual(len(config["parameters"]["minimum_confidence"]["values"]), 8)
        self.assertIn("doc_ufcn_page_mask", PRECOMPUTED_EVIDENCE_PREPARERS)
        self.assertIn("doc_ufcn_page_mask", PRECOMPUTED_EVIDENCE_LOADERS)
        for workflow in ("regress-detector.yml", "execution-optimizer.yml"):
            text = (ROOT / ".github/workflows" / workflow).read_text(encoding="utf-8")
            self.assertIn("          - doc_ufcn_page_mask\n", text)
            self.assertIn("HTH_NEED_DOC_UFCN", text)

    def test_parameter_count_is_2000(self):
        from hth.regression.strategies.cartesian import generate
        config = json.loads((ROOT / "config/detectors/doc_ufcn_page_mask.json").read_text(encoding="utf-8"))
        self.assertEqual(len(generate(config)), 8 * 5 * 5 * 10)

    def test_detect_selects_largest_qualifying_page_polygon(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        evidence = (
            {"confidence": 0.9, "polygon": [[10, 10], [10, 90], [110, 90], [110, 10]], "area": 8000.0},
            {"confidence": 0.99, "polygon": [[1, 1], [1, 5], [5, 5], [5, 1]], "area": 16.0},
        )
        provenance = {"model_id": "doc-ufcn-generic-page", "model_sha256": "abc"}
        with patch.object(detector, "_infer_evidence", return_value=evidence), patch.object(detector, "_provenance", return_value=provenance):
            candidate = detector.detect(
                image_bgr=image,
                mask=None,
                parameters={
                    "minimum_confidence": 0.5,
                    "minimum_component_area_fraction": 0.001,
                    "minimum_page_area_fraction": 0.1,
                    "page_padding_fraction": 0.0,
                },
            )
        self.assertEqual(candidate.method, "doc_ufcn_page_mask")
        self.assertIsNotNone(candidate.corners)
        self.assertGreater(candidate.score, 0.5)


    def test_multi_component_spread_envelope_joins_facing_page_leaves(self):
        image = np.zeros((1000, 1400, 3), dtype=np.uint8)
        evidence = (
            {"confidence": 0.95, "polygon": [[80, 80], [80, 930], [650, 930], [650, 80]], "area": 484500.0},
            {"confidence": 0.93, "polygon": [[720, 90], [720, 925], [1320, 925], [1320, 90]], "area": 501000.0},
        )
        values = detector._parameters({
            "minimum_confidence": 0.5,
            "minimum_component_area_fraction": 0.0005,
            "minimum_page_area_fraction": 0.12,
            "page_padding_fraction": 0.0,
        })
        with patch.object(detector, "_infer_evidence", return_value=evidence):
            selected, diagnostics = detector._select_page_envelope(image, values)
        self.assertIsNotNone(selected)
        self.assertEqual(diagnostics["decision"], "multi-component-spread-envelope")
        self.assertEqual(diagnostics["joined_component_count"], 2)
        polygon = selected[2]
        xs = np.asarray(polygon)[:, 0]
        self.assertLess(float(xs.min()), 100.0)
        self.assertGreater(float(xs.max()), 1300.0)

    def test_multi_component_spread_envelope_rejects_unrelated_local_component(self):
        image = np.zeros((1000, 1400, 3), dtype=np.uint8)
        evidence = (
            {"confidence": 0.95, "polygon": [[80, 80], [80, 930], [900, 930], [900, 80]], "area": 697000.0},
            {"confidence": 0.99, "polygon": [[1100, 50], [1100, 160], [1300, 160], [1300, 50]], "area": 22000.0},
        )
        values = detector._parameters({
            "minimum_confidence": 0.5,
            "minimum_component_area_fraction": 0.0005,
            "minimum_page_area_fraction": 0.12,
            "page_padding_fraction": 0.0,
        })
        with patch.object(detector, "_infer_evidence", return_value=evidence):
            selected, diagnostics = detector._select_page_envelope(image, values)
        self.assertIsNotNone(selected)
        self.assertEqual(diagnostics["decision"], "single-component")
        self.assertEqual(diagnostics["joined_component_count"], 0)


    def test_single_leaf_spread_completion_recovers_image_proven_missing_side(self):
        import cv2
        image = np.zeros((1000, 1400, 3), dtype=np.uint8)
        # A strong left leaf spans the physical height; the far right page edge
        # is independently visible in the source image.
        cv2.line(image, (1230, 70), (1230, 940), (255, 255, 255), 3)
        evidence = (
            {"confidence": 0.96, "polygon": [[90, 70], [90, 940], [760, 940], [760, 70]], "area": 582900.0},
        )
        values = detector._parameters({
            "minimum_confidence": 0.5,
            "minimum_component_area_fraction": 0.0005,
            "minimum_page_area_fraction": 0.12,
            "page_padding_fraction": 0.0,
        })
        with patch.object(detector, "_infer_evidence", return_value=evidence):
            selected, diagnostics = detector._select_page_envelope(image, values)
        self.assertIsNotNone(selected)
        self.assertEqual(diagnostics["decision"], "image-supported-single-leaf-spread-completion")
        completion = diagnostics["single_leaf_spread_completion"]
        self.assertEqual(completion["missing_side"], "right")
        self.assertGreater(completion["boundary_x"], 1200.0)
        xs = np.asarray(selected[2])[:, 0]
        self.assertGreater(float(xs.max()), 1200.0)

    def test_single_leaf_spread_completion_rejects_missing_side_without_image_edge(self):
        image = np.zeros((1000, 1400, 3), dtype=np.uint8)
        evidence = (
            {"confidence": 0.96, "polygon": [[90, 70], [90, 940], [760, 940], [760, 70]], "area": 582900.0},
        )
        values = detector._parameters({
            "minimum_confidence": 0.5,
            "minimum_component_area_fraction": 0.0005,
            "minimum_page_area_fraction": 0.12,
            "page_padding_fraction": 0.0,
        })
        with patch.object(detector, "_infer_evidence", return_value=evidence):
            selected, diagnostics = detector._select_page_envelope(image, values)
        self.assertIsNotNone(selected)
        self.assertEqual(diagnostics["decision"], "single-component")
        self.assertEqual(
            diagnostics["single_leaf_spread_completion"]["decision"],
            "image-boundary-not-proven",
        )

    def test_shared_evidence_round_trip(self):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        key = detector._image_key(image)
        evidence = ({"confidence": 0.8, "polygon": [[1, 1], [1, 30], [46, 30], [46, 1]], "area": 1305.0},)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, patch.object(detector, "precompute_golden_set_evidence", return_value=(key,)):
            with detector._EVIDENCE_CACHE_LOCK:
                detector._EVIDENCE_CACHE[key] = evidence
            detector.export_precomputed_golden_set_evidence([image], tmp)
            with detector._EVIDENCE_CACHE_LOCK:
                detector._EVIDENCE_CACHE.clear()
            detector.load_precomputed_golden_set_evidence(tmp, [image])
            with detector._EVIDENCE_CACHE_LOCK:
                loaded = detector._EVIDENCE_CACHE[key]
            self.assertEqual(loaded[0]["confidence"], 0.8)

    def test_boundary_supported_padding_restores_baseline_when_outer_page_edge_is_stronger(self):
        import cv2
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(image, (28, 28), (72, 72), (255, 255, 255), 2)
        raw = np.asarray([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.float32)
        corners, diagnostics = detector._boundary_supported_padding(
            image, raw, requested_fraction=0.01
        )
        self.assertEqual(diagnostics["decision"], "baseline-padding-boundary-supported")
        self.assertLess(float(corners[:, 0].min()), 29.0)
        self.assertGreater(float(corners[:, 0].max()), 71.0)

    def test_boundary_supported_padding_preserves_requested_without_outer_edge_evidence(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        raw = np.asarray([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.float32)
        corners, diagnostics = detector._boundary_supported_padding(
            image, raw, requested_fraction=0.01
        )
        self.assertEqual(diagnostics["decision"], "requested-padding")
        expected = detector._pad_corners(raw, width=100, height=100, fraction=0.01)
        self.assertTrue(np.allclose(corners, expected))

    def test_single_leaf_spread_completion_prefers_outermost_robust_page_edge(self):
        import cv2
        image = np.zeros((1000, 1400, 3), dtype=np.uint8)
        # A strong interior rule precedes the actual outer paper edge.  Both are
        # robust, but damaged-spread completion must prefer the farther physical
        # boundary instead of stopping at the strongest interior edge.
        cv2.line(image, (990, 70), (990, 940), (255, 255, 255), 5)
        cv2.line(image, (1230, 70), (1230, 940), (220, 220, 220), 3)
        evidence = (
            {"confidence": 0.96, "polygon": [[90, 70], [90, 940], [760, 940], [760, 70]], "area": 582900.0},
        )
        values = detector._parameters({
            "minimum_confidence": 0.5,
            "minimum_component_area_fraction": 0.0005,
            "minimum_page_area_fraction": 0.12,
            "page_padding_fraction": 0.0,
        })
        with patch.object(detector, "_infer_evidence", return_value=evidence):
            selected, diagnostics = detector._select_page_envelope(image, values)
        self.assertIsNotNone(selected)
        completion = diagnostics["single_leaf_spread_completion"]
        self.assertEqual(completion["boundary_selection"], "outermost-robust-boundary")
        self.assertGreater(completion["robust_boundary_candidates"], 1)
        self.assertGreater(completion["boundary_x"], 1200.0)

    def test_single_leaf_spread_completion_uses_farther_outer_background_proof(self):
        import cv2
        image = np.zeros((1000, 1400, 3), dtype=np.uint8)
        # Strong interior fold makes Sobel prefer an early boundary; the secondary
        # sustained-background proof independently locates the farther sheet edge.
        cv2.line(image, (990, 70), (990, 940), (255, 255, 255), 5)
        evidence = (
            {"confidence": 0.96, "polygon": [[90, 70], [90, 940], [760, 940], [760, 70]], "area": 582900.0},
        )
        values = detector._parameters({
            "minimum_confidence": 0.5,
            "minimum_component_area_fraction": 0.0005,
            "minimum_page_area_fraction": 0.12,
            "page_padding_fraction": 0.0,
        })
        proof = (1232.0, {"accepted": True, "reason": "sustained-outer-background-transition", "boundary_x": 1232.0})
        with patch.object(detector, "_infer_evidence", return_value=evidence), patch.object(detector, "_outer_background_boundary", return_value=proof):
            selected, diagnostics = detector._select_page_envelope(image, values)
        self.assertIsNotNone(selected)
        completion = diagnostics["single_leaf_spread_completion"]
        self.assertEqual(completion["boundary_selection"], "outer-background-transition")
        self.assertTrue(completion["outer_background_boundary"]["accepted"])
        self.assertGreater(completion["boundary_x"], 1200.0)


if __name__ == "__main__":
    unittest.main()
