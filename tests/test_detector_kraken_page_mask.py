import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from hth.geometry import detector_kraken_page_mask as detector


class KrakenPageMaskTests(unittest.TestCase):
    def setUp(self):
        detector._EVIDENCE_CACHE.clear()
        detector._RUNTIME_DIAGNOSTICS.clear()

    def test_parameter_grid_is_exactly_10000_sets(self):
        import json
        payload=json.loads(Path("config/detectors/kraken_page_mask.json").read_text(encoding="utf-8"))
        count=1
        for spec in payload["parameters"].values():
            count *= len(spec["values"])
        self.assertEqual(count, 10000)

    def test_extracts_regions_lines_and_baselines(self):
        segmentation=SimpleNamespace(
            text_direction="horizontal-lr",
            regions={"text":[SimpleNamespace(boundary=[(10,10),(90,10),(90,90),(10,90),(10,10)])]},
            lines=[SimpleNamespace(
                boundary=[(15,20),(85,20),(85,30),(15,30),(15,20)],
                baseline=[(15,28),(85,28)],
            )],
        )
        evidence=detector._extract_evidence(segmentation)
        self.assertEqual(len(evidence["regions"]),1)
        self.assertEqual(len(evidence["lines"]),1)
        self.assertEqual(len(evidence["baselines"]),1)

    def test_detect_converts_kraken_layout_to_page_quad(self):
        image=np.zeros((200,200,3),dtype=np.uint8)
        evidence={
            "regions":[[(40,40),(160,40),(160,160),(40,160),(40,40)]],
            "lines":[],
            "baselines":[],
            "text_direction":"horizontal-lr",
        }
        with tempfile.TemporaryDirectory() as td:
            prov=Path(td)/"prov.json"
            prov.write_text('{"model_id":"kraken-blla-default-7.0.2"}',encoding="utf-8")
            with patch.dict(os.environ,{detector.PROVENANCE_ENV:str(prov)}), \
                 patch.object(detector,"_infer_evidence",return_value=evidence):
                candidate=detector.detect(
                    image_bgr=image,
                    mask=None,
                    parameters={
                        "include_lines":0,
                        "dilation_fraction":0.0,
                        "close_kernel_fraction":0.0,
                        "page_padding_fraction":0.05,
                        "minimum_page_area_fraction":0.08,
                        "fill_holes":1,
                    },
                )
        self.assertEqual(candidate.status,"ok")
        self.assertEqual(len(candidate.corners),4)
        self.assertGreater(candidate.diagnostics["page_area_fraction"],0.3)

    def test_no_layout_evidence_returns_no_candidate(self):
        image=np.zeros((100,100,3),dtype=np.uint8)
        evidence={"regions":[],"lines":[],"baselines":[],"text_direction":"horizontal-lr"}
        with tempfile.TemporaryDirectory() as td:
            prov=Path(td)/"prov.json"
            prov.write_text('{"model_id":"kraken-blla-default-7.0.2"}',encoding="utf-8")
            with patch.dict(os.environ,{detector.PROVENANCE_ENV:str(prov)}), \
                 patch.object(detector,"_infer_evidence",return_value=evidence):
                candidate=detector.detect(image_bgr=image,mask=None)
        self.assertEqual(candidate.status,"no_candidate")
        self.assertEqual(candidate.diagnostics["reason"],"no_kraken_layout_evidence")

    def test_canonical_quad_repairs_crossed_corner_order(self):
        crossed = np.array(
            [[20, 20], [180, 180], [180, 20], [20, 180]],
            dtype=np.float32,
        )
        quad = detector._canonical_quad(crossed, width=200, height=200)
        self.assertIsNotNone(quad)
        self.assertEqual(quad.shape, (4, 2))
        self.assertTrue(cv2.isContourConvex(quad.astype(np.float32)))
        self.assertGreater(abs(cv2.contourArea(quad.astype(np.float32))), 1000)

    def test_canonical_quad_rejects_degenerate_geometry(self):
        degenerate = np.array(
            [[10, 10], [10, 10], [10, 10], [10, 10]],
            dtype=np.float32,
        )
        self.assertIsNone(detector._canonical_quad(degenerate, width=100, height=100))

    def test_runtime_chatter_filter_retains_diagnostics_but_replays_other_stderr(self):
        import os
        import tempfile

        saved = os.dup(2)
        capture = tempfile.TemporaryFile(mode="w+b")
        try:
            os.dup2(capture.fileno(), 2)
            with detector._capture_kraken_runtime_chatter() as diagnostics:
                os.write(2, b"TopologyException: side location conflict at 1 2\n")
                os.write(2, b"Polygonizer failed on line 0: invalid geometry\n")
                os.write(2, b"real kraken failure remains visible\n")
            capture.seek(0)
            emitted = capture.read().decode("utf-8")
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            capture.close()

        self.assertEqual(diagnostics["kraken_polygonizer_warnings"], 2)
        self.assertIn("real kraken failure remains visible", emitted)
        self.assertNotIn("TopologyException", emitted)
        self.assertNotIn("Polygonizer failed", emitted)

    def test_detect_reports_invalid_page_quad_as_no_candidate(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        evidence = {
            "regions": [[(10, 10), (90, 10), (90, 90), (10, 90), (10, 10)]],
            "lines": [],
            "baselines": [],
            "text_direction": "horizontal-lr",
        }
        with tempfile.TemporaryDirectory() as td:
            prov = Path(td) / "prov.json"
            prov.write_text('{"model_id":"kraken-blla-default-7.0.2"}', encoding="utf-8")
            with patch.dict(os.environ, {detector.PROVENANCE_ENV: str(prov)}), \
                 patch.object(detector, "_proposal", return_value=(evidence, np.zeros((100, 100), dtype=np.uint8), np.array([[[1,1]]], dtype=np.int32), None, 0.0)):
                candidate = detector.detect(image_bgr=image, mask=None)
        self.assertEqual(candidate.status, "no_candidate")
        self.assertEqual(candidate.diagnostics["reason"], "invalid_page_quadrilateral")


if __name__ == "__main__":
    unittest.main()
