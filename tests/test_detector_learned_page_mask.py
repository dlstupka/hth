import json, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import cv2, numpy as np
from hth.geometry import detector_learned_page_mask as detector
from hth.regression.parameter_space import exhaustive_parameter_sets

class FakeNet:
    def setInput(self,blob): self.blob=blob
    def forward(self,name):
        out=np.zeros((1,1,256,256),np.float32); out[:,:,40:220,35:225]=0.95; return out

class LearnedPageMaskTests(unittest.TestCase):
    def test_generation_4_calibration_space_has_50000_sets_and_retains_baseline(self):
        config_path=Path(__file__).resolve().parents[1]/"config"/"detectors"/"learned_page_mask.json"
        config=json.loads(config_path.read_text(encoding="utf-8"))
        parameter_sets=exhaustive_parameter_sets(config)
        self.assertEqual(len(parameter_sets),50000)
        baseline=config["profiles"]["baseline"]
        self.assertIn(baseline,parameter_sets)

        thresholds=config["parameters"]["mask_threshold"]["values"]
        self.assertEqual(len(thresholds),25)
        self.assertEqual(thresholds[0],0.204)
        self.assertEqual(thresholds[-2],0.25)
        self.assertEqual(thresholds[-1],0.5)
        self.assertIn(0.24,thresholds)
        self.assertIn(0.22,thresholds)

        self.assertEqual(config["parameters"]["minimum_mask_area_fraction"]["values"],[0.04,0.15])
        self.assertEqual(config["parameters"]["close_kernel_fraction"]["values"],[0.0,0.006])

        epsilons=config["parameters"]["polygon_epsilon_fraction"]["values"]
        self.assertEqual(len(epsilons),20)
        self.assertEqual(epsilons[0],0.012)
        self.assertEqual(epsilons[1],0.014)
        self.assertEqual(epsilons[-1],0.023)
        self.assertIn(0.018,epsilons)

        paddings=config["parameters"]["bbox_padding_fraction"]["values"]
        self.assertEqual(len(paddings),25)
        self.assertEqual(paddings[0],0.0)
        self.assertEqual(paddings[1],0.02)
        self.assertEqual(paddings[-1],0.043)
        self.assertIn(0.03,paddings)

    def test_detector_uses_lifecycle_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); proto=root/"d.prototxt"; weights=root/"w.caffemodel"; prov=root/"p.json"
            proto.write_text('input: "data"\n',encoding="utf-8"); weights.write_bytes(b"w")
            prov.write_text(json.dumps({"model_id":"pagenet-ohio","weights_sha256":"abc","license":"BSD-3-Clause","upstream_repository":"https://github.com/ctensmeyer/pagenet"}),encoding="utf-8")
            env={detector.PROTOTXT_ENV:str(proto),detector.WEIGHTS_ENV:str(weights),detector.PROVENANCE_ENV:str(prov),detector.OUTPUT_LAYER_ENV:"baselines_7_prob_0"}
            with patch.dict(os.environ,env,clear=False), patch.object(detector,"_network",return_value=FakeNet()):
                detector._THREAD_LOCAL.key=None
                c=detector.detect(image_bgr=np.full((300,500,3),255,np.uint8),mask=np.zeros((300,500),np.uint8))
            self.assertEqual(c.status,"ok"); self.assertEqual(c.diagnostics["model_id"],"pagenet-ohio")
    def test_network_uses_generic_caffe_loader(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); proto=root/"d.prototxt"; weights=root/"w.caffemodel"
            proto.write_text('input: "data"\\n',encoding="utf-8"); weights.write_bytes(b"w")
            fake=FakeNet()
            detector._THREAD_LOCAL.key=None
            with patch.object(cv2.dnn,"readNet",return_value=fake) as loader:
                resolved=detector._network(proto,weights)
            self.assertIs(resolved,fake)
            loader.assert_called_once_with(str(weights),str(proto),"Caffe")

    def test_failed_network_load_does_not_poison_thread_local_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); proto=root/"d.prototxt"; weights=root/"w.caffemodel"
            proto.write_text('input: "data"\n',encoding="utf-8"); weights.write_bytes(b"w")
            detector._THREAD_LOCAL.key=None
            if hasattr(detector._THREAD_LOCAL,"net"): del detector._THREAD_LOCAL.net
            fake=FakeNet()
            with patch.object(cv2.dnn,"readNet",side_effect=[RuntimeError("backend unavailable"),fake]) as loader:
                with self.assertRaisesRegex(RuntimeError,"backend unavailable"):
                    detector._network(proto,weights)
                resolved=detector._network(proto,weights)
            self.assertIs(resolved,fake)
            self.assertEqual(loader.call_count,2)
    def test_missing_assets_raise_configuration_error(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,"lifecycle did not set"):
                detector.detect(image_bgr=np.zeros((50,50,3),np.uint8),mask=np.zeros((50,50),np.uint8))
    def test_detector_reports_probability_diagnostics_for_rejected_mask(self):
        class EmptyNet:
            def setInput(self, blob): self.blob = blob
            def forward(self, name): return np.zeros((1,1,256,256), np.float32)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); proto=root/"d.prototxt"; weights=root/"w.caffemodel"; prov=root/"p.json"
            proto.write_text('input: "data"\n',encoding="utf-8"); weights.write_bytes(b"w")
            prov.write_text(json.dumps({"model_id":"pagenet-ohio"}),encoding="utf-8")
            env={detector.PROTOTXT_ENV:str(proto),detector.WEIGHTS_ENV:str(weights),detector.PROVENANCE_ENV:str(prov),detector.OUTPUT_LAYER_ENV:"baselines_7_prob_0"}
            with patch.dict(os.environ,env,clear=False), patch.object(detector,"_network",return_value=EmptyNet()):
                c=detector.detect(image_bgr=np.full((300,500,3),255,np.uint8),mask=np.zeros((300,500),np.uint8))
        self.assertEqual(c.status,"no_candidate")
        self.assertEqual(c.diagnostics["reason"],"no_learned_page_region")
        self.assertEqual(c.diagnostics["probability_max"],0.0)
        self.assertEqual(c.diagnostics["thresholded_fraction"],0.0)

    def test_thresholding_occurs_at_native_pagenet_resolution(self):
        prob=np.zeros((256,256),np.float32); prob[32:224,48:208]=0.75
        binary, contour=detector._postprocess(prob, detector._parameters(None))
        self.assertEqual(binary.shape,(256,256))
        self.assertIsNotNone(contour)
        self.assertGreater(cv2.contourArea(contour), 0)

if __name__=="__main__": unittest.main()
