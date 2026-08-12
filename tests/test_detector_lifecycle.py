import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from hth import detector_lifecycle as lifecycle
TRAIN = 'input: "data"\ninput_dim: 1\ninput_dim: 3\ninput_dim: 256\ninput_dim: 256\n\ninput: "gt"\ninput_dim: 1\ninput_dim: 1\ninput_dim: 256\ninput_dim: 256\n\nlayer {\n name: "baselines_7_prob_0"\n type: "Sigmoid"\n bottom: "data"\n top: "out"\n}\nlayer {\n name: "Silence"\n type: "Silence"\n bottom: "out"\n}\nlayer {\n name: "baselines_7_loss_0"\n type: "SigmoidCrossEntropyLoss"\n bottom: "out"\n bottom: "gt"\n top: "loss"\n}\n'
class LifecycleTests(unittest.TestCase):
    def test_deploy_transform(self):
        d=lifecycle.build_pagenet_deploy_prototxt(TRAIN)
        self.assertIn('top: "out"',d)
        self.assertNotIn('input: "gt"',d)
        self.assertNotIn('SigmoidCrossEntropyLoss',d)
    def test_prepare_downloads_then_reuses(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); calls=[]
            def fake(url,target):
                calls.append(url); target.parent.mkdir(parents=True,exist_ok=True)
                if target.suffix==".prototxt":
                    target.write_text(TRAIN,encoding="utf-8")
                else:
                    target.write_bytes(b"weights")
            with patch.object(lifecycle,"_download",side_effect=fake):
                lifecycle.prepare_detector("learned_page_mask",results_root=root)
                lifecycle.prepare_detector("learned_page_mask",results_root=root)
            self.assertEqual(len(calls),2)
    def test_noop_for_ordinary_detector(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(lifecycle.prepare_detector("radial_edge",results_root=Path(td))["prepared"])
if __name__=="__main__":
    unittest.main()
