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
    def test_config_drives_named_prepare_and_finalize_hooks(self):
        import json
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            config=root/"detector.json"
            config.write_text(json.dumps({
                "detector":"learned_page_mask",
                "lifecycle":{
                    "prepare":"learned_page_mask",
                    "finalize":"learned_page_mask",
                    "model_policy":"reuse"
                }
            }),encoding="utf-8")
            env_file=root/"detector.env"
            with patch.object(lifecycle,"_prepare_learned_page_mask_hook",return_value={"prepared":True}) as prepare_hook, \
                 patch.object(lifecycle,"_finalize_learned_page_mask_hook",return_value={"finalized":True}) as finalize_hook:
                # Patch registry entries because the registries hold function objects.
                with patch.dict(lifecycle._PREPARE_HOOKS,{"learned_page_mask":prepare_hook}), \
                     patch.dict(lifecycle._FINALIZE_HOOKS,{"learned_page_mask":finalize_hook}):
                    lifecycle.prepare_config(config,results_root=root,env_file=env_file)
                    lifecycle.finalize_config(config,results_root=root)
            prepare_hook.assert_called_once()
            finalize_hook.assert_called_once()

    def test_shell_environment_writer_quotes_paths(self):
        with tempfile.TemporaryDirectory() as td:
            env_file=Path(td)/"hook.env"
            lifecycle._write_env(env_file,{"MODEL_PATH":"C:/Program Files/HTH/model file.bin"})
            text=env_file.read_text(encoding="utf-8")
            self.assertIn("MODEL_PATH='C:/Program Files/HTH/model file.bin'",text)

if __name__=="__main__":
    unittest.main()
