import cv2
import numpy as np
import pytest
from hth.geometry import detector_learned_page_mask as detector

class FakeNet:
    def setInput(self,blob): self.blob=blob
    def forward(self):
        out=np.full((1,1,512,512),-8.0,np.float32); out[:,:,80:440,70:450]=8.0; return out

def reset_cache():
    detector._THREAD_LOCAL.model_key=None; detector._THREAD_LOCAL.network=None

def test_learned_page_mask_generates_polygon_and_model_provenance(tmp_path,monkeypatch):
    model=tmp_path/"page.onnx"; model.write_bytes(b"fake-model")
    monkeypatch.setenv(detector.MODEL_ENV,str(model)); monkeypatch.setattr(cv2.dnn,"readNetFromONNX",lambda path:FakeNet()); reset_cache()
    c=detector.detect(image_bgr=np.full((300,500,3),255,np.uint8),mask=np.zeros((300,500),np.uint8))
    assert c.status=="ok" and c.method=="learned_page_mask"
    assert c.diagnostics["model_contract"]==detector.MODEL_CONTRACT
    assert len(c.diagnostics["model_sha256"])==64

def test_learned_page_mask_missing_model_is_configuration_error(monkeypatch):
    monkeypatch.delenv(detector.MODEL_ENV,raising=False)
    with pytest.raises(RuntimeError,match="requires an ONNX model"):
        detector.detect(image_bgr=np.zeros((50,50,3),np.uint8),mask=np.zeros((50,50),np.uint8))

def test_learned_page_mask_rejects_multichannel_output(tmp_path,monkeypatch):
    class BadNet(FakeNet):
        def forward(self): return np.zeros((1,2,512,512),np.float32)
    model=tmp_path/"page.onnx"; model.write_bytes(b"fake-model")
    monkeypatch.setenv(detector.MODEL_ENV,str(model)); monkeypatch.setattr(cv2.dnn,"readNetFromONNX",lambda path:BadNet()); reset_cache()
    with pytest.raises(RuntimeError,match="one foreground channel"):
        detector.detect(image_bgr=np.zeros((50,50,3),np.uint8),mask=np.zeros((50,50),np.uint8))
