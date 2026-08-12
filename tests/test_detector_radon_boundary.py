import cv2, numpy as np
from hth.geometry import detector_radon_boundary
def test_radon_boundary_detects_simple_document():
    image=np.full((300,500,3),255,np.uint8); cv2.rectangle(image,(60,40),(440,260),(0,0,0),4)
    mask=np.zeros((300,500),np.uint8)
    c=detector_radon_boundary.detect(image_bgr=image,mask=mask,parameters={"minimum_peak_prominence":1.0})
    assert c.method=="radon_boundary"
    assert c.status=="ok"
def test_radon_unknown_parameter_rejected():
    image=np.zeros((50,50,3),np.uint8); mask=np.zeros((50,50),np.uint8)
    try: detector_radon_boundary.detect(image_bgr=image,mask=mask,parameters={"x":1})
    except ValueError: return
    assert False
