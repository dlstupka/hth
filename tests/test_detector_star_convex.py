import cv2
import numpy as np
from hth.geometry import detector_star_convex

def test_detector_contract():
    image=np.zeros((320,520,3),np.uint8); mask=np.zeros((320,520),np.uint8)
    cv2.rectangle(mask,(70,45),(450,275),255,-1); cv2.rectangle(image,(70,45),(450,275),(220,220,220),-1)
    candidate=detector_star_convex.detect(image_bgr=image,mask=mask)
    assert candidate.method == "star_convex"
    assert candidate.status in ("ok","no_candidate")

def test_unknown_parameter_rejected():
    image=np.zeros((100,100,3),np.uint8); mask=np.zeros((100,100),np.uint8)
    try: detector_star_convex.detect(image_bgr=image,mask=mask,parameters={"mystery":1})
    except ValueError: return
    raise AssertionError("unknown parameter accepted")
