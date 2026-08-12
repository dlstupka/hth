import cv2, numpy as np
from hth.geometry import detector_text_flow
def test_text_flow_detects_text_bands():
    image=np.zeros((300,500,3),np.uint8); mask=np.zeros((300,500),np.uint8)
    for y in (70,100,130,160,190):
        for x in range(90,410,28): cv2.rectangle(mask,(x,y),(x+12,y+8),255,-1)
    c=detector_text_flow.detect(image_bgr=image,mask=mask,parameters={"minimum_text_coverage_fraction":0.01,"line_join_fraction":0.03})
    assert c.method=="text_flow"; assert c.status=="ok"
def test_text_flow_blank_is_miss():
    image=np.zeros((100,100,3),np.uint8); mask=np.zeros((100,100),np.uint8)
    assert detector_text_flow.detect(image_bgr=image,mask=mask).status=="no_candidate"
