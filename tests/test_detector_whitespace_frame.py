import cv2, numpy as np
from hth.geometry import detector_whitespace_frame
def test_whitespace_frame_detects_enclosed_page():
    image=np.full((300,500,3),255,np.uint8); cv2.rectangle(image,(60,40),(440,260),(180,180,180),-1)
    mask=np.zeros((300,500),np.uint8)
    c=detector_whitespace_frame.detect(image_bgr=image,mask=mask,parameters={"background_threshold":245})
    assert c.method=="whitespace_frame"; assert c.status=="ok"
def test_whitespace_frame_rejects_nonbackground_border():
    image=np.zeros((100,100,3),np.uint8); mask=np.zeros((100,100),np.uint8)
    assert detector_whitespace_frame.detect(image_bgr=image,mask=mask).status=="no_candidate"
