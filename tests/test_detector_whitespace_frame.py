import cv2, numpy as np
from hth.geometry import detector_whitespace_frame
def test_whitespace_frame_detects_enclosed_page():
    image=np.full((300,500,3),255,np.uint8); cv2.rectangle(image,(60,40),(440,260),(180,180,180),-1)
    mask=np.zeros((300,500),np.uint8)
    c=detector_whitespace_frame.detect(image_bgr=image,mask=mask,parameters={"background_threshold":245})
    assert c.method=="whitespace_frame"; assert c.status=="ok"
    assert c.diagnostics["background_polarity"]=="light"
    assert c.diagnostics["bright_border_background_fraction"]>0.99
def test_whitespace_frame_detects_enclosed_page_on_dark_surround():
    image=np.zeros((300,500,3),np.uint8); cv2.rectangle(image,(60,40),(440,260),(180,180,180),-1)
    mask=np.zeros((300,500),np.uint8)
    c=detector_whitespace_frame.detect(image_bgr=image,mask=mask,parameters={"background_threshold":245})
    assert c.method=="whitespace_frame"; assert c.status=="ok"
    assert c.diagnostics["background_polarity"]=="dark"
    assert c.diagnostics["dark_background_threshold"]==10
    assert c.diagnostics["dark_border_background_fraction"]>0.99
def test_whitespace_frame_rejects_nonbackground_border():
    image=np.zeros((100,100,3),np.uint8); mask=np.zeros((100,100),np.uint8)
    candidate=detector_whitespace_frame.detect(image_bgr=image,mask=mask)
    assert candidate.status=="no_candidate"
    assert candidate.diagnostics["reason"]=="no_enclosed_page_region"

def test_whitespace_frame_rejects_border_without_dominant_polarity():
    image=np.full((100,100,3),128,np.uint8)
    alternating=np.where(np.arange(100)%2==0,0,255).astype(np.uint8)
    image[0,:,:]=alternating[:,None]; image[-1,:,:]=alternating[:,None]
    image[:,0,:]=alternating[:,None]; image[:,-1,:]=alternating[:,None]
    mask=np.zeros((100,100),np.uint8)
    candidate=detector_whitespace_frame.detect(image_bgr=image,mask=mask)
    assert candidate.status=="no_candidate"
    assert candidate.diagnostics["reason"]=="border_not_background"
    assert candidate.diagnostics["border_background_fraction"]<0.55
