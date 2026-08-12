import cv2, numpy as np
from hth.geometry import detector_joint_rectangle_vote
def test_joint_rectangle_vote_detects_box():
    image=np.full((300,500,3),255,np.uint8); cv2.rectangle(image,(60,40),(440,260),(0,0,0),4)
    mask=np.zeros((300,500),np.uint8)
    c=detector_joint_rectangle_vote.detect(image_bgr=image,mask=mask,parameters={"hough_threshold":80,"minimum_side_support":0.18})
    assert c.method=="joint_rectangle_vote"; assert c.status=="ok"
def test_joint_rectangle_blank_is_miss():
    image=np.full((100,100,3),255,np.uint8); mask=np.zeros((100,100),np.uint8)
    assert detector_joint_rectangle_vote.detect(image_bgr=image,mask=mask).status=="no_candidate"
