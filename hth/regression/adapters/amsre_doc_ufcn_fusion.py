from __future__ import annotations
from hth.geometry.registry import run_registered_detector


def detect(*, image_bgr, mask, parameters=None):
    return run_registered_detector("amsre_doc_ufcn_fusion", image_bgr=image_bgr, mask=mask, parameters=parameters)
