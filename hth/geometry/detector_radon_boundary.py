from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from .model import Candidate

METHOD = "radon_boundary"
BASELINE_PARAMETERS = {
    "angle_limit_degrees": 8.0,
    "angle_step_degrees": 1.0,
    "edge_percentile": 82.0,
    "projection_smooth_fraction": 0.012,
    "minimum_peak_prominence": 1.25,
    "bbox_padding_fraction": 0.0,
}

def _parameters(overrides):
    values = dict(BASELINE_PARAMETERS)
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise ValueError(f"Unknown Radon Boundary parameters: {', '.join(unknown)}")
    values.update(overrides)
    for key in values:
        values[key] = float(values[key])
    if values["angle_step_degrees"] <= 0:
        raise ValueError("angle_step_degrees must be > 0")
    return values

def _gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

def _smooth(profile, fraction):
    width = max(1, int(round(len(profile) * fraction)))
    if width <= 1:
        return profile.astype(np.float32)
    kernel = np.ones(width, np.float32) / width
    return np.convolve(profile.astype(np.float32), kernel, mode="same")

def _outer_pair(profile, prominence):
    p = _smooth(profile, 0.0)
    n = len(p)
    if n < 8:
        return None
    baseline = float(np.median(p)) + 1e-6
    left_region = p[: n // 2]
    right_region = p[n // 2 :]
    li = int(np.argmax(left_region))
    ri = int(np.argmax(right_region)) + n // 2
    if float(p[li]) / baseline < prominence or float(p[ri]) / baseline < prominence:
        return None
    return li, ri

def _evaluate(image, values):
    g = _gray(image)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    threshold = np.percentile(mag, values["edge_percentile"])
    evidence = np.where(mag >= threshold, mag, 0).astype(np.float32)
    h, w = g.shape
    center = (w / 2.0, h / 2.0)
    best = None
    angles = np.arange(
        -values["angle_limit_degrees"],
        values["angle_limit_degrees"] + values["angle_step_degrees"] * 0.5,
        values["angle_step_degrees"],
    )
    for angle in angles:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(evidence, matrix, (w, h), flags=cv2.INTER_LINEAR)
        xproj = _smooth(rotated.sum(axis=0), values["projection_smooth_fraction"])
        yproj = _smooth(rotated.sum(axis=1), values["projection_smooth_fraction"])
        xp = _outer_pair(xproj, values["minimum_peak_prominence"])
        yp = _outer_pair(yproj, values["minimum_peak_prominence"])
        if xp is None or yp is None:
            continue
        score = float(xproj[xp[0]] + xproj[xp[1]] + yproj[yp[0]] + yproj[yp[1]])
        if best is None or score > best["score"]:
            best = {"angle": float(angle), "xp": xp, "yp": yp, "score": score, "matrix": matrix}
    return evidence, best

def detect(*, image_bgr, mask, parameters=None):
    del mask
    values = _parameters(parameters)
    evidence, best = _evaluate(image_bgr, values)
    h, w = evidence.shape
    if best is None:
        return Candidate(METHOD, None, None, 0.0, 0.0, {"parameters": values, "reason": "no_projection_boundary_pair"}, status="no_candidate")
    x1, x2 = best["xp"]
    y1, y2 = best["yp"]
    inverse = cv2.invertAffineTransform(best["matrix"])
    corners_rot = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.float32).reshape(-1,1,2)
    corners = cv2.transform(corners_rot, inverse).reshape(4,2)
    x, y, bw, bh = cv2.boundingRect(corners.astype(np.float32))
    pad = int(round(min(h, w) * values["bbox_padding_fraction"]))
    bbox = [max(0,x-pad), max(0,y-pad), min(w,x+bw+pad), min(h,y+bh+pad)]
    score = min(1.0, best["score"] / max(float(evidence.sum()), 1.0) * 6.0)
    return Candidate(METHOD, bbox, corners.astype(float).tolist(), score, score, {
        "parameters": values, "angle_degrees": best["angle"], "projection_score": best["score"],
        "evidence": "projection_angle_boundary_integration",
    })

def debug_images(*, image_bgr, mask, parameters=None, candidate_corners=None, verbose=False):
    del mask, verbose
    values = _parameters(parameters)
    evidence, best = _evaluate(image_bgr, values)
    norm = cv2.normalize(evidence, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    overlay = image_bgr.copy()
    if candidate_corners is not None:
        pts = np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1,1,2)
        cv2.polylines(overlay, [pts], True, (0,0,255), 3)
    return {"radon-evidence.png": norm, "radon-boundary.png": overlay}

__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
