from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from . import detector_learned_page_mask as _impl
from .model import Candidate

METHOD = "pagenet_page_mask"
BASELINE_PARAMETERS = dict(_impl.BASELINE_PARAMETERS)
_CACHE: OrderedDict[str, tuple[np.ndarray, dict]] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_CACHE_LIMIT = 16


def _image_key(image: np.ndarray) -> str:
    array = np.ascontiguousarray(image)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(memoryview(array))
    return digest.hexdigest()


def _probability(image: np.ndarray) -> tuple[np.ndarray, dict]:
    key = _image_key(image)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached
    with _INFERENCE_LOCK:
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
            if cached is not None:
                _CACHE.move_to_end(key)
                return cached
        probability, provenance = _impl._probability_256(image)
        probability = np.array(probability, dtype=np.float32, copy=True)
        probability.setflags(write=False)
        cached = (probability, dict(provenance))
        with _CACHE_LOCK:
            _CACHE[key] = cached
            _CACHE.move_to_end(key)
            while len(_CACHE) > _CACHE_LIMIT:
                _CACHE.popitem(last=False)
        return cached


def precompute_golden_set_evidence(images, *, progress=None):
    keys = []
    total = len(images)
    for index, image in enumerate(images, 1):
        key = _image_key(image)
        started = time.perf_counter()
        if progress:
            progress("start", index, total, key, 0.0)
        _probability(image)
        if progress:
            progress("finish", index, total, key, time.perf_counter() - started)
        keys.append(key)
    return tuple(keys)


def export_precomputed_golden_set_evidence(images, output_dir, *, progress=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = precompute_golden_set_evidence(images, progress=progress)
    records = []
    with _CACHE_LOCK:
        for key in keys:
            probability, provenance = _CACHE[key]
            filename = f"{key}.npy"
            np.save(output_dir / filename, np.asarray(probability))
            records.append({"image_key": key, "file": filename, "model_id": provenance.get("model_id", "pagenet-ohio")})
    payload = {
        "schema_version": "0.1",
        "detector": METHOD,
        "representation": "pagenet-ohio-page-probability-256",
        "records": records,
    }
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_precomputed_golden_set_evidence(output_dir, images):
    output_dir = Path(output_dir)
    payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("detector") != METHOD:
        raise ValueError("Shared PageNet evidence detector mismatch")
    records = {record["image_key"]: record for record in payload.get("records", [])}
    expected = tuple(_image_key(image) for image in images)
    if any(key not in records for key in expected):
        raise ValueError("Shared PageNet evidence does not match the Golden Set images")
    _, _, provenance = _impl._assets()
    with _CACHE_LOCK:
        for key in expected:
            array = np.load(output_dir / records[key]["file"])
            array = np.array(array, dtype=np.float32, copy=False)
            array.setflags(write=False)
            _CACHE[key] = (array, dict(provenance))
            _CACHE.move_to_end(key)
    return expected


def detect(*, image_bgr, mask, parameters=None):
    del mask
    values = _impl._parameters(parameters)
    probability, provenance = _probability(image_bgr)
    binary, contour = _impl._postprocess(probability, values)
    height, width = image_bgr.shape[:2]
    selected = probability[binary > 0]
    diagnostics = {
        "parameters": values,
        "model_id": provenance.get("model_id", "pagenet-ohio"),
        "model_family": "PageNet",
        "model_weights_sha256": provenance.get("weights_sha256"),
        "model_license": provenance.get("license"),
        "model_source": provenance.get("upstream_repository"),
        "inference_backend": "opencv-dnn-caffe",
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "probability_mean": float(probability.mean()),
        "thresholded_fraction": float(np.count_nonzero(binary)) / float(binary.size),
        "explicit_detector": METHOD,
    }
    if contour is None:
        return Candidate(METHOD, None, None, 0, 0, {**diagnostics, "reason": "no_learned_page_region"}, status="no_candidate")

    area_fraction = float(cv2.contourArea(contour)) / float(256 * 256)
    diagnostics["mask_area_fraction"] = area_fraction
    if area_fraction < values["minimum_mask_area_fraction"]:
        return Candidate(METHOD, None, None, 0, 0, {**diagnostics, "reason": "learned_mask_too_small"}, status="no_candidate")

    perimeter = max(cv2.arcLength(contour, True), 1.0)
    approx = cv2.approxPolyDP(contour, values["polygon_epsilon_fraction"] * perimeter, True)
    corners256 = (
        approx.reshape(4, 2).astype(np.float32)
        if len(approx) == 4 and cv2.isContourConvex(approx)
        else cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    )
    corners = _impl._scale_points(corners256, width, height)
    x, y, box_width, box_height = cv2.boundingRect(corners)
    padding = int(round(min(height, width) * values["bbox_padding_fraction"]))
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(width, x + box_width + padding), min(height, y + box_height + padding)
    mean_probability = float(selected.mean()) if selected.size else 0.0
    score = min(1.0, 0.65 * mean_probability + 0.35 * min(1.0, area_fraction / 0.5))
    diagnostics.update(
        {
            "mean_page_probability": mean_probability,
            "evidence": "pagenet_learned_page_segmentation",
            "postprocess_resolution": "256x256",
        }
    )
    return Candidate(METHOD, [x1, y1, x2, y2], corners.astype(float).tolist(), score, score, diagnostics)


def debug_images(*, image_bgr, mask, parameters=None, candidate_corners=None, verbose=False):
    del mask, verbose
    values = _impl._parameters(parameters)
    probability, _ = _probability(image_bgr)
    binary, contour = _impl._postprocess(probability, values)
    height, width = image_bgr.shape[:2]
    probability_full = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    mask_full = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    overlay = image_bgr.copy()
    if contour is not None:
        scaled = _impl._scale_points(contour.reshape(-1, 2), width, height)
        cv2.polylines(overlay, [np.rint(scaled).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2)
    if candidate_corners is not None:
        cv2.polylines(overlay, [np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3)
    return {
        "pagenet-page-probability.png": np.rint(probability_full * 255).astype(np.uint8),
        "pagenet-page-mask.png": mask_full,
        "pagenet-page-boundary.png": overlay,
    }


__all__ = [
    "BASELINE_PARAMETERS",
    "METHOD",
    "debug_images",
    "detect",
    "export_precomputed_golden_set_evidence",
    "load_precomputed_golden_set_evidence",
    "precompute_golden_set_evidence",
]
