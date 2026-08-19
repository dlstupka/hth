from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import os
import threading
import time

import cv2
import numpy as np

from .model import Candidate

METHOD = "doc_ufcn_page_mask"
MODEL_ENV = "HTH_DOC_UFCN_PAGE_MODEL"
PROVENANCE_ENV = "HTH_DOC_UFCN_PAGE_PROVENANCE"

BASELINE_PARAMETERS = {
    "minimum_confidence": 0.50,
    "minimum_component_area_fraction": 0.0005,
    "minimum_page_area_fraction": 0.12,
    "page_padding_fraction": 0.05,
}

_MODEL = None
_MODEL_KEY = None
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_EVIDENCE_CACHE: OrderedDict[str, tuple[dict, ...]] = OrderedDict()
_EVIDENCE_CACHE_LOCK = threading.Lock()
_EVIDENCE_CACHE_LIMIT = 16


def _parameters(parameters):
    values = dict(BASELINE_PARAMETERS)
    parameters = parameters or {}
    unknown = sorted(set(parameters) - set(values))
    if unknown:
        raise ValueError(f"Unknown Doc-UFCN Page-Mask parameters: {', '.join(unknown)}")
    values.update(parameters)
    for key in values:
        values[key] = float(values[key])
    return values


def _model_path() -> Path:
    raw = os.environ.get(MODEL_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{METHOD} lifecycle did not set {MODEL_ENV}")
    path = Path(raw)
    if not path.is_file():
        raise RuntimeError(f"{METHOD} model does not exist: {path}")
    return path


def _provenance() -> dict:
    raw = os.environ.get(PROVENANCE_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{METHOD} lifecycle did not set {PROVENANCE_ENV}")
    path = Path(raw)
    if not path.is_file():
        raise RuntimeError(f"{METHOD} provenance does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model():
    global _MODEL, _MODEL_KEY
    model_path = _model_path().resolve()
    key = str(model_path)
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_KEY == key:
            return _MODEL
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from doc_ufcn.main import DocUFCN

        provenance = _provenance()
        classes = list(provenance.get("classes") or ["background", "page"])
        input_size = int(provenance.get("input_size") or 768)
        mean = [int(v) for v in provenance.get("mean") or [190, 182, 165]]
        std = [int(v) for v in provenance.get("std") or [48, 48, 45]]
        model = DocUFCN(len(classes), input_size, "cpu")
        model.load(model_path, mean, std, mode="eval")
        _MODEL = model
        _MODEL_KEY = key
        print(
            f"Doc-UFCN Page-Mask ready for inference: model={model_path.name} "
            f"backend=doc-ufcn device=cpu input_size={input_size}",
            flush=True,
        )
        return model


def _image_key(image_bgr: np.ndarray) -> str:
    arr = np.ascontiguousarray(image_bgr)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(memoryview(arr))
    return digest.hexdigest()


def _normalize_polygons(predicted) -> tuple[dict, ...]:
    records = []
    for item in (predicted or {}).get(1, []) or []:
        polygon = np.asarray(item.get("polygon") or [], dtype=np.float32).reshape(-1, 2)
        if len(polygon) < 3 or not np.isfinite(polygon).all():
            continue
        records.append({
            "confidence": float(item.get("confidence") or 0.0),
            "polygon": polygon.astype(float).tolist(),
            "area": abs(float(cv2.contourArea(polygon))),
        })
    records.sort(key=lambda record: (-record["area"], -record["confidence"]))
    return tuple(records)


def _infer_evidence(image_bgr: np.ndarray) -> tuple[dict, ...]:
    key = _image_key(image_bgr)
    with _EVIDENCE_CACHE_LOCK:
        cached = _EVIDENCE_CACHE.get(key)
        if cached is not None:
            _EVIDENCE_CACHE.move_to_end(key)
            return cached

    with _INFERENCE_LOCK:
        with _EVIDENCE_CACHE_LOCK:
            cached = _EVIDENCE_CACHE.get(key)
            if cached is not None:
                _EVIDENCE_CACHE.move_to_end(key)
                return cached
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predicted, _, _, _ = _load_model().predict(rgb, min_cc=1)
        evidence = _normalize_polygons(predicted)
        with _EVIDENCE_CACHE_LOCK:
            _EVIDENCE_CACHE[key] = evidence
            _EVIDENCE_CACHE.move_to_end(key)
            while len(_EVIDENCE_CACHE) > _EVIDENCE_CACHE_LIMIT:
                _EVIDENCE_CACHE.popitem(last=False)
        return evidence


def precompute_golden_set_evidence(images, *, progress=None):
    keys = []
    total = len(images)
    for index, image_bgr in enumerate(images, 1):
        key = _image_key(image_bgr)
        started = time.perf_counter()
        if progress is not None:
            progress("start", index, total, key, 0.0)
        _infer_evidence(image_bgr)
        elapsed = time.perf_counter() - started
        if progress is not None:
            progress("finish", index, total, key, elapsed)
        keys.append(key)
    return tuple(keys)


def export_precomputed_golden_set_evidence(images, output_dir, *, progress=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = precompute_golden_set_evidence(images, progress=progress)
    records = []
    with _EVIDENCE_CACHE_LOCK:
        for key in keys:
            records.append({"image_key": key, "polygons": list(_EVIDENCE_CACHE[key])})
    payload = {
        "schema_version": "0.1",
        "detector": METHOD,
        "representation": "doc-ufcn-page-polygons",
        "page_count": len(records),
        "records": records,
    }
    target = output_dir / "manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, target)
    return target


def load_precomputed_golden_set_evidence(output_dir, images):
    output_dir = Path(output_dir)
    payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("detector") != METHOD:
        raise ValueError(f"Shared evidence detector mismatch: {payload.get('detector')} != {METHOD}")
    records = {
        str(record["image_key"]): tuple(record.get("polygons") or [])
        for record in payload.get("records", [])
        if isinstance(record, dict) and record.get("image_key")
    }
    expected = tuple(_image_key(image_bgr) for image_bgr in images)
    missing = [key for key in expected if key not in records]
    if missing:
        raise ValueError(f"Shared Doc-UFCN evidence is missing {len(missing)} Golden Set page(s)")
    with _EVIDENCE_CACHE_LOCK:
        for key in expected:
            _EVIDENCE_CACHE[key] = records[key]
            _EVIDENCE_CACHE.move_to_end(key)
            while len(_EVIDENCE_CACHE) > _EVIDENCE_CACHE_LIMIT:
                _EVIDENCE_CACHE.popitem(last=False)
    return expected


def _pad_corners(corners: np.ndarray, *, width: int, height: int, fraction: float) -> np.ndarray:
    if fraction <= 0:
        return corners.astype(np.float32)
    center = corners.mean(axis=0)
    scale = 1.0 + 2.0 * float(fraction)
    padded = center + (corners - center) * scale
    padded[:, 0] = np.clip(padded[:, 0], 0, max(0, width - 1))
    padded[:, 1] = np.clip(padded[:, 1], 0, max(0, height - 1))
    return padded.astype(np.float32)


def _select_polygon(image_bgr: np.ndarray, values):
    height, width = image_bgr.shape[:2]
    image_area = float(max(1, height * width))
    minimum_component_area = values["minimum_component_area_fraction"] * image_area
    candidates = []
    for record in _infer_evidence(image_bgr):
        confidence = float(record.get("confidence") or 0.0)
        polygon = np.asarray(record.get("polygon") or [], dtype=np.float32).reshape(-1, 2)
        area = abs(float(cv2.contourArea(polygon))) if len(polygon) >= 3 else 0.0
        if confidence < values["minimum_confidence"] or area < minimum_component_area:
            continue
        candidates.append((area, confidence, polygon))
    return max(candidates, key=lambda item: (item[0], item[1])) if candidates else None


def detect(*, image_bgr, mask, parameters=None):
    del mask
    values = _parameters(parameters)
    height, width = image_bgr.shape[:2]
    image_area = float(max(1, height * width))
    provenance = _provenance()
    selected = _select_polygon(image_bgr, values)
    diagnostics = {
        "parameters": values,
        "model_id": provenance.get("model_id", "doc-ufcn-generic-page"),
        "model_family": "Doc-UFCN",
        "model_sha256": provenance.get("model_sha256"),
        "model_source": provenance.get("model_url"),
        "upstream_repository": provenance.get("upstream_repository"),
        "upstream_license": provenance.get("license"),
        "doc_ufcn_version": provenance.get("doc_ufcn_version"),
        "input_size": provenance.get("input_size", 768),
        "inference_backend": "doc-ufcn-pytorch",
        "raw_polygon_count": len(_infer_evidence(image_bgr)),
    }
    if selected is None:
        return Candidate(METHOD, None, None, 0, 0, {**diagnostics, "reason": "no_doc_ufcn_page_polygon"}, status="no_candidate")

    area, confidence, polygon = selected
    area_fraction = area / image_area
    diagnostics.update({"selected_confidence": confidence, "selected_area_fraction": area_fraction})
    if area_fraction < values["minimum_page_area_fraction"]:
        return Candidate(METHOD, None, None, 0, 0, {**diagnostics, "reason": "doc_ufcn_page_polygon_too_small"}, status="no_candidate")

    corners = cv2.boxPoints(cv2.minAreaRect(polygon)).astype(np.float32)
    corners = _pad_corners(corners, width=width, height=height, fraction=values["page_padding_fraction"])
    x, y, bw, bh = cv2.boundingRect(corners)
    score = min(1.0, 0.75 * confidence + 0.25 * min(1.0, area_fraction / 0.5))
    diagnostics["evidence"] = "doc_ufcn_generic_page_polygon"
    return Candidate(
        METHOD,
        [int(x), int(y), int(x + bw), int(y + bh)],
        corners.astype(float).tolist(),
        score,
        score,
        diagnostics,
    )


def debug_images(*, image_bgr, mask, parameters=None, candidate_corners=None, verbose=False):
    del mask, verbose
    values = _parameters(parameters)
    overlay = image_bgr.copy()
    for record in _infer_evidence(image_bgr):
        polygon = np.rint(np.asarray(record.get("polygon") or [], dtype=np.float32)).astype(np.int32)
        if len(polygon) >= 3:
            cv2.polylines(overlay, [polygon.reshape(-1, 1, 2)], True, (0, 255, 255), 2)
    if candidate_corners is not None:
        corners = np.rint(np.asarray(candidate_corners, dtype=np.float32)).astype(np.int32)
        cv2.polylines(overlay, [corners.reshape(-1, 1, 2)], True, (0, 0, 255), 3)
    return {"doc-ufcn-page-polygons.png": overlay}


__all__ = [
    "BASELINE_PARAMETERS",
    "METHOD",
    "MODEL_ENV",
    "PROVENANCE_ENV",
    "debug_images",
    "detect",
    "precompute_golden_set_evidence",
    "export_precomputed_golden_set_evidence",
    "load_precomputed_golden_set_evidence",
]
