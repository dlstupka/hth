from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import threading
import tempfile
import time
import warnings
from typing import Any

import cv2
import numpy as np

from .model import Candidate

METHOD = "kraken_page_mask"
MODEL_ENV = "HTH_KRAKEN_PAGE_MODEL"
PROVENANCE_ENV = "HTH_KRAKEN_PAGE_PROVENANCE"

BASELINE_PARAMETERS = {
    "include_lines": 1,
    "dilation_fraction": 0.010,
    "close_kernel_fraction": 0.006,
    "page_padding_fraction": 0.050,
    "minimum_page_area_fraction": 0.12,
    "fill_holes": 1,
}

_MODEL = None
_MODEL_KEY = None
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_STDERR_CAPTURE_LOCK = threading.Lock()
_EVIDENCE_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_EVIDENCE_CACHE_LOCK = threading.Lock()
_EVIDENCE_CACHE_LIMIT = 16

_RUNTIME_DIAGNOSTICS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_RUNTIME_DIAGNOSTICS_LOCK = threading.Lock()


@contextmanager
def _capture_kraken_runtime_chatter():
    """Thread-safe capture of known native Kraken/GEOS stderr chatter.

    fd 2 is process-global, so redirection itself cannot be thread-local. Protect
    the complete redirect/restore interval with a dedicated lock. Regression
    parameter threads normally never enter this path because Golden Set evidence
    is precomputed once before parameter concurrency begins.
    """
    with _STDERR_CAPTURE_LOCK:
        saved_fd = os.dup(2)
        captured = tempfile.TemporaryFile(mode="w+b")
        diagnostics = {
            "lightning_srun_advisories": 0,
            "kraken_polygonizer_warnings": 0,
            "filtered_messages": [],
        }
        try:
            os.dup2(captured.fileno(), 2)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*The `srun` command is available on your system but is not used.*",
                )
                yield diagnostics
        finally:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            captured.seek(0)
            text = captured.read().decode("utf-8", errors="replace")
            captured.close()

            replay = []
            for line in text.splitlines():
                stripped = line.strip()
                if "The `srun` command is available on your system but is not used" in stripped:
                    diagnostics["lightning_srun_advisories"] += 1
                    if len(diagnostics["filtered_messages"]) < 8:
                        diagnostics["filtered_messages"].append(stripped)
                    continue
                if "TopologyException: side location conflict" in stripped or "Polygonizer failed on line" in stripped:
                    diagnostics["kraken_polygonizer_warnings"] += 1
                    if len(diagnostics["filtered_messages"]) < 8:
                        diagnostics["filtered_messages"].append(stripped)
                    continue
                replay.append(line)

            if replay:
                os.write(2, ("\n".join(replay) + "\n").encode("utf-8", errors="replace"))


def _runtime_diagnostics_for(key):
    with _RUNTIME_DIAGNOSTICS_LOCK:
        return dict(_RUNTIME_DIAGNOSTICS.get(key) or {})


def _store_runtime_diagnostics(key, diagnostics):
    with _RUNTIME_DIAGNOSTICS_LOCK:
        _RUNTIME_DIAGNOSTICS[key] = dict(diagnostics)
        _RUNTIME_DIAGNOSTICS.move_to_end(key)
        while len(_RUNTIME_DIAGNOSTICS) > _EVIDENCE_CACHE_LIMIT:
            _RUNTIME_DIAGNOSTICS.popitem(last=False)


def _canonical_quad(corners, *, width, height):
    """Return a clipped, clockwise, convex 4-corner page quadrilateral."""
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return None

    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, width - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, height - 1))

    hull = cv2.convexHull(pts, clockwise=True, returnPoints=True)
    if hull is None:
        return None
    hull = hull.reshape(-1, 2)
    if hull.shape != (4, 2):
        return None
    if abs(float(cv2.contourArea(hull))) < 1.0:
        return None

    center = hull.mean(axis=0)
    angles = np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0])
    ordered = hull[np.argsort(angles)]
    if cv2.contourArea(ordered.astype(np.float32), oriented=True) < 0:
        ordered = ordered[::-1]
    return ordered.astype(np.float32)




def _parameters(parameters):
    values = dict(BASELINE_PARAMETERS)
    parameters = parameters or {}
    unknown = sorted(set(parameters) - set(values))
    if unknown:
        raise ValueError(f"Unknown Kraken Page-Mask parameters: {', '.join(unknown)}")
    values.update(parameters)
    values["include_lines"] = int(values["include_lines"])
    values["fill_holes"] = int(values["fill_holes"])
    if values["include_lines"] not in (0, 1):
        raise ValueError("include_lines must be 0 or 1")
    if values["fill_holes"] not in (0, 1):
        raise ValueError("fill_holes must be 0 or 1")
    for key in (
        "dilation_fraction",
        "close_kernel_fraction",
        "page_padding_fraction",
        "minimum_page_area_fraction",
    ):
        values[key] = float(values[key])
    return values


def _model_path():
    raw = os.environ.get(MODEL_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{METHOD} lifecycle did not set {MODEL_ENV}")
    path = Path(raw)
    if not path.is_file():
        raise RuntimeError(f"{METHOD} default BLLA model does not exist: {path}")
    return path


def _provenance():
    raw = os.environ.get(PROVENANCE_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{METHOD} lifecycle did not set {PROVENANCE_ENV}")
    path = Path(raw)
    if not path.is_file():
        raise RuntimeError(f"{METHOD} provenance does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model():
    global _MODEL, _MODEL_KEY
    path = _model_path().resolve()
    key = str(path)
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_KEY == key:
            return _MODEL
        # Kraken 7 task API is the supported non-deprecated inference interface.
        # The package's bundled blla.mlmodel is the normal/default segmentation
        # model used by Kraken when baseline segmentation is requested without
        # an explicit model.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from kraken.tasks import SegmentationTaskModel
        _MODEL = SegmentationTaskModel.load_model(path)
        _MODEL_KEY = key
        print(f"Kraken Page-Mask ready for inference: model={path.name} backend=kraken-task-api device=cpu")
        return _MODEL


def _image_key(image_bgr):
    arr = np.ascontiguousarray(image_bgr)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(memoryview(arr))
    return h.hexdigest()


def _point_list(value):
    if not value:
        return []
    out = []
    for point in value:
        if point is None or len(point) < 2:
            continue
        out.append((int(round(float(point[0]))), int(round(float(point[1])))))
    return out


def _extract_evidence(segmentation):
    regions = []
    for group in (getattr(segmentation, "regions", None) or {}).values():
        for region in group or []:
            pts = _point_list(getattr(region, "boundary", None))
            if len(pts) >= 3:
                regions.append(pts)

    lines = []
    baselines = []
    for line in getattr(segmentation, "lines", None) or []:
        boundary = _point_list(getattr(line, "boundary", None))
        if len(boundary) >= 3:
            lines.append(boundary)
        baseline = _point_list(getattr(line, "baseline", None))
        if len(baseline) >= 2:
            baselines.append(baseline)

    return {
        "regions": regions,
        "lines": lines,
        "baselines": baselines,
        "text_direction": str(getattr(segmentation, "text_direction", "horizontal-lr")),
    }


def _freeze_evidence(evidence):
    """Return an immutable-by-convention evidence snapshot safe for all threads."""
    return {
        "regions": tuple(tuple(tuple(point) for point in polygon) for polygon in evidence["regions"]),
        "lines": tuple(tuple(tuple(point) for point in polygon) for polygon in evidence["lines"]),
        "baselines": tuple(tuple(tuple(point) for point in baseline) for baseline in evidence["baselines"]),
        "text_direction": str(evidence["text_direction"]),
    }


def _infer_evidence(image_bgr):
    key = _image_key(image_bgr)
    with _EVIDENCE_CACHE_LOCK:
        cached = _EVIDENCE_CACHE.get(key)
        if cached is not None:
            _EVIDENCE_CACHE.move_to_end(key)
            return cached

    # Single-flight cache fill. A second thread that missed before the first
    # inference completed rechecks after acquiring the inference lock and uses
    # the now-populated immutable snapshot instead of repeating model.predict().
    with _INFERENCE_LOCK:
        with _EVIDENCE_CACHE_LOCK:
            cached = _EVIDENCE_CACHE.get(key)
            if cached is not None:
                _EVIDENCE_CACHE.move_to_end(key)
                return cached

        from PIL import Image
        from kraken.configs import SegmentationInferenceConfig

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        model = _load_model()
        config = SegmentationInferenceConfig()
        with _capture_kraken_runtime_chatter() as runtime_diagnostics:
            segmentation = model.predict(im=image, config=config)
        evidence = _freeze_evidence(_extract_evidence(segmentation))
        _store_runtime_diagnostics(key, runtime_diagnostics)

        with _EVIDENCE_CACHE_LOCK:
            _EVIDENCE_CACHE[key] = evidence
            _EVIDENCE_CACHE.move_to_end(key)
            while len(_EVIDENCE_CACHE) > _EVIDENCE_CACHE_LIMIT:
                _EVIDENCE_CACHE.popitem(last=False)
        return evidence


def precompute_golden_set_evidence(images, *, progress=None):
    """Populate immutable Kraken evidence once before parameter-thread fan-out."""
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
    """Persist one process-independent immutable Kraken Golden Set evidence set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = precompute_golden_set_evidence(images, progress=progress)
    records = []
    with _EVIDENCE_CACHE_LOCK:
        for key in keys:
            evidence = _EVIDENCE_CACHE[key]
            records.append({
                "image_key": key,
                "evidence": evidence,
                "runtime_diagnostics": _runtime_diagnostics_for(key),
            })
    payload = {
        "schema_version": "0.1",
        "detector": METHOD,
        "representation": "immutable-json",
        "page_count": len(records),
        "records": records,
    }
    target = output_dir / "manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, target)
    return target


def load_precomputed_golden_set_evidence(output_dir, images):
    """Load parent-precomputed Kraken evidence without loading/running Kraken."""
    manifest = Path(output_dir) / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("detector") != METHOD:
        raise ValueError(f"Shared evidence detector mismatch: {payload.get('detector')} != {METHOD}")
    records = {
        str(record["image_key"]): record
        for record in payload.get("records", [])
        if isinstance(record, dict) and record.get("image_key")
    }
    expected = tuple(_image_key(image_bgr) for image_bgr in images)
    missing = [key for key in expected if key not in records]
    if missing:
        raise ValueError(f"Shared Kraken evidence is missing {len(missing)} Golden Set page(s)")
    with _EVIDENCE_CACHE_LOCK:
        for key in expected:
            evidence = _freeze_evidence(records[key]["evidence"])
            _EVIDENCE_CACHE[key] = evidence
            _EVIDENCE_CACHE.move_to_end(key)
            while len(_EVIDENCE_CACHE) > _EVIDENCE_CACHE_LIMIT:
                _EVIDENCE_CACHE.popitem(last=False)
            diagnostics = records[key].get("runtime_diagnostics")
            if isinstance(diagnostics, dict):
                _store_runtime_diagnostics(key, diagnostics)
    return expected


def _odd_kernel(size):
    size = max(1, int(round(size)))
    return size if size % 2 else size + 1


def _fill_holes(mask):
    flood = mask.copy()
    h, w = flood.shape
    padded = cv2.copyMakeBorder(flood, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = padded.copy()
    cv2.floodFill(ff, None, (0, 0), 255)
    holes = cv2.bitwise_not(ff[1:h + 1, 1:w + 1])
    return cv2.bitwise_or(mask, holes)


def _evidence_mask(image_shape, evidence, values):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for polygon in evidence["regions"]:
        pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)

    if values["include_lines"]:
        for polygon in evidence["lines"]:
            pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)
        thickness = max(2, int(round(min(h, w) * 0.003)))
        for baseline in evidence["baselines"]:
            pts = np.asarray(baseline, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts], False, 255, thickness=thickness)

    scale = min(h, w)
    dilation = _odd_kernel(scale * values["dilation_fraction"]) if values["dilation_fraction"] > 0 else 1
    if dilation > 1:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation)))

    close = _odd_kernel(scale * values["close_kernel_fraction"]) if values["close_kernel_fraction"] > 0 else 1
    if close > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close, close)))

    if values["fill_holes"]:
        mask = _fill_holes(mask)
    return mask


def _select_evidence_envelope(contours, *, image_shape):
    """Return the contour set that best represents the learned page extent.

    Kraken can emit several disconnected regions on sparse or damaged pages.
    Keep the dominant component, admit substantial secondary components, and
    also admit smaller learned components when they materially extend the
    dominant envelope.  The latter recovers sparse page-side evidence without
    broadly accepting isolated noise.

    This remains parameter-free so the declared calibration grid is unchanged.
    """
    if not contours:
        return [], {
            "mode": "none",
            "external_contours": 0,
            "selected_contours": 0,
        }

    h, w = image_shape[:2]
    image_area = float(h * w)
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)
    dominant = ranked[0]
    dominant_area = float(cv2.contourArea(dominant))
    absolute_floor = max(4.0, image_area * 0.00025)
    relative_floor = dominant_area * 0.015
    area_floor = max(absolute_floor, relative_floor)

    x, y, cw, ch = cv2.boundingRect(dominant)
    dominant_box = (float(x), float(y), float(x + cw - 1), float(y + ch - 1))
    extension_x = max(8.0, float(w) * 0.035)
    extension_y = max(8.0, float(h) * 0.035)
    extension_area_floor = max(4.0, image_area * 0.00005)

    selected = [dominant]
    admitted_by_extension = 0
    for contour in ranked[1:]:
        area = float(cv2.contourArea(contour))
        if area >= area_floor:
            selected.append(contour)
            continue
        if area < extension_area_floor:
            continue

        sx, sy, sw, sh = cv2.boundingRect(contour)
        secondary_box = (
            float(sx),
            float(sy),
            float(sx + sw - 1),
            float(sy + sh - 1),
        )
        materially_extends = (
            secondary_box[0] < dominant_box[0] - extension_x
            or secondary_box[2] > dominant_box[2] + extension_x
            or secondary_box[1] < dominant_box[1] - extension_y
            or secondary_box[3] > dominant_box[3] + extension_y
        )
        if materially_extends:
            selected.append(contour)
            admitted_by_extension += 1

    if len(selected) == 1:
        return selected, {
            "mode": "dominant",
            "external_contours": len(ranked),
            "selected_contours": 1,
            "dominant_area": dominant_area,
            "component_area_floor": area_floor,
            "extension_area_floor": extension_area_floor,
            "extension_admissions": 0,
        }

    points = np.concatenate(selected, axis=0)
    hull = cv2.convexHull(points)
    return [hull], {
        "mode": "multi-region-envelope",
        "external_contours": len(ranked),
        "selected_contours": len(selected),
        "dominant_area": dominant_area,
        "component_area_floor": area_floor,
        "extension_area_floor": extension_area_floor,
        "extension_admissions": admitted_by_extension,
    }


def _recover_sparse_vertical_extent(contour, *, image_shape):
    """Conservatively recover a missing top/bottom page extent.

    When Kraken evidence already spans nearly the full image width and reaches
    one vertical image edge, a premature stop at the opposite end is usually
    sparse-content truncation rather than a real page boundary.  Add only the
    missing image-edge corners; otherwise leave the learned envelope untouched.
    """
    h, w = image_shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    width_fraction = float(cw) / float(w) if w else 0.0
    height_fraction = float(ch) / float(h) if h else 0.0
    top_gap = float(y) / float(h) if h else 1.0
    bottom_gap = float(h - (y + ch)) / float(h) if h else 1.0

    diagnostics = {
        "applied": False,
        "width_fraction": width_fraction,
        "height_fraction": height_fraction,
        "top_gap_fraction": top_gap,
        "bottom_gap_fraction": bottom_gap,
    }

    # Require strong horizontal page evidence and a substantial existing
    # vertical span.  Recover only one missing end, and only when the opposite
    # end is already anchored very near the image boundary.
    if width_fraction < 0.85 or height_fraction < 0.55:
        return contour, diagnostics

    left = max(0, x)
    right = min(w - 1, x + cw - 1)
    additions = []
    recovered = None
    if top_gap <= 0.05 and bottom_gap >= 0.08:
        additions = [[[left, h - 1]], [[right, h - 1]]]
        recovered = "bottom"
    elif bottom_gap <= 0.05 and top_gap >= 0.08:
        additions = [[[left, 0]], [[right, 0]]]
        recovered = "top"

    if not additions:
        return contour, diagnostics

    extra = np.asarray(additions, dtype=np.int32)
    hull = cv2.convexHull(np.concatenate([contour, extra], axis=0))
    diagnostics["applied"] = True
    diagnostics["recovered_edge"] = recovered
    return hull, diagnostics

def _proposal(image_bgr, values):
    evidence = _infer_evidence(image_bgr)
    mask = _evidence_mask(image_bgr.shape, evidence, values)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    image_area = float(h * w)
    selected, envelope_diagnostics = _select_evidence_envelope(
        contours,
        image_shape=mask.shape,
    )
    if not selected:
        return evidence, mask, None, None, 0.0, envelope_diagnostics

    contour = selected[0]
    contour, vertical_recovery = _recover_sparse_vertical_extent(
        contour,
        image_shape=mask.shape,
    )
    envelope_diagnostics["vertical_extent_recovery"] = vertical_recovery
    rect = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rect).astype(np.float32)

    center = corners.mean(axis=0)
    vectors = corners - center
    lengths = np.linalg.norm(vectors, axis=1)
    pad = float(min(h, w)) * values["page_padding_fraction"]
    safe = np.maximum(lengths, 1e-6)
    corners = center + vectors * ((safe + pad) / safe)[:, None]
    corners = _canonical_quad(corners, width=w, height=h)
    if corners is None:
        return evidence, mask, contour, None, 0.0, envelope_diagnostics

    polygon_area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    page_area_fraction = polygon_area / image_area if image_area else 0.0
    return evidence, mask, contour, corners, page_area_fraction, envelope_diagnostics


def detect(*, image_bgr, mask, parameters=None):
    del mask
    values = _parameters(parameters)
    evidence, evidence_mask, contour, corners, area_fraction, envelope_diagnostics = _proposal(image_bgr, values)

    runtime_diagnostics = _runtime_diagnostics_for(_image_key(image_bgr))
    diagnostics = {
        "parameters": values,
        "evidence": "kraken_default_blla_regions_and_lines",
        "envelope": envelope_diagnostics,
        "region_count": len(evidence["regions"]),
        "line_polygon_count": len(evidence["lines"]),
        "baseline_count": len(evidence["baselines"]),
        "text_direction": evidence["text_direction"],
        "page_area_fraction": area_fraction,
        "model": _provenance().get("model_id"),
        "canonical_quad": corners is not None,
        "kraken_polygonizer_warnings": int(runtime_diagnostics.get("kraken_polygonizer_warnings", 0)),
        "lightning_srun_advisories": int(runtime_diagnostics.get("lightning_srun_advisories", 0)),
        "kraken_filtered_messages": list(runtime_diagnostics.get("filtered_messages", [])),
    }

    if not evidence["regions"] and not evidence["lines"] and not evidence["baselines"]:
        diagnostics["reason"] = "no_kraken_layout_evidence"
        return Candidate(METHOD, None, None, 0, 0, diagnostics, status="no_candidate")
    if contour is None:
        diagnostics["reason"] = "no_connected_layout_region"
        return Candidate(METHOD, None, None, 0, 0, diagnostics, status="no_candidate")
    if corners is None:
        diagnostics["reason"] = "invalid_page_quadrilateral"
        return Candidate(METHOD, None, None, 0, 0, diagnostics, status="no_candidate")
    if area_fraction < values["minimum_page_area_fraction"]:
        diagnostics["reason"] = "page_area_below_minimum"
        return Candidate(METHOD, None, None, 0, 0, diagnostics, status="no_candidate")

    x, y, bw, bh = cv2.boundingRect(corners.astype(np.float32))
    h, w = evidence_mask.shape
    bbox = [max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)]
    support = float(np.count_nonzero(evidence_mask)) / max(1.0, float(h * w))
    score = min(1.0, 0.5 + 0.5 * min(1.0, support / max(area_fraction, 1e-6)))
    diagnostics["evidence_mask_fraction"] = support
    return Candidate(
        METHOD,
        bbox,
        corners.astype(float).tolist(),
        score,
        score,
        diagnostics,
    )


def debug_images(*, image_bgr, mask, parameters=None, candidate_corners=None, verbose=False):
    del mask, verbose
    values = _parameters(parameters)
    evidence, evidence_mask, contour, corners, _, _ = _proposal(image_bgr, values)
    overlay = image_bgr.copy()

    for polygon in evidence["regions"]:
        cv2.polylines(overlay, [np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2)
    if values["include_lines"]:
        for polygon in evidence["lines"]:
            cv2.polylines(overlay, [np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)], True, (255, 255, 0), 1)
        for baseline in evidence["baselines"]:
            cv2.polylines(overlay, [np.asarray(baseline, dtype=np.int32).reshape(-1, 1, 2)], False, (255, 0, 255), 1)

    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
    chosen = candidate_corners if candidate_corners is not None else corners
    if chosen is not None:
        pts = np.rint(np.asarray(chosen)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], True, (0, 0, 255), 3)

    return {
        "kraken-layout-evidence.png": evidence_mask,
        "kraken-page-proposal.png": overlay,
    }


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect", "precompute_golden_set_evidence", "export_precomputed_golden_set_evidence", "load_precomputed_golden_set_evidence"]
