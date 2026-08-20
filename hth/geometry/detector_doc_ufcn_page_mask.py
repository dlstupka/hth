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


def _quad_boundary_support(image_bgr: np.ndarray, corners: np.ndarray) -> dict:
    """Measure source-image edge support along a proposed page quadrilateral.

    This is deliberately parameter-free and is used only to decide whether a
    smaller calibrated padding should yield to the detector baseline padding.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    pts = np.rint(np.asarray(corners, dtype=np.float32)).astype(np.int32)
    edge_scores = []
    for index in range(4):
        mask = np.zeros(gray.shape, dtype=np.uint8)
        a = tuple(int(v) for v in pts[index])
        b = tuple(int(v) for v in pts[(index + 1) % 4])
        cv2.line(mask, a, b, 255, 3, cv2.LINE_AA)
        values = magnitude[mask > 0]
        edge_scores.append(float(np.mean(values)) if values.size else 0.0)
    robust = float(np.median(edge_scores)) if edge_scores else 0.0
    background = float(np.median(magnitude))
    return {
        "edge_scores": [round(value, 4) for value in edge_scores],
        "robust_score": robust,
        "background_score": background,
        "contrast_ratio": robust / max(1.0, background),
    }


def _boundary_supported_padding(image_bgr: np.ndarray, corners: np.ndarray, *, requested_fraction: float):
    """Prefer baseline padding only when the source image independently supports it."""
    height, width = image_bgr.shape[:2]
    requested = _pad_corners(corners, width=width, height=height, fraction=requested_fraction)
    baseline_fraction = float(BASELINE_PARAMETERS["page_padding_fraction"])
    diagnostics = {
        "requested_padding_fraction": float(requested_fraction),
        "baseline_padding_fraction": baseline_fraction,
        "decision": "requested-padding",
    }
    if requested_fraction >= baseline_fraction:
        return requested, diagnostics

    baseline = _pad_corners(corners, width=width, height=height, fraction=baseline_fraction)
    requested_support = _quad_boundary_support(image_bgr, requested)
    baseline_support = _quad_boundary_support(image_bgr, baseline)
    diagnostics.update({
        "requested_boundary_support": requested_support,
        "baseline_boundary_support": baseline_support,
    })
    # The larger envelope must have independent, materially stronger boundary
    # evidence. This preserves the calibrated value unless the image itself
    # argues that the neural polygon terminates inside the physical page.
    if (
        baseline_support["contrast_ratio"] >= 1.25
        and baseline_support["robust_score"] >= requested_support["robust_score"] * 1.12
    ):
        diagnostics["decision"] = "baseline-padding-boundary-supported"
        return baseline, diagnostics
    return requested, diagnostics


def _qualifying_polygons(image_bgr: np.ndarray, values):
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
        x, y, bw, bh = cv2.boundingRect(polygon)
        candidates.append({
            "area": area,
            "confidence": confidence,
            "polygon": polygon,
            "bounds": (float(x), float(y), float(x + bw), float(y + bh)),
        })
    candidates.sort(key=lambda item: (-item["area"], -item["confidence"]))
    return candidates


def _select_polygon(image_bgr: np.ndarray, values):
    """Retain the historic single-component selector for compatibility/tests."""
    candidates = _qualifying_polygons(image_bgr, values)
    if not candidates:
        return None
    item = candidates[0]
    return item["area"], item["confidence"], item["polygon"]


def _outer_background_boundary(gray: np.ndarray, *, side: str, y0: int, y1: int, start: int, stop: int):
    """Find a sustained paper-to-outer-background transition on one horizontal side.

    Damaged historical leaves can have a faint/torn physical edge that is not a
    strong median Sobel peak.  The scanner background outside the paper is often
    much more stable, though, so use robust column luminance to locate the last
    sustained foreground-to-background transition.  This is intentionally only
    a secondary proof used after the single-leaf spread gates have fired.
    """
    if y1 <= y0 + 8 or stop <= start + 8:
        return None, {"accepted": False, "reason": "insufficient-region"}
    band = gray[y0:y1, :].astype(np.float32)
    # Median suppresses handwriting/cracks while preserving the broad paper vs
    # scanner-background separation.
    column_level = np.median(band, axis=0)
    column_level = cv2.GaussianBlur(column_level.reshape(1, -1), (1, 0), sigmaX=3.0).reshape(-1)
    width = gray.shape[1]
    edge_band = max(8, int(round(width * 0.025)))
    if side == "right":
        bg_sample = column_level[max(0, width - edge_band):width]
    else:
        bg_sample = column_level[:min(width, edge_band)]
    if bg_sample.size == 0:
        return None, {"accepted": False, "reason": "no-background-sample"}
    bg = float(np.median(bg_sample))
    bg_mad = float(np.median(np.abs(bg_sample - bg)))
    # Require the paper side of the transition to differ materially from the
    # stable outer background.  Absolute floor handles nearly-black scanners.
    delta = max(10.0, 4.0 * max(1.0, bg_mad))
    sustain = max(6, int(round(width * 0.006)))
    lo, hi = max(1, start), min(width - 1, stop)
    candidates = []
    for x in range(lo + sustain, hi - sustain):
        if side == "right":
            inside = float(np.median(column_level[x - sustain:x]))
            outside = float(np.median(column_level[x:x + sustain]))
        else:
            outside = float(np.median(column_level[x - sustain:x]))
            inside = float(np.median(column_level[x:x + sustain]))
        contrast = abs(inside - outside)
        outside_bg_distance = abs(outside - bg)
        inside_bg_distance = abs(inside - bg)
        if outside_bg_distance <= max(6.0, 3.0 * max(1.0, bg_mad)) and inside_bg_distance >= delta and contrast >= delta:
            candidates.append((x, contrast, inside, outside))
    if not candidates:
        return None, {
            "accepted": False, "reason": "no-sustained-background-transition",
            "background": round(bg, 3), "background_mad": round(bg_mad, 3),
            "required_delta": round(delta, 3),
        }
    # The qualifying transition nearest the physical image edge is the paper
    # boundary; interior folds/rules do not have sustained scanner background
    # on their outer side.
    chosen = max(candidates, key=lambda row: row[0]) if side == "right" else min(candidates, key=lambda row: row[0])
    x, contrast, inside, outside = chosen
    return float(x), {
        "accepted": True, "reason": "sustained-outer-background-transition",
        "boundary_x": float(x), "contrast": round(float(contrast), 3),
        "inside_level": round(float(inside), 3), "outside_level": round(float(outside), 3),
        "background": round(bg, 3), "background_mad": round(bg_mad, 3),
        "candidate_count": len(candidates),
    }


def _single_leaf_spread_completion(image_bgr: np.ndarray, primary: dict):
    """Complete one learned page leaf only when the image proves the missing spread edge.

    Some damaged open-volume pages yield one strong Doc-UFCN leaf and no usable
    partner component.  When that leaf already spans most of the document height,
    sits near one physical side, and leaves a large horizontal region unexplained,
    search the unexplained side for a persistent vertical source-image boundary.
    The learned polygon remains the seed; the missing side is synthesized only
    when independent image evidence supports a far outer page edge.
    """
    height, width = image_bgr.shape[:2]
    px0, py0, px1, py1 = primary["bounds"]
    span_x = max(1.0, px1 - px0)
    span_y = max(1.0, py1 - py0)
    margins = {
        "left": px0 / max(1.0, float(width)),
        "right": (width - px1) / max(1.0, float(width)),
        "top": py0 / max(1.0, float(height)),
        "bottom": (height - py1) / max(1.0, float(height)),
    }
    diagnostics = {
        "attempted": False,
        "decision": "not-applicable",
        "leaf_width_fraction": span_x / max(1.0, float(width)),
        "leaf_height_fraction": span_y / max(1.0, float(height)),
        "physical_margin_fractions": {k: round(float(v), 4) for k, v in margins.items()},
    }

    # A spread leaf must already explain the vertical page extent.  This keeps
    # local text/page fragments from becoming full-width documents.
    if span_y < 0.72 * height or margins["top"] > 0.18 or margins["bottom"] > 0.18:
        diagnostics["decision"] = "insufficient-vertical-page-support"
        return None, diagnostics
    if span_x > 0.72 * width:
        diagnostics["decision"] = "already-broad"
        return None, diagnostics

    side = None
    if margins["left"] <= 0.18 and margins["right"] >= 0.25:
        side = "right"
    elif margins["right"] <= 0.18 and margins["left"] >= 0.25:
        side = "left"
    if side is None:
        diagnostics["decision"] = "no-single-sided-spread-shape"
        return None, diagnostics
    diagnostics["attempted"] = True
    diagnostics["missing_side"] = side

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    y0 = max(0, int(round(py0 + 0.08 * span_y)))
    y1 = min(height, int(round(py1 - 0.08 * span_y)))
    if y1 <= y0 + 8:
        diagnostics["decision"] = "insufficient-boundary-sampling-height"
        return None, diagnostics
    profile = np.median(gx[y0:y1, :], axis=0).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (1, 0), sigmaX=2.0).reshape(-1)
    background = float(np.median(profile))
    mad = float(np.median(np.abs(profile - background)))

    minimum_gap = max(8, int(round(max(0.08 * width, 0.15 * span_x))))
    if side == "right":
        start = min(width - 1, int(round(px1)) + minimum_gap)
        stop = max(start + 1, int(round(width * 0.98)))
        outer_limit = 0.70 * width
    else:
        start = max(1, int(round(width * 0.02)))
        stop = max(start + 1, int(round(px0)) - minimum_gap)
        outer_limit = 0.30 * width
    if stop <= start:
        diagnostics["decision"] = "no-boundary-search-region"
        return None, diagnostics

    segment = profile[start:stop]
    if segment.size == 0:
        diagnostics["decision"] = "no-boundary-search-region"
        return None, diagnostics
    threshold = max(background + 3.0 * max(1.0, mad), background * 1.8, 6.0)

    # Prefer the farthest independently strong outer-sheet boundary rather than
    # the single strongest vertical edge.  Damaged spreads often contain a
    # strong interior fold/rule before the faint physical paper edge; choosing
    # argmax() alone stops the recovered envelope too early.  Restrict the
    # choice to robust peaks so ordinary text strokes cannot pull the envelope
    # outward.
    strong = np.flatnonzero(segment >= threshold)
    if strong.size:
        # Collapse adjacent above-threshold samples into runs and retain the
        # strongest sample from each run.
        runs = np.split(strong, np.where(np.diff(strong) > 1)[0] + 1)
        peaks = []
        for run in runs:
            if not len(run):
                continue
            best = int(run[np.argmax(segment[run])])
            peaks.append(best)
        local_index = (max(peaks) if side == "right" else min(peaks)) if peaks else int(np.argmax(segment))
        selection = "outermost-robust-boundary"
    else:
        local_index = int(np.argmax(segment))
        selection = "strongest-boundary-fallback"
    boundary_x = float(start + local_index)
    score = float(segment[local_index])

    # A torn/faded physical edge may be weaker than an interior fold in the
    # Sobel profile.  Independently look for a sustained transition into the
    # scanner background and prefer that farther physical boundary when proven.
    background_boundary_x, background_diagnostics = _outer_background_boundary(
        gray, side=side, y0=y0, y1=y1, start=start, stop=stop
    )
    diagnostics["outer_background_boundary"] = background_diagnostics
    if background_boundary_x is not None:
        minimum_override = max(6.0, 0.005 * width)
        farther = (
            background_boundary_x >= boundary_x + minimum_override
            if side == "right" else background_boundary_x <= boundary_x - minimum_override
        )
        if farther:
            boundary_x = float(background_boundary_x)
            # Background-transition proof is independent of the Sobel threshold;
            # preserve the observed Sobel score only for diagnostics.
            selection = "outer-background-transition"

    outer_enough = boundary_x >= outer_limit if side == "right" else boundary_x <= outer_limit
    combined_x0 = min(px0, boundary_x)
    combined_x1 = max(px1, boundary_x)
    span_gain = (combined_x1 - combined_x0) / span_x
    diagnostics.update({
        "boundary_x": round(boundary_x, 3),
        "boundary_score": round(score, 4),
        "boundary_background": round(background, 4),
        "boundary_mad": round(mad, 4),
        "boundary_threshold": round(threshold, 4),
        "boundary_contrast_ratio": round(score / max(1.0, background), 4),
        "boundary_selection": selection,
        "robust_boundary_candidates": int(len(peaks)) if strong.size else 0,
        "outer_boundary": bool(outer_enough),
        "span_gain": round(float(span_gain), 4),
    })
    boundary_proven = score >= threshold or selection == "outer-background-transition"
    if not boundary_proven or not outer_enough or span_gain < 1.30:
        diagnostics["decision"] = "image-boundary-not-proven"
        return None, diagnostics

    polygon = np.asarray(primary["polygon"], dtype=np.float32).reshape(-1, 2)
    support = np.asarray([[boundary_x, py0], [boundary_x, py1]], dtype=np.float32)
    completed = cv2.convexHull(np.concatenate([polygon, support], axis=0).reshape(-1, 1, 2)).reshape(-1, 2)
    completed_area = abs(float(cv2.contourArea(completed)))
    diagnostics["decision"] = "image-supported-single-leaf-spread-completion"
    diagnostics["completed_area"] = completed_area
    diagnostics["completed_bounds"] = [
        round(float(combined_x0), 3), round(float(py0), 3),
        round(float(combined_x1), 3), round(float(py1), 3),
    ]
    return (completed_area, float(primary["confidence"]), completed), diagnostics


def _select_page_envelope(image_bgr: np.ndarray, values):
    """Select a Doc-UFCN page envelope, joining credible facing-page leaves.

    Doc-UFCN can emit separate class-page polygons for the two leaves of an
    open historical volume.  Treating only the largest leaf as the physical
    document truncates roughly half the spread.  A second component is joined
    only when it is substantial, vertically coextensive, similarly tall, and
    materially expands the horizontal document span.  The rule is intentionally
    parameter-free and leaves ordinary single-page predictions untouched.
    """
    height, width = image_bgr.shape[:2]
    candidates = _qualifying_polygons(image_bgr, values)
    diagnostics = {
        "qualifying_component_count": len(candidates),
        "decision": "no-component" if not candidates else "single-component",
        "joined_component_count": 0,
    }
    if not candidates:
        return None, diagnostics

    primary = candidates[0]
    joined = [primary]
    px0, py0, px1, py1 = primary["bounds"]
    primary_width = max(1.0, px1 - px0)
    primary_height = max(1.0, py1 - py0)
    primary_center_x = (px0 + px1) * 0.5

    accepted = []
    for other in candidates[1:]:
        ox0, oy0, ox1, oy1 = other["bounds"]
        other_height = max(1.0, oy1 - oy0)
        overlap = max(0.0, min(py1, oy1) - max(py0, oy0))
        vertical_overlap = overlap / max(1.0, min(primary_height, other_height))
        height_ratio = min(primary_height, other_height) / max(primary_height, other_height)
        other_center_x = (ox0 + ox1) * 0.5
        center_separation = abs(other_center_x - primary_center_x)
        substantial = other["area"] >= primary["area"] * 0.12
        horizontally_distinct = center_separation >= max(width * 0.12, primary_width * 0.30)
        combined_x0, combined_x1 = min(px0, ox0), max(px1, ox1)
        span_gain = (combined_x1 - combined_x0) / primary_width
        if (
            substantial
            and vertical_overlap >= 0.55
            and height_ratio >= 0.55
            and horizontally_distinct
            and span_gain >= 1.30
        ):
            joined.append(other)
            accepted.append({
                "area_fraction_of_primary": other["area"] / max(1.0, primary["area"]),
                "vertical_overlap": vertical_overlap,
                "height_ratio": height_ratio,
                "horizontal_center_separation_fraction": center_separation / max(1.0, float(width)),
                "span_gain": span_gain,
            })

    if len(joined) == 1:
        completed, completion_diagnostics = _single_leaf_spread_completion(image_bgr, primary)
        diagnostics["single_leaf_spread_completion"] = completion_diagnostics
        if completed is not None:
            area, confidence, polygon = completed
            diagnostics.update({
                "decision": "image-supported-single-leaf-spread-completion",
                "selected_confidence": confidence,
                "selected_component_area": area,
            })
            return completed, diagnostics
        diagnostics.update({
            "selected_confidence": primary["confidence"],
            "selected_component_area": primary["area"],
        })
        return (primary["area"], primary["confidence"], primary["polygon"]), diagnostics

    points = np.concatenate([item["polygon"] for item in joined], axis=0).astype(np.float32)
    hull = cv2.convexHull(points.reshape(-1, 1, 2)).reshape(-1, 2)
    hull_area = abs(float(cv2.contourArea(hull)))
    confidence = float(min(item["confidence"] for item in joined))
    diagnostics.update({
        "decision": "multi-component-spread-envelope",
        "joined_component_count": len(joined),
        "joined_components": accepted,
        "selected_confidence": confidence,
        "selected_component_area": hull_area,
    })
    return (hull_area, confidence, hull), diagnostics


def detect(*, image_bgr, mask, parameters=None):
    del mask
    values = _parameters(parameters)
    height, width = image_bgr.shape[:2]
    image_area = float(max(1, height * width))
    provenance = _provenance()
    selected, envelope_diagnostics = _select_page_envelope(image_bgr, values)
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
        "page_envelope": envelope_diagnostics,
    }
    if selected is None:
        return Candidate(METHOD, None, None, 0, 0, {**diagnostics, "reason": "no_doc_ufcn_page_polygon"}, status="no_candidate")

    area, confidence, polygon = selected
    area_fraction = area / image_area
    diagnostics.update({"selected_confidence": confidence, "selected_area_fraction": area_fraction})
    if area_fraction < values["minimum_page_area_fraction"]:
        return Candidate(METHOD, None, None, 0, 0, {**diagnostics, "reason": "doc_ufcn_page_polygon_too_small"}, status="no_candidate")

    raw_corners = cv2.boxPoints(cv2.minAreaRect(polygon)).astype(np.float32)
    corners, padding_diagnostics = _boundary_supported_padding(
        image_bgr, raw_corners, requested_fraction=values["page_padding_fraction"]
    )
    diagnostics["padding_arbitration"] = padding_diagnostics
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
