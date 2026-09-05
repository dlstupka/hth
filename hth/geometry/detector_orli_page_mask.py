from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import threading
import time
import warnings
from typing import Any

import cv2
import numpy as np

from .model import Candidate
try:
    # Package mode: imported as hth.geometry.*.
    from hth.thread_safe_stderr import capture_native_stderr
except ModuleNotFoundError as exc:
    # Script mode: hth/detect_geometry_candidates.py is executed directly,
    # so the hth/ directory itself is on sys.path rather than its parent.
    if exc.name != "hth":
        raise
    from thread_safe_stderr import capture_native_stderr

METHOD = "orli_page_mask"
MODEL_ENV = "HTH_ORLI_PAGE_MODEL"
PROVENANCE_ENV = "HTH_ORLI_PAGE_PROVENANCE"

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
_EVIDENCE_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_EVIDENCE_CACHE_LOCK = threading.Lock()
_EVIDENCE_CACHE_LIMIT = 16

_RUNTIME_DIAGNOSTICS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_RUNTIME_DIAGNOSTICS_LOCK = threading.Lock()


@contextmanager
def _capture_orli_runtime_chatter():
    """Capture known native Orli/GEOS chatter through HTH's global fd2 lock.

    fd 2 is process-global, so all detector/runtime redirection shares the same
    lock in ``hth.thread_safe_stderr``. Parameter threads normally never enter
    this path because learned Golden Set evidence is precomputed before fan-out.
    """
    diagnostics = {
        "lightning_srun_advisories": 0,
        "orli_runtime_warnings": 0,
        "filtered_messages": [],
    }
    with capture_native_stderr() as captured:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*The `srun` command is available on your system but is not used.*",
            )
            yield diagnostics

        captured.seek(0)
        text = captured.read().decode("utf-8", errors="replace")

    replay = []
    for line in text.splitlines():
        stripped = line.strip()
        if "The `srun` command is available on your system but is not used" in stripped:
            diagnostics["lightning_srun_advisories"] += 1
            if len(diagnostics["filtered_messages"]) < 8:
                diagnostics["filtered_messages"].append(stripped)
            continue
        if "TopologyException: side location conflict" in stripped or "Polygonizer failed on line" in stripped:
            diagnostics["orli_runtime_warnings"] += 1
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
        raise ValueError(f"Unknown Orli Page-Mask parameters: {', '.join(unknown)}")
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
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # Orli is a Kraken 7 model plugin. Keep the immutable model path as the
        # cached model identity; orli.pred.segment owns model construction.
        import orli.pred  # noqa: F401 - fail PREPARE/inference early if plugin is broken
        _MODEL = path
        _MODEL_KEY = key
        print(f"Orli Page-Mask ready for inference: model={path.name} backend=orli-plugin device=cpu")
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
        from orli.pred import segment

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        model = _load_model()
        with _capture_orli_runtime_chatter() as runtime_diagnostics:
            segmentation = segment(image, str(model))
        evidence = _freeze_evidence(_extract_evidence(segmentation))
        _store_runtime_diagnostics(key, runtime_diagnostics)

        with _EVIDENCE_CACHE_LOCK:
            _EVIDENCE_CACHE[key] = evidence
            _EVIDENCE_CACHE.move_to_end(key)
            while len(_EVIDENCE_CACHE) > _EVIDENCE_CACHE_LIMIT:
                _EVIDENCE_CACHE.popitem(last=False)
        return evidence


def precompute_golden_set_evidence(images, *, progress=None):
    """Populate immutable Orli evidence once before parameter-thread fan-out."""
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
    """Persist one process-independent immutable Orli Golden Set evidence set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    total = len(images)
    for index, image_bgr in enumerate(images, 1):
        key = _image_key(image_bgr)
        started = time.perf_counter()
        if progress is not None:
            progress("start", index, total, key, 0.0)
        # Capture the returned immutable snapshot immediately. The process cache
        # is intentionally bounded and may evict early Golden Set pages before a
        # collection larger than _EVIDENCE_CACHE_LIMIT finishes precomputation.
        evidence = _infer_evidence(image_bgr)
        records.append({
            "image_key": key,
            "evidence": evidence,
            "runtime_diagnostics": _runtime_diagnostics_for(key),
        })
        if progress is not None:
            progress("finish", index, total, key, time.perf_counter() - started)
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
    """Load parent-precomputed Orli evidence without loading/running Orli."""
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
        raise ValueError(f"Shared Orli evidence is missing {len(missing)} Golden Set page(s)")
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


def _select_evidence_envelope(contours, *, image_area):
    """Return the contour set that best represents the learned page extent.

    Orli can emit several disconnected regions on sparse or damaged pages.
    Taking only the largest postprocessed contour can collapse the proposal onto
    a single text block.  Keep the dominant component, then admit substantial
    disconnected components that are large enough to be document evidence
    rather than isolated noise.  This is deliberately parameter-free so it
    improves envelope reconstruction without changing the declared calibration
    grid.
    """
    if not contours:
        return [], {
            "mode": "none",
            "external_contours": 0,
            "selected_contours": 0,
        }

    ranked = sorted(contours, key=cv2.contourArea, reverse=True)
    dominant_area = float(cv2.contourArea(ranked[0]))
    absolute_floor = max(4.0, float(image_area) * 0.00025)
    relative_floor = dominant_area * 0.015
    area_floor = max(absolute_floor, relative_floor)

    selected = [ranked[0]]
    selected.extend(
        contour for contour in ranked[1:]
        if float(cv2.contourArea(contour)) >= area_floor
    )

    # If secondary learned components survive the conservative area gate, use
    # their joint convex envelope. Otherwise preserve the historic dominant-
    # contour behavior exactly.
    if len(selected) == 1:
        return selected, {
            "mode": "dominant",
            "external_contours": len(ranked),
            "selected_contours": 1,
            "dominant_area": dominant_area,
            "component_area_floor": area_floor,
        }

    points = np.concatenate(selected, axis=0)
    hull = cv2.convexHull(points)
    return [hull], {
        "mode": "multi-region-envelope",
        "external_contours": len(ranked),
        "selected_contours": len(selected),
        "dominant_area": dominant_area,
        "component_area_floor": area_floor,
    }



def _learned_geometry_envelope(evidence, *, image_shape):
    """Build a robust outer envelope directly from Orli layout geometry.

    Orli's historical base model can emit hundreds of disconnected baselines
    without line polygons or regions. Rasterizing those baselines and then
    selecting a connected contour can therefore collapse a page proposal onto
    one dense text block or one column.  The learned geometry itself already
    carries the global document extent, so construct a consensus envelope from
    all substantial baseline/line/region geometry before morphology can break
    it into components.

    The filter is deliberately parameter-free: it removes only extremely short
    baseline fragments relative to the page scale, then fits the minimum-area
    rectangle around the surviving learned points.  Calibration parameters
    continue to control only the existing morphology, padding, and area gate.
    """
    h, w = image_shape[:2]
    min_dim = float(max(1, min(h, w)))
    point_sets = []
    baseline_lengths = []

    for polygon in evidence.get("regions", ()):
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(pts) >= 3:
            point_sets.append(pts)

    for polygon in evidence.get("lines", ()):
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(pts) >= 3:
            point_sets.append(pts)

    baseline_sets = []
    for baseline in evidence.get("baselines", ()):
        pts = np.asarray(baseline, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 2:
            continue
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        baseline_sets.append((pts, length))
        baseline_lengths.append(length)

    if baseline_sets:
        # Reject only obvious tiny fragments.  Use both an image-scale floor and
        # a very small fraction of the median learned-line length so sparse short
        # entries remain represented while isolated model specks cannot dominate
        # the global envelope.
        median_length = float(np.median(baseline_lengths)) if baseline_lengths else 0.0
        length_floor = max(2.0, min_dim * 0.0025, median_length * 0.05)
        for pts, length in baseline_sets:
            if length >= length_floor:
                point_sets.append(pts)
    else:
        median_length = 0.0
        length_floor = 0.0

    if not point_sets:
        return None, {
            "available": False,
            "point_sets": 0,
            "baseline_count": len(baseline_sets),
            "baseline_length_floor": length_floor,
        }

    points = np.concatenate(point_sets, axis=0)
    if len(points) < 4 or not np.isfinite(points).all():
        return None, {
            "available": False,
            "point_sets": len(point_sets),
            "baseline_count": len(baseline_sets),
            "baseline_length_floor": length_floor,
        }

    # A convex hull makes the consensus insensitive to the order in which Orli
    # emitted individual baselines while retaining the true outer layout extent.
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    if hull is None or len(hull) < 3:
        return None, {
            "available": False,
            "point_sets": len(point_sets),
            "baseline_count": len(baseline_sets),
            "baseline_length_floor": length_floor,
        }

    rect = cv2.minAreaRect(hull)
    corners = _canonical_quad(cv2.boxPoints(rect), width=w, height=h)
    area = abs(float(cv2.contourArea(corners))) if corners is not None else 0.0
    return corners, {
        "available": corners is not None,
        "point_sets": len(point_sets),
        "point_count": int(len(points)),
        "baseline_count": len(baseline_sets),
        "baseline_median_length": median_length,
        "baseline_length_floor": length_floor,
        "area": area,
    }



def _learned_document_frame(evidence, *, image_shape):
    """Infer a page-oriented frame from broad Orli baseline layout support.

    Baselines describe text, not the physical paper edge.  When they span a
    substantial document axis, use their dominant orientation plus the source
    image margins on that axis to estimate the missing orthogonal page extent.
    This is deliberately parameter-free and conservative: narrow/localized
    evidence is rejected rather than extrapolated.
    """
    h, w = image_shape[:2]
    min_dim = float(max(1, min(h, w)))
    baselines = []
    lengths = []
    directions = []

    for baseline in evidence.get("baselines", ()):
        pts = np.asarray(baseline, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 2 or not np.isfinite(pts).all():
            continue
        segs = np.diff(pts, axis=0)
        length = float(np.linalg.norm(segs, axis=1).sum())
        if length <= 0.0:
            continue
        chord = pts[-1] - pts[0]
        chord_norm = float(np.linalg.norm(chord))
        if chord_norm <= 1e-6:
            continue
        baselines.append((pts, length))
        lengths.append(length)
        directions.append(chord / chord_norm)

    if len(baselines) < 4:
        return None, {
            "available": False,
            "reason": "insufficient-baselines",
            "baseline_count": len(baselines),
        }

    median_length = float(np.median(lengths)) if lengths else 0.0
    length_floor = max(2.0, min_dim * 0.0025, median_length * 0.05)
    substantial = [(pts, length, direction) for (pts, length), direction in zip(baselines, directions) if length >= length_floor]
    if len(substantial) < 4:
        return None, {
            "available": False,
            "reason": "insufficient-substantial-baselines",
            "baseline_count": len(baselines),
            "substantial_baselines": len(substantial),
            "baseline_length_floor": length_floor,
        }

    # Baseline orientation is axial: theta and theta+pi are equivalent.  Average
    # doubled angles so opposite endpoint order cannot cancel the estimate.
    angles = np.asarray([np.arctan2(direction[1], direction[0]) for _, _, direction in substantial], dtype=np.float64)
    mean_cos = float(np.mean(np.cos(2.0 * angles)))
    mean_sin = float(np.mean(np.sin(2.0 * angles)))
    orientation_coherence = float(np.hypot(mean_cos, mean_sin))
    theta = 0.5 * float(np.arctan2(mean_sin, mean_cos))
    u = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
    v = np.asarray([-u[1], u[0]], dtype=np.float64)

    points = np.concatenate([pts for pts, _, _ in substantial], axis=0).astype(np.float64)
    u_proj = points @ u
    v_proj = points @ v
    row_centers = np.asarray([float(np.median(pts.astype(np.float64) @ v)) for pts, _, _ in substantial], dtype=np.float64)

    image_corners = np.asarray([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]], dtype=np.float64)
    image_u = image_corners @ u
    image_v = image_corners @ v
    image_u_min, image_u_max = float(image_u.min()), float(image_u.max())
    image_v_min, image_v_max = float(image_v.min()), float(image_v.max())
    image_u_span = max(1e-6, image_u_max - image_u_min)
    image_v_span = max(1e-6, image_v_max - image_v_min)

    # Quantiles keep one bad Orli endpoint from defining the frame while still
    # retaining nearly all learned support.
    text_u_min, text_u_max = [float(x) for x in np.quantile(u_proj, [0.01, 0.99])]
    text_v_min, text_v_max = [float(x) for x in np.quantile(v_proj, [0.01, 0.99])]
    text_u_span = max(1e-6, text_u_max - text_u_min)
    text_v_span = max(1e-6, text_v_max - text_v_min)
    main_axis_coverage = text_u_span / image_u_span

    # Estimate typical row spacing only for diagnostics and a minimum sensible
    # margin.  Collapse near-duplicate row centers emitted by dense Orli output.
    sorted_rows = np.sort(row_centers)
    coarse_gap = max(1.0, min_dim * 0.002)
    collapsed = []
    for value in sorted_rows:
        if not collapsed or value - collapsed[-1] >= coarse_gap:
            collapsed.append(float(value))
        else:
            collapsed[-1] = (collapsed[-1] + float(value)) * 0.5
    row_diffs = np.diff(np.asarray(collapsed, dtype=np.float64)) if len(collapsed) >= 2 else np.asarray([], dtype=np.float64)
    row_diffs = row_diffs[row_diffs > coarse_gap]
    row_spacing = float(np.median(row_diffs)) if row_diffs.size else 0.0

    # Extrapolation is justified only when the learned text support is broad
    # along the document axis.  Local text blocks remain ordinary learned
    # envelopes and cannot manufacture a page-sized frame.
    broad_frame = main_axis_coverage >= 0.55
    coherent_borderline_frame = (
        main_axis_coverage >= 0.48
        and len(substantial) >= 8
        and orientation_coherence >= 0.92
    )
    if not (broad_frame or coherent_borderline_frame):
        return None, {
            "available": False,
            "reason": "insufficient-main-axis-coverage",
            "baseline_count": len(baselines),
            "substantial_baselines": len(substantial),
            "dominant_angle_degrees": float(np.degrees(theta)),
            "orientation_coherence": orientation_coherence,
            "main_axis_coverage": main_axis_coverage,
            "row_spacing": row_spacing,
        }

    left_margin = max(0.0, text_u_min - image_u_min)
    right_margin = max(0.0, image_u_max - text_u_max)
    side_margin = float(np.median([left_margin, right_margin]))
    minimum_margin = max(min_dim * 0.02, row_spacing * 1.5 if row_spacing > 0.0 else 0.0)
    maximum_margin = min_dim * 0.18
    inferred_margin = float(np.clip(side_margin, minimum_margin, maximum_margin))

    # Keep the observed main-axis text extent; infer only the orthogonal paper
    # extent from the broad text-frame side margins. Existing page padding still
    # provides the calibrated final expansion after arbitration.
    frame_u_min, frame_u_max = text_u_min, text_u_max
    frame_v_min = image_v_min + inferred_margin
    frame_v_max = image_v_max - inferred_margin
    frame_v_min = min(frame_v_min, text_v_min)
    frame_v_max = max(frame_v_max, text_v_max)
    if frame_v_max - frame_v_min < text_v_span:
        frame_v_min, frame_v_max = text_v_min, text_v_max

    uv = np.asarray([
        [frame_u_min, frame_v_min],
        [frame_u_max, frame_v_min],
        [frame_u_max, frame_v_max],
        [frame_u_min, frame_v_max],
    ], dtype=np.float64)
    # [u v] is an orthonormal basis, so x/y = u*u_coord + v*v_coord.
    xy = uv[:, :1] * u[None, :] + uv[:, 1:] * v[None, :]
    corners = _canonical_quad(xy.astype(np.float32), width=w, height=h)
    area = abs(float(cv2.contourArea(corners))) if corners is not None else 0.0
    return corners, {
        "available": corners is not None,
        "reason": (
            "broad-baseline-document-frame"
            if broad_frame and corners is not None
            else "coherent-borderline-baseline-document-frame"
            if corners is not None
            else "invalid-frame"
        ),
        "baseline_count": len(baselines),
        "substantial_baselines": len(substantial),
        "baseline_length_floor": length_floor,
        "baseline_median_length": median_length,
        "dominant_angle_degrees": float(np.degrees(theta)),
        "orientation_coherence": orientation_coherence,
        "main_axis_coverage": main_axis_coverage,
        "text_main_span": text_u_span,
        "text_cross_span": text_v_span,
        "image_main_span": image_u_span,
        "image_cross_span": image_v_span,
        "row_spacing": row_spacing,
        "side_margins": {"leading": left_margin, "trailing": right_margin},
        "inferred_cross_margin": inferred_margin,
        "area": area,
    }



def _quad_axis_bounds(corners):
    if corners is None:
        return None
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4 or not np.isfinite(pts).all():
        return None
    return {
        "left": float(pts[:, 0].min()),
        "top": float(pts[:, 1].min()),
        "right": float(pts[:, 0].max()),
        "bottom": float(pts[:, 1].max()),
        "width": float(pts[:, 0].max() - pts[:, 0].min()),
        "height": float(pts[:, 1].max() - pts[:, 1].min()),
    }


def _arbitrate_envelopes(
    contour_corners, learned_corners, *, image_shape, frame_corners=None, frame_diagnostics=None
):
    """Choose between morphology and global learned geometry without area bias.

    A wide-but-truncated contour can have more area than the Orli geometry that
    actually spans the document.  Prefer the learned envelope when it contributes
    a materially larger document-axis span while retaining a substantial fraction
    of the contour's orthogonal span.  Otherwise retain the historic area fallback.

    This is intentionally parameter-free: it changes envelope arbitration only,
    not the calibration search space or persisted neural evidence contract.
    """
    h, w = image_shape[:2]
    contour_bounds = _quad_axis_bounds(contour_corners)
    learned_bounds = _quad_axis_bounds(learned_corners)
    contour_area = abs(float(cv2.contourArea(contour_corners))) if contour_corners is not None else 0.0
    learned_area = abs(float(cv2.contourArea(learned_corners))) if learned_corners is not None else 0.0
    frame_bounds = _quad_axis_bounds(frame_corners)
    frame_area = abs(float(cv2.contourArea(frame_corners))) if frame_corners is not None else 0.0

    diagnostics = {
        "contour_bounds": contour_bounds,
        "learned_bounds": learned_bounds,
        "contour_area": contour_area,
        "learned_area": learned_area,
        "frame_bounds": frame_bounds,
        "frame_area": frame_area,
        "decision": "none",
        "reason": "no-envelope",
    }
    if learned_corners is None:
        diagnostics.update(decision="contour", reason="learned-unavailable")
        return contour_corners, "contour", diagnostics
    if contour_corners is None:
        if frame_corners is not None:
            diagnostics.update(decision="frame", reason="contour-unavailable-document-frame")
            return frame_corners, "frame", diagnostics
        diagnostics.update(decision="learned", reason="contour-unavailable")
        return learned_corners, "learned", diagnostics

    min_dim = float(max(1, min(h, w)))

    # A learned document frame represents inferred paper extent rather than only
    # observed text support. Prefer it when it materially restores the cross-axis
    # span while retaining substantial contour extent on the document axis.
    if frame_corners is not None and frame_bounds is not None:
        c_width0 = max(1e-6, float(contour_bounds["width"]))
        c_height0 = max(1e-6, float(contour_bounds["height"]))
        f_width = max(1e-6, float(frame_bounds["width"]))
        f_height = max(1e-6, float(frame_bounds["height"]))
        width_ratio0 = f_width / c_width0
        height_ratio0 = f_height / c_height0
        diagnostics.update({
            "frame_to_contour_width_ratio": width_ratio0,
            "frame_to_contour_height_ratio": height_ratio0,
        })
        frame_reason = str((frame_diagnostics or {}).get("reason") or "")
        borderline = frame_reason == "coherent-borderline-baseline-document-frame"
        ratio_floor = 1.04 if borderline else 1.08
        pixel_floor = max(4.0, min_dim * (0.015 if borderline else 0.02))
        diagnostics.update({
            "frame_reason": frame_reason or None,
            "frame_extent_ratio_floor": ratio_floor,
            "frame_extent_pixel_floor": pixel_floor,
        })
        frame_horizontal = width_ratio0 >= ratio_floor and (f_width - c_width0) >= pixel_floor and height_ratio0 >= 0.60
        frame_vertical = height_ratio0 >= ratio_floor and (f_height - c_height0) >= pixel_floor and width_ratio0 >= 0.60
        if frame_horizontal or frame_vertical:
            axis = "both" if frame_horizontal and frame_vertical else "horizontal" if frame_horizontal else "vertical"
            diagnostics.update(decision="frame", reason=f"extrapolated-{axis}-document-extent")
            return frame_corners, "frame", diagnostics
    material_pixels = max(4.0, min_dim * 0.02)
    c_width = max(1e-6, float(contour_bounds["width"]))
    c_height = max(1e-6, float(contour_bounds["height"]))
    l_width = max(1e-6, float(learned_bounds["width"]))
    l_height = max(1e-6, float(learned_bounds["height"]))
    width_ratio = l_width / c_width
    height_ratio = l_height / c_height
    diagnostics.update({
        "material_span_pixels": material_pixels,
        "learned_to_contour_width_ratio": width_ratio,
        "learned_to_contour_height_ratio": height_ratio,
        "learned_extensions": {
            "left": max(0.0, contour_bounds["left"] - learned_bounds["left"]),
            "top": max(0.0, contour_bounds["top"] - learned_bounds["top"]),
            "right": max(0.0, learned_bounds["right"] - contour_bounds["right"]),
            "bottom": max(0.0, learned_bounds["bottom"] - contour_bounds["bottom"]),
        },
    })

    # A >=8% span gain plus an absolute 2%-of-page gain is enough to identify
    # the collapse seen in verbose Orli runs. Require at least 60% retention of
    # the orthogonal contour span so a single remote outlier cannot win merely by
    # stretching one axis.
    width_gain = l_width - c_width
    height_gain = l_height - c_height
    learned_has_horizontal_extent = width_ratio >= 1.08 and width_gain >= material_pixels and height_ratio >= 0.60
    learned_has_vertical_extent = height_ratio >= 1.08 and height_gain >= material_pixels and width_ratio >= 0.60
    if learned_has_horizontal_extent or learned_has_vertical_extent:
        axis = "both" if learned_has_horizontal_extent and learned_has_vertical_extent else "horizontal" if learned_has_horizontal_extent else "vertical"
        diagnostics.update(decision="learned", reason=f"material-{axis}-extent")
        return learned_corners, "learned", diagnostics

    if learned_area > contour_area:
        diagnostics.update(decision="learned", reason="larger-area-fallback")
        return learned_corners, "learned", diagnostics

    diagnostics.update(decision="contour", reason="contour-retains-document-extent")
    return contour_corners, "contour", diagnostics



def _directional_page_completion(corners, *, image_shape):
    """Complete missing page sides from strong image-edge anchors.

    Orli sometimes describes only an upper/side portion of a historical page:
    the observed envelope can have an accurate top/right (or top/left) edge but
    terminate hundreds of pixels before the opposite paper boundaries.  When a
    proposal is anchored near at least two source-image sides, complete only the
    materially truncated axes by mirroring the trusted opposite margin in the
    proposal's own orthogonal basis.

    The rule is deliberately parameter-free and conservative.  It does not grow
    broad proposals, and a localized text block with fewer than two edge anchors
    is never promoted to a page-sized quadrilateral.
    """
    if corners is None:
        return None, {
            "available": False,
            "reason": "no-envelope",
            "changed_axes": [],
        }

    h, w = image_shape[:2]
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if len(pts) != 4 or not np.isfinite(pts).all():
        return corners, {
            "available": False,
            "reason": "invalid-envelope",
            "changed_axes": [],
        }

    rect = cv2.minAreaRect(pts.astype(np.float32))
    box = np.asarray(cv2.boxPoints(rect), dtype=np.float64)
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    if not np.isfinite(lengths).all() or float(lengths.max()) <= 1e-6:
        return corners, {
            "available": False,
            "reason": "degenerate-envelope",
            "changed_axes": [],
        }

    # Use one rectangle edge as u and its perpendicular as v.  The subsequent
    # image projections make the logic orientation-independent.
    edge_index = int(np.argmax(lengths))
    u = edges[edge_index] / max(float(lengths[edge_index]), 1e-6)
    v = np.asarray([-u[1], u[0]], dtype=np.float64)

    image_corners = np.asarray(
        [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
        dtype=np.float64,
    )
    image_u = image_corners @ u
    image_v = image_corners @ v
    cand_u = pts @ u
    cand_v = pts @ v

    axes = {
        "u": {
            "image_min": float(image_u.min()),
            "image_max": float(image_u.max()),
            "cand_min": float(cand_u.min()),
            "cand_max": float(cand_u.max()),
        },
        "v": {
            "image_min": float(image_v.min()),
            "image_max": float(image_v.max()),
            "cand_min": float(cand_v.min()),
            "cand_max": float(cand_v.max()),
        },
    }

    anchor_fraction = 0.14
    truncated_span_fraction = 0.72
    missing_fraction = 0.25
    for axis in axes.values():
        axis["image_span"] = max(1e-6, axis["image_max"] - axis["image_min"])
        axis["span"] = max(1e-6, axis["cand_max"] - axis["cand_min"])
        axis["span_fraction"] = axis["span"] / axis["image_span"]
        axis["low_margin"] = max(0.0, axis["cand_min"] - axis["image_min"])
        axis["high_margin"] = max(0.0, axis["image_max"] - axis["cand_max"])
        axis["low_anchor"] = axis["low_margin"] <= anchor_fraction * axis["image_span"]
        axis["high_anchor"] = axis["high_margin"] <= anchor_fraction * axis["image_span"]

    anchor_count = sum(
        int(axis[side])
        for axis in axes.values()
        for side in ("low_anchor", "high_anchor")
    )
    diagnostics = {
        "available": True,
        "reason": "no-directional-completion",
        "anchor_count": anchor_count,
        "anchor_fraction": anchor_fraction,
        "truncated_span_fraction": truncated_span_fraction,
        "missing_fraction": missing_fraction,
        "axes": axes,
        "changed_axes": [],
    }

    # A strongly localized corner fragment can defeat the rotated-basis rule:
    # projection onto the proposal basis may make one physical image-edge anchor
    # look broad enough that only one axis is completed.  Recognize the stricter
    # physical case directly: two adjacent image-side anchors, with both opposite
    # dimensions materially absent.  This is not a generic localized-block
    # promotion; the observed fragment must actually touch a source-image corner.
    bounds = _quad_axis_bounds(corners)
    physical = {
        "left_margin": max(0.0, float(bounds["left"])),
        "top_margin": max(0.0, float(bounds["top"])),
        "right_margin": max(0.0, float((w - 1) - bounds["right"])),
        "bottom_margin": max(0.0, float((h - 1) - bounds["bottom"])),
    }
    x_anchor = anchor_fraction * max(1.0, float(w - 1))
    y_anchor = anchor_fraction * max(1.0, float(h - 1))
    x_missing = missing_fraction * max(1.0, float(w - 1))
    y_missing = missing_fraction * max(1.0, float(h - 1))
    corner_cases = (
        ("top-left", "left_margin", "top_margin", "right_margin", "bottom_margin"),
        ("top-right", "right_margin", "top_margin", "left_margin", "bottom_margin"),
        ("bottom-right", "right_margin", "bottom_margin", "left_margin", "top_margin"),
        ("bottom-left", "left_margin", "bottom_margin", "right_margin", "top_margin"),
    )
    for corner_name, x_near, y_near, x_far, y_far in corner_cases:
        if (
            physical[x_near] <= x_anchor
            and physical[y_near] <= y_anchor
            and physical[x_far] >= x_missing
            and physical[y_far] >= y_missing
        ):
            left = physical["left_margin"] if x_near == "left_margin" else physical[x_near]
            right = physical["right_margin"] if x_near == "right_margin" else physical[x_near]
            top = physical["top_margin"] if y_near == "top_margin" else physical[y_near]
            bottom = physical["bottom_margin"] if y_near == "bottom_margin" else physical[y_near]
            # Preserve the trusted corner margins and mirror them to the two
            # missing sides.  The final canonicalization clips to the image.
            if "left" in corner_name:
                x0, x1 = left, (w - 1) - left
            else:
                x0, x1 = right, (w - 1) - right
            if "top" in corner_name:
                y0, y1 = top, (h - 1) - top
            else:
                y0, y1 = bottom, (h - 1) - bottom
            corner_completed = _canonical_quad(
                np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32),
                width=w, height=h,
            )
            if corner_completed is not None:
                diagnostics.update({
                    "reason": "corner-anchored-two-axis-completion",
                    "corner_anchor": corner_name,
                    "physical_margins": physical,
                    "changed_axes": [x_far.replace("_margin", ""), y_far.replace("_margin", "")],
                    "completed_bounds": _quad_axis_bounds(corner_completed),
                    "completed_area": abs(float(cv2.contourArea(corner_completed))),
                })
                return corner_completed, diagnostics

    # The physical-corner fallback is intentionally independent of the rotated
    # proposal-basis anchor count.  A localized, rotated upper-right (or other
    # corner) fragment can project to only one trusted basis-side even though its
    # axis-aligned bounds clearly touch two adjacent source-image sides.  The
    # previous early return made that fallback unreachable for exactly this
    # HTH-0001 page-10 failure mode.  If no physical corner qualified, preserve
    # the stricter historic rule before attempting ordinary one-axis completion.
    if anchor_count < 2:
        diagnostics["reason"] = "insufficient-edge-anchors"
        diagnostics["physical_margins"] = physical
        return corners, diagnostics

    completed = {name: dict(axis) for name, axis in axes.items()}
    changed = []
    for name, axis in axes.items():
        if axis["span_fraction"] >= truncated_span_fraction:
            continue
        large_missing = missing_fraction * axis["image_span"]
        # If one side is trusted and the opposite side is materially absent,
        # mirror the trusted margin across the image projection.  This converts
        # an upper/right text fragment into a full page proposal without making
        # any claim about unanchored localized blocks.
        if axis["low_anchor"] and axis["high_margin"] >= large_missing:
            completed[name]["cand_max"] = axis["image_max"] - axis["low_margin"]
            changed.append(f"{name}-high")
        elif axis["high_anchor"] and axis["low_margin"] >= large_missing:
            completed[name]["cand_min"] = axis["image_min"] + axis["high_margin"]
            changed.append(f"{name}-low")

    if not changed:
        diagnostics["reason"] = "anchored-envelope-not-materially-truncated"
        return corners, diagnostics

    uv = np.asarray([
        [completed["u"]["cand_min"], completed["v"]["cand_min"]],
        [completed["u"]["cand_max"], completed["v"]["cand_min"]],
        [completed["u"]["cand_max"], completed["v"]["cand_max"]],
        [completed["u"]["cand_min"], completed["v"]["cand_max"]],
    ], dtype=np.float64)
    xy = uv[:, :1] * u[None, :] + uv[:, 1:] * v[None, :]
    completed_corners = _canonical_quad(xy.astype(np.float32), width=w, height=h)
    if completed_corners is None:
        diagnostics["reason"] = "invalid-directional-completion"
        return corners, diagnostics

    diagnostics.update({
        "reason": "directional-edge-anchor-completion",
        "changed_axes": changed,
        "completed_bounds": _quad_axis_bounds(completed_corners),
        "completed_area": abs(float(cv2.contourArea(completed_corners))),
    })
    return completed_corners, diagnostics




def _axis_edge_profile(image_bgr):
    """Return cached image-gradient profiles for conservative page-edge recovery.

    The learned Orli fragment can identify document orientation/location without
    observing the whole physical sheet.  This image-derived evidence is used only
    as a late fallback to *prove* missing physical boundaries; it never replaces
    the learned detector as the source of the proposal.
    """
    key = _image_key(image_bgr)
    cache_key = f"boundary-profile:{key}"
    with _RUNTIME_DIAGNOSTICS_LOCK:
        cached = _RUNTIME_DIAGNOSTICS.get(cache_key)
        if isinstance(cached, dict) and "x_profile" in cached and "y_profile" in cached:
            return cached

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    # A physical paper/background transition persists across a substantial part
    # of the orthogonal image dimension.  Use the 75th percentile rather than a
    # simple mean so dense text strokes cannot dominate a boundary profile.
    x_profile = np.percentile(gx, 75.0, axis=0).astype(np.float32)
    y_profile = np.percentile(gy, 75.0, axis=1).astype(np.float32)
    smooth = max(3, int(round(min(gray.shape) * 0.006)))
    if smooth % 2 == 0:
        smooth += 1
    kernel = np.ones(smooth, dtype=np.float32) / float(smooth)
    x_profile = np.convolve(x_profile, kernel, mode="same").astype(np.float32)
    y_profile = np.convolve(y_profile, kernel, mode="same").astype(np.float32)

    payload = {
        "x_profile": x_profile,
        "y_profile": y_profile,
        "smooth_window": smooth,
    }
    with _RUNTIME_DIAGNOSTICS_LOCK:
        _RUNTIME_DIAGNOSTICS[cache_key] = payload
        _RUNTIME_DIAGNOSTICS.move_to_end(cache_key)
        # Boundary profiles are image-static and tiny compared with model
        # evidence.  Keep a modest independent allowance without disturbing the
        # existing runtime-warning diagnostics.
        while sum(1 for k in _RUNTIME_DIAGNOSTICS if str(k).startswith("boundary-profile:")) > 16:
            for old_key in list(_RUNTIME_DIAGNOSTICS):
                if str(old_key).startswith("boundary-profile:") and old_key != cache_key:
                    _RUNTIME_DIAGNOSTICS.pop(old_key, None)
                    break
    return payload


def _robust_profile_peak(profile, lo, hi, *, side, dimension):
    """Return a strong outer-image edge peak, or None when evidence is weak."""
    n = int(len(profile))
    lo = max(0, min(n - 1, int(round(lo))))
    hi = max(lo + 1, min(n, int(round(hi))))
    values = np.asarray(profile[lo:hi], dtype=np.float64)
    if values.size < 4 or not np.isfinite(values).all():
        return None

    median = float(np.median(profile))
    mad = float(np.median(np.abs(np.asarray(profile, dtype=np.float64) - median)))
    robust_scale = max(1e-6, 1.4826 * mad)
    positions = np.arange(lo, hi, dtype=np.float64)
    z = (values - median) / robust_scale

    # Favor physically plausible outer-sheet boundaries without forcing the
    # outermost image edge.  This keeps the fallback useful when the capture has
    # a modest black/background border around the actual paper.
    if side in {"left", "top"}:
        outer = 1.0 - positions / max(1.0, float(dimension - 1))
    else:
        outer = positions / max(1.0, float(dimension - 1))
    score = z + 0.75 * outer
    index = int(np.argmax(score))
    peak_z = float(z[index])
    peak = int(round(positions[index]))
    if peak_z < 3.0:
        return None
    return {
        "position": peak,
        "z": peak_z,
        "profile": float(values[index]),
        "median": median,
        "robust_scale": robust_scale,
    }


def _image_supported_boundary_recovery(image_bgr, corners):
    """Recover missing physical page sides using raw-image boundary evidence.

    This is a late, parameter-free fallback for the remaining Orli failure mode:
    a coherent learned fragment identifies the document but covers only a small
    portion of the physical page.  The routine does *not* extrapolate blindly.
    It keeps well-supported seed sides and extends only missing sides for which a
    strong image-gradient boundary is independently observed.
    """
    if corners is None:
        return corners, {"available": False, "reason": "no-envelope", "recovered_sides": []}

    h, w = image_bgr.shape[:2]
    bounds = _quad_axis_bounds(corners)
    if bounds is None:
        return corners, {"available": False, "reason": "invalid-envelope", "recovered_sides": []}

    # The fallback is intentionally limited to near-axis-aligned historical
    # captures.  Rotated geometry remains governed by the learned/frame logic.
    rect = cv2.minAreaRect(np.asarray(corners, dtype=np.float32).reshape(-1, 2))
    angle = float(rect[2])
    rw, rh = rect[1]
    if rw < rh:
        angle += 90.0
    axis_angle = min(abs(angle), abs(abs(angle) - 90.0), abs(abs(angle) - 180.0))

    area = abs(float(cv2.contourArea(np.asarray(corners, dtype=np.float32))))
    area_fraction = area / max(1.0, float(h * w))
    span_x = bounds["width"] / max(1.0, float(w - 1))
    span_y = bounds["height"] / max(1.0, float(h - 1))
    diagnostics = {
        "available": True,
        "reason": "image-boundary-recovery-not-needed",
        "axis_angle_degrees": axis_angle,
        "seed_area_fraction": area_fraction,
        "seed_span_fraction": {"x": span_x, "y": span_y},
        "seed_bounds": bounds,
        "recovered_sides": [],
    }
    if axis_angle > 12.0:
        diagnostics["reason"] = "seed-too-rotated-for-image-boundary-recovery"
        return corners, diagnostics
    if area_fraction >= 0.40 or (span_x >= 0.72 and span_y >= 0.72):
        diagnostics["reason"] = "seed-already-broad"
        return corners, diagnostics

    margins = {
        "left": max(0.0, bounds["left"]),
        "top": max(0.0, bounds["top"]),
        "right": max(0.0, (w - 1) - bounds["right"]),
        "bottom": max(0.0, (h - 1) - bounds["bottom"]),
    }
    diagnostics["seed_margins"] = margins
    near_fraction = 0.20
    missing_fraction = 0.25
    near = {
        "left": margins["left"] <= near_fraction * max(1.0, w - 1),
        "right": margins["right"] <= near_fraction * max(1.0, w - 1),
        "top": margins["top"] <= near_fraction * max(1.0, h - 1),
        "bottom": margins["bottom"] <= near_fraction * max(1.0, h - 1),
    }
    if not any(near.values()):
        diagnostics["reason"] = "no-trusted-seed-side"
        return corners, diagnostics

    missing = {
        "left": margins["left"] >= missing_fraction * max(1.0, w - 1),
        "right": margins["right"] >= missing_fraction * max(1.0, w - 1),
        "top": margins["top"] >= missing_fraction * max(1.0, h - 1),
        "bottom": margins["bottom"] >= missing_fraction * max(1.0, h - 1),
    }
    if not any(missing.values()):
        diagnostics["reason"] = "no-materially-missing-side"
        return corners, diagnostics

    profiles = _axis_edge_profile(image_bgr)
    gap_x = max(8, int(round(w * 0.03)))
    gap_y = max(8, int(round(h * 0.03)))
    searches = {}
    if missing["left"]:
        searches["left"] = _robust_profile_peak(
            profiles["x_profile"], 0, max(1, bounds["left"] - gap_x), side="left", dimension=w
        )
    if missing["right"]:
        searches["right"] = _robust_profile_peak(
            profiles["x_profile"], min(w - 1, bounds["right"] + gap_x), w, side="right", dimension=w
        )
    if missing["top"]:
        searches["top"] = _robust_profile_peak(
            profiles["y_profile"], 0, max(1, bounds["top"] - gap_y), side="top", dimension=h
        )
    if missing["bottom"]:
        searches["bottom"] = _robust_profile_peak(
            profiles["y_profile"], min(h - 1, bounds["bottom"] + gap_y), h, side="bottom", dimension=h
        )
    diagnostics["boundary_search"] = searches

    recovered = {side: result for side, result in searches.items() if result is not None}
    # One proved missing side is useful for a one-axis truncation.  For a tiny
    # two-axis fragment, require both absent sides to be independently proved so
    # a strong text rule cannot inflate an arbitrary local block into a page.
    missing_count = sum(int(value) for value in missing.values())
    required = 2 if missing_count >= 2 else 1
    if len(recovered) < required:
        diagnostics["reason"] = "insufficient-image-boundary-evidence"
        return corners, diagnostics

    x0, y0, x1, y1 = bounds["left"], bounds["top"], bounds["right"], bounds["bottom"]
    if "left" in recovered:
        x0 = float(recovered["left"]["position"])
    if "right" in recovered:
        x1 = float(recovered["right"]["position"])
    if "top" in recovered:
        y0 = float(recovered["top"]["position"])
    if "bottom" in recovered:
        y1 = float(recovered["bottom"]["position"])

    recovered_corners = _canonical_quad(
        np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32),
        width=w,
        height=h,
    )
    if recovered_corners is None:
        diagnostics["reason"] = "invalid-image-boundary-recovery"
        return corners, diagnostics

    diagnostics.update({
        "reason": "image-supported-boundary-recovery",
        "recovered_sides": sorted(recovered),
        "completed_bounds": _quad_axis_bounds(recovered_corners),
        "completed_area": abs(float(cv2.contourArea(recovered_corners))),
        "profile_smooth_window": int(profiles["smooth_window"]),
    })
    return recovered_corners, diagnostics


def _image_supported_overexpansion_trim(image_bgr, corners):
    """Trim padded page sides only when a stronger inward physical edge is proved.

    The late Orli recovery stages can correctly restore a missing document but
    the calibrated padding may then push one or more sides past the physical
    sheet.  This parameter-free guard searches only a narrow inward band from
    sides already close to the source-image border and trims a side when a
    substantially stronger robust gradient peak is present.  It never expands
    geometry and therefore cannot resurrect the historic truncated-page failure.
    """
    if corners is None:
        return corners, {"available": False, "reason": "no-envelope", "trimmed_sides": []}

    h, w = image_bgr.shape[:2]
    bounds = _quad_axis_bounds(corners)
    if bounds is None:
        return corners, {"available": False, "reason": "invalid-envelope", "trimmed_sides": []}

    profiles = _axis_edge_profile(image_bgr)
    xprof = np.asarray(profiles["x_profile"], dtype=np.float64)
    yprof = np.asarray(profiles["y_profile"], dtype=np.float64)
    diagnostics = {
        "available": True,
        "reason": "no-supported-overexpansion",
        "input_bounds": bounds,
        "trimmed_sides": [],
        "searches": {},
    }

    def candidate(profile, current, *, side, dimension):
        border_margin = current if side in {"left", "top"} else (dimension - 1) - current
        # Only question sides already within the outer 10% of the capture.
        if border_margin > 0.10 * max(1.0, dimension - 1):
            return None
        inward = max(12, int(round(0.10 * dimension)))
        gap = max(5, int(round(0.008 * dimension)))
        if side in {"left", "top"}:
            lo = int(round(current + gap))
            hi = min(dimension, int(round(current + inward)))
        else:
            lo = max(0, int(round(current - inward)))
            hi = int(round(current - gap))
        if hi <= lo + 3:
            return None
        peak = _robust_profile_peak(profile, lo, hi, side=side, dimension=dimension)
        if peak is None:
            return None
        cur_i = max(0, min(dimension - 1, int(round(current))))
        current_score = float(profile[cur_i])
        shift = abs(float(peak["position"]) - float(current))
        minimum_shift = max(4.0, 0.012 * dimension)
        stronger = float(peak["profile"]) >= max(current_score * 1.30, current_score + float(peak["robust_scale"]) * 1.5)
        if shift < minimum_shift or not stronger:
            return None
        return {**peak, "current_profile": current_score, "shift": shift}

    searches = {
        "left": candidate(xprof, bounds["left"], side="left", dimension=w),
        "right": candidate(xprof, bounds["right"], side="right", dimension=w),
        "top": candidate(yprof, bounds["top"], side="top", dimension=h),
        "bottom": candidate(yprof, bounds["bottom"], side="bottom", dimension=h),
    }
    diagnostics["searches"] = searches

    x0, y0, x1, y1 = bounds["left"], bounds["top"], bounds["right"], bounds["bottom"]
    if searches["left"] is not None:
        x0 = float(searches["left"]["position"]); diagnostics["trimmed_sides"].append("left")
    if searches["right"] is not None:
        x1 = float(searches["right"]["position"]); diagnostics["trimmed_sides"].append("right")
    if searches["top"] is not None:
        y0 = float(searches["top"]["position"]); diagnostics["trimmed_sides"].append("top")
    if searches["bottom"] is not None:
        y1 = float(searches["bottom"]["position"]); diagnostics["trimmed_sides"].append("bottom")
    if not diagnostics["trimmed_sides"]:
        return corners, diagnostics
    if x1 <= x0 + 8 or y1 <= y0 + 8:
        diagnostics["reason"] = "invalid-trimmed-envelope"
        diagnostics["trimmed_sides"] = []
        return corners, diagnostics

    trimmed = _canonical_quad(
        np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32),
        width=w, height=h,
    )
    if trimmed is None:
        diagnostics["reason"] = "invalid-trimmed-envelope"
        diagnostics["trimmed_sides"] = []
        return corners, diagnostics
    diagnostics["reason"] = "image-supported-overexpansion-trim"
    diagnostics["output_bounds"] = _quad_axis_bounds(trimmed)
    return trimmed, diagnostics


def _pad_quad(corners, *, image_shape, padding_fraction):
    if corners is None:
        return None
    h, w = image_shape[:2]
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    center = corners.mean(axis=0)
    vectors = corners - center
    lengths = np.linalg.norm(vectors, axis=1)
    pad = float(min(h, w)) * float(padding_fraction)
    safe = np.maximum(lengths, 1e-6)
    padded = center + vectors * ((safe + pad) / safe)[:, None]
    return _canonical_quad(padded, width=w, height=h)

def _proposal(image_bgr, values):
    evidence = _infer_evidence(image_bgr)
    mask = _evidence_mask(image_bgr.shape, evidence, values)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    image_area = float(h * w)
    selected, contour_diagnostics = _select_evidence_envelope(
        contours,
        image_area=image_area,
    )

    contour = selected[0] if selected else None
    contour_corners = None
    contour_area = 0.0
    if contour is not None:
        contour_corners = _canonical_quad(cv2.boxPoints(cv2.minAreaRect(contour)), width=w, height=h)
        if contour_corners is not None:
            contour_area = abs(float(cv2.contourArea(contour_corners)))

    learned_corners, learned_diagnostics = _learned_geometry_envelope(
        evidence,
        image_shape=image_bgr.shape,
    )
    learned_area = abs(float(cv2.contourArea(learned_corners))) if learned_corners is not None else 0.0
    frame_corners, frame_diagnostics = _learned_document_frame(
        evidence,
        image_shape=image_bgr.shape,
    )
    frame_area = abs(float(cv2.contourArea(frame_corners))) if frame_corners is not None else 0.0

    raw_corners, arbitration_source, arbitration_diagnostics = _arbitrate_envelopes(
        contour_corners,
        learned_corners,
        image_shape=image_bgr.shape,
        frame_corners=frame_corners,
        frame_diagnostics=frame_diagnostics,
    )
    if arbitration_source == "frame":
        envelope_mode = "learned-document-frame"
    elif arbitration_source == "learned":
        envelope_mode = "learned-geometry-consensus"
    else:
        envelope_mode = contour_diagnostics.get("mode", "none")

    completed_corners, completion_diagnostics = _directional_page_completion(
        raw_corners,
        image_shape=image_bgr.shape,
    )
    if completion_diagnostics.get("changed_axes"):
        raw_corners = completed_corners
        envelope_mode = f"{envelope_mode}+directional-completion"

    if completion_diagnostics.get("changed_axes"):
        boundary_corners = raw_corners
        boundary_diagnostics = {
            "available": False,
            "reason": "directional-completion-already-succeeded",
            "recovered_sides": [],
        }
    else:
        boundary_corners, boundary_diagnostics = _image_supported_boundary_recovery(
            image_bgr, raw_corners
        )
        if boundary_diagnostics.get("recovered_sides"):
            raw_corners = boundary_corners
            envelope_mode = f"{envelope_mode}+image-boundary-recovery"

    envelope_diagnostics = {
        "mode": envelope_mode,
        "contour": contour_diagnostics,
        "learned_geometry": learned_diagnostics,
        "learned_document_frame": frame_diagnostics,
        "arbitration": arbitration_diagnostics,
        "directional_completion": completion_diagnostics,
        "image_boundary_recovery": boundary_diagnostics,
        "contour_quad_area": contour_area,
        "learned_quad_area": learned_area,
        "learned_document_frame_area": frame_area,
    }
    if raw_corners is None:
        return evidence, mask, contour, None, 0.0, envelope_diagnostics

    corners = _pad_quad(
        raw_corners,
        image_shape=image_bgr.shape,
        padding_fraction=values["page_padding_fraction"],
    )
    if corners is None:
        return evidence, mask, contour, None, 0.0, envelope_diagnostics

    corners, trim_diagnostics = _image_supported_overexpansion_trim(image_bgr, corners)
    envelope_diagnostics["overexpansion_trim"] = trim_diagnostics

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
        "evidence": "orli_base_historical_baselines",
        "envelope": envelope_diagnostics,
        "region_count": len(evidence["regions"]),
        "line_polygon_count": len(evidence["lines"]),
        "baseline_count": len(evidence["baselines"]),
        "text_direction": evidence["text_direction"],
        "page_area_fraction": area_fraction,
        "model": _provenance().get("model_id"),
        "canonical_quad": corners is not None,
        "orli_runtime_warnings": int(runtime_diagnostics.get("orli_runtime_warnings", 0)),
        "lightning_srun_advisories": int(runtime_diagnostics.get("lightning_srun_advisories", 0)),
        "orli_filtered_messages": list(runtime_diagnostics.get("filtered_messages", [])),
    }

    if not evidence["regions"] and not evidence["lines"] and not evidence["baselines"]:
        diagnostics["reason"] = "no_orli_layout_evidence"
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
        "orli-layout-evidence.png": evidence_mask,
        "orli-page-proposal.png": overlay,
    }


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect", "precompute_golden_set_evidence", "export_precomputed_golden_set_evidence", "load_precomputed_golden_set_evidence"]
