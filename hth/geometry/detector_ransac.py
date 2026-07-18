from __future__ import annotations

import numpy as np
from skimage.measure import LineModelND, ransac

from .common import bbox_from_points, candidate_score
from .model import Candidate

METHOD = "ransac"


def _scan_boundary_points(mask: np.ndarray) -> dict[str, np.ndarray]:
    height, width = mask.shape
    row_step = max(1, height // 220)
    col_step = max(1, width // 220)
    minimum_row_pixels = max(6, width // 80)
    minimum_col_pixels = max(6, height // 80)
    left, right, top, bottom = [], [], [], []

    for y in range(0, height, row_step):
        xs = np.flatnonzero(mask[y] > 0)
        if len(xs) >= minimum_row_pixels:
            left.append((float(xs[0]), float(y)))
            right.append((float(xs[-1]), float(y)))

    for x in range(0, width, col_step):
        ys = np.flatnonzero(mask[:, x] > 0)
        if len(ys) >= minimum_col_pixels:
            top.append((float(x), float(ys[0])))
            bottom.append((float(x), float(ys[-1])))

    return {
        "left": np.asarray(left, dtype=float),
        "right": np.asarray(right, dtype=float),
        "top": np.asarray(top, dtype=float),
        "bottom": np.asarray(bottom, dtype=float),
    }


def _fit_line(points: np.ndarray, threshold: float) -> tuple[LineModelND, np.ndarray] | None:
    if len(points) < 10:
        return None
    try:
        model, inliers = ransac(
            points,
            LineModelND,
            min_samples=2,
            residual_threshold=threshold,
            max_trials=400,
            stop_probability=0.999,
            rng=42,
        )
    except Exception:
        return None
    if model is None or inliers is None or int(np.sum(inliers)) < 6:
        return None
    return model, inliers


def _intersection(a: LineModelND, b: LineModelND) -> np.ndarray | None:
    origin_a, direction_a = a.params
    origin_b, direction_b = b.params
    matrix = np.column_stack((direction_a, -direction_b))
    if abs(np.linalg.det(matrix)) < 1e-8:
        return None
    t, _ = np.linalg.solve(matrix, origin_b - origin_a)
    return origin_a + direction_a * t


def detect(*, image_bgr: np.ndarray, mask: np.ndarray) -> Candidate:
    del image_bgr
    height, width = mask.shape
    points = _scan_boundary_points(mask)
    threshold = max(2.0, min(width, height) * 0.008)
    fitted: dict[str, tuple[LineModelND, np.ndarray]] = {}

    for name, values in points.items():
        result = _fit_line(values, threshold)
        if result is not None:
            fitted[name] = result

    if set(fitted) != {"left", "right", "top", "bottom"}:
        return Candidate(
            METHOD, None, None, 0.0, 0.0,
            {
                "reason": "insufficient_edge_models",
                "fitted_edges": sorted(fitted),
                "sample_counts": {k: int(len(v)) for k, v in points.items()},
            },
        )

    tl = _intersection(fitted["left"][0], fitted["top"][0])
    tr = _intersection(fitted["right"][0], fitted["top"][0])
    br = _intersection(fitted["right"][0], fitted["bottom"][0])
    bl = _intersection(fitted["left"][0], fitted["bottom"][0])
    if any(point is None for point in (tl, tr, br, bl)):
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "parallel_edge_models"})

    corners_np = np.asarray([tl, tr, br, bl], dtype=float)
    box = bbox_from_points(corners_np, width, height)
    score = candidate_score(mask, box)
    inlier_ratio = float(
        np.mean([np.mean(fitted[name][1]) for name in ("left", "right", "top", "bottom")])
    )
    combined = 0.65 * score + 0.35 * inlier_ratio

    return Candidate(
        METHOD,
        box,
        corners_np.tolist(),
        round(combined, 6),
        round(combined, 6),
        {
            "residual_threshold_px": round(threshold, 3),
            "mean_inlier_ratio": round(inlier_ratio, 6),
            "sample_counts": {k: int(len(v)) for k, v in points.items()},
            "inlier_counts": {k: int(np.sum(fitted[k][1])) for k in fitted},
        },
    )
