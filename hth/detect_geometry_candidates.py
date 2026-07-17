#!/usr/bin/env python3
"""
Generate multiple physical-document geometry candidates for HTH page images.

Candidates:
  - contour: largest plausible foreground contour / quadrilateral
  - ransac: robust four-edge fit from scanline boundary samples
  - hough: dominant horizontal/vertical line envelope

The existing HTH detector remains in page-analysis.json as the "current"
candidate. This script augments each analysis record with geometry_candidates.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from skimage.measure import LineModelND, ransac


@dataclass
class Candidate:
    method: str
    bbox: list[int] | None
    corners: list[list[float]] | None
    confidence: float
    score: float
    diagnostics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dimension", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def valid_bbox(box: Any) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def resize_for_analysis(image: np.ndarray, maximum: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, maximum / max(width, height))
    if scale == 1.0:
        return image, 1.0
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def border_background_lab(image_bgr: np.ndarray, fraction: float = 0.035) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    height, width = lab.shape[:2]
    sx = max(2, round(width * fraction))
    sy = max(2, round(height * fraction))
    strips = np.concatenate(
        [
            lab[:sy, :, :].reshape(-1, 3),
            lab[height - sy :, :, :].reshape(-1, 3),
            lab[sy : height - sy, :sx, :].reshape(-1, 3),
            lab[sy : height - sy, width - sx :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(strips, axis=0)


def document_mask(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    bg = border_background_lab(image_bgr)
    distance = np.linalg.norm(lab.astype(np.float32) - bg.astype(np.float32), axis=2)

    border = np.concatenate(
        [
            distance[: max(2, distance.shape[0] // 30), :].ravel(),
            distance[-max(2, distance.shape[0] // 30) :, :].ravel(),
            distance[:, : max(2, distance.shape[1] // 30)].ravel(),
            distance[:, -max(2, distance.shape[1] // 30) :].ravel(),
        ]
    )
    threshold = max(18.0, float(np.percentile(border, 99)) + 5.0)
    mask = (distance >= threshold).astype(np.uint8) * 255

    # Join interrupted paper regions while suppressing isolated noise.
    short = max(3, (min(mask.shape) // 180) | 1)
    long = max(7, (min(mask.shape) // 70) | 1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (long, long))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (short, short))
    )

    return mask, {
        "background_lab": [round(float(v), 3) for v in bg],
        "color_distance_threshold": round(threshold, 3),
        "foreground_fraction": round(float(np.mean(mask > 0)), 6),
    }


def scale_bbox(box: list[int], inverse_scale: float, width: int, height: int) -> list[int]:
    return [
        max(0, min(width, round(box[0] * inverse_scale))),
        max(0, min(height, round(box[1] * inverse_scale))),
        max(0, min(width, round(box[2] * inverse_scale))),
        max(0, min(height, round(box[3] * inverse_scale))),
    ]


def bbox_from_points(points: np.ndarray, width: int, height: int) -> list[int] | None:
    if points.size == 0:
        return None
    xs = points[:, 0]
    ys = points[:, 1]
    box = [
        max(0, int(math.floor(float(xs.min())))),
        max(0, int(math.floor(float(ys.min())))),
        min(width, int(math.ceil(float(xs.max())))),
        min(height, int(math.ceil(float(ys.max())))),
    ]
    return box if valid_bbox(box) else None


def candidate_score(mask: np.ndarray, box: list[int] | None) -> float:
    if not valid_bbox(box):
        return 0.0
    height, width = mask.shape[:2]
    left, top, right, bottom = box
    inside = mask[top:bottom, left:right]
    if inside.size == 0:
        return 0.0

    interior = float(np.mean(inside > 0))
    area_fraction = ((right - left) * (bottom - top)) / max(1, width * height)

    outside_mask = np.ones(mask.shape, dtype=bool)
    outside_mask[top:bottom, left:right] = False
    outside_values = mask[outside_mask]
    outside_foreground = float(np.mean(outside_values > 0)) if outside_values.size else 0.0

    size_score = min(1.0, area_fraction / 0.55)
    score = 0.50 * interior + 0.30 * size_score + 0.20 * (1.0 - outside_foreground)
    return max(0.0, min(1.0, score))


def detect_contour(mask: np.ndarray) -> Candidate:
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, np.ndarray, list[int], np.ndarray] | None = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < width * height * 0.12:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        x, y, w, h = cv2.boundingRect(contour)
        box = [x, y, x + w, y + h]
        score = candidate_score(mask, box)
        rectangularity = area / max(1.0, w * h)
        combined = score * 0.75 + min(1.0, rectangularity) * 0.25
        if best is None or combined > best[0]:
            best = (combined, contour, box, approx)

    if best is None:
        return Candidate("contour", None, None, 0.0, 0.0, {"reason": "no_plausible_contour"})

    combined, contour, box, approx = best
    corners = None
    if len(approx) == 4:
        corners = [[float(p[0][0]), float(p[0][1])] for p in approx]
    else:
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect).astype(float).tolist()

    return Candidate(
        "contour",
        box,
        corners,
        round(combined, 6),
        round(combined, 6),
        {
            "contour_area": round(float(cv2.contourArea(contour)), 3),
            "polygon_vertices": int(len(approx)),
        },
    )


def scan_boundary_points(mask: np.ndarray) -> dict[str, np.ndarray]:
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


def fit_ransac_line(points: np.ndarray, threshold: float) -> tuple[LineModelND, np.ndarray] | None:
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


def line_intersection(a: LineModelND, b: LineModelND) -> np.ndarray | None:
    origin_a, direction_a = a.params
    origin_b, direction_b = b.params
    matrix = np.column_stack((direction_a, -direction_b))
    if abs(np.linalg.det(matrix)) < 1e-8:
        return None
    t, _ = np.linalg.solve(matrix, origin_b - origin_a)
    return origin_a + direction_a * t


def detect_ransac(mask: np.ndarray) -> Candidate:
    height, width = mask.shape
    points = scan_boundary_points(mask)
    threshold = max(2.0, min(width, height) * 0.008)
    fitted: dict[str, tuple[LineModelND, np.ndarray]] = {}

    for name, values in points.items():
        result = fit_ransac_line(values, threshold)
        if result is not None:
            fitted[name] = result

    if set(fitted) != {"left", "right", "top", "bottom"}:
        return Candidate(
            "ransac",
            None,
            None,
            0.0,
            0.0,
            {
                "reason": "insufficient_edge_models",
                "fitted_edges": sorted(fitted),
                "sample_counts": {k: int(len(v)) for k, v in points.items()},
            },
        )

    tl = line_intersection(fitted["left"][0], fitted["top"][0])
    tr = line_intersection(fitted["right"][0], fitted["top"][0])
    br = line_intersection(fitted["right"][0], fitted["bottom"][0])
    bl = line_intersection(fitted["left"][0], fitted["bottom"][0])
    if any(point is None for point in (tl, tr, br, bl)):
        return Candidate("ransac", None, None, 0.0, 0.0, {"reason": "parallel_edge_models"})

    corners_np = np.asarray([tl, tr, br, bl], dtype=float)
    box = bbox_from_points(corners_np, width, height)
    score = candidate_score(mask, box)
    inlier_ratio = float(
        np.mean([np.mean(fitted[name][1]) for name in ("left", "right", "top", "bottom")])
    )
    combined = 0.65 * score + 0.35 * inlier_ratio

    return Candidate(
        "ransac",
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


def detect_hough(image_bgr: np.ndarray, mask: np.ndarray) -> Candidate:
    height, width = mask.shape
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=max(35, min(width, height) // 18),
        minLineLength=max(45, min(width, height) // 5),
        maxLineGap=max(20, min(width, height) // 18),
    )

    if lines is None:
        return Candidate("hough", None, None, 0.0, 0.0, {"reason": "no_hough_lines"})

    vertical, horizontal = [], []
    for item in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, item)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180
        if angle > 90:
            angle = 180 - angle
        if angle >= 68:
            vertical.append((x1, y1, x2, y2, length))
        elif angle <= 22:
            horizontal.append((x1, y1, x2, y2, length))

    if len(vertical) < 2 or len(horizontal) < 2:
        return Candidate(
            "hough",
            None,
            None,
            0.0,
            0.0,
            {
                "reason": "insufficient_axis_lines",
                "vertical_lines": len(vertical),
                "horizontal_lines": len(horizontal),
            },
        )

    # Weighted midpoint positions; choose robust low/high envelopes.
    vx = np.asarray([((x1 + x2) / 2, length) for x1, _, x2, _, length in vertical])
    hy = np.asarray([((y1 + y2) / 2, length) for _, y1, _, y2, length in horizontal])

    left = int(round(np.percentile(vx[:, 0], 10)))
    right = int(round(np.percentile(vx[:, 0], 90)))
    top = int(round(np.percentile(hy[:, 0], 10)))
    bottom = int(round(np.percentile(hy[:, 0], 90)))
    box = [max(0, left), max(0, top), min(width, right), min(height, bottom)]

    if not valid_bbox(box):
        return Candidate("hough", None, None, 0.0, 0.0, {"reason": "invalid_line_envelope"})

    score = candidate_score(mask, box)
    support = min(1.0, (len(vertical) + len(horizontal)) / 24.0)
    combined = 0.72 * score + 0.28 * support
    corners = [
        [float(box[0]), float(box[1])],
        [float(box[2]), float(box[1])],
        [float(box[2]), float(box[3])],
        [float(box[0]), float(box[3])],
    ]
    return Candidate(
        "hough",
        box,
        corners,
        round(combined, 6),
        round(combined, 6),
        {
            "vertical_lines": len(vertical),
            "horizontal_lines": len(horizontal),
            "support_score": round(support, 6),
        },
    )


def run_detectors(image_path: Path, maximum: int) -> list[dict[str, Any]]:
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    original_height, original_width = original.shape[:2]
    image, scale = resize_for_analysis(original, maximum)
    mask, mask_diag = document_mask(image)

    candidates = [
        detect_contour(mask),
        detect_ransac(mask),
        detect_hough(image, mask),
    ]

    inverse = 1.0 / scale
    for candidate in candidates:
        candidate.diagnostics["mask"] = mask_diag
        candidate.diagnostics["analysis_scale"] = round(scale, 8)
        if candidate.bbox is not None:
            candidate.bbox = scale_bbox(
                candidate.bbox, inverse, original_width, original_height
            )
        if candidate.corners is not None:
            candidate.corners = [
                [
                    round(max(0.0, min(original_width, point[0] * inverse)), 3),
                    round(max(0.0, min(original_height, point[1] * inverse)), 3),
                ]
                for point in candidate.corners
            ]

    return [asdict(candidate) for candidate in candidates]


def path_for_record(record: dict[str, Any], image_root: Path) -> Path:
    for key in (
        "extracted_file",
        "raw_file",
        "image_file",
        "output_file",
        "path",
    ):
        value = record.get(key)
        if value:
            candidate = image_root / str(value)
            if candidate.exists():
                return candidate

    ordinal = int(record["global_ordinal"])
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"):
        candidate = image_root / "raw" / f"fs_{ordinal:04d}{suffix}"
        if candidate.exists():
            return candidate
        candidate = image_root / f"fs_{ordinal:04d}{suffix}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No image found for global ordinal {ordinal}")


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists; pass --overwrite: {args.output}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))

    manifest_records = {
        int(record["global_ordinal"]): record
        for record in manifest.get("records", [])
    }

    processed = 0
    errors = 0
    for record in analysis.get("records", []):
        ordinal = int(record["global_ordinal"])
        manifest_record = manifest_records.get(ordinal, {"global_ordinal": ordinal})
        try:
            image_path = path_for_record(manifest_record, args.image_root)
            record["geometry_candidates"] = run_detectors(
                image_path, args.max_dimension
            )
            record["geometry_candidate_status"] = "complete"
            processed += 1
            
        except Exception as exc:
            import traceback

            print(f"\nERROR processing page {ordinal}")
            traceback.print_exc()

            record["geometry_candidates"] = []
            record["geometry_candidate_status"] = "error"
            record["geometry_candidate_error"] = traceback.format_exc()

            errors += 1
    
    analysis["geometry_candidate_summary"] = {
        "processed": processed,
        "errors": errors,
        "methods": ["contour", "ransac", "hough"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            analysis["geometry_candidate_summary"], indent=2, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
