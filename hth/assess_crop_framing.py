#!/usr/bin/env python3
"""Compare initial crop/framing transforms on an approved Golden Set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ALGORITHMS = ("axis-aligned", "rotation-crop", "projective")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_image(root: Path, ordinal: int) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"):
        for candidate in (root / f"fs_{ordinal:04d}{suffix}", root / "raw" / f"fs_{ordinal:04d}{suffix}"):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No Golden Set image found for page {ordinal}")


def order_corners(points: Any) -> np.ndarray:
    corners = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(corners).all():
        raise ValueError("Framing corners must be finite")
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    ordered = corners[np.argsort(angles)]
    ordered = np.roll(ordered, -int(np.argmin(ordered[:, 0] + ordered[:, 1])), axis=0)
    if abs(float(cv2.contourArea(ordered))) < 1.0:
        raise ValueError("Framing corners are degenerate")
    return ordered


def _destination(corners: np.ndarray) -> tuple[np.ndarray, int, int]:
    top = float(np.linalg.norm(corners[1] - corners[0]))
    bottom = float(np.linalg.norm(corners[2] - corners[3]))
    left = float(np.linalg.norm(corners[3] - corners[0]))
    right = float(np.linalg.norm(corners[2] - corners[1]))
    width = max(2, int(round(max(top, bottom))))
    height = max(2, int(round(max(left, right))))
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    return destination, width, height


def framing_transform(corners: Any, algorithm: str, image_shape: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    ordered = order_corners(corners)
    image_height, image_width = image_shape[:2]
    if algorithm == "axis-aligned":
        x1 = max(0, int(math.floor(float(np.min(ordered[:, 0])))))
        y1 = max(0, int(math.floor(float(np.min(ordered[:, 1])))))
        x2 = min(image_width, int(math.ceil(float(np.max(ordered[:, 0])))))
        y2 = min(image_height, int(math.ceil(float(np.max(ordered[:, 1])))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Axis-aligned crop is empty")
        source = np.array([[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]], dtype=np.float32)
    elif algorithm == "rotation-crop":
        source = order_corners(cv2.boxPoints(cv2.minAreaRect(ordered)))
    elif algorithm == "projective":
        source = ordered
    else:
        raise ValueError(f"Unknown crop/framing algorithm: {algorithm}")
    destination, width, height = _destination(source)
    matrix = cv2.getPerspectiveTransform(source, destination)
    return matrix, (width, height), source


def apply_framing(image: np.ndarray, corners: Any, algorithm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix, size, source = framing_transform(corners, algorithm, image.shape)
    output = cv2.warpPerspective(
        image,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return output, matrix, source


def approved_box_retention(box: Any, matrix: np.ndarray, output_shape: tuple[int, ...]) -> float:
    x1, y1, x2, y2 = (float(value) for value in box)
    reference = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(reference, matrix)[0]
    area = abs(float(cv2.contourArea(transformed)))
    if area <= 0:
        return 0.0
    height, width = output_shape[:2]
    boundary = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    intersection, _ = cv2.intersectConvexConvex(transformed.astype(np.float32), boundary)
    return max(0.0, min(1.0, float(intersection) / area))


def reprojection_error(source: np.ndarray, matrix: np.ndarray) -> float:
    inverse = np.linalg.inv(matrix)
    projected = cv2.perspectiveTransform(source.reshape(1, 4, 2), matrix)
    restored = cv2.perspectiveTransform(projected, inverse)[0]
    return float(np.max(np.linalg.norm(restored - source, axis=1)))


def _candidate(record: dict[str, Any], detector: str) -> dict[str, Any]:
    candidates = record.get("geometry_candidates") or []
    match = next((item for item in candidates if item.get("method") == detector), None)
    if not isinstance(match, dict) or match.get("status") != "ok" or not match.get("corners"):
        raise ValueError(f"Page {record.get('global_ordinal')} has no successful {detector} quadrilateral")
    return match


def _fit_panel(image: np.ndarray, width: int = 420, height: int = 520) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def _contact_sheet(original: np.ndarray, box: Any, corners: Any, variants: dict[str, np.ndarray], ordinal: int) -> np.ndarray:
    overlay = original.copy()
    x1, y1, x2, y2 = (int(round(float(value))) for value in box)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), max(2, round(min(original.shape[:2]) / 500)))
    cv2.polylines(overlay, [np.rint(order_corners(corners)).astype(np.int32)], True, (255, 80, 0), max(2, round(min(original.shape[:2]) / 500)))
    images = [overlay, *(variants[name] for name in ALGORITHMS)]
    labels = ["Original: approved box (red), detector quad (blue)", *ALGORITHMS]
    panels = []
    for image, label in zip(images, labels):
        panel = _fit_panel(image)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (20, 20, 20), -1)
        cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    sheet = np.concatenate(panels, axis=1)
    cv2.putText(sheet, f"GS0002 page {ordinal}", (10, sheet.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return sheet


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["global_ordinal", "algorithm", "width", "height", "output_area_ratio", "approved_box_retention_proxy", "reprojection_error_px", "output_sha256"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _summary(payload: dict[str, Any]) -> str:
    detector = payload["detector_selection"]
    lines = [
        "# GS0002 Crop / Framing Assessment",
        "",
        "> This is diagnostic evidence, not a production normalization decision. GS0002 stores approved axis-aligned boxes, not four-corner perspective truth. Approved-box retention is therefore a clipping proxy; perspective quality requires review of the contact sheets.",
        "",
        "## Assessment provenance",
        "",
        f"- Golden Set: `{payload['golden_set']['id']}` (`{payload['golden_set']['sha256']}`)",
        f"- Pages: `{payload['page_count']}`",
        f"- Detector: `{detector.get('display_name') or detector.get('detector')}` (`{detector.get('detector')}`)",
        f"- Parameter Set ID: `{detector.get('parameter_set_id')}`",
        f"- Calibration ID: `{detector.get('calibration_id')}`",
        "",
        "## Aggregate comparison",
        "",
        "| Algorithm | Pages | Mean approved-box retention proxy | Minimum retention | Mean output/source area | Maximum reprojection error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        item = payload["algorithms"][algorithm]
        lines.append(
            f"| {algorithm} | {item['pages']} | {item['mean_approved_box_retention_proxy']:.4f} | "
            f"{item['minimum_approved_box_retention_proxy']:.4f} | {item['mean_output_area_ratio']:.4f} | "
            f"{item['maximum_reprojection_error_px']:.6f} px |"
        )
    lines.extend([
        "",
        "## Review artifact",
        "",
        "Open `index.html` from the extracted artifact. Each contact sheet shows the original image with the approved box in red and the calibrated detector quadrilateral in blue, followed by the three framing variants.",
        "",
    ])
    return "\n".join(lines)


def _write_html(path: Path, payload: dict[str, Any]) -> None:
    cards = []
    for ordinal in payload["ordinals"]:
        cards.append(
            f'<article><h2>Golden Set page {ordinal}</h2><a href="contact-sheets/fs_{ordinal:04d}.jpg">'
            f'<img loading="lazy" src="contact-sheets/fs_{ordinal:04d}.jpg" alt="Crop and framing comparison for page {ordinal}"></a></article>'
        )
    detector = payload["detector_selection"]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GS0002 Crop / Framing Assessment</title><style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#111820;color:#e6edf3}}main{{max-width:1800px;margin:auto}}
article{{margin:2rem 0;padding:1rem;background:#18212b;border:1px solid #34404c;border-radius:.5rem}}img{{width:100%;height:auto}}
code{{background:#26313c;padding:.15rem .35rem;border-radius:.25rem}}.warning{{padding:1rem;border-left:4px solid #d29922;background:#2b2415}}
</style></head><body><main><h1>GS0002 Crop / Framing Assessment</h1>
<p>Detector: <code>{html.escape(str(detector.get('display_name') or detector.get('detector')))}</code>; parameter set <code>{html.escape(str(detector.get('parameter_set_id')))}</code>.</p>
<p class="warning">GS0002 contains approved axis-aligned boxes rather than perspective-corner truth. Red boxes support clipping checks; visual comparison is required to judge perspective correction.</p>
{''.join(cards)}</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def assess(golden_set_path: Path, image_root: Path, analysis_path: Path, output: Path) -> dict[str, Any]:
    golden_set = _read_json(golden_set_path)
    analysis = _read_json(analysis_path)
    selection = analysis.get("document_detector") or {}
    detector = str(selection.get("detector") or "").strip()
    if not detector:
        raise ValueError("Analysis has no resolved document_detector")
    analysis_by_ordinal = {int(item["global_ordinal"]): item for item in analysis.get("records") or []}
    pages = [item for item in golden_set.get("pages") or [] if item.get("review_status") == "approved" and item.get("physical_document_bbox")]
    if not pages:
        raise ValueError("Golden Set has no approved physical-document boxes")

    output.mkdir(parents=True, exist_ok=True)
    (output / "contact-sheets").mkdir(exist_ok=True)
    for algorithm in ALGORITHMS:
        (output / "variants" / algorithm).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    ordinals: list[int] = []
    for page in sorted(pages, key=lambda item: int(item["global_ordinal"])):
        ordinal = int(page["global_ordinal"])
        image_path = _find_image(image_root, ordinal)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode Golden Set image: {image_path}")
        candidate = _candidate(analysis_by_ordinal.get(ordinal, {}), detector)
        corners = candidate["corners"]
        variants: dict[str, np.ndarray] = {}
        for algorithm in ALGORITHMS:
            framed, matrix, source = apply_framing(image, corners, algorithm)
            target = output / "variants" / algorithm / f"fs_{ordinal:04d}.png"
            if not cv2.imwrite(str(target), framed, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
                raise ValueError(f"Could not write framing output: {target}")
            variants[algorithm] = framed
            rows.append({
                "global_ordinal": ordinal,
                "algorithm": algorithm,
                "width": int(framed.shape[1]),
                "height": int(framed.shape[0]),
                "output_area_ratio": round((framed.shape[0] * framed.shape[1]) / (image.shape[0] * image.shape[1]), 8),
                "approved_box_retention_proxy": round(approved_box_retention(page["physical_document_bbox"], matrix, framed.shape), 8),
                "reprojection_error_px": round(reprojection_error(source, matrix), 8),
                "output_sha256": _sha256(target),
                "source_transform": [[round(float(value), 10) for value in line] for line in matrix.tolist()],
            })
        sheet = _contact_sheet(image, page["physical_document_bbox"], corners, variants, ordinal)
        if not cv2.imwrite(str(output / "contact-sheets" / f"fs_{ordinal:04d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ValueError(f"Could not write contact sheet for page {ordinal}")
        ordinals.append(ordinal)

    algorithms: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        matching = [row for row in rows if row["algorithm"] == algorithm]
        algorithms[algorithm] = {
            "pages": len(matching),
            "mean_approved_box_retention_proxy": sum(row["approved_box_retention_proxy"] for row in matching) / len(matching),
            "minimum_approved_box_retention_proxy": min(row["approved_box_retention_proxy"] for row in matching),
            "mean_output_area_ratio": sum(row["output_area_ratio"] for row in matching) / len(matching),
            "maximum_reprojection_error_px": max(row["reprojection_error_px"] for row in matching),
        }
    payload = {
        "schema_version": "1.0",
        "assessment_type": "crop-framing-comparison",
        "status": "diagnostic-only",
        "truth_limitation": "Golden Set reference geometry is axis-aligned; approved-box retention is a clipping proxy and does not score perspective correctness.",
        "golden_set": {"id": golden_set.get("collection_id"), "sha256": _sha256(golden_set_path)},
        "detector_selection": selection,
        "page_count": len(ordinals),
        "ordinals": ordinals,
        "algorithms": algorithms,
        "pages": rows,
    }
    (output / "assessment.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(output / "assessment.csv", rows)
    summary = _summary(payload)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    _write_html(output / "index.html", payload)
    return payload


def prepare_inputs(golden_set_path: Path, image_root: Path, manifest_path: Path, analysis_path: Path) -> None:
    golden_set = _read_json(golden_set_path)
    records = []
    analysis_records = []
    for page in sorted(golden_set.get("pages") or [], key=lambda item: int(item["global_ordinal"])):
        ordinal = int(page["global_ordinal"])
        image = _find_image(image_root, ordinal)
        records.append({"global_ordinal": ordinal, "raw_file": image.relative_to(image_root).as_posix()})
        analysis_records.append({"global_ordinal": ordinal})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"schema_version": "1.0", "records": records}, indent=2) + "\n", encoding="utf-8")
    analysis_path.write_text(json.dumps({"schema_version": "1.0", "records": analysis_records}, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Create detector input manifests for Golden Set images")
    prepare.add_argument("--golden-set", type=Path, required=True)
    prepare.add_argument("--image-root", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--analysis", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate", help="Generate framing variants and comparison evidence")
    evaluate.add_argument("--golden-set", type=Path, required=True)
    evaluate.add_argument("--image-root", type=Path, required=True)
    evaluate.add_argument("--analysis", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--github-summary", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_inputs(args.golden_set, args.image_root, args.manifest, args.analysis)
    else:
        payload = assess(args.golden_set, args.image_root, args.analysis, args.output)
        if args.github_summary:
            with args.github_summary.open("a", encoding="utf-8") as handle:
                handle.write(_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
