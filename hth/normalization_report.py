#!/usr/bin/env python3
"""Render non-canonical review and reporting surfaces for normalization."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPORT_SCHEMA_VERSION = "1.0"


def should_render_review(
    page_index: int,
    page_count: int,
    cadence: int,
    transform_decision: str,
) -> bool:
    """Return whether a page belongs on the bounded visual-review surface."""
    regular_review_page = cadence > 0 and (
        page_index == 0 or page_index == page_count - 1 or page_index % cadence == 0
    )
    return regular_review_page or transform_decision == "apply"


def _display_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _fit_panel(image: np.ndarray, width: int = 700, height: int = 620) -> np.ndarray:
    image = _display_image(image)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def write_contact_sheet(
    path: Path,
    original: np.ndarray,
    normalized: np.ndarray,
    bounds: tuple[int, int, int, int],
    ordinal: int,
    collection_label: str,
) -> None:
    overlay = _display_image(original.copy())
    left, top, right, bottom = bounds
    cv2.rectangle(
        overlay,
        (left, top),
        (right - 1, bottom - 1),
        (0, 0, 255),
        max(2, round(min(original.shape[:2]) / 500)),
    )
    panels = []
    for image, label in (
        (overlay, "Canonical source + crop boundary"),
        (normalized, "Canonical normalized image"),
    ):
        panel = _fit_panel(image)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (20, 20, 20), -1)
        cv2.putText(
            panel,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    sheet = np.concatenate(panels, axis=1)
    cv2.putText(
        sheet,
        f"{collection_label} page {ordinal}",
        (12, sheet.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise ValueError(f"Could not write normalization contact sheet for page {ordinal}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "global_ordinal", "source_sha256", "output_sha256", "output_pixel_sha256",
        "base_crop_pixel_sha256", "transform_decision", "deskew_correction_degrees",
        "crop_left", "crop_top", "crop_right_exclusive", "crop_bottom_exclusive",
        "source_width", "source_height", "output_width", "output_height", "detector_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def render_summary(payload: dict[str, Any]) -> str:
    detector = payload["detector_selection"]
    label = payload["collection"]["id"]
    scope = "Golden Set validation" if payload["target"]["type"] == "golden-set" else "complete collection"
    transform = payload.get("transform_summary") or {}
    transform_description = (
        "Axis-aligned framing with conservative Hough deskew on pages that pass every safety gate. Gross page orientation is preserved."
        if transform.get("policy_requested") else
        "Axis-aligned framing only. No rotation or geometric resampling was performed."
    )
    lines = [
        f"# {label} Canonical Normalization",
        "",
        f"> Canonical {scope} normalization. {transform_description}",
        "",
        "## Result",
        "",
        f"- Status: `{payload['status']}`",
        f"- Policy: `{payload['policy']['id']}`",
        f"- Pages normalized: `{payload['page_count']}`",
        f"- Detector evidence: `{detector.get('detector')}` / `{detector.get('parameter_set_id')}`",
        f"- Canonical preprocess build: `{payload['canonical_preprocess']['effective_build_identity']}`",
        f"- Canonical preprocess result: `{payload['canonical_preprocess']['canonical_result_identity']}`",
        f"- Normalization identity: `{payload['normalization_identity']}`",
        f"- Canonical normalization result: `{payload['canonical_result_identity']}`",
    ]
    if transform.get("policy_requested"):
        lines.extend([
            f"- Pages conservatively deskewed: `{transform['pages_transformed']}`",
            f"- Pages preserved without resampling: `{transform['pages_preserved']}`",
        ])
    lines.extend([
        "",
        "## Review artifact",
        "",
        "Download and extract the review artifact, then open `index.html` locally. The red rectangle is the exact half-open crop boundary; the normalized panel shows the final policy output.",
        "",
    ])
    return "\n".join(lines)


def _write_html(path: Path, payload: dict[str, Any], review_ordinals: list[int]) -> None:
    cards = "".join(
        f'<article><h2>Page {ordinal}</h2><a href="contact-sheets/fs_{ordinal:04d}.jpg">'
        f'<img loading="lazy" src="contact-sheets/fs_{ordinal:04d}.jpg" alt="Normalization review for page {ordinal}"></a></article>'
        for ordinal in review_ordinals
    )
    policy = html.escape(payload["policy"]["id"])
    path.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(payload['collection']['id']))} Canonical Normalization</title><style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#111820;color:#e6edf3}}main{{max-width:1500px;margin:auto}}
article{{margin:2rem 0;padding:1rem;background:#18212b;border:1px solid #34404c;border-radius:.5rem}}img{{width:100%;height:auto}}
code{{background:#26313c;padding:.15rem .35rem;border-radius:.25rem}}
</style></head><body><main><h1>{html.escape(str(payload['collection']['id']))} Canonical Normalization</h1>
<p>Policy: <code>{policy}</code>. Gross page orientation is preserved; deskew is applied only when every recorded safety gate passes.</p>{cards}</main></body></html>""", encoding="utf-8")


def write_reports(
    output: Path,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    review_ordinals: list[int],
    cadence: int,
) -> None:
    """Write disposable review metadata and human-readable report surfaces."""
    review_manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "normalization-review",
        "contact_sheet_cadence": cadence,
        "contact_sheet_ordinals": review_ordinals,
    }
    (output / "review-manifest.json").write_text(
        json.dumps(review_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "normalization-manifest.csv", rows)
    (output / "summary.md").write_text(render_summary(payload), encoding="utf-8")
    _write_html(output / "index.html", payload, review_ordinals)
