#!/usr/bin/env python3
"""HTH Stage 2 physical page analysis.

Analyzes extracted page images produced by hth/preprocess.py and emits compact,
reproducible JSON/CSV reports plus optional annotated previews.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import statistics
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageFilter,
    ImageOps,
    ImageStat,
    UnidentifiedImageError,
)

SCHEMA_VERSION = "0.1"


@dataclass
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)


@dataclass
class PageAnalysis:
    global_ordinal: int
    source_docx: str
    source_ordinal: int
    extracted_file: str
    sha256: str
    source_repository: str
    source_commit: str
    pipeline_repository: str
    pipeline_commit: str
    width_px: int
    height_px: int
    analysis_scale: float
    exif_orientation_applied: bool
    estimated_orientation_degrees: int
    estimated_skew_degrees: float
    skew_confidence: float
    content_left_px: int
    content_top_px: int
    content_right_px: int
    content_bottom_px: int
    content_width_px: int
    content_height_px: int
    margin_left_px: int
    margin_top_px: int
    margin_right_px: int
    margin_bottom_px: int
    brightness_mean: float
    contrast_stddev: float
    dynamic_range_p05_p95: float
    sharpness_score: float
    entropy_bits: float
    dark_pixel_fraction: float
    light_pixel_fraction: float
    background_uniformity_score: float
    bleed_through_proxy: float
    content_fraction: float
    border_contact: bool
    likely_blank: bool
    likely_low_contrast: bool
    likely_blurry: bool
    likely_overexposed: bool
    likely_underexposed: bool
    quality_score: float
    quality_status: str
    review_reasons: str
    analysis_error: str = ""


@dataclass
class AnalysisConfig:
    max_analysis_dimension: int = 1800
    background_threshold: int = 238
    minimum_component_area_fraction: float = 0.00002
    bbox_padding_fraction: float = 0.008
    skew_search_degrees: float = 3.0
    skew_step_degrees: float = 0.25
    blank_dark_fraction_max: float = 0.004
    low_contrast_stddev: float = 22.0
    blurry_sharpness_threshold: float = 8.0
    overexposed_brightness: float = 244.0
    underexposed_brightness: float = 65.0
    review_quality_threshold: float = 0.62
    fail_quality_threshold: float = 0.35


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze physical page geometry and image quality.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--annotated-previews", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--source-repository", default="")
    p.add_argument("--source-commit", default="")
    p.add_argument("--pipeline-repository", default="")
    p.add_argument("--pipeline-commit", default="")
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return p.parse_args()


def load_config(path: Path | None) -> AnalysisConfig:
    cfg = AnalysisConfig()
    if path is None:
        return cfg
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = {f.name for f in fields(AnalysisConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    values = asdict(cfg)
    values.update(raw)
    return AnalysisConfig(**values)


def percentile(hist: list[int], fraction: float) -> int:
    total = sum(hist)
    if total <= 0:
        return 0
    target = total * fraction
    cumulative = 0
    for value, count in enumerate(hist):
        cumulative += count
        if cumulative >= target:
            return value
    return 255


def entropy(hist: list[int]) -> float:
    total = sum(hist)
    if total <= 0:
        return 0.0
    value = 0.0
    for count in hist:
        if count:
            probability = count / total
            value -= probability * math.log2(probability)
    return value


def resize_for_analysis(image: Image.Image, max_dimension: int) -> tuple[Image.Image, float]:
    longest = max(image.size)
    if longest <= max_dimension:
        return image.copy(), 1.0
    scale = max_dimension / longest
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    ), scale


def projection_score(mask: Image.Image) -> float:
    width, height = mask.size
    pixels = mask.load()
    rows = [sum(1 for x in range(width) if pixels[x, y] > 0) for y in range(height)]
    if len(rows) < 3:
        return 0.0
    differences = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
    return sum(v * v for v in differences) / max(1, width * width)


def estimate_skew(gray: Image.Image, cfg: AnalysisConfig) -> tuple[float, float]:
    threshold = min(cfg.background_threshold, 215)
    mask = gray.point(lambda v: 255 if v < threshold else 0, mode="1").convert("L")
    if max(mask.size) > 1000:
        mask.thumbnail((1000, 1000), Image.Resampling.BILINEAR)
    candidates: list[tuple[float, float]] = []
    angle = -cfg.skew_search_degrees
    while angle <= cfg.skew_search_degrees + 1e-9:
        rotated = mask.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=0)
        candidates.append((projection_score(rotated), angle))
        angle += cfg.skew_step_degrees
    candidates.sort(reverse=True)
    best_score, best_angle = candidates[0]
    runner_up = candidates[1][0] if len(candidates) > 1 else 0.0
    confidence = 0.0 if best_score <= 0 else max(0.0, min(1.0, (best_score - runner_up) / best_score * 6.0))
    return round(-best_angle, 3), round(confidence, 3)


def content_bbox(mask: Image.Image, cfg: AnalysisConfig) -> BoundingBox | None:
    joined = mask.filter(ImageFilter.MaxFilter(5))
    bbox = joined.getbbox()
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    if (right - left) * (bottom - top) < mask.width * mask.height * cfg.minimum_component_area_fraction:
        return None
    px = round(mask.width * cfg.bbox_padding_fraction)
    py = round(mask.height * cfg.bbox_padding_fraction)
    return BoundingBox(max(0, left - px), max(0, top - py), min(mask.width, right + px), min(mask.height, bottom + py))


def scale_bbox(box: BoundingBox, inverse_scale: float, width: int, height: int) -> BoundingBox:
    return BoundingBox(
        max(0, min(width, round(box.left * inverse_scale))),
        max(0, min(height, round(box.top * inverse_scale))),
        max(0, min(width, round(box.right * inverse_scale))),
        max(0, min(height, round(box.bottom * inverse_scale))),
    )


def background_uniformity(gray: Image.Image) -> float:
    width, height = gray.size
    border = max(2, round(min(width, height) * 0.04))
    samples = [
        gray.crop((0, 0, width, border)),
        gray.crop((0, height - border, width, height)),
        gray.crop((0, 0, border, height)),
        gray.crop((width - border, 0, width, height)),
    ]
    mean_stddev = statistics.fmean(ImageStat.Stat(sample).stddev[0] for sample in samples)
    return max(0.0, min(1.0, 1.0 - mean_stddev / 55.0))


def bleed_proxy(gray: Image.Image) -> float:
    small = gray.copy()
    if max(small.size) > 800:
        small.thumbnail((800, 800), Image.Resampling.LANCZOS)
    background = small.filter(ImageFilter.GaussianBlur(radius=10))
    residual = ImageOps.autocontrast(ImageChops.difference(small, background))
    hist = residual.histogram()
    total = sum(hist) or 1
    return max(0.0, min(1.0, (sum(hist[20:85]) / total) * 2.5))


def assess_quality(brightness: float, contrast: float, sharpness: float, dark_fraction: float,
                   light_fraction: float, content_fraction: float, border_contact: bool,
                   cfg: AnalysisConfig) -> tuple[float, str, list[str], dict[str, bool]]:
    flags = {
        "likely_blank": dark_fraction <= cfg.blank_dark_fraction_max,
        "likely_low_contrast": contrast < cfg.low_contrast_stddev,
        "likely_blurry": sharpness < cfg.blurry_sharpness_threshold,
        "likely_overexposed": brightness > cfg.overexposed_brightness or light_fraction > 0.93,
        "likely_underexposed": brightness < cfg.underexposed_brightness,
    }
    reasons = []
    labels = {
        "likely_blank": "possible_blank_page",
        "likely_low_contrast": "low_contrast",
        "likely_blurry": "possible_blur",
        "likely_overexposed": "possible_overexposure",
        "likely_underexposed": "possible_underexposure",
    }
    reasons.extend(labels[k] for k, value in flags.items() if value)
    if border_contact:
        reasons.append("content_touches_border")
    if content_fraction < 0.05 and not flags["likely_blank"]:
        reasons.append("very_small_content_region")
    score = (
        max(0.0, 1.0 - abs(brightness - 185.0) / 150.0) * 0.22
        + max(0.0, min(1.0, contrast / 60.0)) * 0.30
        + max(0.0, min(1.0, sharpness / 35.0)) * 0.30
        + max(0.0, min(1.0, content_fraction / 0.28)) * 0.18
    )
    if border_contact:
        score -= 0.08
    if flags["likely_blank"]:
        score = min(score, 0.25)
    score = round(max(0.0, min(1.0, score)), 3)
    status = "fail" if score < cfg.fail_quality_threshold else "review" if score < cfg.review_quality_threshold or reasons else "pass"
    return score, status, reasons, flags


def analyze_record(record: dict[str, Any], image_root: Path, cfg: AnalysisConfig,
                   args: argparse.Namespace) -> tuple[PageAnalysis, Image.Image | None, BoundingBox | None]:
    ordinal = int(record.get("global_ordinal", 0))
    relative = Path(str(record["extracted_file"]))
    path = image_root / relative
    base = dict(
        global_ordinal=ordinal,
        source_docx=str(record.get("source_docx", "")),
        source_ordinal=int(record.get("source_ordinal", 0)),
        extracted_file=relative.as_posix(),
        sha256=str(record.get("sha256", "")),
        source_repository=str(record.get("source_repository") or args.source_repository),
        source_commit=str(record.get("source_commit") or args.source_commit),
        pipeline_repository=str(record.get("pipeline_repository") or args.pipeline_repository),
        pipeline_commit=str(record.get("pipeline_commit") or args.pipeline_commit),
    )
    try:
        with Image.open(path) as opened:
            opened.load()
            before = opened.size
            oriented = ImageOps.exif_transpose(opened)
            exif_applied = oriented.size != before
            rgb = oriented.convert("RGB")
        width, height = rgb.size
        working, scale = resize_for_analysis(rgb, cfg.max_analysis_dimension)
        gray = ImageOps.grayscale(working)
        hist = gray.histogram()
        stat = ImageStat.Stat(gray)
        brightness = float(stat.mean[0])
        contrast = float(stat.stddev[0])
        p05, p95 = percentile(hist, 0.05), percentile(hist, 0.95)
        total = max(1, gray.width * gray.height)
        dark_fraction = sum(hist[:80]) / total
        light_fraction = sum(hist[245:]) / total
        sharpness = float(ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).var[0])
        mask = gray.point(lambda v: 255 if v < cfg.background_threshold else 0, mode="1").convert("L")
        small_box = content_bbox(mask, cfg)
        full_box = BoundingBox(0, 0, 0, 0) if small_box is None else scale_bbox(small_box, 1.0 / scale, width, height)
        content_fraction = full_box.width * full_box.height / max(1, width * height)
        tolerance = max(3, round(min(width, height) * 0.005))
        border_contact = full_box.width > 0 and (
            full_box.left <= tolerance or full_box.top <= tolerance
            or width - full_box.right <= tolerance or height - full_box.bottom <= tolerance
        )
        skew, skew_confidence = estimate_skew(gray, cfg)
        score, status, reasons, flags = assess_quality(
            brightness, contrast, sharpness, dark_fraction, light_fraction,
            content_fraction, border_contact, cfg
        )
        result = PageAnalysis(
            **base,
            width_px=width, height_px=height, analysis_scale=round(scale, 6),
            exif_orientation_applied=exif_applied, estimated_orientation_degrees=0,
            estimated_skew_degrees=skew, skew_confidence=skew_confidence,
            content_left_px=full_box.left, content_top_px=full_box.top,
            content_right_px=full_box.right, content_bottom_px=full_box.bottom,
            content_width_px=full_box.width, content_height_px=full_box.height,
            margin_left_px=full_box.left, margin_top_px=full_box.top,
            margin_right_px=max(0, width - full_box.right), margin_bottom_px=max(0, height - full_box.bottom),
            brightness_mean=round(brightness, 3), contrast_stddev=round(contrast, 3),
            dynamic_range_p05_p95=float(p95 - p05), sharpness_score=round(sharpness, 3),
            entropy_bits=round(entropy(hist), 4), dark_pixel_fraction=round(dark_fraction, 6),
            light_pixel_fraction=round(light_fraction, 6),
            background_uniformity_score=round(background_uniformity(gray), 3),
            bleed_through_proxy=round(bleed_proxy(gray), 3),
            content_fraction=round(content_fraction, 6), border_contact=border_contact,
            likely_blank=flags["likely_blank"], likely_low_contrast=flags["likely_low_contrast"],
            likely_blurry=flags["likely_blurry"], likely_overexposed=flags["likely_overexposed"],
            likely_underexposed=flags["likely_underexposed"], quality_score=score,
            quality_status=status, review_reasons=";".join(reasons),
        )
        return result, rgb, full_box
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        logging.exception("Failed page %s", ordinal)
        zero = dict(width_px=int(record.get("width_px", 0)), height_px=int(record.get("height_px", 0)),
                    analysis_scale=1.0, exif_orientation_applied=False, estimated_orientation_degrees=0,
                    estimated_skew_degrees=0.0, skew_confidence=0.0, content_left_px=0,
                    content_top_px=0, content_right_px=0, content_bottom_px=0, content_width_px=0,
                    content_height_px=0, margin_left_px=0, margin_top_px=0, margin_right_px=0,
                    margin_bottom_px=0, brightness_mean=0.0, contrast_stddev=0.0,
                    dynamic_range_p05_p95=0.0, sharpness_score=0.0, entropy_bits=0.0,
                    dark_pixel_fraction=0.0, light_pixel_fraction=0.0,
                    background_uniformity_score=0.0, bleed_through_proxy=0.0,
                    content_fraction=0.0, border_contact=False, likely_blank=False,
                    likely_low_contrast=False, likely_blurry=False, likely_overexposed=False,
                    likely_underexposed=False, quality_score=0.0, quality_status="error",
                    review_reasons="analysis_error", analysis_error=str(exc))
        return PageAnalysis(**base, **zero), None, None


def save_preview(image: Image.Image, box: BoundingBox | None, destination: Path, result: PageAnalysis) -> None:
    preview = image.copy()
    preview.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(preview)
    if box and box.width > 0:
        sx, sy = preview.width / image.width, preview.height / image.height
        draw.rectangle((round(box.left * sx), round(box.top * sy), round(box.right * sx), round(box.bottom * sy)), outline=(220, 30, 30), width=3)
    label = f"FS {result.global_ordinal:04d} | {result.quality_status.upper()} | score {result.quality_score:.3f} | skew {result.estimated_skew_degrees:+.2f} deg"
    draw.rectangle((0, 0, preview.width, 34), fill=(255, 255, 255))
    draw.text((8, 8), label, fill=(0, 0, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    preview.save(destination, "JPEG", quality=88, optimize=True)


def write_csv(records: Iterable[PageAnalysis], path: Path) -> None:
    rows = [asdict(r) for r in records]
    names = [f.name for f in fields(PageAnalysis)]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records_in = manifest.get("records")
    if not isinstance(records_in, list):
        raise ValueError("Manifest does not contain a records list")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; use --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    results: list[PageAnalysis] = []
    for index, record in enumerate(records_in, start=1):
        logging.info("Analyzing page %s (%d/%d)", record.get("global_ordinal", index), index, len(records_in))
        result, image, box = analyze_record(record, args.image_root, cfg, args)
        results.append(result)
        if args.annotated_previews and image is not None:
            save_preview(image, box, args.output / "annotated-previews" / f"fs_{result.global_ordinal:04d}_page-analysis.jpg", result)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "collection_id": manifest.get("collection_id", ""),
        "collection_title": manifest.get("collection_title", ""),
        "page_count": len(results),
        "records": [asdict(r) for r in results],
    }
    (args.output / "page-analysis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(results, args.output / "page-analysis.csv")
    review = [r for r in results if r.quality_status in {"review", "fail", "error"}]
    write_csv(review, args.output / "review-queue.csv")
    scores = [r.quality_score for r in results if r.quality_status != "error"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "collection_id": manifest.get("collection_id", ""),
        "collection_title": manifest.get("collection_title", ""),
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "page_count": len(results),
        "quality_status_counts": {status: sum(r.quality_status == status for r in results) for status in ("pass", "review", "fail", "error")},
        "mean_quality_score": round(statistics.fmean(scores), 3) if scores else None,
        "median_quality_score": round(statistics.median(scores), 3) if scores else None,
        "review_queue_count": len(review),
        "limitations": [
            "estimated_orientation_degrees is fixed at 0 after EXIF transpose",
            "bleed_through_proxy is a triage heuristic, not a diagnosis",
            "quality flags require human validation",
        ],
    }
    (args.output / "analysis-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Stage 2 complete: %d pages, %d queued for review", len(results), len(review))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logging.exception("Page analysis failed: %s", exc)
        raise SystemExit(1)
