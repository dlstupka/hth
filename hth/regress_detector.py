#!/usr/bin/env python3
"""Black-box detector regression against the approved Golden Set."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any

import cv2

try:
    from hth.geometry.common import document_mask, resize_for_analysis, scale_bbox, valid_bbox
    from hth.geometry.detector_grabcut import detect as detect_grabcut
except ModuleNotFoundError as exc:
    if exc.name != "hth":
        raise
    from geometry.common import document_mask, resize_for_analysis, scale_bbox, valid_bbox
    from geometry.detector_grabcut import detect as detect_grabcut

DETECTORS = {"grabcut": detect_grabcut}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--golden-set", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=("exhaustive", "binary-refine"), default="exhaustive")
    parser.add_argument("--max-dimension", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Development-only limit on parameter sets.")
    return parser.parse_args()


def canonical_parameters(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def parameter_set_id(parameters: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_parameters(parameters).encode("utf-8")).hexdigest()[:12]


def bbox_iou(a: list[int], b: list[int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def edge_errors(predicted: list[int], approved: list[int]) -> dict[str, int | float]:
    absolute = [abs(predicted[index] - approved[index]) for index in range(4)]
    return {
        "left": absolute[0],
        "top": absolute[1],
        "right": absolute[2],
        "bottom": absolute[3],
        "mean": sum(absolute) / 4.0,
        "maximum": max(absolute),
    }


def find_image(image_root: Path, ordinal: int) -> Path:
    candidates: list[Path] = []
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"):
        candidates.extend(
            (
                image_root / "raw" / f"fs_{ordinal:04d}{suffix}",
                image_root / f"fs_{ordinal:04d}{suffix}",
            )
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for Golden Set ordinal {ordinal}")


def load_pages(golden_set_path: Path, image_root: Path, maximum: int) -> list[dict[str, Any]]:
    golden_set = json.loads(golden_set_path.read_text(encoding="utf-8"))
    pages: list[dict[str, Any]] = []
    for page in golden_set.get("pages", []):
        approved = page.get("physical_document_bbox")
        if not valid_bbox(approved):
            continue
        ordinal = int(page["global_ordinal"])
        path = find_image(image_root, ordinal)
        original = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError(f"Could not read image: {path}")
        original_height, original_width = original.shape[:2]
        image, scale = resize_for_analysis(original, maximum)
        mask, mask_diagnostics = document_mask(image)
        pages.append(
            {
                "global_ordinal": ordinal,
                "label": page.get("label", f"page_{ordinal}"),
                "layout_type": page.get("layout_type", "other"),
                "approved_bbox": [int(value) for value in approved],
                "image_path": str(path),
                "image": image,
                "mask": mask,
                "mask_diagnostics": mask_diagnostics,
                "scale": scale,
                "original_width": original_width,
                "original_height": original_height,
            }
        )
    if not pages:
        raise ValueError("Golden Set contains no approved pages with valid bounding boxes")
    return pages


def exhaustive_parameter_sets(config: dict[str, Any]) -> list[dict[str, Any]]:
    names = list(config["parameters"])
    values = [config["parameters"][name]["values"] for name in names]
    return [dict(zip(names, combination, strict=True)) for combination in itertools.product(*values)]


def _value_index(values: list[Any], value: Any) -> int:
    try:
        return values.index(value)
    except ValueError:
        return min(range(len(values)), key=lambda index: abs(float(values[index]) - float(value)))


def evaluate_set(
    detector: Any,
    parameters: dict[str, Any],
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    page_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for page in pages:
        page_started = time.perf_counter()
        try:
            candidate = detector(
                image_bgr=page["image"], mask=page["mask"], parameters=parameters
            )
            elapsed_ms = (time.perf_counter() - page_started) * 1000.0
            if candidate.bbox is None:
                page_results.append(
                    {
                        "global_ordinal": page["global_ordinal"],
                        "label": page["label"],
                        "layout_type": page["layout_type"],
                        "status": candidate.status if candidate.status != "ok" else "no_candidate",
                        "iou": 0.0,
                        "edge_error_mean_px": None,
                        "edge_error_maximum_px": None,
                        "elapsed_ms": round(elapsed_ms, 3),
                        "candidate": asdict(candidate),
                    }
                )
                continue

            predicted = scale_bbox(
                candidate.bbox,
                1.0 / page["scale"],
                page["original_width"],
                page["original_height"],
            )
            approved = page["approved_bbox"]
            errors = edge_errors(predicted, approved)
            page_results.append(
                {
                    "global_ordinal": page["global_ordinal"],
                    "label": page["label"],
                    "layout_type": page["layout_type"],
                    "status": "ok",
                    "approved_bbox": approved,
                    "predicted_bbox": predicted,
                    "iou": round(bbox_iou(predicted, approved), 8),
                    "edge_error_mean_px": round(float(errors["mean"]), 3),
                    "edge_error_maximum_px": int(errors["maximum"]),
                    "elapsed_ms": round(elapsed_ms, 3),
                    "candidate": asdict(candidate),
                }
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - page_started) * 1000.0
            page_results.append(
                {
                    "global_ordinal": page["global_ordinal"],
                    "label": page["label"],
                    "layout_type": page["layout_type"],
                    "status": "error",
                    "iou": 0.0,
                    "edge_error_mean_px": None,
                    "edge_error_maximum_px": None,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )

    successful = [result for result in page_results if result["status"] == "ok"]
    ious = [float(result["iou"]) for result in page_results]
    edge_means = [float(result["edge_error_mean_px"]) for result in successful]
    elapsed = [float(result["elapsed_ms"]) for result in page_results]
    return {
        "parameter_set_id": parameter_set_id(parameters),
        "parameters": parameters,
        "summary": {
            "page_count": len(page_results),
            "success_count": len(successful),
            "failure_count": len(page_results) - len(successful),
            "mean_iou": round(sum(ious) / len(ious), 8),
            "minimum_iou": round(min(ious), 8),
            "mean_edge_error_px": round(sum(edge_means) / len(edge_means), 3) if edge_means else None,
            "elapsed_ms_total": round(sum(elapsed), 3),
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "pages": page_results,
    }


def ranking_key(result: dict[str, Any]) -> tuple[float, float, int, float]:
    summary = result["summary"]
    edge = summary["mean_edge_error_px"]
    return (
        -float(summary["mean_iou"]),
        -float(summary["minimum_iou"]),
        int(summary["failure_count"]),
        float(edge) if edge is not None else float("inf"),
    )


def binary_refine_parameter_sets(
    config: dict[str, Any],
    detector: Any,
    pages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Interval-halving coordinate search followed by a local Cartesian pass.

    The runner treats every parameter as an ordered black-box value list. It
    does not interpret detector behavior or parameter meaning.
    """
    parameter_specs = config["parameters"]
    current = dict(config["profiles"]["baseline"])
    evaluated: dict[str, dict[str, Any]] = {}

    def evaluate(parameters: dict[str, Any]) -> dict[str, Any]:
        key = canonical_parameters(parameters)
        if key not in evaluated:
            evaluated[key] = evaluate_set(detector, dict(parameters), pages)
        return evaluated[key]

    evaluate(current)
    passes = int(config.get("binary_refine", {}).get("passes", 3))
    for _ in range(passes):
        changed = False
        for name, spec in parameter_specs.items():
            values = list(spec["values"])
            if len(values) < 2:
                continue
            low = 0
            high = len(values) - 1
            best_parameters = dict(current)
            best_result = evaluate(best_parameters)
            while low <= high:
                middle = (low + high) // 2
                indices = sorted({low, middle, high})
                trial_results: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
                for index in indices:
                    trial = dict(current)
                    trial[name] = values[index]
                    trial_results.append((index, trial, evaluate(trial)))
                index, trial, result = min(trial_results, key=lambda item: ranking_key(item[2]))
                if ranking_key(result) < ranking_key(best_result):
                    best_parameters, best_result = trial, result
                if high - low <= 2:
                    break
                if index <= middle:
                    high = middle
                else:
                    low = middle
            if canonical_parameters(best_parameters) != canonical_parameters(current):
                current = best_parameters
                changed = True
        if not changed:
            break

    radius = int(config.get("binary_refine", {}).get("local_exhaustive_radius", 1))
    names = list(parameter_specs)
    local_values: list[list[Any]] = []
    for name in names:
        values = list(parameter_specs[name]["values"])
        center = _value_index(values, current[name])
        local_values.append(values[max(0, center - radius) : min(len(values), center + radius + 1)])
    local_sets = [dict(zip(names, combination, strict=True)) for combination in itertools.product(*local_values)]
    for parameters in local_sets:
        evaluate(parameters)
    return local_sets, list(evaluated.values())


def write_csv(path: Path, ranked: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "parameter_set_id",
        "profile",
        "mean_iou",
        "minimum_iou",
        "mean_edge_error_px",
        "failure_count",
        "elapsed_ms_total",
        "parameters_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(ranked, 1):
            summary = result["summary"]
            writer.writerow(
                {
                    "rank": rank,
                    "parameter_set_id": result["parameter_set_id"],
                    "profile": result.get("profile", ""),
                    "mean_iou": summary["mean_iou"],
                    "minimum_iou": summary["minimum_iou"],
                    "mean_edge_error_px": summary["mean_edge_error_px"],
                    "failure_count": summary["failure_count"],
                    "elapsed_ms_total": summary["elapsed_ms_total"],
                    "parameters_json": canonical_parameters(result["parameters"]),
                }
            )


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output directory exists; pass --overwrite: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.detector_config.read_text(encoding="utf-8"))
    detector_name = str(config["detector"])
    if detector_name not in DETECTORS:
        raise SystemExit(f"Unsupported detector: {detector_name}")
    detector = DETECTORS[detector_name]
    pages = load_pages(args.golden_set, args.image_root, args.max_dimension)

    if args.strategy == "exhaustive":
        parameter_sets = exhaustive_parameter_sets(config)
        if args.limit is not None:
            parameter_sets = parameter_sets[: args.limit]
        results = [evaluate_set(detector, parameters, pages) for parameters in parameter_sets]
    else:
        _, results = binary_refine_parameter_sets(config, detector, pages)

    profiles = {
        canonical_parameters(parameters): name
        for name, parameters in config.get("profiles", {}).items()
    }
    for result in results:
        result["profile"] = profiles.get(canonical_parameters(result["parameters"]))

    ranked = sorted(results, key=ranking_key)
    for rank, result in enumerate(ranked, 1):
        result["rank"] = rank

    baseline = next((result for result in ranked if result.get("profile") == "baseline"), None)
    report = {
        "schema_version": "0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "detector": detector_name,
        "strategy": args.strategy,
        "golden_set": str(args.golden_set),
        "image_root": str(args.image_root),
        "page_ordinals": [page["global_ordinal"] for page in pages],
        "parameter_set_count": len(ranked),
        "winner": ranked[0],
        "baseline": baseline,
        "results": ranked,
    }
    (args.output / "regression-results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "regression-ranking.csv", ranked)
    print(json.dumps({key: report[key] for key in ("detector", "strategy", "page_ordinals", "parameter_set_count", "winner", "baseline")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
