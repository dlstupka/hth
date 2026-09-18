"""Assess, validate, and integrate deterministic contrast/tonal normalization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .canonical_build_evidence import canonical_hash
from .normalize_document_images import _pixel_sha256

SCHEMA_VERSION = "1.0"
RELEASE_TYPE = "canonical-tonal-normalized-collection"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_path(collection: Path, ordinal: int) -> Path:
    candidates = tuple((collection / "photometric-normalized").glob(f"fs_{ordinal:04d}.*"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one photometric input image for page {ordinal}, found {len(candidates)}")
    return candidates[0]


def luminance(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32) / 255.0
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) / 255.0


def measure(image: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    light = luminance(image)
    p = config["percentiles"]
    values = np.percentile(light, [p["black"], p["shadow"], p["median"], p["highlight"], p["white"]])
    black, shadow, median, highlight, white = (float(value) for value in values)
    dark_clip = float(np.mean(light <= 1.0 / 255.0))
    bright_clip = float(np.mean(light >= 254.0 / 255.0))
    span = highlight - shadow
    gates = config["candidate_gates"]
    reasons: list[str] = []
    if span < float(gates["minimum_candidate_tonal_span"]):
        decision = "review"
        reasons.append("extremely-compressed-tonal-range")
    elif dark_clip + bright_clip > float(gates["maximum_existing_endpoint_clipping_fraction"]):
        decision = "review"
        reasons.append("existing-endpoint-clipping")
    elif span < float(gates["minimum_adequate_tonal_span"]):
        decision = "correction-candidate"
        reasons.append("compressed-tonal-range")
    else:
        decision = "preserve"
        reasons.append("tonal-range-adequate")
    return {
        "black_percentile": black,
        "shadow_percentile": shadow,
        "median_luminance": median,
        "highlight_percentile": highlight,
        "white_percentile": white,
        "tonal_span": span,
        "dark_clipping_fraction": dark_clip,
        "bright_clipping_fraction": bright_clip,
        "decision": decision,
        "decision_reasons": reasons,
        "pipeline_action": "evaluate-correction" if decision == "correction-candidate" else "preserve-and-continue",
    }


def apply_method(image: np.ndarray, method: dict[str, Any]) -> np.ndarray:
    if method.get("mode") != "percentile-stretch":
        raise ValueError(f"Unsupported tonal method: {method.get('mode')!r}")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) if image.ndim == 3 else None
    channel = (lab[:, :, 0] if lab is not None else image).astype(np.float32) / 255.0
    low, high = np.percentile(channel, [float(method["low_percentile"]), float(method["high_percentile"])])
    if high <= low + 1e-6:
        return image.copy()
    mapped = (channel - low) / (high - low)
    mapped = float(method["output_low"]) + np.clip(mapped, 0.0, 1.0) * (
        float(method["output_high"]) - float(method["output_low"])
    )
    strength = float(method["strength"])
    output_l = np.clip(channel * (1.0 - strength) + mapped * strength, 0.0, 1.0)
    encoded = np.rint(output_l * 255.0).astype(np.uint8)
    if lab is None:
        return encoded
    lab[:, :, 0] = encoded
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _high_frequency_correlation(before: np.ndarray, after: np.ndarray) -> float:
    first = cv2.Laplacian(luminance(before), cv2.CV_32F).ravel()
    second = cv2.Laplacian(luminance(after), cv2.CV_32F).ravel()
    if float(np.std(first)) < 1e-8 or float(np.std(second)) < 1e-8:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(np.corrcoef(first, second)[0, 1])


def evaluate(before: np.ndarray, after: np.ndarray, assessment_config: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    original = measure(before, assessment_config)
    result = measure(after, assessment_config)
    new_clip = max(
        0.0,
        result["dark_clipping_fraction"] + result["bright_clipping_fraction"]
        - original["dark_clipping_fraction"] - original["bright_clipping_fraction"],
    )
    metrics = {
        "tonal_span_gain": result["tonal_span"] - original["tonal_span"],
        "high_frequency_correlation": _high_frequency_correlation(before, after),
        "new_endpoint_clipping_fraction": new_clip,
        "absolute_median_luminance_shift": abs(result["median_luminance"] - original["median_luminance"]),
    }
    checks = {
        "tonal_span_gain": metrics["tonal_span_gain"] >= float(gates["minimum_tonal_span_gain"]),
        "detail_correlation": metrics["high_frequency_correlation"] >= float(gates["minimum_high_frequency_correlation"]),
        "endpoint_clipping": metrics["new_endpoint_clipping_fraction"] <= float(gates["maximum_new_endpoint_clipping_fraction"]),
        "median_shift": metrics["absolute_median_luminance_shift"] <= float(gates["maximum_absolute_median_luminance_shift"]),
    }
    return {**metrics, "gates": checks, "safe": all(checks.values())}


def _upstream_pages(collection: Path, manifest: dict[str, Any]) -> list[tuple[int, dict[str, Any], np.ndarray]]:
    rows = []
    for page in sorted(manifest.get("pages") or [], key=lambda item: int(item["global_ordinal"])):
        ordinal = int(page["global_ordinal"])
        image = cv2.imread(str(_image_path(collection, ordinal)), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not decode photometric page {ordinal}")
        if _pixel_sha256(image) != str(page.get("output_pixel_sha256") or ""):
            raise ValueError(f"Photometric page {ordinal} does not match its pixel identity")
        rows.append((ordinal, page, image))
    if not rows:
        raise ValueError("Photometric manifest contains no pages")
    return rows


def assess(collection: Path, upstream: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    pages = []
    modulus = int(config["partition"]["development_modulus"])
    residue = int(config["partition"]["development_residue"])
    for ordinal, source, image in _upstream_pages(collection, upstream):
        result = measure(image, config)
        partition = "development" if ordinal % modulus == residue else "held-out"
        pages.append({
            "global_ordinal": ordinal,
            "input_pixel_sha256": source["output_pixel_sha256"],
            "partition": partition,
            "measurement": result,
        })
    counts = {name: sum(page["measurement"]["decision"] == name for page in pages) for name in ("correction-candidate", "preserve", "review")}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "assessment_type": config["assessment_type"],
        "upstream_photometric_result_identity": upstream["photometric_result_identity"],
        "config": config,
        "aggregate": {"page_count": len(pages), **counts},
        "pages": pages,
    }
    payload["assessment_identity"] = canonical_hash(payload)
    return payload


def compare(collection: Path, upstream: dict[str, Any], assessment: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    by_ordinal = {ordinal: image for ordinal, _, image in _upstream_pages(collection, upstream)}
    candidates = [page for page in assessment["pages"] if page["partition"] == "development" and page["measurement"]["decision"] == "correction-candidate"]
    pages = []
    for page in candidates:
        ordinal = int(page["global_ordinal"])
        variants = []
        for method in config["methods"]:
            output = apply_method(by_ordinal[ordinal], method)
            result = evaluate(by_ordinal[ordinal], output, assessment["config"], config["safety_gates"])
            variants.append({"method_id": method["id"], "output_pixel_sha256": _pixel_sha256(output), **result})
        pages.append({"global_ordinal": ordinal, "input_pixel_sha256": page["input_pixel_sha256"], "variants": variants})
    globally_safe = [
        method["id"] for method in config["methods"]
        if pages and all(next(v for v in page["variants"] if v["method_id"] == method["id"])["safe"] for page in pages)
    ]
    recommended = globally_safe[0] if globally_safe else None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "assessment_type": config["assessment_type"],
        "upstream_photometric_result_identity": upstream["photometric_result_identity"],
        "tonal_assessment_identity": assessment["assessment_identity"],
        "config": config,
        "candidate_count": len(pages),
        "globally_safe_methods": globally_safe,
        "recommended_method_id": recommended,
        "pages": pages,
    }
    payload["method_assessment_identity"] = canonical_hash(payload)
    return payload


def validate(collection: Path, upstream: dict[str, Any], assessment: dict[str, Any], comparison: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    images = {ordinal: image for ordinal, _, image in _upstream_pages(collection, upstream)}
    method_id = comparison.get("recommended_method_id")
    method = next((item for item in comparison["config"]["methods"] if item["id"] == method_id), None)
    candidates = [page for page in assessment["pages"] if page["partition"] == "held-out" and page["measurement"]["decision"] == "correction-candidate"]
    pages = []
    for page in candidates:
        ordinal = int(page["global_ordinal"])
        if method is None:
            pages.append({"global_ordinal": ordinal, "safe": False, "reason": "no-development-method"})
            continue
        output = apply_method(images[ordinal], method)
        result = evaluate(images[ordinal], output, assessment["config"], comparison["config"]["safety_gates"])
        pages.append({"global_ordinal": ordinal, "input_pixel_sha256": page["input_pixel_sha256"], "output_pixel_sha256": _pixel_sha256(output), **result})
    development_count = int(comparison["candidate_count"])
    safe = method is not None and all(page.get("safe") is True for page in pages)
    gains = [float(page["tonal_span_gain"]) for page in pages if "tonal_span_gain" in page]
    mean_gain = float(np.mean(gains)) if gains else 0.0
    enough_gain = not pages or mean_gain >= float(config["gates"]["minimum_mean_tonal_span_gain"])
    decision = "apply" if safe and enough_gain and (development_count + len(pages) > 0) else "preserve"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "validation_type": config["validation_type"],
        "upstream_photometric_result_identity": upstream["photometric_result_identity"],
        "tonal_assessment_identity": assessment["assessment_identity"],
        "method_assessment_identity": comparison["method_assessment_identity"],
        "config": config,
        "method": method,
        "aggregate": {"held_out_candidates": len(pages), "safe_candidates": sum(page.get("safe") is True for page in pages), "mean_tonal_span_gain": mean_gain},
        "decision": decision,
        "pages": pages,
    }
    payload["validation_identity"] = canonical_hash(payload)
    return payload


def integrate(collection: Path, upstream: dict[str, Any], assessment: dict[str, Any], comparison: dict[str, Any], validation: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Tonal output is not empty: {output}")
    images_out = output / "tonal-normalized"
    images_out.mkdir(parents=True, exist_ok=True)
    comparison_pages = {int(page["global_ordinal"]): page for page in comparison["pages"]}
    validation_pages = {int(page["global_ordinal"]): page for page in validation["pages"]}
    method = validation.get("method")
    apply_enabled = validation.get("decision") == "apply" and method is not None
    rows = []
    for ordinal, source, image in _upstream_pages(collection, upstream):
        assessed = next(page for page in assessment["pages"] if int(page["global_ordinal"]) == ordinal)
        candidate = assessed["measurement"]["decision"] == "correction-candidate"
        apply = apply_enabled and candidate
        result = apply_method(image, method) if apply else image
        evidence = comparison_pages.get(ordinal) or validation_pages.get(ordinal)
        if apply:
            if not evidence:
                raise ValueError(f"Tonal candidate page {ordinal} has no method evidence")
            expected = next((item["output_pixel_sha256"] for item in evidence.get("variants", []) if item["method_id"] == method["id"]), evidence.get("output_pixel_sha256"))
            if _pixel_sha256(result) != expected:
                raise ValueError(f"Tonal output page {ordinal} does not reproduce validated evidence")
        target = images_out / f"fs_{ordinal:04d}.png"
        if not cv2.imwrite(str(target), result, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise ValueError(f"Could not write tonal page {ordinal}")
        round_trip = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
        if round_trip is None or not np.array_equal(round_trip, result):
            raise ValueError(f"Tonal page {ordinal} failed lossless round-trip")
        rows.append({
            "global_ordinal": ordinal,
            "route": "apply" if apply else "preserve",
            "pipeline_action": "corrected-and-continue" if apply else "preserve-and-continue",
            "decision_before": assessed["measurement"]["decision"],
            "input_pixel_sha256": source["output_pixel_sha256"],
            "output_pixel_sha256": _pixel_sha256(result),
            "output_sha256": _sha256(target),
            "output_width": int(result.shape[1]),
            "output_height": int(result.shape[0]),
        })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "integration_type": "contrast-tonal-normalization",
        "status": "complete",
        "upstream_photometric_result_identity": upstream["photometric_result_identity"],
        "tonal_assessment_identity": assessment["assessment_identity"],
        "method_assessment_identity": comparison["method_assessment_identity"],
        "validation_identity": validation["validation_identity"],
        "method": method,
        "aggregate": {"page_count": len(rows), "corrected_pages": sum(page["route"] == "apply" for page in rows), "preserved_pages": sum(page["route"] == "preserve" for page in rows)},
        "pages": rows,
    }
    payload["tonal_result_identity"] = canonical_hash(payload)
    _write(output / "tonal-normalization-manifest.json", payload)
    with (output / "tonal-normalization-manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.md").write_text(
        "# HTH Contrast and Tonal Normalization\n\n"
        f"- Input pages: {len(rows)}\n- Corrected pages: {payload['aggregate']['corrected_pages']}\n"
        f"- Preserved pages: {payload['aggregate']['preserved_pages']}\n- Result identity: `{payload['tonal_result_identity']}`\n",
        encoding="utf-8",
    )
    return payload


def package_release(collection: Path, asset: Path, tag: str) -> dict[str, Any]:
    manifest = _read(collection / "tonal-normalization-manifest.json")
    identity = str(manifest.get("tonal_result_identity") or "")
    if tag != f"HTH-TONAL-{identity}":
        raise ValueError("Tonal release tag does not match result identity")
    asset.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(asset, "w", allowZip64=True) as archive:
        for path in sorted(item for item in collection.rglob("*") if item.is_file() and item.name != "release.json"):
            info = zipfile.ZipInfo(path.relative_to(collection).as_posix(), _ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return {"schema_version": SCHEMA_VERSION, "release_type": RELEASE_TYPE, "tonal_result_identity": identity, "tag": tag, "asset": asset.name, "asset_sha256": _sha256(asset)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("assess", "compare", "validate", "integrate"):
        command = commands.add_parser(name)
        command.add_argument("--collection", type=Path, required=True)
        command.add_argument("--upstream-manifest", type=Path, required=True)
        command.add_argument("--assessment", type=Path)
        command.add_argument("--comparison", type=Path)
        command.add_argument("--validation", type=Path)
        command.add_argument("--config", type=Path)
        command.add_argument("--output", type=Path, required=True)
    package = commands.add_parser("package")
    package.add_argument("--collection", type=Path, required=True)
    package.add_argument("--asset", type=Path, required=True)
    package.add_argument("--tag", required=True)
    package.add_argument("--output-record", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "package":
        record = package_release(args.collection, args.asset, args.tag)
        _write(args.output_record, record)
        return 0
    upstream = _read(args.upstream_manifest)
    config = _read(args.config) if args.config else None
    if args.command == "assess":
        payload = assess(args.collection, upstream, config)
    elif args.command == "compare":
        payload = compare(args.collection, upstream, _read(args.assessment), config)
    elif args.command == "validate":
        payload = validate(args.collection, upstream, _read(args.assessment), _read(args.comparison), config)
    else:
        payload = integrate(args.collection, upstream, _read(args.assessment), _read(args.comparison), _read(args.validation), args.output)
        return 0
    _write(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
