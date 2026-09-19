"""Assess and selectively integrate restoration and binarization normalization."""
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
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
DOMAINS = {
    "denoising": {
        "upstream_folder": "chromatic-normalized",
        "upstream_manifest": "chromatic-normalization-manifest.json",
        "upstream_identity": "chromatic_result_identity",
        "output_folder": "denoising-normalized",
        "manifest": "denoising-normalization-manifest.json",
        "csv": "denoising-normalization-manifest.csv",
        "identity": "denoising_result_identity",
        "release_prefix": "HTH-DENOISING",
    },
    "sharpening": {
        "upstream_folder": "denoising-normalized",
        "upstream_manifest": "denoising-normalization-manifest.json",
        "upstream_identity": "denoising_result_identity",
        "output_folder": "sharpening-normalized",
        "manifest": "sharpening-normalization-manifest.json",
        "csv": "sharpening-normalization-manifest.csv",
        "identity": "sharpening_result_identity",
        "release_prefix": "HTH-SHARPENING",
    },
    "binarization": {
        "upstream_folder": "sharpening-normalized",
        "upstream_manifest": "sharpening-normalization-manifest.json",
        "upstream_identity": "sharpening_result_identity",
        "output_folder": "binarization-normalized",
        "manifest": "binarization-normalization-manifest.json",
        "csv": "binarization-normalization-manifest.csv",
        "identity": "binarization_result_identity",
        "release_prefix": "HTH-BINARIZATION",
    },
}


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


def _gray(image: np.ndarray) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    a, b = first.astype(np.float32).ravel(), second.astype(np.float32).ravel()
    if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def metrics(image: np.ndarray) -> dict[str, float]:
    gray = _gray(image)
    gray_u8 = np.rint(gray * 255).astype(np.uint8)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)
    residual = gray - smooth
    median = cv2.medianBlur(gray_u8, 3).astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    threshold, binary = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = binary == 0
    dark = gray[foreground]
    light = gray[~foreground]
    foreground_background_contrast = (
        float(np.mean(light) - np.mean(dark)) if dark.size and light.size else 0.0
    )
    return {
        "noise_sigma": float(np.median(np.abs(residual - np.median(residual))) / 0.6745),
        "impulse_fraction": float(np.mean(np.abs(gray - median) >= 0.10)),
        "detail_energy": float(np.std(laplacian)),
        "edge_fraction": float(np.mean(np.abs(laplacian) >= 0.08)),
        "median_luminance": float(np.median(gray)),
        "clipping_fraction": float(np.mean((image <= 0) | (image >= 255))),
        "luminance_span": float(np.percentile(gray, 95) - np.percentile(gray, 5)),
        "otsu_threshold": float(threshold / 255.0),
        "foreground_fraction": float(np.mean(foreground)),
        "foreground_background_contrast": foreground_background_contrast,
    }


def measure(image: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    domain = config["domain"]
    values = metrics(image)
    gates = config["candidate_gates"]
    reasons: list[str] = []
    if domain == "denoising":
        candidate = (
            values["noise_sigma"] >= float(gates["minimum_noise_sigma"])
            or values["impulse_fraction"] >= float(gates["minimum_impulse_fraction"])
        )
        decision = "correction-candidate" if candidate else "preserve"
        reasons.append("measurable-noise-or-artifacts" if candidate else "noise-within-bounds")
    elif domain == "sharpening":
        if values["noise_sigma"] > float(gates["maximum_safe_noise_sigma"]):
            decision, reasons = "review", ["residual-noise-may-be-amplified"]
        elif values["detail_energy"] <= float(gates["maximum_candidate_detail_energy"]):
            decision, reasons = "correction-candidate", ["low-detail-energy"]
        else:
            decision, reasons = "preserve", ["detail-energy-adequate"]
    elif domain == "binarization":
        plausible_ink = (
            float(gates["minimum_foreground_fraction"])
            <= values["foreground_fraction"]
            <= float(gates["maximum_foreground_fraction"])
        )
        separable = (
            values["foreground_background_contrast"]
            >= float(gates["minimum_foreground_background_contrast"])
            and values["luminance_span"] >= float(gates["minimum_luminance_span"])
        )
        if not plausible_ink:
            decision, reasons = "review", ["implausible-foreground-coverage"]
        elif separable:
            decision, reasons = "correction-candidate", ["foreground-background-separable"]
        else:
            decision, reasons = "review", ["foreground-background-not-safely-separable"]
    else:
        raise ValueError(f"Unsupported restoration domain: {domain}")
    return {
        **values,
        "decision": decision,
        "decision_reasons": reasons,
        "pipeline_action": "evaluate-correction" if decision == "correction-candidate" else "preserve-and-continue",
    }


def apply_method(image: np.ndarray, method: dict[str, Any]) -> np.ndarray:
    alpha = None
    working = image
    if image.ndim == 3 and image.shape[2] == 4:
        working = image[:, :, :3]
        alpha = image[:, :, 3].copy()
    mode = method["mode"]
    if mode == "bilateral":
        sigma = float(method["sigma"])
        result = cv2.bilateralFilter(working, 5, sigma, sigma)
    elif mode == "median":
        result = cv2.medianBlur(working, int(method.get("kernel", 3)))
    elif mode == "unsharp-mask":
        strength = float(method["strength"])
        blurred = cv2.GaussianBlur(working, (0, 0), float(method.get("radius", 1.0)))
        enhanced = working.astype(np.float32) + strength * (
            working.astype(np.float32) - blurred.astype(np.float32)
        )
        result = np.clip(np.rint(enhanced), 0, 255).astype(np.uint8)
    elif mode == "global-otsu":
        gray = np.rint(_gray(working) * 255).astype(np.uint8)
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        alpha = None
    elif mode == "adaptive-gaussian":
        gray = np.rint(_gray(working) * 255).astype(np.uint8)
        block_size = int(method["block_size"])
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError("adaptive-gaussian block_size must be odd and at least 3")
        result = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            block_size, float(method["c"]),
        )
        alpha = None
    elif mode == "sauvola":
        gray = _gray(working).astype(np.float32)
        window = int(method["window"])
        if window < 3 or window % 2 == 0:
            raise ValueError("sauvola window must be odd and at least 3")
        mean = cv2.boxFilter(gray, cv2.CV_32F, (window, window), normalize=True)
        square_mean = cv2.boxFilter(gray * gray, cv2.CV_32F, (window, window), normalize=True)
        deviation = np.sqrt(np.maximum(square_mean - mean * mean, 0.0))
        threshold = mean * (1.0 + float(method["k"]) * (deviation / float(method.get("r", 0.5)) - 1.0))
        result = np.where(gray > threshold, 255, 0).astype(np.uint8)
        alpha = None
    else:
        raise ValueError(f"Unsupported restoration method: {mode}")
    if alpha is not None:
        return np.dstack((result, alpha))
    return result


def evaluate(before: np.ndarray, after: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    domain, gates = config["domain"], config["safety_gates"]
    original, result = metrics(before), metrics(after)
    before_gray, after_gray = _gray(before), _gray(after)
    before_lap = cv2.Laplacian(before_gray, cv2.CV_32F)
    after_lap = cv2.Laplacian(after_gray, cv2.CV_32F)
    detail_ratio = result["detail_energy"] / max(original["detail_energy"], 1e-8)
    values = {
        "noise_reduction_fraction": (original["noise_sigma"] - result["noise_sigma"]) / max(original["noise_sigma"], 1e-8),
        "detail_gain_fraction": detail_ratio - 1.0,
        "detail_correlation": _correlation(before_lap, after_lap),
        "edge_retention_fraction": result["edge_fraction"] / max(original["edge_fraction"], 1e-8),
        "new_clipping_fraction": max(0.0, result["clipping_fraction"] - original["clipping_fraction"]),
        "absolute_median_luminance_shift": abs(result["median_luminance"] - original["median_luminance"]),
    }
    if domain == "binarization":
        before_u8 = np.rint(before_gray * 255).astype(np.uint8)
        _, reference = cv2.threshold(before_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        reference_ink = reference == 0
        result_ink = after_gray < 0.5
        union = int(np.count_nonzero(reference_ink | result_ink))
        intersection = int(np.count_nonzero(reference_ink & result_ink))
        reference_components = max(1, cv2.connectedComponents(reference_ink.astype(np.uint8), 8)[0] - 1)
        result_components = max(1, cv2.connectedComponents(result_ink.astype(np.uint8), 8)[0] - 1)
        values.update({
            "foreground_agreement": intersection / union if union else 1.0,
            "output_foreground_fraction": float(np.mean(result_ink)),
            "component_inflation_ratio": result_components / reference_components,
            "edge_correlation": _correlation(before_lap, after_lap),
        })
    if domain == "denoising":
        checks = {
            "noise_reduction": values["noise_reduction_fraction"] >= float(gates["minimum_noise_reduction_fraction"]),
            "detail_correlation": values["detail_correlation"] >= float(gates["minimum_detail_correlation"]),
            "edge_retention": values["edge_retention_fraction"] >= float(gates["minimum_edge_retention_fraction"]),
            "clipping": values["new_clipping_fraction"] <= float(gates["maximum_new_clipping_fraction"]),
            "median_luminance_shift": values["absolute_median_luminance_shift"] <= float(gates["maximum_absolute_median_luminance_shift"]),
        }
    elif domain == "sharpening":
        checks = {
            "minimum_detail_gain": values["detail_gain_fraction"] >= float(gates["minimum_detail_gain_fraction"]),
            "maximum_detail_gain": values["detail_gain_fraction"] <= float(gates["maximum_detail_gain_fraction"]),
            "detail_correlation": values["detail_correlation"] >= float(gates["minimum_detail_correlation"]),
            "noise_amplification": values["noise_reduction_fraction"] >= -float(gates["maximum_noise_amplification_fraction"]),
            "clipping": values["new_clipping_fraction"] <= float(gates["maximum_new_clipping_fraction"]),
            "median_luminance_shift": values["absolute_median_luminance_shift"] <= float(gates["maximum_absolute_median_luminance_shift"]),
        }
    else:
        checks = {
            "foreground_agreement": values["foreground_agreement"] >= float(gates["minimum_foreground_agreement"]),
            "minimum_foreground_fraction": values["output_foreground_fraction"] >= float(gates["minimum_output_foreground_fraction"]),
            "maximum_foreground_fraction": values["output_foreground_fraction"] <= float(gates["maximum_output_foreground_fraction"]),
            "component_inflation": values["component_inflation_ratio"] <= float(gates["maximum_component_inflation_ratio"]),
            "edge_correlation": values["edge_correlation"] >= float(gates["minimum_edge_correlation"]),
        }
    return {**values, "gates": checks, "safe": all(checks.values())}


def _pages(domain: str, collection: Path, manifest: dict[str, Any]):
    meta = DOMAINS[domain]
    rows = []
    for page in sorted(manifest.get("pages") or [], key=lambda item: int(item["global_ordinal"])):
        ordinal = int(page["global_ordinal"])
        matches = tuple((collection / meta["upstream_folder"]).glob(f"fs_{ordinal:04d}.*"))
        if len(matches) != 1:
            raise ValueError(f"Expected one {domain} input image for page {ordinal}, found {len(matches)}")
        image = cv2.imread(str(matches[0]), cv2.IMREAD_UNCHANGED)
        if image is None or _pixel_sha256(image) != str(page.get("output_pixel_sha256") or ""):
            raise ValueError(f"Upstream page {ordinal} does not match its pixel identity")
        rows.append((ordinal, page, image))
    if not rows:
        raise ValueError("Upstream manifest contains no pages")
    return rows


def _balance_candidate_partitions(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Guarantee usable development/held-out evidence when candidates permit it."""
    candidates = [
        page for page in pages
        if page["measurement"]["decision"] == "correction-candidate"
    ]
    reassigned: list[dict[str, Any]] = []
    if len(candidates) >= 2:
        development = [page for page in candidates if page["partition"] == "development"]
        held_out = [page for page in candidates if page["partition"] == "held-out"]
        if not development:
            selected = candidates[0]
            selected["partition"] = "development"
            reassigned.append({
                "global_ordinal": selected["global_ordinal"],
                "from": "held-out",
                "to": "development",
                "reason": "ensure-development-candidate",
            })
        elif not held_out:
            selected = candidates[-1]
            selected["partition"] = "held-out"
            reassigned.append({
                "global_ordinal": selected["global_ordinal"],
                "from": "development",
                "to": "held-out",
                "reason": "ensure-held-out-candidate",
            })
    return {
        "candidate_count": len(candidates),
        "development_candidates": sum(page["partition"] == "development" for page in candidates),
        "held_out_candidates": sum(page["partition"] == "held-out" for page in candidates),
        "reassigned_candidates": reassigned,
    }


def assess(domain: str, collection: Path, upstream: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    modulus, residue = int(config["partition"]["development_modulus"]), int(config["partition"]["development_residue"])
    pages = [{"global_ordinal": ordinal, "input_pixel_sha256": source["output_pixel_sha256"], "partition": "development" if ordinal % modulus == residue else "held-out", "measurement": measure(image, config)} for ordinal, source, image in _pages(domain, collection, upstream)]
    partition = _balance_candidate_partitions(pages)
    counts = {name: sum(p["measurement"]["decision"] == name for p in pages) for name in ("correction-candidate", "preserve", "review")}
    payload = {"schema_version": SCHEMA_VERSION, "domain": domain, "assessment_type": config["assessment_type"], "upstream_result_identity": upstream[DOMAINS[domain]["upstream_identity"]], "config": config, "aggregate": {"page_count": len(pages), **counts, **partition}, "pages": pages}
    payload["assessment_identity"] = canonical_hash(payload)
    return payload


def compare(domain: str, collection: Path, upstream: dict[str, Any], assessment: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    images = {ordinal: image for ordinal, _, image in _pages(domain, collection, upstream)}
    candidates = [p for p in assessment["pages"] if p["partition"] == "development" and p["measurement"]["decision"] == "correction-candidate"]
    pages = []
    for page in candidates:
        ordinal = int(page["global_ordinal"])
        variants = []
        for method in config["methods"]:
            output = apply_method(images[ordinal], method)
            variants.append({"method_id": method["id"], "output_pixel_sha256": _pixel_sha256(output), **evaluate(images[ordinal], output, config)})
        pages.append({"global_ordinal": ordinal, "input_pixel_sha256": page["input_pixel_sha256"], "variants": variants})
    safe = [m["id"] for m in config["methods"] if pages and all(next(v for v in p["variants"] if v["method_id"] == m["id"])["safe"] for p in pages)]
    payload = {"schema_version": SCHEMA_VERSION, "domain": domain, "assessment_type": config["assessment_type"], "upstream_result_identity": upstream[DOMAINS[domain]["upstream_identity"]], "assessment_identity": assessment["assessment_identity"], "config": config, "candidate_count": len(pages), "globally_safe_methods": safe, "recommended_method_id": safe[0] if safe else None, "pages": pages}
    payload["method_assessment_identity"] = canonical_hash(payload)
    return payload


def validate(domain: str, collection: Path, upstream: dict[str, Any], assessment: dict[str, Any], comparison: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    images = {ordinal: image for ordinal, _, image in _pages(domain, collection, upstream)}
    method = next((m for m in comparison["config"]["methods"] if m["id"] == comparison.get("recommended_method_id")), None)
    candidates = [p for p in assessment["pages"] if p["partition"] == "held-out" and p["measurement"]["decision"] == "correction-candidate"]
    pages = []
    for page in candidates:
        ordinal = int(page["global_ordinal"])
        if method is None:
            pages.append({"global_ordinal": ordinal, "safe": False, "reason": "no-development-method"})
        else:
            output = apply_method(images[ordinal], method)
            pages.append({"global_ordinal": ordinal, "input_pixel_sha256": page["input_pixel_sha256"], "output_pixel_sha256": _pixel_sha256(output), **evaluate(images[ordinal], output, comparison["config"])})
    all_safe = method is not None and all(p.get("safe") is True for p in pages)
    decision = "apply" if all_safe and (int(comparison["candidate_count"]) + len(pages) > 0) else "preserve"
    payload = {"schema_version": SCHEMA_VERSION, "domain": domain, "validation_type": config["validation_type"], "upstream_result_identity": upstream[DOMAINS[domain]["upstream_identity"]], "assessment_identity": assessment["assessment_identity"], "method_assessment_identity": comparison["method_assessment_identity"], "config": config, "method": method, "aggregate": {"held_out_candidates": len(pages), "safe_candidates": sum(p.get("safe") is True for p in pages)}, "decision": decision, "pages": pages}
    payload["validation_identity"] = canonical_hash(payload)
    return payload


def integrate(domain: str, collection: Path, upstream: dict[str, Any], assessment: dict[str, Any], comparison: dict[str, Any], validation: dict[str, Any], output: Path) -> dict[str, Any]:
    meta = DOMAINS[domain]
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{domain} output is not empty: {output}")
    images_out = output / meta["output_folder"]
    images_out.mkdir(parents=True, exist_ok=True)
    evidence = {int(p["global_ordinal"]): p for p in comparison["pages"] + validation["pages"]}
    method = validation.get("method")
    apply_enabled = validation.get("decision") == "apply" and method is not None
    rows = []
    by_assessment = {int(p["global_ordinal"]): p for p in assessment["pages"]}
    for ordinal, source, image in _pages(domain, collection, upstream):
        assessed = by_assessment[ordinal]
        apply = apply_enabled and assessed["measurement"]["decision"] == "correction-candidate"
        result = apply_method(image, method) if apply else image
        if apply:
            proof = evidence.get(ordinal)
            expected = next((v["output_pixel_sha256"] for v in proof.get("variants", []) if v["method_id"] == method["id"]), proof.get("output_pixel_sha256")) if proof else None
            if not expected or _pixel_sha256(result) != expected:
                raise ValueError(f"{domain} output page {ordinal} does not reproduce validated evidence")
        target = images_out / f"fs_{ordinal:04d}.png"
        if not cv2.imwrite(str(target), result, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise ValueError(f"Could not write {domain} page {ordinal}")
        round_trip = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
        if round_trip is None or not np.array_equal(round_trip, result):
            raise ValueError(f"{domain} page {ordinal} failed lossless round-trip")
        rows.append({"global_ordinal": ordinal, "route": "apply" if apply else "preserve", "pipeline_action": "corrected-and-continue" if apply else "preserve-and-continue", "decision_before": assessed["measurement"]["decision"], "input_pixel_sha256": source["output_pixel_sha256"], "output_pixel_sha256": _pixel_sha256(result), "output_sha256": _sha256(target), "output_width": int(result.shape[1]), "output_height": int(result.shape[0])})
    payload = {"schema_version": SCHEMA_VERSION, "domain": domain, "integration_type": f"{domain}-normalization", "status": "complete", "upstream_result_identity": upstream[meta["upstream_identity"]], "assessment_identity": assessment["assessment_identity"], "method_assessment_identity": comparison["method_assessment_identity"], "validation_identity": validation["validation_identity"], "method": method, "aggregate": {"page_count": len(rows), "corrected_pages": sum(p["route"] == "apply" for p in rows), "preserved_pages": sum(p["route"] == "preserve" for p in rows)}, "pages": rows}
    payload[meta["identity"]] = canonical_hash(payload)
    _write(output / meta["manifest"], payload)
    with (output / meta["csv"]).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "summary.md").write_text(f"# HTH {domain.title()} Normalization\n\n- Input pages: {len(rows)}\n- Corrected pages: {payload['aggregate']['corrected_pages']}\n- Preserved pages: {payload['aggregate']['preserved_pages']}\n- Result identity: `{payload[meta['identity']]}`\n", encoding="utf-8")
    return payload


def package_release(domain: str, collection: Path, asset: Path, tag: str) -> dict[str, Any]:
    meta = DOMAINS[domain]
    manifest = _read(collection / meta["manifest"])
    identity = str(manifest.get(meta["identity"]) or "")
    if tag != f"{meta['release_prefix']}-{identity}":
        raise ValueError(f"{domain} release tag does not match result identity")
    asset.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(asset, "w", allowZip64=True) as archive:
        for path in sorted(p for p in collection.rglob("*") if p.is_file() and p.name != "release.json"):
            info = zipfile.ZipInfo(path.relative_to(collection).as_posix(), _ZIP_TIMESTAMP)
            info.compress_type, info.external_attr = zipfile.ZIP_STORED, 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return {"schema_version": SCHEMA_VERSION, "release_type": f"canonical-{domain}-normalized-collection", meta["identity"]: identity, "tag": tag, "asset": asset.name, "asset_sha256": _sha256(asset)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=tuple(DOMAINS), required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("assess", "compare", "validate", "integrate"):
        command = commands.add_parser(name)
        command.add_argument("--collection", type=Path, required=True); command.add_argument("--upstream-manifest", type=Path, required=True)
        command.add_argument("--assessment", type=Path); command.add_argument("--comparison", type=Path); command.add_argument("--validation", type=Path); command.add_argument("--config", type=Path); command.add_argument("--output", type=Path, required=True)
    package = commands.add_parser("package")
    package.add_argument("--collection", type=Path, required=True); package.add_argument("--asset", type=Path, required=True); package.add_argument("--tag", required=True); package.add_argument("--output-record", type=Path, required=True)
    args = parser.parse_args(argv); domain = args.domain
    if args.command == "package":
        _write(args.output_record, package_release(domain, args.collection, args.asset, args.tag)); return 0
    upstream, config = _read(args.upstream_manifest), _read(args.config) if args.config else None
    if args.command == "assess": result = assess(domain, args.collection, upstream, config)
    elif args.command == "compare": result = compare(domain, args.collection, upstream, _read(args.assessment), config)
    elif args.command == "validate": result = validate(domain, args.collection, upstream, _read(args.assessment), _read(args.comparison), config)
    else: result = integrate(domain, args.collection, upstream, _read(args.assessment), _read(args.comparison), _read(args.validation), args.output)
    if args.command != "integrate": _write(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
