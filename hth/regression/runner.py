"""Execute a reproducible detector regression run."""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import cv2
from hth.geometry.common import document_mask, resize_for_analysis, scale_bbox, valid_bbox
from hth.geometry import detector_components, detector_consensus_quad, detector_contour_components, detector_contour_grabcut, detector_grabcut_contour, detector_contour_projection, detector_contour_quad, detector_ransac
from .adapters.components import (
    detect as components_detect,
    pre_regression_report_sections as components_pre_regression_report_sections,
)
from .adapters.contour import detect as contour_detect
from .adapters.contour_quad import detect as contour_quad_detect
from .adapters.contour_components import detect as contour_components_detect
from .adapters.consensus_quad import detect as consensus_quad_detect
from .adapters.contour_projection import detect as contour_projection_detect
from .adapters.contour_grabcut import detect as contour_grabcut_detect
from .adapters.grabcut_contour import detect as grabcut_contour_detect
from .adapters.edge_contour import detect as edge_contour_detect
from .adapters.grabcut import detect as grabcut_detect
from .adapters.hough import (
    detect as hough_detect,
    pre_regression_report_sections as hough_pre_regression_report_sections,
)
from .adapters.lsd import (
    detect as lsd_detect,
    pre_regression_report_sections as lsd_pre_regression_report_sections,
)
from .adapters.ransac import (
    detect as ransac_detect,
    pre_regression_report_sections as ransac_pre_regression_report_sections,
)
from .io import create_run_directory, environment_info, utc_now, write_json
from .metrics import bbox_iou, edge_errors
from .parameter_space import canonical_parameters, parameter_set_id, exhaustive_parameter_sets
from .reports import ranking_key, write_rankings, write_raw_results
from .strategies.cartesian import generate as cartesian_generate
from .strategies.binary_refine import search as binary_search
from .progress import ProgressReporter
from .performance import PerformanceSampler, peak_rss_bytes
from .calibration_intelligence import build_calibration_intelligence

DETECTORS={"components":components_detect,"contour":contour_detect,"contour_quad":contour_quad_detect,"contour_components":contour_components_detect,"contour_grabcut":contour_grabcut_detect,"grabcut_contour":grabcut_contour_detect,"contour_projection":contour_projection_detect,"consensus_quad":consensus_quad_detect,"edge_contour":edge_contour_detect,"grabcut":grabcut_detect,"hough":hough_detect,"lsd":lsd_detect,"ransac":ransac_detect}
ALLOWED_THREAD_COUNTS=(1,2,4,8,16,32,48,64,96,128,256,512,1024)

PRE_REGRESSION_REPORTERS={
    "components":components_pre_regression_report_sections,
    "hough":hough_pre_regression_report_sections,
    "lsd":lsd_pre_regression_report_sections,
    "ransac":ransac_pre_regression_report_sections,
}


def repository_root(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists():
            return directory
    return Path.cwd()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file as lowercase hexadecimal."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_environment_banner(*, environment: dict[str, Any], detector: str, golden_set: Path, golden_set_sha256: str, source_commit: str | None = None) -> None:
    """Print the execution environment once before a long regression begins."""
    print("Detector Regression Environment")
    print("=" * 31)
    print(f"Execution Environment : {environment.get('execution_target') or '--'}")
    print(f"Runner                : {environment.get('runner_name') or '--'} ({environment.get('runner_environment') or '--'})")
    print(f"CPU                   : {environment.get('cpu_model') or '--'}")
    print(f"Physical Cores        : {environment.get('physical_core_count') or '--'}")
    print(f"Logical CPUs          : {environment.get('logical_cpu_count') or '--'}")
    print(f"Available CPUs        : {environment.get('available_cpu_count') or '--'}")
    smt = environment.get("smt_enabled")
    print(f"SMT Enabled           : {'yes' if smt is True else 'no' if smt is False else '--'}")
    memory = environment.get('memory_gib')
    print(f"Memory                : {memory:.2f} GiB" if isinstance(memory, (int, float)) else "Memory           : --")
    print(f"OS / Architecture     : {environment.get('platform') or '--'} / {environment.get('runner_arch') or '--'}")
    print(f"Python                : {environment.get('python_version') or '--'}")
    print(f"OpenCV                : {environment.get('opencv_version') or '--'}")
    print(f"NumPy                 : {environment.get('numpy_version') or '--'}")
    print(f"Pipeline Commit       : {environment.get('pipeline_commit') or '--'}")
    print(f"Source Commit         : {source_commit or '--'}")
    print(f"Golden Set            : {golden_set.as_posix()}")
    print(f"Golden Set SHA-256    : {golden_set_sha256}")
    print(f"Detector              : {detector}")
    # GitHub Actions can visually collapse truly empty log records. A single
    # space preserves the intended blank separator in both Actions and terminals.
    print(" ")

def print_report_sections(sections: list[dict[str, Any]]) -> None:
    """Print ordered optional research-tuning sections without implementation noise."""
    for section in sections:
        title = str(section.get("title") or "").strip()
        rows = section.get("rows") or []
        if not title or not rows:
            continue
        label_width = max(len(str(label)) for label, _ in rows)
        print(title)
        print("=" * len(title))
        for label, value in rows:
            print(f"{str(label):<{label_width}} : {value}")
        print(" ")

EFFECT_STRATEGY_KEYS = {
    "non-dormant": "non_dormant",
    "low+": "low_plus",
    "moderate+": "moderate_plus",
    "important+": "important_plus",
    "critical": "critical",
}
EFFECT_FALLBACK_ORDER = ["critical", "important+", "moderate+", "low+", "non-dormant", "exhaustive"]


def _resolve_effect_strategy(requested: str, metadata: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None, str | None]:
    if requested not in EFFECT_STRATEGY_KEYS:
        return requested, None, None
    domains = metadata.get("domain_space", {}) if isinstance(metadata, dict) else {}
    start = EFFECT_FALLBACK_ORDER.index(requested)
    for candidate in EFFECT_FALLBACK_ORDER[start:]:
        if candidate == "exhaustive":
            return "exhaustive", None, f"No available parameter sets for {requested}; fell back to exhaustive."
        domain = domains.get(EFFECT_STRATEGY_KEYS[candidate]) if isinstance(domains, dict) else None
        if isinstance(domain, dict) and int(domain.get("parameter_set_count", 0) or 0) > 0:
            reason = None if candidate == requested else f"No available parameter sets for {requested}; fell back to {candidate}."
            return candidate, domain, reason
    return "exhaustive", None, f"No available parameter sets for {requested}; fell back to exhaustive."


def _filter_parameter_sets(parameter_sets: list[dict[str, Any]], domain: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not domain:
        return parameter_sets
    included = set(str(name) for name in domain.get("included_parameters", []))
    fixed = domain.get("fixed_parameters", {}) if isinstance(domain.get("fixed_parameters"), dict) else {}
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for parameters in parameter_sets:
        if any(parameters.get(name) != value for name, value in fixed.items()):
            continue
        reduced = {name: value for name, value in parameters.items() if name in included or name in fixed}
        key = canonical_parameters(reduced)
        if key not in seen:
            seen.add(key)
            filtered.append(parameters)
    return filtered


def parse_args(argv: list[str] | None=None) -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--detector-config",type=Path,required=True)
    p.add_argument("--golden-set",type=Path,required=True)
    p.add_argument("--image-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True,help="Regression root; a detector/run-* directory is created below it.")
    p.add_argument("--strategy",choices=("exhaustive","binary-refine","non-dormant","low+","moderate+","important+","critical"),default="exhaustive")
    p.add_argument("--calibration-intelligence",type=Path,default=None,help="Prior calibration-intelligence.json used for effect-size-domain strategies.")
    p.add_argument("--max-dimension",type=int,default=1800)
    p.add_argument("--limit",type=int,default=None)
    p.add_argument("--top",type=int,default=20)
    p.add_argument("--threads",type=int,choices=ALLOWED_THREAD_COUNTS,default=1,help="Parallel exhaustive-search threads; default: 1.")
    p.add_argument("--run-id",default=None)
    p.add_argument(
        "--debug-artifacts",
        choices=("none", "failures", "winner", "all"),
        default=None,
        help="Debug image policy; defaults to detector configuration or failures.",
    )
    return p.parse_args(argv)

def find_image(root:Path, ordinal:int)->Path:
    for suffix in (".png",".jpg",".jpeg",".tif",".tiff",".webp"):
        for p in (root/"raw"/f"fs_{ordinal:04d}{suffix}",root/f"fs_{ordinal:04d}{suffix}"):
            if p.exists(): return p
    raise FileNotFoundError(f"No image found for Golden Set ordinal {ordinal}")

def load_pages(path:Path,root:Path,maximum:int)->list[dict[str,Any]]:
    data=json.loads(path.read_text(encoding="utf-8")); pages=[]
    for page in data.get("pages",[]):
        approved=page.get("physical_document_bbox")
        if not valid_bbox(approved): continue
        ordinal=int(page["global_ordinal"]); image_path=find_image(root,ordinal)
        original=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        if original is None: raise RuntimeError(f"Could not read image: {image_path}")
        oh,ow=original.shape[:2]; image,scale=resize_for_analysis(original,maximum); mask,diag=document_mask(image)
        pages.append({"global_ordinal":ordinal,"label":page.get("label",f"page_{ordinal}"),"layout_type":page.get("layout_type","other"),"approved_bbox":[int(v) for v in approved],"image_path":str(image_path),"image":image,"mask":mask,"mask_diagnostics":diag,"scale":scale,"original_width":ow,"original_height":oh})
    if not pages: raise ValueError("Golden Set contains no approved pages with valid bounding boxes")
    return pages

def evaluate_set(detector:Any, parameters:dict[str,Any], pages:list[dict[str,Any]])->dict[str,Any]:
    page_results=[]; started=time.perf_counter()
    for page in pages:
        page_started=time.perf_counter()
        try:
            candidate=detector(image_bgr=page["image"],mask=page["mask"],parameters=parameters)
            elapsed=(time.perf_counter()-page_started)*1000
            if candidate.bbox is None:
                page_results.append({"global_ordinal":page["global_ordinal"],"label":page["label"],"layout_type":page["layout_type"],"status":candidate.status if candidate.status!="ok" else "no_candidate","iou":0.0,"edge_error_mean_px":None,"edge_error_maximum_px":None,"elapsed_ms":round(elapsed,3),"candidate":asdict(candidate)})
                continue
            predicted=scale_bbox(candidate.bbox,1.0/page["scale"],page["original_width"],page["original_height"])
            approved=page["approved_bbox"]; errors=edge_errors(predicted,approved)
            page_results.append({"global_ordinal":page["global_ordinal"],"label":page["label"],"layout_type":page["layout_type"],"status":"ok","approved_bbox":approved,"predicted_bbox":predicted,"iou":round(bbox_iou(predicted,approved),8),"edge_errors":errors,"edge_error_mean_px":round(float(errors["mean"]),3),"edge_error_maximum_px":int(errors["maximum"]),"elapsed_ms":round(elapsed,3),"candidate":asdict(candidate)})
        except Exception as exc:
            elapsed=(time.perf_counter()-page_started)*1000
            page_results.append({"global_ordinal":page["global_ordinal"],"label":page["label"],"layout_type":page["layout_type"],"status":"error","iou":0.0,"edge_error_mean_px":None,"edge_error_maximum_px":None,"elapsed_ms":round(elapsed,3),"error":{"type":type(exc).__name__,"message":str(exc)}})
    successful=[r for r in page_results if r["status"]=="ok"]; ious=[float(r["iou"]) for r in page_results]; edges=[float(r["edge_error_mean_px"]) for r in successful]; elapsed=[float(r["elapsed_ms"]) for r in page_results]
    return {"parameter_set_id":parameter_set_id(parameters),"parameters":parameters,"summary":{"page_count":len(page_results),"success_count":len(successful),"failure_count":len(page_results)-len(successful),"mean_iou":round(sum(ious)/len(ious),8),"minimum_iou":round(min(ious),8),"stddev_iou":round(statistics.pstdev(ious),8),"mean_edge_error_px":round(sum(edges)/len(edges),3) if edges else None,"elapsed_ms_total":round(sum(elapsed),3),"wall_ms":round((time.perf_counter()-started)*1000,3)},"pages":page_results}


def build_winner_page_report(
    winner: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    poor_match_iou_below: float = 0.50,
    regression_delta_below: float = -0.001,
) -> dict[str, Any]:
    """Build the canonical per-page analysis for the winning parameter set."""
    baseline_pages = {
        int(page["global_ordinal"]): page
        for page in (baseline or {}).get("pages", [])
    }
    parameter_set = winner.get("profile") or str(winner.get("parameter_set_id", "unknown"))[:12]
    rows: list[dict[str, Any]] = []
    counts = {
        "unprocessed_pages": 0,
        "no_polygon_found": 0,
        "zero_overlap": 0,
        "poor_matches": 0,
        "regressions": 0,
    }

    for page in winner.get("pages", []):
        ordinal = int(page["global_ordinal"])
        baseline_page = baseline_pages.get(ordinal, {})
        baseline_iou = float(baseline_page.get("iou", 0.0) or 0.0)
        winner_iou = float(page.get("iou", 0.0) or 0.0)
        delta_iou = winner_iou - baseline_iou
        winner_status = str(page.get("status", "unknown"))
        reasons: list[str] = []

        if winner_status == "error":
            status = "Unprocessed"
            reasons.append("Unprocessed")
            counts["unprocessed_pages"] += 1
        elif winner_status != "ok":
            status = "No polygon found"
            reasons.append("No polygon found")
            counts["no_polygon_found"] += 1
        elif winner_iou == 0.0:
            status = "Zero overlap"
            reasons.append("Zero overlap")
            counts["zero_overlap"] += 1
        elif baseline_iou == 0.0 and winner_iou > 0.0:
            status = "Recovered"
        elif delta_iou > abs(regression_delta_below):
            status = "Improved"
        elif delta_iou < regression_delta_below:
            status = "Regressed"
        else:
            status = "Unchanged"

        if winner_status == "ok" and 0.0 < winner_iou < poor_match_iou_below:
            reasons.append("Poor match")
            counts["poor_matches"] += 1
        if delta_iou < regression_delta_below:
            reasons.append("Regressed")
            counts["regressions"] += 1

        rows.append({
            "golden_set_page": ordinal,
            "baseline_iou": round(baseline_iou, 8),
            "winner_iou": round(winner_iou, 8),
            "delta_iou": round(delta_iou, 8),
            "status": status,
            "parameter_set": parameter_set,
            "parameter_set_id": winner.get("parameter_set_id"),
            "problem": bool(reasons),
            "problem_reasons": reasons,
            "detector_status": winner_status,
        })

    return {
        "schema_version": "0.1",
        "thresholds": {
            "poor_match_iou_below": poor_match_iou_below,
            "regression_delta_below": regression_delta_below,
        },
        "counts": counts,
        "pages": sorted(
            rows,
            key=lambda row: (-float(row["winner_iou"]), int(row["golden_set_page"])),
        ),
    }

def _safe_name(value: Any) -> str:
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def _write_debug_page(
    root: Path,
    *,
    page: dict[str, Any],
    result: dict[str, Any],
    parameter_set_id_value: str,
) -> None:
    ordinal = int(page["global_ordinal"])
    page_dir = root / _safe_name(parameter_set_id_value) / f"page-{ordinal:04d}"
    page_dir.mkdir(parents=True, exist_ok=True)

    original = cv2.imread(str(page["image_path"]), cv2.IMREAD_COLOR)
    if original is None:
        raise RuntimeError(f"Could not read debug image: {page['image_path']}")
    overlay = original.copy()
    approved = result.get("approved_bbox") or page.get("approved_bbox")
    predicted = result.get("predicted_bbox")
    if valid_bbox(approved):
        x1, y1, x2, y2 = (int(v) for v in approved)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)
    if valid_bbox(predicted):
        x1, y1, x2, y2 = (int(v) for v in predicted)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 4)

    cv2.imwrite(str(page_dir / "01-original.jpg"), original)
    cv2.imwrite(str(page_dir / "02-input-mask.png"), page["mask"])
    candidate = result.get("candidate") if isinstance(result.get("candidate"), dict) else {}
    if candidate.get("method") == "components":
        diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
        parameters = diagnostics.get("parameters") if isinstance(diagnostics.get("parameters"), dict) else None
        numbered_component_images = {
            "after-morphology.png": "03-after-morphology.png",
            "component-labels.png": "04-component-labels.png",
            "significant-components.png": "05-significant-components.png",
            "selected-components.png": "06-selected-components.png",
            "candidate-envelope.png": "07-candidate-envelope.png",
        }
        for filename, debug_image in detector_components.debug_images(
            mask=page["mask"],
            parameters=parameters,
            diagnostics=diagnostics,
            candidate_bbox=candidate.get("bbox"),
        ).items():
            cv2.imwrite(str(page_dir / numbered_component_images[filename]), debug_image)
        overlay_name = "08-overlay.jpg"
        diagnostics_name = "09-diagnostics.json"
    elif candidate.get("method") == "contour_quad":
        diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
        parameters = diagnostics.get("parameters") if isinstance(diagnostics.get("parameters"), dict) else None
        numbered_contour_quad_images = {
            "after-morphology.png": "03-after-morphology.png",
            "contour-hypotheses.png": "04-contour-hypotheses.png",
            "quadrilateral-hypotheses.png": "05-quadrilateral-hypotheses.png",
            "edge-evidence.png": "06-edge-evidence.png",
            "selected-quadrilateral.png": "07-selected-quadrilateral.png",
        }
        for filename, debug_image in detector_contour_quad.debug_images(
            image_bgr=original,
            mask=page["mask"],
            parameters=parameters,
            candidate_corners=candidate.get("corners"),
        ).items():
            cv2.imwrite(str(page_dir / numbered_contour_quad_images[filename]), debug_image)
        overlay_name = "08-overlay.jpg"
        diagnostics_name = "09-diagnostics.json"
    elif candidate.get("method") == "contour_components":
        diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
        parameters = diagnostics.get("parameters") if isinstance(diagnostics.get("parameters"), dict) else None
        numbered_component_images = {
            "contour-hypotheses.png": "03-contour-hypotheses.png",
            "component-labels.png": "04-component-labels.png",
            "selected-components.png": "05-selected-components.png",
            "component-envelope.png": "06-component-envelope.png",
            "component-evidence.png": "07-component-evidence.png",
            "selected-quadrilateral.png": "08-selected-quadrilateral.png",
        }
        for filename, debug_image in detector_contour_components.debug_images(
            image_bgr=original,
            mask=page["mask"],
            parameters=parameters,
            candidate_corners=candidate.get("corners"),
            diagnostics=diagnostics,
        ).items():
            cv2.imwrite(str(page_dir / numbered_component_images[filename]), debug_image)
        overlay_name = "09-overlay.jpg"
        diagnostics_name = "10-diagnostics.json"
    elif candidate.get("method") == "contour_grabcut":
        diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
        parameters = diagnostics.get("parameters") if isinstance(diagnostics.get("parameters"), dict) else None
        numbered_hybrid_images = {
            "contour-candidate.png": "03-contour-candidate.png",
            "grabcut-candidate.png": "04-grabcut-candidate.png",
            "agreement-overlay.png": "05-agreement-overlay.png",
            "selected-quadrilateral.png": "06-selected-quadrilateral.png",
        }
        for filename, debug_image in detector_contour_grabcut.debug_images(
            image_bgr=original,
            mask=page["mask"],
            parameters=parameters,
            candidate_corners=candidate.get("corners"),
            diagnostics=diagnostics,
        ).items():
            cv2.imwrite(str(page_dir / numbered_hybrid_images[filename]), debug_image)
        overlay_name = "07-overlay.jpg"
        diagnostics_name = "08-diagnostics.json"
    elif candidate.get("method") == "grabcut_contour":
        diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
        parameters = diagnostics.get("parameters") if isinstance(diagnostics.get("parameters"), dict) else None
        numbered_hybrid_images = {
            "grabcut-candidate.png": "03-grabcut-candidate.png",
            "contour-candidate.png": "04-contour-candidate.png",
            "agreement-overlay.png": "05-agreement-overlay.png",
            "selected-quadrilateral.png": "06-selected-quadrilateral.png",
        }
        for filename, debug_image in detector_grabcut_contour.debug_images(
            image_bgr=original,
            mask=page["mask"],
            parameters=parameters,
            candidate_corners=candidate.get("corners"),
            diagnostics=diagnostics,
        ).items():
            cv2.imwrite(str(page_dir / numbered_hybrid_images[filename]), debug_image)
        overlay_name = "07-overlay.jpg"
        diagnostics_name = "08-diagnostics.json"
    elif candidate.get("method") == "contour_projection":
        diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
        parameters = diagnostics.get("parameters") if isinstance(diagnostics.get("parameters"), dict) else None
        numbered_projection_images = {
            "contour-hypotheses.png": "03-contour-hypotheses.png",
            "warped-candidate.png": "04-warped-candidate.png",
            "projection-binary.png": "05-projection-binary.png",
            "horizontal-projection.png": "06-horizontal-projection.png",
            "selected-quadrilateral.png": "07-selected-quadrilateral.png",
        }
        for filename, debug_image in detector_contour_projection.debug_images(
            image_bgr=original, mask=page["mask"], parameters=parameters,
            candidate_corners=candidate.get("corners"),
        ).items():
            cv2.imwrite(str(page_dir / numbered_projection_images[filename]), debug_image)
        overlay_name = "08-overlay.jpg"
        diagnostics_name = "09-diagnostics.json"
    elif candidate.get("method") == "consensus_quad":
        diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
        numbered_consensus_images = {
            "contour-quad-vote.png": "03-contour-quad-vote.png",
            "edge-contour-vote.png": "04-edge-contour-vote.png",
            "agreement-overlay.png": "05-agreement-overlay.png",
            "selected-consensus.png": "06-selected-consensus.png",
        }
        for filename, debug_image in detector_consensus_quad.debug_images(
            image_bgr=original,
            diagnostics=diagnostics,
            candidate_corners=candidate.get("corners"),
        ).items():
            cv2.imwrite(str(page_dir / numbered_consensus_images[filename]), debug_image)
        overlay_name = "07-overlay.jpg"
        diagnostics_name = "08-diagnostics.json"
    elif candidate.get("method") == "ransac":
        diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
        parameters = diagnostics.get("parameters") if isinstance(diagnostics.get("parameters"), dict) else None
        numbered_ransac_images = {
            "boundary-samples.png": "03-boundary-samples.png",
            "fitted-edge-models.png": "04-fitted-edge-models.png",
            "ransac-inliers.png": "05-ransac-inliers.png",
            "candidate-quadrilateral.png": "06-candidate-quadrilateral.png",
        }
        for filename, debug_image in detector_ransac.debug_images(
            mask=page["mask"], parameters=parameters
        ).items():
            cv2.imwrite(str(page_dir / numbered_ransac_images[filename]), debug_image)
        overlay_name = "07-overlay.jpg"
        diagnostics_name = "08-diagnostics.json"
    else:
        overlay_name = "03-overlay.jpg"
        diagnostics_name = "04-diagnostics.json"
    cv2.imwrite(str(page_dir / overlay_name), overlay)
    write_json(
        page_dir / diagnostics_name,
        {
            "global_ordinal": ordinal,
            "label": page.get("label"),
            "layout_type": page.get("layout_type"),
            "image_path": page.get("image_path"),
            "parameter_set_id": parameter_set_id_value,
            "result": result,
            "overlay_legend": {"approved_bbox": "green", "predicted_bbox": "red"},
        },
    )


def write_debug_artifacts(
    regression_root: Path,
    detector: str,
    run_id: str,
    *,
    policy: str,
    ranked: list[dict[str, Any]],
    pages: list[dict[str, Any]],
) -> list[str]:
    debug_root = regression_root / "debug" / _safe_name(detector) / _safe_name(run_id)
    debug_root.mkdir(parents=True, exist_ok=False)
    page_by_ordinal = {int(page["global_ordinal"]): page for page in pages}
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if policy == "all":
        selected = [(parameter_set, page_result) for parameter_set in ranked for page_result in parameter_set["pages"]]
    elif policy in {"winner", "failures"}:
        winner = ranked[0]
        selected = [(winner, page_result) for page_result in winner["pages"]]
        if policy == "failures":
            selected = [item for item in selected if item[1].get("status") != "ok"]

    for parameter_set, page_result in selected:
        page = page_by_ordinal[int(page_result["global_ordinal"])]
        _write_debug_page(
            debug_root,
            page=page,
            result=page_result,
            parameter_set_id_value=str(parameter_set["parameter_set_id"]),
        )

    readme = [
        "HTH detector regression debug artifacts",
        "",
        f"Policy: {policy}",
        f"Pages written: {len(selected)}",
        "",
        "Each page directory uses numeric prefixes to preserve analysis order.",
        "Common files:",
        "- 01-original.jpg: source image",
        "- 02-input-mask.png: mask supplied to the detector",
        "Connected Components stages:",
        "- 03-after-morphology.png: mask after closing and dilation",
        "- 04-component-labels.png: all connected-component labels in distinct colors",
        "- 05-significant-components.png: components surviving the configured area filter",
        "- 06-selected-components.png: components merged into the final candidate",
        "- 07-candidate-envelope.png: selected components with the analysis-space envelope",
        "- 08-overlay.jpg: approved bbox in green; predicted bbox in red",
        "- 09-diagnostics.json: complete page result and detector diagnostics",
        "Contour Quadrilateral stages:",
        "- 03-after-morphology.png: mask after configured contour-closing morphology",
        "- 04-contour-hypotheses.png: external contours and optional merged hull",
        "- 05-quadrilateral-hypotheses.png: plausible convex quadrilateral approximations",
        "- 06-edge-evidence.png: combined image and mask edge-support evidence",
        "- 07-selected-quadrilateral.png: winning quadrilateral and ordered corners",
        "- 08-overlay.jpg: approved bbox in green; predicted bbox in red",
        "- 09-diagnostics.json: complete page result and detector diagnostics",
        "Consensus Quad stages:",
        "- 03-contour-quad-vote.png: Contour Quadrilateral voter result",
        "- 04-edge-contour-vote.png: Edge-Contour voter result",
        "- 05-agreement-overlay.png: both voter polygons overlaid for comparison",
        "- 06-selected-consensus.png: confidence-weighted consensus quadrilateral",
        "- 07-overlay.jpg: approved bbox in green; predicted bbox in red",
        "- 08-diagnostics.json: complete page result, voter evidence, and consensus diagnostics",
        "RANSAC stages:",
        "- 03-boundary-samples.png: left/right/top/bottom observations sampled from the mask",
        "- 04-fitted-edge-models.png: robust line models fitted to each edge family",
        "- 05-ransac-inliers.png: observations accepted by each fitted edge model",
        "- 06-candidate-quadrilateral.png: line intersections and derived candidate envelope",
        "- 07-overlay.jpg: approved bbox in green; predicted bbox in red",
        "- 08-diagnostics.json: complete page result and detector diagnostics",
        "Other detectors use 03-overlay.jpg and 04-diagnostics.json.",
        "",
    ]
    (debug_root / "README.txt").write_text("\n".join(readme), encoding="utf-8")
    outputs = [(debug_root / "README.txt").relative_to(regression_root).as_posix()]
    outputs.extend(
        path.relative_to(regression_root).as_posix()
        for path in sorted(debug_root.rglob("*"))
        if path.is_file() and path.name != "README.txt"
    )
    return outputs


def print_parameter_scope(*, strategy: str, possible_sets: int, planned_sets: int | None, golden_pages: int, threads: int, limit: int | None) -> None:
    print("Regression Scope")
    print("=" * 16)
    rows = [
        ("Search Strategy", strategy),
        ("Possible Parameter Sets", possible_sets),
        ("Planned Parameter Sets", planned_sets if planned_sets is not None else "adaptive / unknown"),
        ("Golden Set Pages", golden_pages),
        (
            "Planned Page Evaluations",
            planned_sets * golden_pages if planned_sets is not None else "adaptive / unknown",
        ),
        ("Parameter-set Limit", limit if limit is not None else "unlimited"),
        ("Threads", threads),
    ]
    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{label_width}} : {value}")
    print(" ")


def run(args:argparse.Namespace)->Path:
    config=json.loads(args.detector_config.read_text(encoding="utf-8")); name=str(config["detector"])
    regression_config = config.get("regression", {}) if isinstance(config.get("regression"), dict) else {}
    debug_policy = args.debug_artifacts or str(regression_config.get("debug_artifacts", "failures"))
    if debug_policy not in {"none", "failures", "winner", "all"}:
        raise ValueError(f"Unsupported debug artifact policy: {debug_policy}")
    if name not in DETECTORS: raise SystemExit(f"Unsupported detector: {name}")
    run_id,run_dir=create_run_directory(args.output,name,args.run_id); started=utc_now(); wall=time.perf_counter()
    environment=environment_info(repository_root(args.detector_config))
    source_commit=os.environ.get("HTH_SOURCE_COMMIT")
    golden_set_sha256=file_sha256(args.golden_set)
    requested_strategy = args.strategy
    effective_strategy = requested_strategy
    strategy_fallback_reason = None
    manifest={"schema_version":"0.1","run_id":run_id,"detector":name,"strategy":effective_strategy,"requested_strategy":requested_strategy,"strategy_fallback_reason":strategy_fallback_reason,"status":"running","started_at_utc":started,"outputs":[]}
    write_json(run_dir/"manifest.json",manifest)
    try:
        pages=load_pages(args.golden_set,args.image_root,args.max_dimension); detector=DETECTORS[name]
        if not callable(detector):
            raise TypeError(
                f"Detector registry entry {name!r} is not callable: "
                f"{type(detector).__name__}"
            )
        profiles={canonical_parameters(p):n for n,p in config.get("profiles",{}).items()}
        baseline_parameters=config.get("profiles",{}).get("baseline")
        if not isinstance(baseline_parameters,dict):
            raise ValueError("Detector configuration must define profiles.baseline")
        baseline_key=canonical_parameters(baseline_parameters)

        all_parameter_sets=cartesian_generate(config)
        possible_parameter_set_count=len(all_parameter_sets)
        calibration_metadata = None
        if args.calibration_intelligence and args.calibration_intelligence.is_file():
            calibration_metadata = json.loads(args.calibration_intelligence.read_text(encoding="utf-8"))
        requested_strategy = args.strategy
        effective_strategy, effect_domain, strategy_fallback_reason = _resolve_effect_strategy(requested_strategy, calibration_metadata)
        if effect_domain is not None:
            all_parameter_sets = _filter_parameter_sets(all_parameter_sets, effect_domain)
        write_json(run_dir/"parameters.json",{"schema_version":"0.4","detector":name,"strategy":effective_strategy,"requested_strategy":requested_strategy,"strategy_fallback_reason":strategy_fallback_reason,"detector_config":str(args.detector_config),"golden_set":str(args.golden_set),"golden_set_sha256":golden_set_sha256,"image_root":str(args.image_root),"max_dimension":args.max_dimension,"limit":args.limit,"threads":args.threads,"configuration":config})
        manifest.update({"strategy": effective_strategy, "requested_strategy": requested_strategy, "strategy_fallback_reason": strategy_fallback_reason})
        write_json(run_dir/"manifest.json", manifest)
        exhaustive_candidates=[
            parameters for parameters in all_parameter_sets
            if canonical_parameters(parameters) != baseline_key
        ]
        if args.limit is not None:
            exhaustive_candidates=exhaustive_candidates[:args.limit]
        planned_parameter_set_count=(
            1 + len(exhaustive_candidates) if effective_strategy=="exhaustive" or effective_strategy in EFFECT_STRATEGY_KEYS else None
        )
        estimated_total=len(exhaustive_candidates) if effective_strategy=="exhaustive" or effective_strategy in EFFECT_STRATEGY_KEYS else max(0,possible_parameter_set_count-1)

        print_environment_banner(environment=environment,detector=name,golden_set=args.golden_set,golden_set_sha256=golden_set_sha256,source_commit=source_commit)
        print_parameter_scope(
            strategy=effective_strategy,
            possible_sets=possible_parameter_set_count,
            planned_sets=planned_parameter_set_count,
            golden_pages=len(pages),
            threads=args.threads,
            limit=args.limit,
        )
        reporter = PRE_REGRESSION_REPORTERS.get(name)
        if reporter is not None:
            print_report_sections(reporter(config))
        progress=ProgressReporter(total=estimated_total,interval_seconds=60.0)
        progress.start()

        progress.begin_evaluation("baseline")
        baseline_result=evaluate_set(detector,dict(baseline_parameters),pages)
        progress.observe_baseline(baseline_result)

        active_lock=threading.Lock()
        active_evaluations=0

        def telemetry_snapshot() -> dict[str, Any]:
            snap=progress.snapshot()
            with active_lock:
                active=active_evaluations
            return {
                "completed_parameter_sets": snap.completed,
                "planned_parameter_sets": snap.total,
                "parameter_sets_per_second": round(snap.eval_rate, 6) if snap.eval_rate is not None else None,
                "completed_page_evaluations": snap.completed * len(pages),
                "page_evaluations_per_second": round(snap.eval_rate * len(pages), 6) if snap.eval_rate is not None else None,
                "active_threads": active,
                "configured_threads": args.threads,
            }

        performance=PerformanceSampler(run_dir/"logs"/"runner-performance.jsonl",snapshot=telemetry_snapshot,interval_seconds=60.0)
        performance.start()

        def evaluate(parameters:dict[str,Any], *, observe: bool=True)->dict[str,Any]:
            nonlocal active_evaluations
            canonical=canonical_parameters(parameters)
            if canonical == baseline_key:
                return baseline_result
            profile=profiles.get(canonical)
            profile_name=profile or parameter_set_id(parameters)[:8]
            progress.begin_evaluation(profile_name)
            with active_lock:
                active_evaluations += 1
            try:
                result=evaluate_set(detector,parameters,pages)
            finally:
                with active_lock:
                    active_evaluations -= 1
            if observe:
                progress.observe(result,profile)
            return result
        if effective_strategy=="exhaustive" or effective_strategy in EFFECT_STRATEGY_KEYS:
            if args.threads == 1:
                candidate_results=[evaluate(p) for p in exhaustive_candidates]
            else:
                indexed_results: list[dict[str,Any] | None]=[None] * len(exhaustive_candidates)
                with ThreadPoolExecutor(max_workers=args.threads,thread_name_prefix="regression") as executor:
                    futures={executor.submit(evaluate,parameters,observe=False): index for index,parameters in enumerate(exhaustive_candidates)}
                    for future in as_completed(futures):
                        index=futures[future]
                        result=future.result()
                        indexed_results[index]=result
                        profile=profiles.get(canonical_parameters(result["parameters"]))
                        progress.observe(result,profile)
                candidate_results=[result for result in indexed_results if result is not None]
            results=[baseline_result,*candidate_results]
        else:
            results=binary_search(config,evaluate,ranking_key)
            if not any(canonical_parameters(r["parameters"]) == baseline_key for r in results):
                results.insert(0,baseline_result)
        progress_snapshot=progress.finish()
        performance_samples=performance.finish()
        for r in results: r["profile"]=profiles.get(canonical_parameters(r["parameters"])); r["run_id"]=run_id
        ranked=sorted(results,key=ranking_key)
        for rank,r in enumerate(ranked,1): r["rank"]=rank
        baseline=next((r for r in ranked if r.get("profile")=="baseline"),None)
        raw=run_dir/"raw"/"results.csv"; rankings=run_dir/"reports"/"rankings.csv"; top=run_dir/"reports"/"top20.csv"
        write_raw_results(raw,ranked); write_rankings(rankings,ranked); write_rankings(top,ranked[:max(0,args.top)])
        winner_pages = build_winner_page_report(ranked[0], baseline)
        summary={"schema_version":"0.8","run_id":run_id,"detector":name,"strategy":effective_strategy,"requested_strategy":requested_strategy,"strategy_fallback_reason":strategy_fallback_reason,"threads":args.threads,"parameter_space":{"possible_parameter_sets":possible_parameter_set_count,"planned_parameter_sets":planned_parameter_set_count,"actual_parameter_sets":len(ranked),"golden_set_pages":len(pages),"planned_page_evaluations":planned_parameter_set_count*len(pages) if planned_parameter_set_count is not None else None,"actual_page_evaluations":len(ranked)*len(pages)},"page_ordinals":[p["global_ordinal"] for p in pages],"parameter_set_count":len(ranked),"page_evaluation_count":len(ranked)*len(pages),"successful_page_evaluation_count":len(ranked)*len(pages)-progress_snapshot.failures,"fully_successful_parameter_set_count":sum(1 for r in ranked if int(r["summary"].get("failure_count", 0) or 0) == 0),"golden_set_sha256":golden_set_sha256,"winner":ranked[0],"baseline":baseline,"top_parameter_sets":ranked[:5],"winner_page_report":winner_pages,"runner":environment,"source_commit":source_commit,"performance":{"sample_count":len(performance_samples),"configured_threads":args.threads,"peak_rss_bytes":peak_rss_bytes(),"samples_file":"logs/runner-performance.jsonl"},"progress":{"estimated_parameter_sets":progress_snapshot.total,"completed_parameter_sets":progress_snapshot.completed,"average_eval_rate":progress_snapshot.eval_rate,"failures":progress_snapshot.failures,"best_mean_iou":progress_snapshot.best_mean_iou,"best_worst_page_iou":progress_snapshot.best_minimum_page_iou,"best_stddev_iou":progress_snapshot.best_stddev_iou,"mean_iou_improvements":progress_snapshot.mean_iou_improvements,"minimum_iou_improvements":progress_snapshot.minimum_iou_improvements,"stddev_improvements":progress_snapshot.stddev_improvements,"total_metric_improvements":progress_snapshot.mean_iou_improvements+progress_snapshot.minimum_iou_improvements+progress_snapshot.stddev_improvements,"parameter_sets_with_improvements":progress_snapshot.parameter_sets_with_improvements,"winner_changes":progress_snapshot.winner_changes,"baseline_surpassed":progress.baseline_surpassed,"winner_first_changed_elapsed_seconds":progress_snapshot.winner_first_changed_elapsed_seconds,"winner_last_changed_elapsed_seconds":progress_snapshot.winner_last_changed_elapsed_seconds,"winner_history":progress_snapshot.winner_history,"last_improvement_elapsed_seconds":progress_snapshot.last_improvement_elapsed_seconds,"time_since_last_improvement_seconds":progress_snapshot.last_improvement_seconds}}
        write_json(run_dir/"reports"/"summary.json",summary)
        write_json(run_dir/"reports"/"winner-pages.json",winner_pages)
        try:
            golden_set_payload = json.loads(args.golden_set.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            golden_set_payload = {}
        source_document = golden_set_payload.get("source_document") if isinstance(golden_set_payload, dict) else None
        golden_set_identity = {
            "configuration": str(args.golden_set),
            "sha256": golden_set_sha256,
            "collection_id": golden_set_payload.get("collection_id") if isinstance(golden_set_payload, dict) else None,
            "schema_version": golden_set_payload.get("schema_version") if isinstance(golden_set_payload, dict) else None,
            "description": golden_set_payload.get("description") if isinstance(golden_set_payload, dict) else None,
            "page_count": len(pages),
            "page_ordinals": [page["global_ordinal"] for page in pages],
        }
        calibration_context = {
            "calibration_run_id": run_id,
            "calibration_schema_version": "1.1",
            "created_at_utc": started,
            "source_document": source_document,
            "golden_set": golden_set_identity,
            "detector_configuration": {
                "detector_id": name,
                "configuration": str(args.detector_config),
                "sha256": file_sha256(args.detector_config),
            },
            "pipeline": {
                "commit": environment.get("pipeline_commit"),
                "source_commit": source_commit,
                "python": environment.get("python_version"),
                "opencv": environment.get("opencv_version"),
            },
        }
        regression_context = {
            "requested_strategy": requested_strategy,
            "resolved_strategy": effective_strategy,
            "strategy_fallback_reason": strategy_fallback_reason,
            "configured_threads": args.threads,
            "possible_parameter_sets": possible_parameter_set_count,
            "planned_parameter_sets": planned_parameter_set_count,
            "evaluated_parameter_sets": len(ranked),
            "golden_set_pages": len(pages),
            "page_evaluations": len(ranked) * len(pages),
            "failed_page_evaluations": progress_snapshot.failures,
            "average_eval_rate": progress_snapshot.eval_rate,
            "execution_environment": environment,
        }
        calibration_intelligence = build_calibration_intelligence(
            ranked,
            detector=name,
            strategy=effective_strategy,
            possible_parameter_sets=possible_parameter_set_count,
            calibration_context=calibration_context,
            regression_context=regression_context,
        )
        write_json(
            run_dir/"reports"/"calibration-intelligence.json",
            calibration_intelligence,
        )
        debug_outputs = [] if debug_policy == "none" else write_debug_artifacts(
            args.output, name, run_id, policy=debug_policy, ranked=ranked, pages=pages
        )
        finished=utc_now(); info={"schema_version":"0.3","run_id":run_id,"detector":name,"strategy":effective_strategy,"requested_strategy":requested_strategy,"strategy_fallback_reason":strategy_fallback_reason,"status":"complete","started_at_utc":started,"finished_at_utc":finished,"elapsed_seconds":round(time.perf_counter()-wall,3),"golden_set":str(args.golden_set),"golden_set_sha256":golden_set_sha256,"detector_config":str(args.detector_config),"debug_artifacts":debug_policy,"source_commit":source_commit,"threads":args.threads,"possible_parameter_sets":possible_parameter_set_count,"planned_parameter_sets":planned_parameter_set_count,"actual_parameter_sets":len(ranked),"performance_samples":len(performance_samples),"peak_rss_bytes":peak_rss_bytes(),**environment}
        write_json(run_dir/"RUN-INFO.json",info)
        manifest.update({
            "status": "complete",
            "finished_at_utc": finished,
            "outputs": [
                "RUN-INFO.json",
                "parameters.json",
                "raw/results.csv",
                "logs/runner-performance.jsonl",
                "reports/summary.json",
                "reports/winner-pages.json",
                "reports/calibration-intelligence.json",
                "reports/rankings.csv",
                "reports/top20.csv",
            ],
            "debug_outputs": debug_outputs,
        }); write_json(run_dir/"manifest.json",manifest)
        # Convenience report at detector root, refreshed on every completed run.
        write_rankings(run_dir.parent/f"{name}-regression-results.csv",ranked)
        winner_summary=ranked[0]["summary"]
        baseline_summary=baseline["summary"] if baseline else None
        winner_profile=ranked[0].get("profile") or ranked[0]["parameter_set_id"][:8]
        progress.announce("Regression complete", emit_status=False)
        # Preserve one visible separator before the summary in GitHub Actions.
        print(" ")
        elapsed_seconds=time.perf_counter()-wall
        def print_key_value_section(title: str, rows: list[tuple[str, object] | None]) -> None:
            label_width = max(len(row[0]) for row in rows if row is not None)
            print(title)
            print("=" * len(title))
            for row in rows:
                if row is None:
                    print()
                    continue
                label, value = row
                print(f"{label:<{label_width}} : {value}")

        summary_rows: list[tuple[str, object] | None] = [
            ("Run", run_id),
            ("Elapsed", f"{elapsed_seconds:.1f}s"),
            ("Average Eval Rate", f"{(len(ranked)/elapsed_seconds if elapsed_seconds else 0.0):.4f}/s"),
            ("Search Strategy", effective_strategy),
            ("Threads", args.threads),
            ("Possible parameter sets", possible_parameter_set_count),
            ("Planned parameter sets", planned_parameter_set_count if planned_parameter_set_count is not None else "adaptive / unknown"),
            None,
            ("Parameter sets evaluated", len(ranked)),
            ("Fully successful parameter sets", summary["fully_successful_parameter_set_count"]),
            None,
            ("Page evaluations", len(ranked) * len(pages)),
            ("Successful page evaluations", len(ranked) * len(pages) - progress_snapshot.failures),
            ("Failed page evaluations", progress_snapshot.failures),
            None,
            ("Winner", winner_profile),
            ("Average Page IoU", f"{winner_summary['mean_iou']:.4f}"),
            ("Minimum Page IoU", f"{winner_summary['minimum_iou']:.4f}"),
            ("Std Dev", f"{winner_summary['stddev_iou']:.4f}"),
        ]
        if baseline_summary:
            summary_rows.extend([
                ("Baseline Average Page IoU", f"{baseline_summary['mean_iou']:.4f}"),
                ("Average Page IoU improvement", f"{winner_summary['mean_iou']-baseline_summary['mean_iou']:+.4f}"),
                ("Minimum Page IoU improvement", f"{winner_summary['minimum_iou']-baseline_summary['minimum_iou']:+.4f}"),
            ])
        print_key_value_section("Regression Summary", summary_rows)
        print(" ")
        print_key_value_section("Regression Statistics", [
            ("Mean IoU improvements", progress_snapshot.mean_iou_improvements),
            ("Minimum IoU improvements", progress_snapshot.minimum_iou_improvements),
            ("Std Dev improvements", progress_snapshot.stddev_improvements),
            ("Total metric improvements", progress_snapshot.mean_iou_improvements + progress_snapshot.minimum_iou_improvements + progress_snapshot.stddev_improvements),
            ("Parameter sets with improvements", progress_snapshot.parameter_sets_with_improvements),
            ("Winner changes", progress_snapshot.winner_changes),
            ("Baseline surpassed", "yes" if progress.baseline_surpassed else "no"),
        ])
        print(json.dumps({"run_id":run_id,"run_directory":str(run_dir),"winner":ranked[0],"baseline":baseline},indent=2))
        return run_dir
    except Exception as exc:
        manifest.update({"status":"failed","finished_at_utc":utc_now(),"error":{"type":type(exc).__name__,"message":str(exc)}}); write_json(run_dir/"manifest.json",manifest)
        raise

def main(argv:list[str]|None=None)->int:
    run(parse_args(argv)); return 0
