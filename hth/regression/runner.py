"""Execute a reproducible detector regression run."""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import cv2
from hth.geometry.common import document_mask, resize_for_analysis, scale_bbox, valid_bbox
from hth.geometry import detector_kraken_page_mask, detector_adaptive_multi_scale_radial_edge, detector_amsre_bfq_spbv_pbg, detector_adaptive_radial_edge, detector_border_energy, detector_border_fusion_quad, detector_components, detector_convex_hull, detector_consensus_quad, detector_contour_components, detector_contour_grabcut, detector_cross_edge_contour, detector_distance_transform, detector_distance_transform_rect, detector_dhsegment_page_mask, detector_polar_boundary_vote, detector_page_background, detector_signed_polar_boundary_vote, detector_segment_supported_polar_vote, detector_star_convex, detector_grabcut, detector_grabcut_contour, detector_gradient_vote, detector_multi_scale_radial_edge, detector_msre_bfq_spbv_pbg, detector_projective_gradient_vote, detector_radial_edge, detector_contour_projection, detector_contour_quad, detector_ransac, detector_radon_boundary, detector_text_flow, detector_whitespace_frame, detector_joint_rectangle_vote, detector_learned_page_mask
from .adapters.convex_hull import detect as convex_hull_detect
from .adapters.distance_transform import detect as distance_transform_detect
from .adapters.distance_transform_rect import detect as distance_transform_rect_detect
from .adapters.polar_boundary_vote import detect as polar_boundary_vote_detect
from .adapters.page_background import detect as page_background_detect
from .adapters.signed_polar_boundary_vote import detect as signed_polar_boundary_vote_detect
from .adapters.segment_supported_polar_vote import detect as segment_supported_polar_vote_detect
from .adapters.radon_boundary import detect as radon_boundary_detect
from .adapters.text_flow import detect as text_flow_detect
from .adapters.whitespace_frame import detect as whitespace_frame_detect
from .adapters.joint_rectangle_vote import detect as joint_rectangle_vote_detect
from .adapters.learned_page_mask import detect as learned_page_mask_detect
from .adapters.dhsegment_page_mask import detect as dhsegment_page_mask_detect
from .adapters.multi_scale_radial_edge import detect as multi_scale_radial_edge_detect
from .adapters.msre_bfq_spbv_pbg import detect as msre_bfq_spbv_pbg_detect
from .adapters.projective_gradient_vote import detect as projective_gradient_vote_detect
from .adapters.border_fusion_quad import detect as border_fusion_quad_detect
from .adapters.star_convex import detect as star_convex_detect
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
from .adapters.cross_edge_contour import detect as cross_edge_contour_detect
from .adapters.gradient_vote import detect as gradient_vote_detect
from .adapters.radial_edge import detect as radial_edge_detect
from .adapters.adaptive_multi_scale_radial_edge import detect as adaptive_multi_scale_radial_edge_detect
from .adapters.amsre_bfq_spbv_pbg import detect as amsre_bfq_spbv_pbg_detect
from .adapters.adaptive_radial_edge import detect as adaptive_radial_edge_detect
from .adapters.border_energy import detect as border_energy_detect
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
from .parameter_provenance import attach_identity, build_provenance
from .reports import ranking_key, write_rankings, write_raw_results
from .strategies.cartesian import generate as cartesian_generate
from .strategies.binary_refine import search as binary_search
from .progress import ProgressReporter
from .performance import PerformanceSampler, peak_rss_bytes
from .calibration_intelligence import build_calibration_intelligence
from hth.regression.result_metrics import aggregate_page_metrics
from hth.domain.result_metrics import baseline_surpassed
from hth.geometry.registry import detector_entrypoint, detector_names

# Backward-compatible read-only-style view for existing callers/tests. The
# authoritative source is hth.geometry.registry; this map is generated from it
# at import time and must never be hand-maintained.
DETECTORS={name: detector_entrypoint(name) for name in detector_names()}
MIN_THREAD_COUNT=1
MAX_THREAD_COUNT=1024

PRE_REGRESSION_REPORTERS={
    "components":components_pre_regression_report_sections,
    "hough":hough_pre_regression_report_sections,
    "lsd":lsd_pre_regression_report_sections,
    "ransac":ransac_pre_regression_report_sections,
}


PRECOMPUTED_EVIDENCE_PREPARERS={
    "kraken_page_mask":detector_kraken_page_mask.precompute_golden_set_evidence,
    "dhsegment_page_mask":detector_dhsegment_page_mask.precompute_golden_set_evidence,
}

PRECOMPUTED_EVIDENCE_LOADERS={
    "kraken_page_mask":detector_kraken_page_mask.load_precomputed_golden_set_evidence,
    "dhsegment_page_mask":detector_dhsegment_page_mask.load_precomputed_golden_set_evidence,
}


def logical_golden_set(pages:list[dict[str,Any]])->list[dict[str,Any]]:
    """Give each parameter evaluation its own page metadata view.

    Large image/mask arrays and learned evidence remain shared read-only inputs;
    the list and page dictionaries are private to the evaluation thread.
    """
    return [dict(page) for page in pages]


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


def load_or_evaluate_shared_baseline(
    path: Path | None,
    evaluator: Callable[[], dict[str, Any]],
    *,
    timeout_seconds: float = 900.0,
) -> tuple[dict[str, Any], bool]:
    """Return a baseline result, evaluating it at most once across sibling shards.

    The cache is run-local.  A small atomic lock file elects one producer while
    sibling processes wait for the JSON result, which keeps the mechanism usable
    on both Linux and Windows runners without another dependency.
    """
    if path is None:
        return evaluator(), False

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    while True:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Shared baseline cache must contain a JSON object: {path}")
            return payload, True
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for shared baseline cache: {path}")
            time.sleep(0.05)
            continue
        try:
            os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
            os.close(fd)
            fd = -1
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError(f"Shared baseline cache must contain a JSON object: {path}")
                return payload, True
            result = evaluator()
            write_json(path, result)
            return result, False
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def parse_args(argv: list[str] | None=None) -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--detector-config",type=Path,required=True)
    p.add_argument("--golden-set",type=Path,required=True)
    p.add_argument("--image-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True,help="Regression root; a detector/run-* directory is created below it.")
    p.add_argument("--strategy",choices=("exhaustive","binary-refine","non-dormant","low+","moderate+","important+","critical"),default="exhaustive")
    p.add_argument("--calibration-intelligence",type=Path,default=None,help="Prior calibration-intelligence.json used for effect-size-domain strategies.")
    p.add_argument("--historic-best",type=Path,default=None,help="Exact historic best-known parameter reference injected into every regression.")
    p.add_argument("--precomputed-evidence",type=Path,default=None,help="Parent-precomputed immutable learned Golden Set evidence shared by all shards.")
    p.add_argument("--max-dimension",type=int,default=1800)
    p.add_argument("--limit",type=int,default=None)
    p.add_argument("--top",type=int,default=20)
    p.add_argument("--threads",type=int,default=1,help="Parallel exhaustive-search threads from 1 through 1024; default: 1.")
    p.add_argument("--run-id",default=None)
    p.add_argument("--shard-index",type=int,default=0,help="Zero-based interleaved exhaustive-search shard index.")
    p.add_argument("--shard-count",type=int,default=1,help="Total interleaved exhaustive-search shard count.")
    p.add_argument("--shared-baseline",type=Path,default=None,help="Optional run-local baseline cache shared by parallel shards.")
    p.add_argument(
        "--debug-artifacts",
        choices=("none", "failures", "winner", "all"),
        default=None,
        help="Page-selection policy for debug artifacts; defaults to detector configuration or failures.",
    )
    p.add_argument(
        "--debug-level",
        choices=("none", "basic", "verbose"),
        default=None,
        help="Debug detail level. none writes no images; basic writes the established artifacts; verbose adds detector-specific evidence.",
    )
    args=p.parse_args(argv)
    if not MIN_THREAD_COUNT <= args.threads <= MAX_THREAD_COUNT:
        p.error(f"--threads must be within [{MIN_THREAD_COUNT}, {MAX_THREAD_COUNT}]")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        p.error("--shard-index must be within [0, --shard-count)")
    if args.shard_count > 1 and args.strategy != "exhaustive":
        p.error("sharding currently requires --strategy exhaustive")
    return args

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
    successful=[r for r in page_results if r["status"]=="ok"]; edges=[float(r["edge_error_mean_px"]) for r in successful]; elapsed=[float(r["elapsed_ms"]) for r in page_results]
    summary = aggregate_page_metrics(page_results)
    summary.update({"mean_edge_error_px":round(sum(edges)/len(edges),3) if edges else None,"elapsed_ms_total":round(sum(elapsed),3),"wall_ms":round((time.perf_counter()-started)*1000,3)})
    return {"parameter_set_id":parameter_set_id(parameters),"parameters":parameters,"summary":summary,"pages":page_results}


def failure_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    """Summarize failed page reasons, evidence, and preserved detector exceptions."""
    reasons: dict[str, int] = {}
    numeric: dict[str, list[float]] = {}
    exceptions: dict[tuple[str, str], dict[str, Any]] = {}
    for page in result.get("pages", []):
        if not isinstance(page, dict) or str(page.get("status", "")) == "ok":
            continue
        candidate = page.get("candidate") if isinstance(page.get("candidate"), dict) else {}
        diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
        error = page.get("error") if isinstance(page.get("error"), dict) else {}
        reason = str(diagnostics.get("reason") or error.get("type") or page.get("status") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1

        exception_type = str(diagnostics.get("exception_type") or error.get("type") or "")
        exception_message = str(diagnostics.get("exception_message") or error.get("message") or "")
        exception_traceback = str(diagnostics.get("traceback") or error.get("traceback") or "")
        if exception_type or exception_message:
            key = (exception_type or "Exception", exception_message or "(no message)")
            item = exceptions.setdefault(
                key,
                {
                    "type": key[0],
                    "message": key[1],
                    "count": 0,
                    "example_page": page.get("global_ordinal"),
                    "traceback": exception_traceback or None,
                },
            )
            item["count"] += 1

        for key in ("probability_min", "probability_max", "probability_mean", "thresholded_fraction", "mask_area_fraction"):
            value = diagnostics.get(key)
            if isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))
    ranges = {key: {"min": min(values), "max": max(values)} for key, values in numeric.items() if values}
    return {
        "reason_counts": reasons,
        "diagnostic_ranges": ranges,
        "exceptions": sorted(
            exceptions.values(),
            key=lambda item: (-int(item["count"]), str(item["type"]), str(item["message"])),
        ),
    }


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
    debug_level: str = "basic",
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
    elif candidate.get("method") in {"radial_edge", "adaptive_multi_scale_radial_edge", "amsre_bfq_spbv_pbg", "msre_bfq_spbv_pbg", "adaptive_radial_edge", "multi_scale_radial_edge", "projective_gradient_vote", "border_fusion_quad", "border_energy", "convex_hull", "distance_transform", "distance_transform_rect", "dhsegment_page_mask", "polar_boundary_vote", "page_background", "signed_polar_boundary_vote", "segment_supported_polar_vote", "star_convex", "radon_boundary", "text_flow", "whitespace_frame", "joint_rectangle_vote", "learned_page_mask"} or (
        debug_level == "verbose" and candidate.get("method") in {"gradient_vote", "grabcut"}
    ):
        diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
        parameters = diagnostics.get("parameters") if isinstance(diagnostics.get("parameters"), dict) else None
        method = candidate.get("method")
        module = {
            "radial_edge": detector_radial_edge,
            "adaptive_multi_scale_radial_edge": detector_adaptive_multi_scale_radial_edge,
            "amsre_bfq_spbv_pbg": detector_amsre_bfq_spbv_pbg,
            "msre_bfq_spbv_pbg": detector_msre_bfq_spbv_pbg,
            "adaptive_radial_edge": detector_adaptive_radial_edge,
            "multi_scale_radial_edge": detector_multi_scale_radial_edge,
            "projective_gradient_vote": detector_projective_gradient_vote,
            "border_fusion_quad": detector_border_fusion_quad,
            "gradient_vote": detector_gradient_vote,
            "grabcut": detector_grabcut,
            "border_energy": detector_border_energy,
            "convex_hull": detector_convex_hull,
            "distance_transform": detector_distance_transform,
            "distance_transform_rect": detector_distance_transform_rect,
            "dhsegment_page_mask": detector_dhsegment_page_mask,
            "polar_boundary_vote": detector_polar_boundary_vote,
            "page_background": detector_page_background,
            "signed_polar_boundary_vote": detector_signed_polar_boundary_vote,
            "segment_supported_polar_vote": detector_segment_supported_polar_vote,
            "radon_boundary": detector_radon_boundary,
            "text_flow": detector_text_flow,
            "whitespace_frame": detector_whitespace_frame,
            "joint_rectangle_vote": detector_joint_rectangle_vote,
            "learned_page_mask": detector_learned_page_mask,
            "star_convex": detector_star_convex,
        }[method]
        basic_names_by_method = {
            "radial_edge": ["radial-gradient.png", "radial-edge-points.png"],
            "adaptive_multi_scale_radial_edge": ["adaptive-multi-scale-gradient.png", "adaptive-multi-scale-radial-points.png"],
            "amsre_bfq_spbv_pbg": ["fusion-gen2-gradient.png", "fusion-gen2-child-quads.png", "fusion-gen2-selected-quad.png"],
            "msre_bfq_spbv_pbg": ["fusion-gen1-gradient.png", "fusion-gen1-child-quads.png", "fusion-gen1-selected-quad.png"],
            "adaptive_radial_edge": ["adaptive-radial-gradient.png", "adaptive-radial-edge-points.png"],
            "multi_scale_radial_edge": ["multi-scale-gradient.png", "multi-scale-radial-points.png"],
            "projective_gradient_vote": ["projective-gradient.png", "projective-line-votes.png"],
            "border_fusion_quad": ["fusion-gradient.png", "fusion-child-quads.png", "fusion-selected-quad.png"],
            "gradient_vote": [],
            "grabcut": [],
            "border_energy": ["border-energy.png", "validated-border.png"],
            "convex_hull": ["convex-hull.png"],
            "distance_transform": ["distance-transform.png", "distance-core.png", "distance-candidate.png"],
            "distance_transform_rect": ["distance-rect-transform.png", "distance-rect-core.png", "distance-rect-proposal.png"],
            "polar_boundary_vote": ["polar-gradient.png", "polar-boundary-votes.png"],
            "page_background": ["page-background-distance.png", "page-background-mask.png", "page-background-candidate.png"],
            "signed_polar_boundary_vote": ["signed-polar-gradient.png", "signed-polar-boundary-votes.png"],
            "segment_supported_polar_vote": ["segment-polar-gradient.png", "segment-supported-polar-votes.png"],
            "radon_boundary": ["radon-evidence.png", "radon-boundary.png"],
            "text_flow": ["text-components.png", "text-lines.png"],
            "whitespace_frame": ["whitespace-background.png", "whitespace-frame.png"],
            "joint_rectangle_vote": ["joint-rectangle-edges.png", "joint-rectangle-votes.png"],
            "learned_page_mask": ["learned-page-probability.png", "learned-page-mask.png", "learned-page-boundary.png"],
            "star_convex": ["star-rays.png", "star-mask.png"],
        }
        basic_names = basic_names_by_method.get(method, [])
        images = module.debug_images(
            image_bgr=original, mask=page["mask"], parameters=parameters,
            candidate_corners=candidate.get("corners"), verbose=debug_level == "verbose",
        )
        next_number = 3
        for filename in basic_names:
            if filename in images:
                cv2.imwrite(str(page_dir / f"{next_number:02d}-{filename}"), images[filename])
                next_number += 1
        for filename, debug_image in images.items():
            if filename in basic_names:
                continue
            cv2.imwrite(str(page_dir / f"{next_number:02d}-{filename}"), debug_image)
            next_number += 1
        overlay_name = f"{next_number:02d}-overlay.jpg"
        diagnostics_name = f"{next_number + 1:02d}-diagnostics.json"
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
    debug_level: str = "basic",
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
        candidate = page_result.get("candidate") if isinstance(page_result.get("candidate"), dict) else {}
        candidate_method = str(candidate.get("method") or "")
        if candidate_method and candidate_method != detector:
            raise RuntimeError(
                f"Debug artifact detector mismatch: run={detector!r}, "
                f"candidate={candidate_method!r}, page={page_result.get('global_ordinal')}"
            )
        page = page_by_ordinal[int(page_result["global_ordinal"])]
        _write_debug_page(
            debug_root,
            page=page,
            result=page_result,
            parameter_set_id_value=str(parameter_set["parameter_set_id"]),
            debug_level=debug_level,
        )

    readme = [
        "HTH detector regression debug artifacts",
        "",
        f"Policy: {policy}",
        f"Debug level: {debug_level}",
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


def print_parameter_scope(*, strategy: str, possible_sets: int, planned_sets: int | None, golden_pages: int, threads: int, limit: int | None, shard_index: int = 0, shard_count: int = 1) -> None:
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
        ("Parameter-set Limit", f"{limit} total (including baseline)" if limit is not None else "unlimited"),
        ("Threads", threads),
        ("Shard", f"{shard_index + 1} of {shard_count}"),
        ("Shard Assignment", "interleaved" if shard_count > 1 else "unsharded"),
    ]
    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{label_width}} : {value}")
    print(" ")


def run(args:argparse.Namespace)->Path:
    config=json.loads(args.detector_config.read_text(encoding="utf-8")); name=str(config["detector"])
    regression_config = config.get("regression", {}) if isinstance(config.get("regression"), dict) else {}
    debug_level = args.debug_level or "basic"
    debug_policy = args.debug_artifacts or str(regression_config.get("debug_artifacts", "failures"))
    if debug_level not in {"none", "basic", "verbose"}:
        raise ValueError(f"Unsupported debug level: {debug_level}")
    if debug_policy not in {"none", "failures", "winner", "all"}:
        raise ValueError(f"Unsupported debug artifact policy: {debug_policy}")
    # Verbose debugging is a request for complete page-level evidence for the
    # selected winner.  Do not let a detector's normal failures-only policy
    # silently suppress successful pages.  An explicit ``all`` policy retains
    # its stronger meaning: every page for every evaluated parameter set.
    if debug_level == "none":
        debug_policy = "none"
    elif debug_level == "verbose" and debug_policy in {"none", "failures"}:
        debug_policy = "winner"
    if name not in detector_names():
        raise SystemExit(f"Unsupported detector: {name}")
    run_id,run_dir=create_run_directory(args.output,name,args.run_id); started=utc_now(); wall=time.perf_counter()
    environment=environment_info(repository_root(args.detector_config))
    source_commit=os.environ.get("HTH_SOURCE_COMMIT")
    golden_set_sha256=file_sha256(args.golden_set)
    detector_config_sha256=file_sha256(args.detector_config)
    requested_strategy = args.strategy
    effective_strategy = requested_strategy
    strategy_fallback_reason = None
    manifest={"schema_version":"0.1","run_id":run_id,"detector":name,"strategy":effective_strategy,"requested_strategy":requested_strategy,"strategy_fallback_reason":strategy_fallback_reason,"status":"running","started_at_utc":started,"outputs":[]}
    detector_pipeline_context = {
        "pipeline_count": int(os.environ.get("HTH_DETECTOR_PIPELINES", "1")),
        "pipeline_number": int(os.environ.get("HTH_DETECTOR_PIPELINE_NUMBER", "1")),
        "stagger_minutes": int(os.environ.get("HTH_PIPELINE_STAGGER_MINUTES", "0")),
        "loading_strategy": os.environ.get("HTH_DETECTOR_LOADING_STRATEGY", "fifo"),
        "runtime_estimate_seconds": os.environ.get("HTH_DETECTOR_RUNTIME_ESTIMATE_SECONDS"),
        "runtime_estimate_source": os.environ.get("HTH_DETECTOR_RUNTIME_ESTIMATE_SOURCE"),
        "queue_position": os.environ.get("HTH_DETECTOR_QUEUE_POSITION"),
        "ranked_quality": os.environ.get("HTH_DETECTOR_RANKED_QUALITY"),
        # Preserve the shape-resolution contract in the durable regression record.
        # The manifest renderer can therefore say whether the run used measured
        # optimizer evidence, a prediction, manual input, or the generic planner.
        "execution_shape_source": os.environ.get("HTH_EXACT_EXECUTION_SHAPE_SOURCE", "auto"),
        "exact_execution_shape": os.environ.get("HTH_EXACT_EXECUTION_SHAPE", "0") == "1",
        "execution_thread_budget": os.environ.get("HTH_EXECUTION_THREAD_BUDGET"),
    }
    write_json(run_dir/"manifest.json",manifest)
    try:
        pages=load_pages(args.golden_set,args.image_root,args.max_dimension); detector=detector_entrypoint(name)
        if not callable(detector):
            raise TypeError(
                f"Detector registry entry {name!r} is not callable: "
                f"{type(detector).__name__}"
            )
        evidence_preparer = PRECOMPUTED_EVIDENCE_PREPARERS.get(name)
        evidence_loader = PRECOMPUTED_EVIDENCE_LOADERS.get(name)
        evidence_precompute_seconds = None
        evidence_source = None
        if args.precomputed_evidence is not None:
            if evidence_loader is None:
                raise ValueError(f"Detector {name} does not support --precomputed-evidence")
            evidence_started = time.perf_counter()
            prepared_keys = evidence_loader(
                args.precomputed_evidence,
                [page["image"] for page in pages],
            )
            evidence_precompute_seconds = time.perf_counter() - evidence_started
            evidence_source = "parent-shared"
            print(
                f"Golden Set evidence        : loaded {len(prepared_keys)} parent-precomputed immutable pages "
                f"in {evidence_precompute_seconds:.2f}s from {args.precomputed_evidence}"
            )
        elif evidence_preparer is not None:
            evidence_started = time.perf_counter()

            def local_progress(event, index, total, image_key, elapsed):
                if event == "start":
                    print(
                        f"Golden Set evidence        : page {index}/{total} START key={image_key[:12]}",
                        flush=True,
                    )
                else:
                    print(
                        f"Golden Set evidence        : page {index}/{total} READY "
                        f"key={image_key[:12]} elapsed={elapsed:.2f}s",
                        flush=True,
                    )

            prepared_keys = evidence_preparer(
                [page["image"] for page in pages],
                progress=local_progress,
            )
            evidence_precompute_seconds = time.perf_counter() - evidence_started
            evidence_source = "process-local-fallback"
            print(
                f"Golden Set evidence        : precomputed {len(prepared_keys)} immutable pages "
                f"in {evidence_precompute_seconds:.2f}s before parameter concurrency"
            )
        profiles={canonical_parameters(p):n for n,p in config.get("profiles",{}).items()}
        baseline_parameters=config.get("profiles",{}).get("baseline")
        if not isinstance(baseline_parameters,dict):
            raise ValueError("Detector configuration must define profiles.baseline")
        baseline_key=canonical_parameters(baseline_parameters)

        historic_best_reference = None
        historic_best_parameters = None
        historic_best_key = None
        if args.historic_best is not None and args.historic_best.is_file():
            historic_best_reference = json.loads(args.historic_best.read_text(encoding="utf-8"))
            candidate = historic_best_reference.get("parameters") if isinstance(historic_best_reference, dict) else None
            if isinstance(candidate, dict):
                historic_best_parameters = dict(candidate)
                historic_best_key = canonical_parameters(historic_best_parameters)

        all_parameter_sets=cartesian_generate(config)
        possible_parameter_set_count=len(all_parameter_sets)
        calibration_metadata = None
        if args.calibration_intelligence and args.calibration_intelligence.is_file():
            calibration_metadata = json.loads(args.calibration_intelligence.read_text(encoding="utf-8"))
        requested_strategy = args.strategy
        effective_strategy, effect_domain, strategy_fallback_reason = _resolve_effect_strategy(requested_strategy, calibration_metadata)
        if effect_domain is not None:
            all_parameter_sets = _filter_parameter_sets(all_parameter_sets, effect_domain)
        requested_search_keys = {canonical_parameters(parameters) for parameters in all_parameter_sets}
        historic_best_in_requested_search = bool(historic_best_key and historic_best_key in requested_search_keys)
        write_json(run_dir/"parameters.json",{"schema_version":"0.4","detector":name,"strategy":effective_strategy,"requested_strategy":requested_strategy,"strategy_fallback_reason":strategy_fallback_reason,"detector_config":str(args.detector_config),"golden_set":str(args.golden_set),"golden_set_sha256":golden_set_sha256,"image_root":str(args.image_root),"max_dimension":args.max_dimension,"limit":args.limit,"threads":args.threads,"precomputed_evidence":str(args.precomputed_evidence) if args.precomputed_evidence is not None else None,"debug_level":debug_level,"debug_artifacts":debug_policy,"detector_pipeline":detector_pipeline_context,"shard":{"index":args.shard_index,"count":args.shard_count,"assignment":"interleaved"},"configuration":config})
        manifest.update({"strategy": effective_strategy, "requested_strategy": requested_strategy, "strategy_fallback_reason": strategy_fallback_reason})
        write_json(run_dir/"manifest.json", manifest)
        exhaustive_candidates=[
            parameters for parameters in all_parameter_sets
            if canonical_parameters(parameters) != baseline_key
            and (historic_best_key is None or canonical_parameters(parameters) != historic_best_key)
        ]
        if args.limit is not None:
            # The execution limit is the total number of parameter sets, including baseline.
            exhaustive_candidates=exhaustive_candidates[:max(0, args.limit - 1)]
        full_exhaustive_candidate_count=len(exhaustive_candidates)
        if args.shard_count > 1:
            exhaustive_candidates=[
                parameters for index, parameters in enumerate(exhaustive_candidates)
                if index % args.shard_count == args.shard_index
            ]
        historic_best_planned = int(
            args.shard_index == 0
            and historic_best_parameters is not None
            and historic_best_key != baseline_key
        )
        planned_parameter_set_count=(
            1 + historic_best_planned + len(exhaustive_candidates)
            if effective_strategy=="exhaustive" or effective_strategy in EFFECT_STRATEGY_KEYS else None
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
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        reporter = PRE_REGRESSION_REPORTERS.get(name)
        if reporter is not None:
            print_report_sections(reporter(config))
        progress_total = planned_parameter_set_count if planned_parameter_set_count is not None else estimated_total + 1
        progress=ProgressReporter(total=progress_total,interval_seconds=60.0)
        progress.start()

        progress.begin_evaluation("baseline")
        baseline_result, baseline_reused = load_or_evaluate_shared_baseline(
            args.shared_baseline,
            lambda: evaluate_set(detector,dict(baseline_parameters),logical_golden_set(pages)),
        )
        if canonical_parameters(baseline_result.get("parameters", {})) != baseline_key:
            raise ValueError("Shared baseline cache does not match this detector baseline")
        attach_identity(baseline_result, name, config)
        progress.observe_baseline(baseline_result)
        if args.shared_baseline is not None:
            print(
                f"Shared baseline            : {'reused' if baseline_reused else 'evaluated'} "
                f"({args.shared_baseline})"
            )

        baseline_result["reference_roles"] = ["baseline"]
        baseline_result["requested_search_member"] = False

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
                result=evaluate_set(detector,parameters,logical_golden_set(pages))
            finally:
                with active_lock:
                    active_evaluations -= 1
            if observe:
                progress.observe(result,profile)
            return result
        historic_best_result = None
        if (
            args.shard_index == 0
            and historic_best_parameters is not None
            and historic_best_key != baseline_key
        ):
            progress.begin_evaluation("historic-best")
            historic_best_result = evaluate(dict(historic_best_parameters), observe=False)
            historic_best_result["reference_roles"] = ["historic_best"]
            historic_best_result["requested_search_member"] = historic_best_in_requested_search
            historic_best_result["historic_reference"] = historic_best_reference
            progress.observe(historic_best_result, "historic-best")

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
            for result in candidate_results:
                result["reference_roles"] = []
                result["requested_search_member"] = True
            results=[baseline_result]
            if historic_best_result is not None:
                results.append(historic_best_result)
            results.extend(candidate_results)
        else:
            results=binary_search(config,evaluate,ranking_key)
            for result in results:
                result["reference_roles"] = []
                result["requested_search_member"] = True
            if not any(canonical_parameters(r["parameters"]) == baseline_key for r in results):
                results.insert(0,baseline_result)
            else:
                for result in results:
                    if canonical_parameters(result["parameters"]) == baseline_key:
                        result["reference_roles"] = ["baseline"]
                        result["requested_search_member"] = False
            if historic_best_result is not None:
                duplicate = next(
                    (result for result in results if canonical_parameters(result["parameters"]) == historic_best_key),
                    None,
                )
                if duplicate is None:
                    results.append(historic_best_result)
                else:
                    duplicate["reference_roles"] = sorted(set(duplicate.get("reference_roles", [])) | {"historic_best"})
                    duplicate["historic_reference"] = historic_best_reference
        progress_snapshot=progress.finish()
        performance_samples=performance.finish()
        for r in results:
            attach_identity(r, name, config)
            r["profile"]=profiles.get(canonical_parameters(r["parameters"]))
            r["run_id"]=run_id
        ranked=sorted(results,key=ranking_key)
        for rank,r in enumerate(ranked,1):
            r["rank"]=rank
        search_ranked = [
            result for result in ranked
            if result.get("requested_search_member")
            and not result.get("reference_roles")
        ]
        for search_rank, result in enumerate(search_ranked, 1):
            result["search_rank"] = search_rank
        historic_best_result = next(
            (result for result in ranked if "historic_best" in (result.get("reference_roles") or [])),
            None,
        )
        complete_cartesian = (
            effective_strategy == "exhaustive"
            and args.limit is None
            and args.shard_count == 1
            and len(ranked) >= possible_parameter_set_count
        )
        parameter_provenance = build_provenance(
            name,
            config,
            ranked,
            strategy=effective_strategy,
            complete_cartesian=complete_cartesian,
        )
        write_json(run_dir/"parameter-provenance.json", parameter_provenance)
        baseline=next((r for r in ranked if r.get("profile")=="baseline"),None)
        raw=run_dir/"raw"/"results.csv"; rankings=run_dir/"reports"/"rankings.csv"; top=run_dir/"reports"/"top20.csv"
        write_raw_results(raw,ranked); write_rankings(rankings,ranked); write_rankings(top,ranked[:max(0,args.top)])
        winner_pages = build_winner_page_report(ranked[0], baseline)
        locally_evaluated_parameter_sets = max(0, len(results) - 1) + (0 if baseline_reused else 1)
        locally_evaluated_page_evaluations = locally_evaluated_parameter_sets * len(pages)
        summary={"schema_version":"0.8","run_id":run_id,"detector":name,"strategy":effective_strategy,"requested_strategy":requested_strategy,"strategy_fallback_reason":strategy_fallback_reason,"threads":args.threads,"shard":{"index":args.shard_index,"count":args.shard_count,"assignment":"interleaved","full_candidate_count":full_exhaustive_candidate_count},"detector_pipeline":detector_pipeline_context,"parameter_space":{"possible_parameter_sets":possible_parameter_set_count,"planned_parameter_sets":planned_parameter_set_count,"actual_parameter_sets":len(ranked),"locally_evaluated_parameter_sets":locally_evaluated_parameter_sets,"locally_evaluated_page_evaluations":locally_evaluated_page_evaluations,"baseline_execution":"shared-cache" if baseline_reused else "evaluated","shard_index":args.shard_index,"shard_count":args.shard_count,"full_exhaustive_candidate_count":full_exhaustive_candidate_count,"golden_set_pages":len(pages),"planned_page_evaluations":planned_parameter_set_count*len(pages) if planned_parameter_set_count is not None else None,"actual_page_evaluations":len(ranked)*len(pages),"locally_evaluated_parameter_sets":locally_evaluated_parameter_sets,"locally_evaluated_page_evaluations":locally_evaluated_page_evaluations,"baseline_execution":"shared-cache" if baseline_reused else "evaluated"},"page_ordinals":[p["global_ordinal"] for p in pages],"parameter_set_count":len(ranked),"page_evaluation_count":len(ranked)*len(pages),"successful_page_evaluation_count":len(ranked)*len(pages)-progress_snapshot.failures,"fully_successful_parameter_set_count":sum(1 for r in ranked if int(r["summary"].get("failure_count", 0) or 0) == 0),"golden_set_sha256":golden_set_sha256,"detector_config_sha256":detector_config_sha256,"max_dimension":args.max_dimension,"winner":ranked[0],"baseline":baseline,"historic_best":historic_best_result,"top_parameter_sets":ranked[:5],"search_top_parameter_sets":search_ranked[:5],"winner_page_report":winner_pages,"runner":environment,"source_commit":source_commit,"performance":{"sample_count":len(performance_samples),"configured_threads":args.threads,"peak_rss_bytes":peak_rss_bytes(),"samples_file":"logs/runner-performance.jsonl","precomputed_evidence":name in PRECOMPUTED_EVIDENCE_PREPARERS,"evidence_source":evidence_source,"evidence_precompute_seconds":round(evidence_precompute_seconds,6) if evidence_precompute_seconds is not None else None},"progress":{"estimated_parameter_sets":progress_snapshot.total,"completed_parameter_sets":progress_snapshot.completed,"average_eval_rate":progress_snapshot.eval_rate,"failures":progress_snapshot.failures,"best_mean_iou":progress_snapshot.best_mean_iou,"best_worst_page_iou":progress_snapshot.best_minimum_page_iou,"best_stddev_iou":progress_snapshot.best_stddev_iou,"mean_iou_improvements":progress_snapshot.mean_iou_improvements,"minimum_iou_improvements":progress_snapshot.minimum_iou_improvements,"stddev_improvements":progress_snapshot.stddev_improvements,"total_metric_improvements":progress_snapshot.mean_iou_improvements+progress_snapshot.minimum_iou_improvements+progress_snapshot.stddev_improvements,"parameter_sets_with_improvements":progress_snapshot.parameter_sets_with_improvements,"winner_changes":progress_snapshot.winner_changes,"baseline_surpassed":baseline_surpassed(ranked[0], baseline),"winner_first_changed_elapsed_seconds":progress_snapshot.winner_first_changed_elapsed_seconds,"winner_last_changed_elapsed_seconds":progress_snapshot.winner_last_changed_elapsed_seconds,"winner_history":progress_snapshot.winner_history,"last_improvement_elapsed_seconds":progress_snapshot.last_improvement_elapsed_seconds,"time_since_last_improvement_seconds":progress_snapshot.last_improvement_seconds}}
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
            "detector_pipeline": detector_pipeline_context,
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
            args.output,
            name,
            run_id,
            policy=debug_policy,
            ranked=ranked,
            pages=pages,
            debug_level=debug_level,
        )
        finished=utc_now(); info={"schema_version":"0.3","run_id":run_id,"detector":name,"strategy":effective_strategy,"requested_strategy":requested_strategy,"strategy_fallback_reason":strategy_fallback_reason,"status":"complete","started_at_utc":started,"finished_at_utc":finished,"elapsed_seconds":round(time.perf_counter()-wall,3),"golden_set":str(args.golden_set),"golden_set_sha256":golden_set_sha256,"detector_config":str(args.detector_config),"detector_config_sha256":detector_config_sha256,"max_dimension":args.max_dimension,"debug_artifacts":debug_policy,"debug_level":debug_level,"source_commit":source_commit,"threads":args.threads,"detector_pipeline":detector_pipeline_context,"possible_parameter_sets":possible_parameter_set_count,"planned_parameter_sets":planned_parameter_set_count,"actual_parameter_sets":len(ranked),"shard_index":args.shard_index,"shard_count":args.shard_count,"full_exhaustive_candidate_count":full_exhaustive_candidate_count,"performance_samples":len(performance_samples),"peak_rss_bytes":peak_rss_bytes(),**environment}
        write_json(run_dir/"RUN-INFO.json",info)
        manifest.update({
            "status": "complete",
            "finished_at_utc": finished,
            "outputs": [
                "RUN-INFO.json",
                "parameters.json",
                "parameter-provenance.json",
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
            ("Detector pipeline", f"{detector_pipeline_context['pipeline_number']} of {detector_pipeline_context['pipeline_count']}"),
            ("Pipeline stagger", f"{detector_pipeline_context['stagger_minutes']}m"),
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
        if progress_snapshot.failures:
            diag = failure_diagnostics(ranked[0])
            print(" ")
            print("Failure Diagnostics (winner)")
            print("============================")
            print(f"Reason counts : {json.dumps(diag['reason_counts'], sort_keys=True)}")
            if diag["diagnostic_ranges"]:
                print(f"Evidence ranges: {json.dumps(diag['diagnostic_ranges'], sort_keys=True)}")
            for item in diag.get("exceptions", []):
                print(
                    f"Exception x{item['count']}: {item['type']}: {item['message']} "
                    f"(example page {item.get('example_page')})"
                )
                if item.get("traceback"):
                    print(str(item["traceback"]).rstrip())
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
