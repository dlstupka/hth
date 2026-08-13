"""Derive calibration intelligence from complete detector regression results."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from typing import Any, Iterable


NEAR_BEST_ABSOLUTE_TOLERANCE = 0.001
EQUIVALENT_ABSOLUTE_TOLERANCE = 0.0001
INTERACTION_SAMPLE_LIMIT = 50_000
INTERACTION_PARAMETER_LIMIT = 6


_EFFECT_GROUP_RANK = {
    "Dormant": 0,
    "Low": 1,
    "Moderate": 2,
    "Important": 3,
    "Critical": 4,
}


_DETECTOR_EVIDENCE: dict[str, dict[str, Any]] = {
    "multi_scale_radial_edge": {"friendly_name":"Multi-Scale Radial Edge Search","short_name":"Multi-Scale Radial","role":"Generator","evidence":[("Scale-space gradient field","Primary","Measures page-boundary transitions across several independently normalized blur scales."),("Center-outward rays","Generator","Samples the fused scale-space evidence along radial paths from the document center."),("Cross-scale persistence","Robustness","Allows a physical boundary to remain strong when fine texture or coarse illumination weakens another scale."),("Minimum-area rectangle","Geometry","Fits a page quadrilateral to angularly distributed multi-scale edge points.")]},
    "projective_gradient_vote": {"friendly_name":"Projective Gradient Vote","short_name":"Projective Gradient","role":"Generator","evidence":[("Long line segments","Primary","Finds extended boundary candidates without assuming horizontal or vertical page sides."),("Sobel gradient support","Scoring","Weights each segment by the image transition carried along it."),("Orientation families","Geometry","Selects two near-orthogonal side families while allowing opposite sides to converge under perspective."),("Line intersections","Generator","Builds a projective quadrilateral from opposing members of both families.")]},
    "border_fusion_quad": {"friendly_name":"Border Fusion Quad","short_name":"Border Fusion","role":"Hybrid (Radial + Polar + Gradient)","evidence":[("Radial/Polar/Gradient child quads","Primary","Supplies independent top/right/bottom/left boundary hypotheses from three detector families."),("Side-level recombination","Generator","Allows each page side to come from the child detector that supports it best."),("Gradient side support","Validation","Requires all four fused sides to coincide with image-gradient evidence."),("Source diversity","Validation","Requires the selected quadrilateral to use evidence from multiple child detectors.")]},
    "learned_page_mask": {"friendly_name":"Learned Page-Mask Detector","short_name":"Learned Page Mask","role":"Generator","evidence":[("PageNet segmentation","Primary","Uses a pretrained historical-document CNN to predict per-pixel page membership."),("Probability threshold","Generator","Converts learned probabilities into a page mask."),("Dominant learned region","Geometry","Fits a quadrilateral to the dominant predicted region."),("Model identity","Provenance","Records released-model source, license, and SHA-256.") ]},
    "radon_boundary": {"friendly_name":"Radon Boundary Projection","short_name":"Radon Boundary","role":"Generator","evidence":[("Projection-angle integration","Primary","Integrates gradient evidence along candidate orientations and offsets."),("Opposing projection peaks","Generator","Selects left/right and top/bottom boundary pairs jointly in projection space."),("Orientation search","Geometry","Searches a bounded skew range before mapping the winning rectangle back to image coordinates.")]},
    "text_flow": {"friendly_name":"Text Flow Envelope","short_name":"Text Flow","role":"Generator","evidence":[("Ink components","Primary","Extracts writing-sized connected components from the document mask."),("Text-line grouping","Generator","Joins nearby components into horizontal writing bands."),("Text envelope","Geometry","Fits an oriented document envelope around the recovered text flow.") ]},
    "whitespace_frame": {"friendly_name":"Whitespace Frame","short_name":"Whitespace Frame","role":"Generator","evidence":[("Border whitespace","Primary","Measures whether the image perimeter is dominated by background."),("Negative-space segmentation","Generator","Inverts the surrounding whitespace to isolate the enclosed page region."),("Page envelope","Geometry","Fits an oriented rectangle to the dominant non-background region.") ]},
    "joint_rectangle_vote": {"friendly_name":"Joint Rectangle Voting","short_name":"Joint Rectangle","role":"Generator","evidence":[("Hough line families","Primary","Detects near-horizontal and near-vertical line evidence."),("Opposing side pairs","Generator","Selects outer opposing lines as a single four-side rectangle hypothesis."),("Side support","Validation","Requires edge support along all four proposed page boundaries."),("Rectangle area","Validation","Rejects geometrically implausible page extents.") ]},
    "polar_boundary_vote": {"friendly_name":"Polar Boundary Voting","short_name":"Polar Boundary Vote","role":"Generator","evidence":[("Polar gradient field","Primary","Samples image-gradient evidence along center-outward polar rays."),("Boundary votes","Generator","Selects strong outer transitions on each ray as page-boundary votes."),("Ray support","Validation","Requires sufficient angular support before fitting geometry."),("Minimum-area rectangle","Geometry","Fits the page proposal around the accepted polar votes.")]},
    "signed_polar_boundary_vote": {"friendly_name":"Signed Polar Boundary Voting","short_name":"Signed Polar Vote","role":"Generator","evidence":[("Signed radial gradient","Primary","Measures transition direction as well as magnitude along center-outward rays."),("Polarity gate","Filtering","Prefers bright-page-to-dark-background, dark-page-to-bright-background, or absolute transitions."),("Boundary votes","Generator","Selects strong outer polarity-consistent transitions on each ray."),("Minimum-area rectangle","Geometry","Fits the page proposal around accepted signed polar votes.")]},
    "segment_supported_polar_vote": {"friendly_name":"Segment-Supported Polar Voting","short_name":"Segment Polar Vote","role":"Hybrid (Polar + LSD)","evidence":[("Polar boundary votes","Primary","Generates radial page-boundary hypotheses."),("Long line segments","Validator","Detects independent straight boundary evidence with OpenCV LSD."),("Vote-to-segment proximity","Validation","Retains polar votes that lie close to sufficiently long line segments."),("Minimum-area rectangle","Geometry","Fits the page proposal to segment-supported polar votes.")]},
    "star_convex": {"friendly_name":"Star-Convex Boundary Optimization","short_name":"Star-Convex","role":"Generator","evidence":[("Foreground center","Anchor","Estimates an interior anchor from the document mask."),("Star rays","Primary","Finds the outer supported foreground extent independently along radial directions."),("Angular smoothing","Optimization","Suppresses isolated radial excursions while preserving star-convex boundary support."),("Boundary envelope","Geometry","Fits a page quadrilateral around the optimized radial boundary.")]},
    "distance_transform_rect": {"friendly_name":"Distance-Transform Rectangle Proposal","short_name":"DT Rectangle","role":"Generator","evidence":[("Distance transform","Primary","Measures robust interior support away from foreground boundaries."),("Interior core","Generator","Thresholds the distance field to obtain a stable document core."),("Rectangle expansion","Proposal","Expands the core envelope into a candidate page rectangle."),("Mask coverage","Validation","Rejects proposals with insufficient foreground support or implausible area.")]},
    "convex_hull": {
        "friendly_name": "Convex Hull Detector",
        "short_name": "Convex Hull",
        "role": "Generator",
        "evidence": [("Foreground fragments", "Primary", "Collects substantial foreground regions from the document mask."), ("Convex hull", "Geometry", "Wraps fragmented foreground evidence in the smallest convex envelope."), ("Solidity", "Validation", "Rejects hulls whose enclosed area is poorly supported by foreground evidence."), ("Quadrilateral fit", "Geometry", "Returns a polygonal or minimum-area rectangular page envelope.")],
    },
    "distance_transform": {
        "friendly_name": "Distance Transform Detector",
        "short_name": "Distance Transform",
        "role": "Generator",
        "evidence": [("Distance transform", "Primary", "Measures interior distance from foreground pixels to the nearest background boundary."), ("Interior core", "Generator", "Selects robust page-interior support away from noisy edges."), ("Core-supported components", "Filtering", "Retains connected foreground regions supported by the interior core."), ("Supported hull", "Geometry", "Fits a page quadrilateral around the selected foreground support.")],
    },
    "components": {"friendly_name": "Connected Components", "short_name": "Components", "role": "Generator", "evidence": [("Connected-component envelope", "Primary", "Generates a page-region hypothesis from grouped foreground components."), ("Morphological grouping", "Supporting", "Controls how fragmented marks are joined before envelope extraction.")]},
    "consensus_quad": {"friendly_name": "Consensus Quadrilateral", "short_name": "Consensus Quad", "role": "Hybrid (Contour Quad + Edge Contour)", "evidence": [("Contour Quad vote", "Primary", "Supplies one geometric quadrilateral hypothesis."), ("Edge Contour vote", "Primary", "Supplies an independently scored edge-supported hypothesis."), ("Polygon agreement", "Decision", "Requires sufficient IoU and corner agreement before fusion.")]},
    "contour": {"friendly_name": "Contour Envelope", "short_name": "Contour", "role": "Generator", "evidence": [("Contour geometry", "Primary", "Generates page-region hypotheses from thresholded contours."), ("Fragment merging", "Supporting", "Attempts to recover page boundaries split across multiple contours.")]},
    "contour_components": {"friendly_name": "Contour + Components", "short_name": "Contour Components", "role": "Hybrid (Contour Quad + Components)", "evidence": [("Contour quadrilateral", "Generator", "Produces candidate page quadrilaterals."), ("Component containment", "Validator", "Measures how well selected components fall within each candidate."), ("Component envelope overlap", "Validator", "Compares each contour candidate with the independent component envelope."), ("Component spread and density", "Validator", "Checks whether foreground evidence is distributed plausibly across the candidate.")]},
    "contour_grabcut": {"friendly_name": "Contour + GrabCut", "short_name": "Contour GrabCut", "role": "Hybrid (Contour Quad + GrabCut)", "evidence": [("Contour quadrilateral", "Generator", "Produces the candidate page geometry."), ("GrabCut foreground segmentation", "Validator", "Provides independent pixel-level foreground evidence."), ("Polygon agreement", "Validation", "Requires sufficient overlap between contour and GrabCut hypotheses."), ("Fusion score", "Scoring", "Combines contour quality, GrabCut quality, and hypothesis agreement.")]},
    "grabcut_contour": {"friendly_name": "GrabCut + Contour", "short_name": "GrabCut Contour", "role": "Hybrid (GrabCut + Contour Quad)", "evidence": [("GrabCut foreground segmentation", "Generator", "Generates the primary page polygon from pixel-level foreground segmentation."), ("Foreground contour geometry", "Geometry", "Converts the GrabCut mask into the returned page quadrilateral."), ("Contour quadrilateral", "Validator", "Provides an independent geometric hypothesis for validation."), ("Polygon agreement", "Validation", "Requires sufficient overlap between GrabCut-derived and contour-derived hypotheses."), ("Fusion score", "Scoring", "Combines GrabCut quality, contour quality, and hypothesis agreement while retaining GrabCut geometry.")]},
    "contour_projection": {"friendly_name": "Contour + Projection", "short_name": "Contour Projection", "role": "Hybrid (Contour Quad + Projection)", "evidence": [("Contour quadrilateral", "Generator", "Produces candidate page quadrilaterals."), ("Horizontal projection profile", "Validator", "Scores text-band structure after candidate normalization."), ("Vertical coverage", "Validator", "Checks whether foreground structure spans the candidate height."), ("Ink density", "Validator", "Rejects implausibly empty or saturated candidate interiors.")]},
    "contour_quad": {"friendly_name": "Contour Quadrilateral", "short_name": "Contour Quad", "role": "Generator", "evidence": [("Contour quadrilaterals", "Primary", "Generates multiple polygonal page hypotheses."), ("Area", "Scoring", "Rewards candidates occupying a plausible image fraction."), ("Rectangularity", "Scoring", "Rewards quadrilateral-like contour geometry."), ("Corner angles", "Scoring", "Rewards near-right-angle page geometry.")]},
    "cross_edge_contour": {
        "friendly_name": "Cross-Edge Contour",
        "short_name": "X-Edge Contour",
        "role": "Hybrid (Contour Quad + Cross-Edge Validation)",
        "evidence": [("Contour quadrilateral", "Generator", "Produces candidate page geometry."), ("Inside/outside intensity samples", "Validator", "Measures the image transition across each proposed boundary."), ("Cross-edge contrast", "Validation", "Rejects geometrically plausible boundaries lacking a real photometric transition."), ("Polarity consistency", "Validation", "Checks that inside-versus-outside transition direction is coherent around the page.")],
    },
    "gradient_vote": {
        "friendly_name": "Gradient Boundary Voting",
        "short_name": "Gradient Vote",
        "role": "Generator",
        "evidence": [("Sobel gradient field", "Primary", "Measures distributed horizontal and vertical intensity transitions."), ("Boundary vote profiles", "Generator", "Accumulates local gradients into opposing page-boundary votes."), ("Peak prominence", "Validation", "Requires selected boundaries to stand out from competing transitions."), ("Boundary span", "Geometry", "Forms a page quadrilateral from the winning left, right, top, and bottom votes.")],
    },
    "radial_edge": {
        "friendly_name": "Radial Edge Search",
        "short_name": "Radial Edge",
        "role": "Generator",
        "evidence": [("Center-outward rays", "Primary", "Samples image gradients along radial paths from the document center."), ("Strongest radial transitions", "Generator", "Selects likely page-boundary points independently on each ray."), ("Minimum-area rectangle", "Geometry", "Fits a quadrilateral to the supported radial edge points."), ("Ray support", "Validation", "Rejects candidates when too few directions provide credible boundary evidence.")],
    },
    "adaptive_radial_edge": {
        "friendly_name": "Adaptive Radial Edge Search",
        "short_name": "Adaptive Radial",
        "role": "Generator",
        "evidence": [("Coarse center-outward rays", "Primary", "Samples the full image at 3-degree spacing."), ("Weak-side support", "Trigger", "Identifies fitted document sides with comparatively sparse boundary confirmation."), ("One-degree angular refinement", "Generator", "Adds a second pass only through weak-side sectors."), ("Refined quadrilateral", "Geometry", "Refits the page boundary from combined coarse and refined evidence.")],
    },
    "border_energy": {
        "friendly_name": "Border Energy Validator",
        "short_name": "Border Energy",
        "role": "Hybrid (Contour Quad + Border Energy)",
        "evidence": [("Contour quadrilateral", "Generator", "Produces candidate page geometry."), ("Sobel border energy", "Validator", "Measures gradient magnitude in a narrow band along each proposed border."), ("Side consistency", "Validation", "Requires all four sides to carry comparable boundary evidence."), ("Fusion score", "Scoring", "Combines contour quality, border energy, and side consistency.")],
    },
    "edge_contour": {"friendly_name": "Edge-Supported Contour", "short_name": "Edge Contour", "role": "Hybrid (Contour Quad + LSD)", "evidence": [("Contour quadrilateral", "Generator", "Produces candidate page quadrilaterals."), ("LSD line segments", "Validator", "Independently detects line support near proposed borders."), ("Edge support", "Validator", "Measures border coverage after configurable dilation."), ("Geometry score", "Scoring", "Combines area, rectangularity, and angle quality.")]},
    "grabcut": {"friendly_name": "GrabCut Segmentation", "short_name": "GrabCut", "role": "Generator", "evidence": [("GrabCut foreground mask", "Primary", "Segments foreground pixels from a border-seeded background model."), ("Morphological cleanup", "Supporting", "Closes and erodes the segmentation before region extraction."), ("Foreground contour", "Geometry", "Converts the segmented region into a page polygon or bounding quadrilateral.")]},
    "hough": {"friendly_name": "Hough Line Borders", "short_name": "Hough", "role": "Generator", "evidence": [("Hough lines", "Primary", "Generates axis-aligned border hypotheses from detected lines."), ("Outer-line percentile", "Scoring", "Selects outer line groups used to form a page box."), ("Axis-angle tolerance", "Filtering", "Restricts candidate lines to near-horizontal or near-vertical orientations.")]},
    "lsd": {"friendly_name": "Line Segment Detector", "short_name": "LSD", "role": "Generator", "evidence": [("LSD segments", "Primary", "Generates border hypotheses directly from line segments."), ("Outer-line percentile", "Scoring", "Selects outer segment groups for page-boundary construction."), ("Axis-angle tolerance", "Filtering", "Limits segments to plausible page-border orientations.")]},
    "ransac": {"friendly_name": "RANSAC Border Fit", "short_name": "RANSAC", "role": "Generator", "evidence": [("Scan foreground samples", "Primary", "Samples likely border evidence along image scans."), ("RANSAC line fitting", "Primary", "Fits robust page-border models while rejecting outliers."), ("Inlier ratio", "Validation", "Requires sufficient support for accepted line models.")]},
}


def detector_characterization(detector: str) -> dict[str, Any]:
    item = _DETECTOR_EVIDENCE.get(detector, {
        "friendly_name": detector.replace("_", " ").title(),
        "short_name": detector,
        "role": "Unknown",
        "evidence": [("Detector output", "Primary", "Evidence characterization has not yet been registered for this detector.")],
    })
    return {
        "friendly_name": item["friendly_name"],
        "short_name": item["short_name"],
        "role": item["role"],
        "evidence": list(item["evidence"]),
    }


def _detector_evidence(detector: str) -> dict[str, Any]:
    item = detector_characterization(detector)
    return {
        "detector_id": detector,
        "friendly_name": item["friendly_name"],
        "short_name": item["short_name"],
        "role": item["role"],
        "evidence_sources": [
            {"source": source, "function": function, "interpretation": interpretation}
            for source, function, interpretation in item["evidence"]
        ],
    }


def _domain_space(parameters_report: list[dict[str, Any]], winner_parameters: dict[str, Any], possible_parameter_sets: int | None) -> dict[str, Any]:
    """Build executable cumulative effect-size parameter domains."""
    exhaustive_count = int(possible_parameter_sets or 0)
    domains: dict[str, Any] = {
        "exhaustive": {
            "parameter_set_count": exhaustive_count,
            "included_parameters": [str(item.get("parameter")) for item in parameters_report],
            "fixed_parameters": {},
        }
    }
    specifications = (
        ("non_dormant", 1),
        ("low_plus", 1),
        ("moderate_plus", 2),
        ("important_plus", 3),
        ("critical", 4),
    )
    for key, minimum_rank in specifications:
        included = [
            item for item in parameters_report
            if _EFFECT_GROUP_RANK.get(str(item.get("classification")), 0) >= minimum_rank
        ]
        count = math.prod(max(1, int(item.get("value_count", 1) or 1)) for item in included) if included else 0
        names = [str(item.get("parameter")) for item in included]
        domains[key] = {
            "parameter_set_count": count,
            "included_parameters": names,
            "fixed_parameters": {
                name: value for name, value in winner_parameters.items() if name not in names
            },
        }
    return domains


def _value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _display_value(key: str) -> str:
    try:
        value = json.loads(key)
    except json.JSONDecodeError:
        return key
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _quantile_desc(values: list[float], quantile: float) -> float | None:
    """Return a quantile from values already ordered from highest to lowest."""
    if not values:
        return None
    ascending_position = quantile * (len(values) - 1)
    descending_position = (len(values) - 1) - ascending_position
    lower = int(math.floor(descending_position))
    upper = int(math.ceil(descending_position))
    if lower == upper:
        return values[lower]
    fraction = descending_position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _eta_squared(groups: dict[str, list[float]], overall_mean: float, total_ss: float) -> float:
    if total_ss <= 0.0:
        return 0.0
    between = sum(bucket[0] * ((bucket[1] / bucket[0]) - overall_mean) ** 2 for bucket in groups.values() if bucket[0])
    return max(0.0, min(1.0, between / total_ss))


def _influence_class(eta_squared: float, mean_range: float) -> str:
    if mean_range < EQUIVALENT_ABSOLUTE_TOLERANCE or eta_squared < 0.001:
        return "Dormant"
    if eta_squared >= 0.25:
        return "Critical"
    if eta_squared >= 0.10:
        return "Important"
    if eta_squared >= 0.03:
        return "Moderate"
    return "Low"


def _confidence(*, exhaustive_complete: bool, success_rate: float, near_best_share: float, set_count: int) -> tuple[str, list[str]]:
    reasons: list[str] = []
    score = 0
    if exhaustive_complete:
        score += 2
        reasons.append("complete exhaustive coverage")
    else:
        reasons.append("partial or adaptive search")
    if success_rate >= 0.90:
        score += 1
        reasons.append("high fully-successful-set rate")
    elif success_rate < 0.50:
        reasons.append("many parameter sets failed at least one page")
    if near_best_share >= 0.01:
        score += 1
        reasons.append("broad near-best basin")
    elif set_count > 20:
        reasons.append("narrow near-best basin")
    if set_count < 20:
        reasons.append("small calibration sample")
    if score >= 4:
        return "High", reasons
    if score >= 2:
        return "Medium", reasons
    return "Low", reasons


def build_calibration_intelligence(
    ranked: list[dict[str, Any]],
    *,
    detector: str,
    strategy: str,
    possible_parameter_sets: int | None,
    calibration_context: dict[str, Any] | None = None,
    regression_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact, machine-readable characterization of a calibration run.

    The conclusions are intentionally scoped to the evaluated Golden Set and
    parameter grid. They do not claim detector behavior outside that corpus.
    """
    if not ranked:
        return {
            "schema_version": "1.0",
            "detector": detector,
            "available": False,
            "reason": "no evaluated parameter sets",
        }

    page_evaluations = [
        page
        for result in ranked
        for page in (result.get("pages", []) if isinstance(result.get("pages"), list) else [])
        if isinstance(page, dict)
    ]
    successful_page_evaluations = sum(1 for page in page_evaluations if str(page.get("status", "")) == "ok")
    failure_reasons = Counter()
    for page in page_evaluations:
        if str(page.get("status", "")) == "ok":
            continue
        candidate = page.get("candidate") if isinstance(page.get("candidate"), dict) else {}
        diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
        error = page.get("error") if isinstance(page.get("error"), dict) else {}
        reason = diagnostics.get("reason") or error.get("type") or page.get("status") or "unknown"
        failure_reasons[str(reason)] += 1
    positive_iou_page_evaluations = sum(
        1 for page in page_evaluations
        if str(page.get("status", "")) == "ok" and float(page.get("iou", 0.0) or 0.0) > 0.0
    )
    summary_positive_signal = any(
        float(result.get("summary", {}).get("mean_iou", 0.0) or 0.0) > 0.0
        for result in ranked
    )
    if not page_evaluations and summary_positive_signal:
        # Older/persisted fixtures may omit page detail while retaining valid
        # aggregate metrics; keep those calibrations analyzable.
        measurement_state = {
            "informative": True,
            "status": "measured",
            "reason": "Calibration contains positive aggregate overlap measurements.",
            "page_evaluations": 0,
            "successful_page_evaluations": 0,
            "positive_iou_page_evaluations": 0,
        }
    elif successful_page_evaluations == 0:
        measurement_state = {
            "informative": False,
            "status": "no_valid_measurements",
            "reason": "No page evaluation produced a valid detector candidate.",
            "page_evaluations": len(page_evaluations),
            "successful_page_evaluations": 0,
            "positive_iou_page_evaluations": 0,
            "failure_reason_counts": dict(failure_reasons.most_common()),
        }
    elif positive_iou_page_evaluations == 0:
        measurement_state = {
            "informative": False,
            "status": "no_overlap_signal",
            "reason": "Valid detector candidates were produced, but none overlapped an approved Golden Set bounding box.",
            "page_evaluations": len(page_evaluations),
            "successful_page_evaluations": successful_page_evaluations,
            "positive_iou_page_evaluations": 0,
            "failure_reason_counts": dict(failure_reasons.most_common()),
        }
    else:
        measurement_state = {
            "informative": True,
            "status": "measured",
            "reason": "Calibration contains valid positive-overlap measurements.",
            "page_evaluations": len(page_evaluations),
            "successful_page_evaluations": successful_page_evaluations,
            "positive_iou_page_evaluations": positive_iou_page_evaluations,
            "failure_reason_counts": dict(failure_reasons.most_common()),
        }

    scores = [float(result.get("summary", {}).get("mean_iou", 0.0) or 0.0) for result in ranked]
    best_score = scores[0]
    count = len(scores)
    overall_mean = sum(scores) / count
    total_ss = sum((score - overall_mean) ** 2 for score in scores)
    successful_count = sum(
        1 for result in ranked
        if int(result.get("summary", {}).get("failure_count", 0) or 0) == 0
    )
    near_best_count = sum(1 for score in scores if best_score - score <= NEAR_BEST_ABSOLUTE_TOLERANCE)
    equivalent_count = sum(1 for score in scores if best_score - score <= EQUIVALENT_ABSOLUTE_TOLERANCE)

    group_accumulators: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, -math.inf]))
    near_best_values: dict[str, set[str]] = defaultdict(set)
    page_accumulators: dict[int, dict[str, float]] = defaultdict(lambda: {
        "count": 0.0, "success": 0.0, "sum": 0.0, "sum_sq": 0.0,
        "minimum": math.inf, "maximum": -math.inf,
    })

    for result, score in zip(ranked, scores):
        parameters = result.get("parameters", {})
        if isinstance(parameters, dict):
            for name, value in parameters.items():
                key = _value_key(value)
                bucket = group_accumulators[str(name)][key]
                bucket[0] += 1.0
                bucket[1] += score
                bucket[2] += score * score
                bucket[3] = max(bucket[3], score)
                if best_score - score <= NEAR_BEST_ABSOLUTE_TOLERANCE:
                    near_best_values[str(name)].add(key)
        for page in result.get("pages", []):
            try:
                ordinal = int(page.get("global_ordinal"))
            except (TypeError, ValueError):
                continue
            accumulator = page_accumulators[ordinal]
            accumulator["count"] += 1.0
            page_iou = float(page.get("iou", 0.0) or 0.0)
            if str(page.get("status", "")) == "ok":
                accumulator["success"] += 1.0
            accumulator["sum"] += page_iou
            accumulator["sum_sq"] += page_iou * page_iou
            accumulator["minimum"] = min(accumulator["minimum"], page_iou)
            accumulator["maximum"] = max(accumulator["maximum"], page_iou)

    parameters_report: list[dict[str, Any]] = []
    for name, groups in group_accumulators.items():
        means = [(key, bucket[1] / bucket[0]) for key, bucket in groups.items() if bucket[0]]
        means.sort(key=lambda item: (-item[1], item[0]))
        mean_values = [mean for _, mean in means]
        mean_range = max(mean_values) - min(mean_values) if mean_values else 0.0
        eta = _eta_squared(groups, overall_mean, total_ss)
        parameters_report.append({
            "parameter": name,
            "value_count": len(groups),
            "eta_squared": eta,
            "mean_iou_range": mean_range,
            "classification": _influence_class(eta, mean_range),
            "near_best_value_coverage": (
                len(near_best_values.get(name, set())) / len(groups) if groups else 0.0
            ),
            "best_values": [
                {"value": _display_value(key), "mean_iou": mean, "count": int(groups[key][0])}
                for key, mean in means[:3]
            ],
        })
    parameters_report.sort(key=lambda item: (-float(item["eta_squared"]), -float(item["mean_iou_range"]), str(item["parameter"])))
    if not measurement_state["informative"]:
        # A field of identical failure/zero-overlap scores is not evidence that
        # parameters are dormant. Withhold influence and reduction claims until
        # the detector produces an actual calibration signal.
        parameters_report = []

    interaction_parameters = [item["parameter"] for item in parameters_report[:INTERACTION_PARAMETER_LIMIT]]
    sample_step = max(1, math.ceil(count / INTERACTION_SAMPLE_LIMIT))
    sample = ranked[::sample_step]
    sample_scores = [float(result.get("summary", {}).get("mean_iou", 0.0) or 0.0) for result in sample]
    sample_mean = sum(sample_scores) / len(sample_scores) if sample_scores else 0.0
    sample_ss = sum((score - sample_mean) ** 2 for score in sample_scores)
    sample_individual_eta: dict[str, float] = {}
    for parameter in interaction_parameters:
        groups: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for result, score in zip(sample, sample_scores):
            parameters = result.get("parameters", {})
            if not isinstance(parameters, dict) or parameter not in parameters:
                continue
            key = _value_key(parameters[parameter])
            groups[key][0] += 1.0
            groups[key][1] += score
        sample_individual_eta[parameter] = _eta_squared(groups, sample_mean, sample_ss)

    interactions: list[dict[str, Any]] = []
    for left_index, left in enumerate(interaction_parameters):
        for right in interaction_parameters[left_index + 1:]:
            groups: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
            for result, score in zip(sample, sample_scores):
                parameters = result.get("parameters", {})
                if not isinstance(parameters, dict) or left not in parameters or right not in parameters:
                    continue
                key = _value_key([parameters[left], parameters[right]])
                groups[key][0] += 1.0
                groups[key][1] += score
            pair_eta = _eta_squared(groups, sample_mean, sample_ss)
            incremental = max(0.0, pair_eta - max(sample_individual_eta.get(left, 0.0), sample_individual_eta.get(right, 0.0)))
            interactions.append({
                "parameters": [left, right],
                "eta_squared": pair_eta,
                "incremental_importance": incremental,
                "sample_size": len(sample),
            })
    interactions.sort(key=lambda item: (-float(item["incremental_importance"]), -float(item["eta_squared"]), item["parameters"]))

    page_report: list[dict[str, Any]] = []
    for ordinal, accumulator in sorted(page_accumulators.items()):
        page_count = int(accumulator["count"])
        mean = accumulator["sum"] / page_count if page_count else 0.0
        variance = max(0.0, accumulator["sum_sq"] / page_count - mean * mean) if page_count else 0.0
        page_report.append({
            "global_ordinal": ordinal,
            "mean_iou": mean,
            "minimum_iou": accumulator["minimum"] if page_count else None,
            "maximum_iou": accumulator["maximum"] if page_count else None,
            "stddev_iou": math.sqrt(variance),
            "success_rate": accumulator["success"] / page_count if page_count else 0.0,
        })

    exhaustive_complete = bool(
        strategy in {"exhaustive", "cartesian"}
        and possible_parameter_sets is not None
        and count >= int(possible_parameter_sets)
    )
    success_rate = successful_count / count
    near_best_share = near_best_count / count
    confidence, confidence_reasons = _confidence(
        exhaustive_complete=exhaustive_complete,
        success_rate=success_rate,
        near_best_share=near_best_share,
        set_count=count,
    )
    dormant = [item["parameter"] for item in parameters_report if item["classification"] == "Dormant"]
    winner_parameters = ranked[0].get("parameters", {}) if isinstance(ranked[0].get("parameters"), dict) else {}
    domain_space = (
        _domain_space(parameters_report, winner_parameters, possible_parameter_sets)
        if measurement_state["informative"] else {}
    )
    calibration_context = dict(calibration_context or {})
    regression_context = dict(regression_context or {})
    winner = ranked[0]
    winner_summary = winner.get("summary", {}) if isinstance(winner.get("summary"), dict) else {}
    winner_parameter_set_id = winner.get("parameter_set_id") or winner.get("parameter_short_name")
    fallback_order = ["critical", "important_plus", "moderate_plus", "low_plus", "non_dormant", "exhaustive"]
    available_domains = [
        name for name, domain in domain_space.items()
        if isinstance(domain, dict) and int(domain.get("parameter_set_count", 0) or 0) > 0
    ]
    parameter_intelligence = {
        "effect_size_method": "one-way eta-squared over Avg IoU",
        "classification_thresholds": {
            "dormant": {"eta_squared_below": 0.001, "or_avg_iou_range_below": EQUIVALENT_ABSOLUTE_TOLERANCE},
            "low": {"eta_squared_minimum": 0.001},
            "moderate": {"eta_squared_minimum": 0.03},
            "important": {"eta_squared_minimum": 0.10},
            "critical": {"eta_squared_minimum": 0.25},
        },
        "parameters": parameters_report,
        "dormant_parameters": dormant,
        "active_parameters": [item["parameter"] for item in parameters_report if item["classification"] != "Dormant"],
        "interactions": interactions[:10],
        "interaction_method": {
            "parameters_considered": interaction_parameters,
            "sample_size": len(sample),
            "sample_step": sample_step,
            "note": "Pairwise interaction importance is estimated from a deterministic sample and is exploratory, not causal.",
        },
        "page_sensitivity": page_report,
    }
    domain_space_intelligence = {
        "domains": domain_space,
        "default_strategy": "exhaustive",
        "fallback_order": fallback_order,
        "available_domains": available_domains,
        "scope": "Golden Set and detector configuration specific",
    }
    detector_selection_intelligence = {
        "recommended_detector_id": detector,
        "recommended_parameter_set_id": winner_parameter_set_id,
        "recommended_parameters": winner_parameters,
        "best_avg_iou": winner_summary.get("mean_iou"),
        "avg_iou_success": winner_summary.get("mean_iou_success", winner_summary.get("mean_iou")),
        "minimum_iou": winner_summary.get("minimum_iou"),
        "stddev_iou": winner_summary.get("stddev_iou"),
        "failure_count": winner_summary.get("failure_count"),
        "near_best_coverage": near_best_share,
        "equivalent_best_coverage": equivalent_count / count,
        "calibration_evidence": {"rating": confidence, "reasons": confidence_reasons},
        "recommended_search_domains": available_domains,
        "applicability": {
            "source_document": calibration_context.get("source_document"),
            "golden_set": calibration_context.get("golden_set"),
            "detector_configuration": calibration_context.get("detector_configuration"),
            "revalidate_when": ["source document changes", "Golden Set changes", "detector configuration changes", "effect-size policy changes"],
        },
    }

    return {
        "schema_version": "1.1",
        "calibration_identity": calibration_context,
        "regression_metadata": regression_context,
        "detector_evidence": _detector_evidence(detector),
        "parameter_intelligence": parameter_intelligence,
        "domain_space_intelligence": domain_space_intelligence,
        "detector_selection_intelligence": detector_selection_intelligence,
        "detector": detector,
        "available": True,
        "measurement_state": measurement_state,
        "scope_note": "All conclusions are specific to the evaluated Golden Set and configured parameter grid.",
        "search": {
            "strategy": strategy,
            "parameter_sets": count,
            "possible_parameter_sets": possible_parameter_sets,
            "exhaustive_complete": exhaustive_complete,
            "fully_successful_parameter_sets": successful_count,
            "fully_successful_rate": success_rate,
        },
        "landscape": {
            "best_mean_iou": best_score,
            "mean_mean_iou": overall_mean,
            "minimum_mean_iou": scores[-1],
            "stddev_mean_iou": math.sqrt(total_ss / count) if count else 0.0,
            "median_mean_iou": _quantile_desc(scores, 0.50),
            "p90_mean_iou": _quantile_desc(scores, 0.90),
            "p95_mean_iou": _quantile_desc(scores, 0.95),
            "p99_mean_iou": _quantile_desc(scores, 0.99),
            "equivalent_winner_count": equivalent_count,
            "equivalent_winner_share": equivalent_count / count,
            "near_best_count": near_best_count,
            "near_best_share": near_best_share,
            "equivalent_tolerance": EQUIVALENT_ABSOLUTE_TOLERANCE,
            "near_best_tolerance": NEAR_BEST_ABSOLUTE_TOLERANCE,
        },
        "parameter_influence": parameters_report,
        "domain_space": domain_space,
        "interactions": interactions[:10],
        "interaction_method": {
            "parameters_considered": interaction_parameters,
            "sample_size": len(sample),
            "sample_step": sample_step,
            "note": "Pairwise interaction importance is estimated from a deterministic sample and is exploratory, not causal.",
        },
        "page_sensitivity": page_report,
        "recommendations": {
            "dormant_parameters": dormant,
            "retain_for_revalidation": bool(dormant),
            "note": "Dormant parameters may be omitted from future searches for this Golden Set, but should be re-evaluated when the Golden Set changes.",
        },
        "calibration_confidence": {
            "rating": confidence,
            "reasons": confidence_reasons,
        },
    }
