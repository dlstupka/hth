from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import time
import traceback

import cv2
import numpy as np

from . import (
    detector_adaptive_multi_scale_radial_edge,
    detector_adaptive_radial_edge,
    detector_border_energy,
    detector_border_fusion_quad,
    detector_components,
    detector_convex_hull,
    detector_contour,
    detector_contour_components,
    detector_contour_grabcut,
    detector_cross_edge_contour,
    detector_distance_transform,
    detector_distance_transform_rect,
    detector_polar_boundary_vote,
    detector_page_background,
    detector_signed_polar_boundary_vote,
    detector_segment_supported_polar_vote,
    detector_radon_boundary,
    detector_star_convex,
    detector_text_flow,
    detector_whitespace_frame,
    detector_contour_quad,
    detector_contour_projection,
    detector_consensus_quad,
    detector_edge_contour,
    detector_grabcut,
    detector_grabcut_contour,
    detector_gradient_vote,
    detector_multi_scale_radial_edge,
    detector_msre_bfq_spbv_pbg,
    detector_projective_gradient_vote,
    detector_joint_rectangle_vote,
    detector_learned_page_mask,
    detector_hough,
    detector_lsd,
    detector_radial_edge,
    detector_ransac,
)
from .model import Candidate
try:
    # Package mode: imported as hth.geometry.registry (tests, installed use).
    from hth.version import HTH_REPOSITORY, HTH_VERSION
except ModuleNotFoundError as exc:
    # Script mode: workflows execute hth/detect_geometry_candidates.py directly,
    # which places the hth/ directory itself on sys.path.
    if exc.name != "hth":
        raise
    from version import HTH_REPOSITORY, HTH_VERSION

Detector = Callable[..., Candidate]


@dataclass(frozen=True)
class DetectorSpec:
    """Small plugin contract; framework services stay outside detectors."""

    method: str
    name: str
    origin: str
    entrypoint: Detector
    foundation: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    version: str = ""
    repository: str = ""

    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.origin})" if self.origin else self.name


# Order is intentional and preserves the pre-registry JSON candidate order.
# Method IDs remain stable for downstream compatibility; names are presentation.
_REGISTRY: tuple[DetectorSpec, ...] = (
    DetectorSpec(
        method=detector_contour.METHOD,
        name="Contour",
        origin="HTH",
        entrypoint=detector_contour.detect,
        foundation=("OpenCV",),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_contour_quad.METHOD,
        name="Contour Quadrilateral",
        origin="HTH",
        entrypoint=detector_contour_quad.detect,
        foundation=("Contour geometry", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_contour_components.METHOD,
        name="Contour + Components",
        origin="HTH",
        entrypoint=detector_contour_components.detect,
        foundation=("Contour geometry", "Connected components", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_contour_grabcut.METHOD,
        name="Contour + GrabCut",
        origin="HTH",
        entrypoint=detector_contour_grabcut.detect,
        foundation=("Contour geometry", "GrabCut", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_grabcut_contour.METHOD,
        name="GrabCut + Contour",
        origin="HTH",
        entrypoint=detector_grabcut_contour.detect,
        foundation=("GrabCut", "Contour geometry", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_contour_projection.METHOD,
        name="Contour + Projection",
        origin="HTH",
        entrypoint=detector_contour_projection.detect,
        foundation=("Contour geometry", "Projection profiles", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_consensus_quad.METHOD,
        name="Consensus Quad",
        origin="HTH",
        entrypoint=detector_consensus_quad.detect,
        foundation=("Contour Quadrilateral", "Edge-Contour Hybrid", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_cross_edge_contour.METHOD,
        name="Cross-Edge Contour",
        origin="HTH",
        entrypoint=detector_cross_edge_contour.detect,
        foundation=("Contour geometry", "Cross-boundary intensity sampling", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_gradient_vote.METHOD,
        name="Gradient Boundary Voting",
        origin="HTH",
        entrypoint=detector_gradient_vote.detect,
        foundation=("Sobel gradients", "Projection voting", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_radial_edge.METHOD,
        name="Radial Edge Search",
        origin="HTH",
        entrypoint=detector_radial_edge.detect,
        foundation=("Radial gradient search", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_adaptive_multi_scale_radial_edge.METHOD,
        name="Adaptive Multi-Scale Radial Edge Search",
        origin="HTH",
        entrypoint=detector_adaptive_multi_scale_radial_edge.detect,
        foundation=("Multi-scale gradients", "Adaptive angular refinement", "Radial gradient search", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_adaptive_radial_edge.METHOD,
        name="Adaptive Radial Edge Search",
        origin="HTH",
        entrypoint=detector_adaptive_radial_edge.detect,
        foundation=("Two-pass radial gradient search", "Adaptive angular refinement", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_multi_scale_radial_edge.METHOD,
        name="Multi-Scale Radial Edge Search",
        origin="HTH",
        entrypoint=detector_multi_scale_radial_edge.detect,
        foundation=("Scale-space gradients", "Radial gradient search", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_msre_bfq_spbv_pbg.METHOD,
        name="Fusion Gen1 — MSRE + BFQ + SPBV + Page Background",
        origin="HTH",
        entrypoint=detector_msre_bfq_spbv_pbg.detect,
        foundation=("Multi-Scale Radial Edge", "Border Fusion Quad", "Signed Polar Boundary Voting", "Page Background", "Side-level consensus", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_projective_gradient_vote.METHOD,
        name="Projective Gradient Vote",
        origin="HTH",
        entrypoint=detector_projective_gradient_vote.detect,
        foundation=("Sobel gradients", "Line Segment Detector", "Projective line intersections", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_border_fusion_quad.METHOD,
        name="Border Fusion Quad",
        origin="HTH",
        entrypoint=detector_border_fusion_quad.detect,
        foundation=("Radial Edge Search", "Polar Boundary Voting", "Gradient Boundary Voting", "Side-level fusion", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_border_energy.METHOD,
        name="Border Energy Validator",
        origin="HTH",
        entrypoint=detector_border_energy.detect,
        foundation=("Contour geometry", "Sobel border energy", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_edge_contour.METHOD,
        name="Edge-Contour Hybrid",
        origin="HTH",
        entrypoint=detector_edge_contour.detect,
        foundation=("Contour geometry", "Line Segment Detector", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_convex_hull.METHOD,
        name="Convex Hull Detector",
        origin="HTH",
        entrypoint=detector_convex_hull.detect,
        foundation=("Convex hull geometry", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_distance_transform.METHOD,
        name="Distance Transform Detector",
        origin="HTH",
        entrypoint=detector_distance_transform.detect,
        foundation=("Distance transform", "Connected components", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(method=detector_polar_boundary_vote.METHOD, name="Polar Boundary Voting", origin="HTH", entrypoint=detector_polar_boundary_vote.detect, foundation=("Polar gradient voting", "OpenCV"), authors=("OpenAI ChatGPT",), version=HTH_VERSION, repository=HTH_REPOSITORY),
    DetectorSpec(method=detector_signed_polar_boundary_vote.METHOD, name="Signed Polar Boundary Voting", origin="HTH", entrypoint=detector_signed_polar_boundary_vote.detect, foundation=("Signed radial gradients", "Polar boundary voting", "OpenCV"), authors=("OpenAI ChatGPT",), version=HTH_VERSION, repository=HTH_REPOSITORY),
    DetectorSpec(method=detector_segment_supported_polar_vote.METHOD, name="Segment-Supported Polar Voting", origin="HTH", entrypoint=detector_segment_supported_polar_vote.detect, foundation=("Polar boundary voting", "Line Segment Detector", "OpenCV"), authors=("OpenAI ChatGPT",), version=HTH_VERSION, repository=HTH_REPOSITORY),
    DetectorSpec(method=detector_star_convex.METHOD, name="Star-Convex Boundary Optimization", origin="HTH", entrypoint=detector_star_convex.detect, foundation=("Star-convex geometry", "Radial mask support", "OpenCV"), authors=("OpenAI ChatGPT",), version=HTH_VERSION, repository=HTH_REPOSITORY),
    DetectorSpec(method=detector_distance_transform_rect.METHOD, name="Distance-Transform Rectangle Proposal", origin="HTH", entrypoint=detector_distance_transform_rect.detect, foundation=("Distance transform", "Rectangle proposal", "OpenCV"), authors=("OpenAI ChatGPT",), version=HTH_VERSION, repository=HTH_REPOSITORY),
    DetectorSpec(
        method=detector_radon_boundary.METHOD,
        name="Radon Boundary Projection",
        origin="HTH",
        entrypoint=detector_radon_boundary.detect,
        foundation=("Projection-angle integration", "Sobel gradients", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_text_flow.METHOD,
        name="Text Flow Envelope",
        origin="HTH",
        entrypoint=detector_text_flow.detect,
        foundation=("Connected components", "Text-line geometry", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_page_background.METHOD,
        name="Page Background",
        origin="HTH",
        entrypoint=detector_page_background.detect,
        foundation=("Robust border background model", "CIE Lab color distance", "Negative-space segmentation", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_whitespace_frame.METHOD,
        name="Whitespace Frame",
        origin="HTH",
        entrypoint=detector_whitespace_frame.detect,
        foundation=("Negative-space segmentation", "Morphology", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_joint_rectangle_vote.METHOD,
        name="Joint Rectangle Voting",
        origin="HTH",
        entrypoint=detector_joint_rectangle_vote.detect,
        foundation=("Hough lines", "Joint rectangle scoring", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_learned_page_mask.METHOD,
        name="Learned Page-Mask Detector",
        origin="PageNet / HTH",
        entrypoint=detector_learned_page_mask.detect,
        foundation=("PageNet", "Learned page segmentation", "OpenCV DNN", "Caffe"),
        authors=("Chris Tensmeyer et al.", "OpenAI ChatGPT"),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_components.METHOD,
        name="Connected Components",
        origin="OpenCV",
        entrypoint=detector_components.detect,
        foundation=("OpenCV",),
        authors=("OpenCV contributors",),
        version=cv2.__version__,
        repository="https://github.com/opencv/opencv",
    ),
    DetectorSpec(
        method=detector_ransac.METHOD,
        name="RANSAC",
        origin="HTH",
        entrypoint=detector_ransac.detect,
        foundation=("RANSAC", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_hough.METHOD,
        name="Hough Lines",
        origin="OpenCV",
        entrypoint=detector_hough.detect,
        foundation=("Hough transform", "OpenCV"),
        authors=("OpenCV contributors",),
        version=cv2.__version__,
        repository="https://github.com/opencv/opencv",
    ),
    DetectorSpec(
        method=detector_lsd.METHOD,
        name="Line Segment Detector",
        origin="OpenCV",
        entrypoint=detector_lsd.detect,
        foundation=("LSD", "OpenCV"),
        authors=("OpenCV contributors",),
        version=cv2.__version__,
        repository="https://github.com/opencv/opencv",
    ),
    DetectorSpec(
        method=detector_grabcut.METHOD,
        name="GrabCut",
        origin="OpenCV",
        entrypoint=detector_grabcut.detect,
        foundation=("GrabCut", "OpenCV"),
        authors=("OpenCV contributors",),
        version=cv2.__version__,
        repository="https://github.com/opencv/opencv",
    ),
)


def detector_specs() -> list[DetectorSpec]:
    return list(_REGISTRY)


def detector_names() -> list[str]:
    return [spec.method for spec in _REGISTRY]


def detector_catalog() -> list[dict[str, Any]]:
    return [
        {
            "method": spec.method,
            "name": spec.name,
            "display_name": spec.display_name,
            "origin": spec.origin,
            "foundation": list(spec.foundation),
            "authors": list(spec.authors),
            "version": spec.version,
            "repository": spec.repository,
        }
        for spec in _REGISTRY
    ]


def _apply_spec(candidate: Candidate, spec: DetectorSpec) -> Candidate:
    candidate.detector_name = spec.name
    candidate.origin = spec.origin
    candidate.foundation = list(spec.foundation)
    candidate.authors = list(spec.authors)
    candidate.version = spec.version
    candidate.repository = spec.repository
    return candidate


def _failed_candidate(
    spec: DetectorSpec,
    exc: BaseException,
    *,
    elapsed_ms: float,
) -> Candidate:
    """Represent a detector exception as data instead of aborting the page."""
    return _apply_spec(
        Candidate(
            method=spec.method,
            bbox=None,
            corners=None,
            confidence=0.0,
            score=0.0,
            diagnostics={
                "reason": "detector_exception",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(limit=8),
                "elapsed_ms": round(elapsed_ms, 3),
            },
            status="error",
        ),
        spec,
    )


def _normalize_candidate(
    spec: DetectorSpec,
    candidate: Candidate,
    *,
    elapsed_ms: float,
) -> Candidate:
    if not isinstance(candidate, Candidate):
        raise TypeError(
            f"Detector {spec.method!r} returned {type(candidate).__name__}, "
            "expected Candidate"
        )
    if candidate.method != spec.method:
        raise ValueError(
            f"Detector registry mismatch: registered as {spec.method!r}, "
            f"returned {candidate.method!r}"
        )

    candidate.diagnostics = dict(candidate.diagnostics or {})
    candidate.diagnostics.setdefault("elapsed_ms", round(elapsed_ms, 3))

    if candidate.status not in {"ok", "no_candidate", "error"}:
        raise ValueError(
            f"Detector {spec.method!r} returned invalid status {candidate.status!r}"
        )

    if candidate.status == "ok" and candidate.bbox is None:
        candidate.status = "no_candidate"

    if candidate.status != "ok":
        candidate.confidence = 0.0
        candidate.score = 0.0

    return _apply_spec(candidate, spec)



def run_registered_detector(
    method: str,
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> Candidate:
    """Run one detector through the authoritative registry contract."""
    spec = next((item for item in _REGISTRY if item.method == method), None)
    if spec is None:
        raise KeyError(f"Unknown detector: {method}")
    started = time.perf_counter()
    try:
        candidate = spec.entrypoint(
            image_bgr=image_bgr,
            mask=mask,
            parameters=parameters,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return _normalize_candidate(spec, candidate, elapsed_ms=elapsed_ms)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return _failed_candidate(spec, exc, elapsed_ms=elapsed_ms)

def run_registered_detectors(*, image_bgr: np.ndarray, mask: np.ndarray) -> list[Candidate]:
    """Run every registered detector independently with timing and isolation."""
    candidates: list[Candidate] = []
    for spec in _REGISTRY:
        started = time.perf_counter()
        try:
            candidate = spec.entrypoint(image_bgr=image_bgr, mask=mask)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            candidate = _normalize_candidate(spec, candidate, elapsed_ms=elapsed_ms)
        except Exception as exc:  # Detector plugins are an isolation boundary.
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            candidate = _failed_candidate(spec, exc, elapsed_ms=elapsed_ms)
        candidates.append(candidate)
    return candidates


def summarize_candidates(candidates: list[Candidate]) -> dict[str, object]:
    counts = Counter(candidate.status for candidate in candidates)
    return {
        "status_counts": {
            "ok": counts.get("ok", 0),
            "no_candidate": counts.get("no_candidate", 0),
            "error": counts.get("error", 0),
        },
        "failed_methods": [
            candidate.method for candidate in candidates if candidate.status == "error"
        ],
        "no_candidate_methods": [
            candidate.method
            for candidate in candidates
            if candidate.status == "no_candidate"
        ],
    }
