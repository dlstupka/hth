#!/usr/bin/env python3
"""Canonical Build Evidence for deterministic HTH processing workflows.

The contract deliberately separates effective inputs, canonical domain results,
and execution observations.  Collection preprocessing is the first consumer;
normalization and later deterministic stages can reuse the same primitives and
CLI by declaring their own scope and artifact profile.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable

from hth.markdown_links import code_link, github_blob_url


SCHEMA_VERSION = "1.0"
EVIDENCE_TYPE = "canonical-build-evidence"
STORE_TYPE = "canonical-build-evidence-store"
RESOURCE_PROVENANCE_VERSION = "1"
PREPROCESS_SCOPE = "hth-preprocess"
NORMALIZATION_SCOPE = "hth-normalization"
PHOTOMETRIC_INTEGRATION_SCOPE = "hth-photometric-integration"
TONAL_ASSESSMENT_SCOPE = "hth-tonal-assessment"
TONAL_METHOD_ASSESSMENT_SCOPE = "hth-tonal-method-assessment"
TONAL_VALIDATION_SCOPE = "hth-tonal-validation"
TONAL_INTEGRATION_SCOPE = "hth-tonal-integration"
CHROMATIC_ASSESSMENT_SCOPE = "hth-chromatic-assessment"
CHROMATIC_METHOD_ASSESSMENT_SCOPE = "hth-chromatic-method-assessment"
CHROMATIC_VALIDATION_SCOPE = "hth-chromatic-validation"
CHROMATIC_INTEGRATION_SCOPE = "hth-chromatic-integration"
DENOISING_ASSESSMENT_SCOPE = "hth-denoising-assessment"
DENOISING_METHOD_ASSESSMENT_SCOPE = "hth-denoising-method-assessment"
DENOISING_VALIDATION_SCOPE = "hth-denoising-validation"
DENOISING_INTEGRATION_SCOPE = "hth-denoising-integration"
SHARPENING_ASSESSMENT_SCOPE = "hth-sharpening-assessment"
SHARPENING_METHOD_ASSESSMENT_SCOPE = "hth-sharpening-method-assessment"
SHARPENING_VALIDATION_SCOPE = "hth-sharpening-validation"
SHARPENING_INTEGRATION_SCOPE = "hth-sharpening-integration"
BINARIZATION_ASSESSMENT_SCOPE = "hth-binarization-assessment"
BINARIZATION_METHOD_ASSESSMENT_SCOPE = "hth-binarization-method-assessment"
BINARIZATION_VALIDATION_SCOPE = "hth-binarization-validation"
BINARIZATION_INTEGRATION_SCOPE = "hth-binarization-integration"
CROP_FRAMING_ASSESSMENT_SCOPE = "hth-crop-framing-assessment"
ORIENTATION_DESKEW_ASSESSMENT_SCOPE = "hth-orientation-deskew-assessment"
PERSPECTIVE_ASSESSMENT_SCOPE = "hth-perspective-assessment"
PHOTOMETRIC_ASSESSMENT_SCOPE = "hth-photometric-assessment"
PHOTOMETRIC_METHOD_ASSESSMENT_SCOPE = "hth-photometric-method-assessment"
PHOTOMETRIC_VALIDATION_SCOPE = "hth-photometric-validation"
COMPACT_EVIDENCE_SCOPES = frozenset({
    CROP_FRAMING_ASSESSMENT_SCOPE,
    ORIENTATION_DESKEW_ASSESSMENT_SCOPE,
    PERSPECTIVE_ASSESSMENT_SCOPE,
    PHOTOMETRIC_ASSESSMENT_SCOPE,
    PHOTOMETRIC_METHOD_ASSESSMENT_SCOPE,
    PHOTOMETRIC_VALIDATION_SCOPE,
    TONAL_ASSESSMENT_SCOPE,
    TONAL_METHOD_ASSESSMENT_SCOPE,
    TONAL_VALIDATION_SCOPE,
    CHROMATIC_ASSESSMENT_SCOPE,
    CHROMATIC_METHOD_ASSESSMENT_SCOPE,
    CHROMATIC_VALIDATION_SCOPE,
    DENOISING_ASSESSMENT_SCOPE,
    DENOISING_METHOD_ASSESSMENT_SCOPE,
    DENOISING_VALIDATION_SCOPE,
    SHARPENING_ASSESSMENT_SCOPE,
    SHARPENING_METHOD_ASSESSMENT_SCOPE,
    SHARPENING_VALIDATION_SCOPE,
    BINARIZATION_ASSESSMENT_SCOPE,
    BINARIZATION_METHOD_ASSESSMENT_SCOPE,
    BINARIZATION_VALIDATION_SCOPE,
})
POLICIES = ("auto", "audit", "force-verify", "rebuild")

# Operation contracts are semantic build inputs, not workflow implementation
# details.  Keeping them beside the scope registry prevents YAML consolidation,
# step renaming, or caller refactors from silently changing build identity.
SCOPE_OPERATION_CONTRACTS: dict[str, tuple[str, ...]] = {
    PREPROCESS_SCOPE: (
        "source-image-extract",
        "word-crop",
        "analysis-derivative",
        "thumbnail",
        "page-quality-analysis",
        "physical-document-detection",
    ),
    CROP_FRAMING_ASSESSMENT_SCOPE: ("approved-detector-crop-framing-assessment",),
    ORIENTATION_DESKEW_ASSESSMENT_SCOPE: ("orientation-and-conservative-deskew-assessment",),
    PERSPECTIVE_ASSESSMENT_SCOPE: ("residual-perspective-assessment",),
    PHOTOMETRIC_ASSESSMENT_SCOPE: ("photometric-assessment",),
    PHOTOMETRIC_METHOD_ASSESSMENT_SCOPE: ("bounded-photometric-method-comparison",),
    PHOTOMETRIC_VALIDATION_SCOPE: ("complete-held-out-photometric-validation",),
    PHOTOMETRIC_INTEGRATION_SCOPE: (
        "reconstruct-proven-canonical-normalized-pages",
        "apply-validated-background-field-correction",
        "preserve-ineligible-pages-and-continue",
        "package-immutable-lossless-collection",
    ),
    TONAL_ASSESSMENT_SCOPE: ("tonal-assess-evidence",),
    TONAL_METHOD_ASSESSMENT_SCOPE: ("tonal-compare-evidence",),
    TONAL_VALIDATION_SCOPE: ("tonal-validate-evidence",),
    TONAL_INTEGRATION_SCOPE: (
        "consume-immutable-photometric-collection",
        "classify-every-page-by-tonal-metrics",
        "apply-only-validated-bounded-tonal-method",
        "preserve-exceptions-and-continue",
        "package-immutable-lossless-collection",
    ),
    CHROMATIC_ASSESSMENT_SCOPE: ("chromatic-assess-evidence",),
    CHROMATIC_METHOD_ASSESSMENT_SCOPE: ("chromatic-compare-evidence",),
    CHROMATIC_VALIDATION_SCOPE: ("chromatic-validate-evidence",),
    CHROMATIC_INTEGRATION_SCOPE: (
        "consume-immutable-tonal-collection",
        "classify-every-page-by-chromatic-metrics",
        "apply-only-validated-bounded-chromatic-method",
        "preserve-exceptions-and-continue",
        "package-immutable-lossless-collection",
    ),
}
for _domain, _scopes in {
    "denoising": (
        DENOISING_ASSESSMENT_SCOPE,
        DENOISING_METHOD_ASSESSMENT_SCOPE,
        DENOISING_VALIDATION_SCOPE,
        DENOISING_INTEGRATION_SCOPE,
    ),
    "sharpening": (
        SHARPENING_ASSESSMENT_SCOPE,
        SHARPENING_METHOD_ASSESSMENT_SCOPE,
        SHARPENING_VALIDATION_SCOPE,
        SHARPENING_INTEGRATION_SCOPE,
    ),
    "binarization": (
        BINARIZATION_ASSESSMENT_SCOPE,
        BINARIZATION_METHOD_ASSESSMENT_SCOPE,
        BINARIZATION_VALIDATION_SCOPE,
        BINARIZATION_INTEGRATION_SCOPE,
    ),
}.items():
    SCOPE_OPERATION_CONTRACTS[_scopes[0]] = (f"{_domain}-assess-evidence",)
    SCOPE_OPERATION_CONTRACTS[_scopes[1]] = (f"{_domain}-compare-evidence",)
    SCOPE_OPERATION_CONTRACTS[_scopes[2]] = (f"{_domain}-validate-evidence",)
    SCOPE_OPERATION_CONTRACTS[_scopes[3]] = (
        f"integrate-validated-{_domain}",
        "preserve-exceptions-and-continue",
        "package-immutable-lossless-collection",
    )

NORMALIZATION_OPERATION_CONTRACTS = frozenset({
    (
        "canonical-source-reconstruction",
        "axis-aligned-document-crop",
        "lossless-png-encoding",
        "pixel-roundtrip-verification",
    ),
    (
        "canonical-source-reconstruction",
        "axis-aligned-document-crop",
        "lossless-png-encoding",
        "pixel-roundtrip-verification",
        "preserve-gross-page-orientation",
        "gated-conservative-hough-deskew",
    ),
})

# These fields are observations or duplicated provenance, not domain results.
# Their authoritative values remain in ``effective_inputs`` and ``execution``.
NON_CANONICAL_RESULT_KEYS = frozenset({
    "elapsed_ms",
    "elapsed_ms_average",
    "elapsed_ms_total",
    "elapsed_seconds",
    "finished_at_utc",
    "generated_at_utc",
    "pipeline_commit",
    "pipeline_repository",
    "prepared_at_utc",
    "processed_at_utc",
    "source_commit",
    "source_repository",
    "started_at_utc",
})


@dataclass(frozen=True)
class ArtifactSpec:
    logical_name: str
    generated_path: str
    published_path: str
    format: str = "json"


PREPROCESS_ARTIFACTS = (
    ArtifactSpec("image-manifest", "metadata/image_manifest.json", "metadata/image_manifest.json"),
    ArtifactSpec("exact-duplicates", "metadata/exact_duplicates.json", "metadata/exact_duplicates.json"),
    ArtifactSpec("preprocess-summary", "summary.json", "reports/preprocess-summary.json"),
    ArtifactSpec("page-analysis", "page-analysis/page-analysis.json", "analysis/page-analysis.json"),
    ArtifactSpec("analysis-summary", "page-analysis/analysis-summary.json", "analysis/analysis-summary.json"),
)

NORMALIZATION_ARTIFACTS = (
    ArtifactSpec(
        "normalization-manifest",
        "normalization-manifest.json",
        "normalization/normalization-manifest.json",
    ),
)

PHOTOMETRIC_INTEGRATION_ARTIFACTS = (
    ArtifactSpec(
        "photometric-normalization-manifest",
        "photometric-normalization-manifest.json",
        "normalization/photometric-integration/photometric-normalization-manifest.json",
    ),
    ArtifactSpec(
        "integration-plan",
        "integration-plan.json",
        "normalization/photometric-integration/integration-plan.json",
    ),
    ArtifactSpec(
        "materialization-evidence",
        "materialization-evidence.json",
        "normalization/photometric-integration/materialization-evidence.json",
    ),
    ArtifactSpec(
        "applied-integration-policy",
        "applied-integration-policy.json",
        "normalization/photometric-integration/applied-integration-policy.json",
    ),
    ArtifactSpec(
        "release-record",
        "release.json",
        "normalization/photometric-integration/release.json",
    ),
)

TONAL_INTEGRATION_ARTIFACTS = (
    ArtifactSpec(
        "tonal-normalization-manifest",
        "tonal-normalization-manifest.json",
        "normalization/tonal-integration/tonal-normalization-manifest.json",
    ),
    ArtifactSpec(
        "tonal-assessment",
        "tonal-assessment.json",
        "normalization/tonal-integration/tonal-assessment.json",
    ),
    ArtifactSpec(
        "tonal-method-assessment",
        "tonal-method-assessment.json",
        "normalization/tonal-integration/tonal-method-assessment.json",
    ),
    ArtifactSpec(
        "tonal-validation",
        "tonal-validation.json",
        "normalization/tonal-integration/tonal-validation.json",
    ),
    ArtifactSpec(
        "release-record",
        "release.json",
        "normalization/tonal-integration/release.json",
    ),
)

TONAL_ASSESSMENT_ARTIFACTS = (
    ArtifactSpec("tonal-assessment", "assessment.json", "normalization/tonal/assessment.json"),
)

TONAL_METHOD_ASSESSMENT_ARTIFACTS = (
    ArtifactSpec(
        "tonal-method-assessment",
        "assessment.json",
        "normalization/tonal-methods/assessment.json",
    ),
)

TONAL_VALIDATION_ARTIFACTS = (
    ArtifactSpec("tonal-validation", "validation.json", "normalization/tonal-validation/validation.json"),
)

CHROMATIC_INTEGRATION_ARTIFACTS = (
    ArtifactSpec(
        "chromatic-normalization-manifest",
        "chromatic-normalization-manifest.json",
        "normalization/chromatic-integration/chromatic-normalization-manifest.json",
    ),
    ArtifactSpec(
        "chromatic-assessment",
        "chromatic-assessment.json",
        "normalization/chromatic-integration/chromatic-assessment.json",
    ),
    ArtifactSpec(
        "chromatic-method-assessment",
        "chromatic-method-assessment.json",
        "normalization/chromatic-integration/chromatic-method-assessment.json",
    ),
    ArtifactSpec(
        "chromatic-validation",
        "chromatic-validation.json",
        "normalization/chromatic-integration/chromatic-validation.json",
    ),
    ArtifactSpec("release-record", "release.json", "normalization/chromatic-integration/release.json"),
)

CHROMATIC_ASSESSMENT_ARTIFACTS = (
    ArtifactSpec("chromatic-assessment", "assessment.json", "normalization/chromatic/assessment.json"),
)

CHROMATIC_METHOD_ASSESSMENT_ARTIFACTS = (
    ArtifactSpec(
        "chromatic-method-assessment",
        "assessment.json",
        "normalization/chromatic-methods/assessment.json",
    ),
)

CHROMATIC_VALIDATION_ARTIFACTS = (
    ArtifactSpec(
        "chromatic-validation",
        "validation.json",
        "normalization/chromatic-validation/validation.json",
    ),
)


def _restoration_artifacts(domain: str) -> tuple[ArtifactSpec, ...]:
    base = f"normalization/{domain}-integration"
    return (
        ArtifactSpec(f"{domain}-normalization-manifest", f"{domain}-normalization-manifest.json", f"{base}/{domain}-normalization-manifest.json"),
        ArtifactSpec(f"{domain}-assessment", f"{domain}-assessment.json", f"{base}/{domain}-assessment.json"),
        ArtifactSpec(f"{domain}-method-assessment", f"{domain}-method-assessment.json", f"{base}/{domain}-method-assessment.json"),
        ArtifactSpec(f"{domain}-validation", f"{domain}-validation.json", f"{base}/{domain}-validation.json"),
        ArtifactSpec("release-record", "release.json", f"{base}/release.json"),
    )


DENOISING_INTEGRATION_ARTIFACTS = _restoration_artifacts("denoising")
SHARPENING_INTEGRATION_ARTIFACTS = _restoration_artifacts("sharpening")
BINARIZATION_INTEGRATION_ARTIFACTS = _restoration_artifacts("binarization")
DENOISING_ASSESSMENT_ARTIFACTS = (ArtifactSpec("denoising-assessment", "assessment.json", "normalization/denoising/assessment.json"),)
DENOISING_METHOD_ASSESSMENT_ARTIFACTS = (ArtifactSpec("denoising-method-assessment", "assessment.json", "normalization/denoising-methods/assessment.json"),)
DENOISING_VALIDATION_ARTIFACTS = (ArtifactSpec("denoising-validation", "validation.json", "normalization/denoising-validation/validation.json"),)
SHARPENING_ASSESSMENT_ARTIFACTS = (ArtifactSpec("sharpening-assessment", "assessment.json", "normalization/sharpening/assessment.json"),)
SHARPENING_METHOD_ASSESSMENT_ARTIFACTS = (ArtifactSpec("sharpening-method-assessment", "assessment.json", "normalization/sharpening-methods/assessment.json"),)
SHARPENING_VALIDATION_ARTIFACTS = (ArtifactSpec("sharpening-validation", "validation.json", "normalization/sharpening-validation/validation.json"),)
BINARIZATION_ASSESSMENT_ARTIFACTS = (ArtifactSpec("binarization-assessment", "assessment.json", "normalization/binarization/assessment.json"),)
BINARIZATION_METHOD_ASSESSMENT_ARTIFACTS = (ArtifactSpec("binarization-method-assessment", "assessment.json", "normalization/binarization-methods/assessment.json"),)
BINARIZATION_VALIDATION_ARTIFACTS = (ArtifactSpec("binarization-validation", "validation.json", "normalization/binarization-validation/validation.json"),)
CROP_FRAMING_ASSESSMENT_ARTIFACTS = (
    ArtifactSpec("crop-framing-assessment", "assessment.json", "normalization/crop-framing/assessment.json"),
    ArtifactSpec("crop-framing-detector-selection", "detector-selection.json", "normalization/crop-framing/detector-selection.json"),
    ArtifactSpec("crop-framing-geometry-evidence", "geometry-evidence.json", "normalization/crop-framing/geometry-evidence.json"),
)
ORIENTATION_DESKEW_ASSESSMENT_ARTIFACTS = (
    ArtifactSpec("orientation-deskew-assessment", "assessment.json", "normalization/orientation-deskew/assessment.json"),
    ArtifactSpec("orientation-deskew-sample-plan", "sample-plan.json", "normalization/orientation-deskew/sample-plan.json"),
    ArtifactSpec("orientation-deskew-materialization-evidence", "materialization-evidence.json", "normalization/orientation-deskew/materialization-evidence.json"),
    ArtifactSpec("orientation-deskew-recommendation", "recommendation.md", "normalization/orientation-deskew/recommendation.md", "binary"),
    ArtifactSpec("orientation-deskew-policy", "normalization-policy.json", "normalization/orientation-deskew/normalization-policy.json"),
)
PERSPECTIVE_ASSESSMENT_ARTIFACTS = (
    ArtifactSpec("perspective-assessment", "assessment.json", "normalization/perspective/assessment.json"),
    ArtifactSpec("perspective-sample-plan", "sample-plan.json", "normalization/perspective/sample-plan.json"),
    ArtifactSpec("perspective-materialization-evidence", "materialization-evidence.json", "normalization/perspective/materialization-evidence.json"),
    ArtifactSpec("perspective-recommendation", "recommendation.md", "normalization/perspective/recommendation.md", "binary"),
    ArtifactSpec("perspective-policy", "perspective-policy.json", "normalization/perspective/perspective-policy.json"),
)
PHOTOMETRIC_ASSESSMENT_ARTIFACTS = (
    ArtifactSpec("photometric-assessment", "assessment.json", "normalization/photometric/assessment.json"),
    ArtifactSpec("photometric-sample-plan", "sample-plan.json", "normalization/photometric/sample-plan.json"),
    ArtifactSpec("photometric-materialization-evidence", "materialization-evidence.json", "normalization/photometric/materialization-evidence.json"),
    ArtifactSpec("photometric-recommendation", "recommendation.md", "normalization/photometric/recommendation.md", "binary"),
    ArtifactSpec("photometric-policy", "photometric-policy.json", "normalization/photometric/photometric-policy.json"),
)
PHOTOMETRIC_METHOD_ASSESSMENT_ARTIFACTS = (
    ArtifactSpec("photometric-method-assessment", "assessment.json", "normalization/photometric-methods/assessment.json"),
    ArtifactSpec("photometric-method-assessment-csv", "assessment.csv", "normalization/photometric-methods/assessment.csv", "binary"),
    ArtifactSpec("photometric-method-sample-plan", "sample-plan.json", "normalization/photometric-methods/sample-plan.json"),
    ArtifactSpec("photometric-method-materialization-evidence", "materialization-evidence.json", "normalization/photometric-methods/materialization-evidence.json"),
    ArtifactSpec("photometric-method-recommendation", "recommendation.md", "normalization/photometric-methods/recommendation.md", "binary"),
    ArtifactSpec("photometric-method-policy", "photometric-method-policy.json", "normalization/photometric-methods/photometric-method-policy.json"),
)
PHOTOMETRIC_VALIDATION_ARTIFACTS = (
    ArtifactSpec("photometric-validation", "validation.json", "normalization/photometric-validation/validation.json"),
    ArtifactSpec("photometric-validation-csv", "validation.csv", "normalization/photometric-validation/validation.csv", "binary"),
    ArtifactSpec("photometric-validation-sample-plan", "sample-plan.json", "normalization/photometric-validation/sample-plan.json"),
    ArtifactSpec("photometric-validation-materialization-evidence", "materialization-evidence.json", "normalization/photometric-validation/materialization-evidence.json"),
    ArtifactSpec("photometric-integration-policy", "photometric-integration-policy.json", "normalization/photometric-validation/photometric-integration-policy.json"),
)

SCOPE_ARTIFACT_PROFILES = {
    PREPROCESS_SCOPE: PREPROCESS_ARTIFACTS,
    NORMALIZATION_SCOPE: NORMALIZATION_ARTIFACTS,
    PHOTOMETRIC_INTEGRATION_SCOPE: PHOTOMETRIC_INTEGRATION_ARTIFACTS,
    TONAL_ASSESSMENT_SCOPE: TONAL_ASSESSMENT_ARTIFACTS,
    TONAL_METHOD_ASSESSMENT_SCOPE: TONAL_METHOD_ASSESSMENT_ARTIFACTS,
    TONAL_VALIDATION_SCOPE: TONAL_VALIDATION_ARTIFACTS,
    TONAL_INTEGRATION_SCOPE: TONAL_INTEGRATION_ARTIFACTS,
    CHROMATIC_ASSESSMENT_SCOPE: CHROMATIC_ASSESSMENT_ARTIFACTS,
    CHROMATIC_METHOD_ASSESSMENT_SCOPE: CHROMATIC_METHOD_ASSESSMENT_ARTIFACTS,
    CHROMATIC_VALIDATION_SCOPE: CHROMATIC_VALIDATION_ARTIFACTS,
    CHROMATIC_INTEGRATION_SCOPE: CHROMATIC_INTEGRATION_ARTIFACTS,
    DENOISING_ASSESSMENT_SCOPE: DENOISING_ASSESSMENT_ARTIFACTS,
    DENOISING_METHOD_ASSESSMENT_SCOPE: DENOISING_METHOD_ASSESSMENT_ARTIFACTS,
    DENOISING_VALIDATION_SCOPE: DENOISING_VALIDATION_ARTIFACTS,
    DENOISING_INTEGRATION_SCOPE: DENOISING_INTEGRATION_ARTIFACTS,
    SHARPENING_ASSESSMENT_SCOPE: SHARPENING_ASSESSMENT_ARTIFACTS,
    SHARPENING_METHOD_ASSESSMENT_SCOPE: SHARPENING_METHOD_ASSESSMENT_ARTIFACTS,
    SHARPENING_VALIDATION_SCOPE: SHARPENING_VALIDATION_ARTIFACTS,
    SHARPENING_INTEGRATION_SCOPE: SHARPENING_INTEGRATION_ARTIFACTS,
    BINARIZATION_ASSESSMENT_SCOPE: BINARIZATION_ASSESSMENT_ARTIFACTS,
    BINARIZATION_METHOD_ASSESSMENT_SCOPE: BINARIZATION_METHOD_ASSESSMENT_ARTIFACTS,
    BINARIZATION_VALIDATION_SCOPE: BINARIZATION_VALIDATION_ARTIFACTS,
    BINARIZATION_INTEGRATION_SCOPE: BINARIZATION_INTEGRATION_ARTIFACTS,
    CROP_FRAMING_ASSESSMENT_SCOPE: CROP_FRAMING_ASSESSMENT_ARTIFACTS,
    ORIENTATION_DESKEW_ASSESSMENT_SCOPE: ORIENTATION_DESKEW_ASSESSMENT_ARTIFACTS,
    PERSPECTIVE_ASSESSMENT_SCOPE: PERSPECTIVE_ASSESSMENT_ARTIFACTS,
    PHOTOMETRIC_ASSESSMENT_SCOPE: PHOTOMETRIC_ASSESSMENT_ARTIFACTS,
    PHOTOMETRIC_METHOD_ASSESSMENT_SCOPE: PHOTOMETRIC_METHOD_ASSESSMENT_ARTIFACTS,
    PHOTOMETRIC_VALIDATION_SCOPE: PHOTOMETRIC_VALIDATION_ARTIFACTS,
}

SCOPE_EVIDENCE_PATHS = {
    PREPROCESS_SCOPE: "metadata/canonical-build-evidence.json",
    NORMALIZATION_SCOPE: "normalization/canonical-build-evidence.json",
    PHOTOMETRIC_INTEGRATION_SCOPE: "normalization/photometric-integration/canonical-build-evidence.json",
    TONAL_ASSESSMENT_SCOPE: "normalization/tonal/canonical-build-evidence.json",
    TONAL_METHOD_ASSESSMENT_SCOPE: "normalization/tonal-methods/canonical-build-evidence.json",
    TONAL_VALIDATION_SCOPE: "normalization/tonal-validation/canonical-build-evidence.json",
    TONAL_INTEGRATION_SCOPE: "normalization/tonal-integration/canonical-build-evidence.json",
    CHROMATIC_ASSESSMENT_SCOPE: "normalization/chromatic/canonical-build-evidence.json",
    CHROMATIC_METHOD_ASSESSMENT_SCOPE: "normalization/chromatic-methods/canonical-build-evidence.json",
    CHROMATIC_VALIDATION_SCOPE: "normalization/chromatic-validation/canonical-build-evidence.json",
    CHROMATIC_INTEGRATION_SCOPE: "normalization/chromatic-integration/canonical-build-evidence.json",
    DENOISING_ASSESSMENT_SCOPE: "normalization/denoising/canonical-build-evidence.json",
    DENOISING_METHOD_ASSESSMENT_SCOPE: "normalization/denoising-methods/canonical-build-evidence.json",
    DENOISING_VALIDATION_SCOPE: "normalization/denoising-validation/canonical-build-evidence.json",
    DENOISING_INTEGRATION_SCOPE: "normalization/denoising-integration/canonical-build-evidence.json",
    SHARPENING_ASSESSMENT_SCOPE: "normalization/sharpening/canonical-build-evidence.json",
    SHARPENING_METHOD_ASSESSMENT_SCOPE: "normalization/sharpening-methods/canonical-build-evidence.json",
    SHARPENING_VALIDATION_SCOPE: "normalization/sharpening-validation/canonical-build-evidence.json",
    SHARPENING_INTEGRATION_SCOPE: "normalization/sharpening-integration/canonical-build-evidence.json",
    BINARIZATION_ASSESSMENT_SCOPE: "normalization/binarization/canonical-build-evidence.json",
    BINARIZATION_METHOD_ASSESSMENT_SCOPE: "normalization/binarization-methods/canonical-build-evidence.json",
    BINARIZATION_VALIDATION_SCOPE: "normalization/binarization-validation/canonical-build-evidence.json",
    BINARIZATION_INTEGRATION_SCOPE: "normalization/binarization-integration/canonical-build-evidence.json",
    CROP_FRAMING_ASSESSMENT_SCOPE: "normalization/crop-framing/canonical-build-evidence.json",
    ORIENTATION_DESKEW_ASSESSMENT_SCOPE: "normalization/orientation-deskew/canonical-build-evidence.json",
    PERSPECTIVE_ASSESSMENT_SCOPE: "normalization/perspective/canonical-build-evidence.json",
    PHOTOMETRIC_ASSESSMENT_SCOPE: "normalization/photometric/canonical-build-evidence.json",
    PHOTOMETRIC_METHOD_ASSESSMENT_SCOPE: "normalization/photometric-methods/canonical-build-evidence.json",
    PHOTOMETRIC_VALIDATION_SCOPE: "normalization/photometric-validation/canonical-build-evidence.json",
}


def artifact_profile(scope: str) -> tuple[ArtifactSpec, ...]:
    try:
        return SCOPE_ARTIFACT_PROFILES[scope]
    except KeyError as exc:
        raise EvidenceError(
            f"Canonical Build Evidence scope has no artifact profile: {scope!r}"
        ) from exc


def evidence_relative_path(scope: str) -> str:
    try:
        return SCOPE_EVIDENCE_PATHS[scope]
    except KeyError as exc:
        raise EvidenceError(
            f"Canonical Build Evidence scope has no evidence path: {scope!r}"
        ) from exc


class EvidenceError(RuntimeError):
    """The persisted evidence contract is missing, corrupt, or contradictory."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def elapsed_seconds(started_at_utc: str, completed_at_utc: str) -> int:
    """Return whole elapsed seconds for two canonical UTC observations."""
    started = datetime.fromisoformat(started_at_utc.replace("Z", "+00:00"))
    completed = datetime.fromisoformat(completed_at_utc.replace("Z", "+00:00"))
    return max(0, int((completed - started).total_seconds()))


def format_elapsed(seconds: int) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def execution_elapsed_seconds(execution: dict[str, Any]) -> int | None:
    persisted = execution.get("elapsed_seconds")
    if isinstance(persisted, (int, float)) and persisted >= 0:
        return int(persisted)
    started = execution.get("evaluated_at_utc")
    completed = execution.get("completed_at_utc")
    if isinstance(started, str) and isinstance(completed, str):
        return elapsed_seconds(started, completed)
    return None


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_result(value: Any) -> Any:
    """Remove execution telemetry while preserving every domain-result field."""
    if isinstance(value, dict):
        return {
            str(key): canonicalize_result(item)
            for key, item in value.items()
            if str(key) not in NON_CANONICAL_RESULT_KEYS
        }
    if isinstance(value, list):
        return [canonicalize_result(item) for item in value]
    return value


def canonical_file_sha256(path: Path, format: str = "json") -> str:
    path = Path(path)
    if not path.is_file():
        raise EvidenceError(f"Required canonical result is missing: {path}")
    if format == "json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return canonical_hash(canonicalize_result(value))
    if format == "binary":
        return file_sha256(path)
    raise EvidenceError(f"Unsupported canonical artifact format: {format}")


def _relative_label(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.name


def fingerprint_paths(paths: Iterable[Path], repository_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for requested in paths:
        path = Path(requested)
        if not path.exists():
            raise EvidenceError(f"Identity input does not exist: {path}")
        candidates = [path] if path.is_file() else sorted(
            (
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and "__pycache__" not in candidate.parts
                and candidate.suffix not in {".pyc", ".pyo"}
            ),
            key=lambda candidate: candidate.as_posix(),
        )
        for candidate in candidates:
            records.append({
                "path": _relative_label(candidate, repository_root),
                "bytes": candidate.stat().st_size,
                "sha256": file_sha256(candidate),
            })
    records.sort(key=lambda record: record["path"])
    labels = [record["path"] for record in records]
    if len(labels) != len(set(labels)):
        raise EvidenceError("Identity inputs contain duplicate stable paths")
    return records


def _logical_source_label(value: str) -> str:
    label = value.strip().replace("\\", "/")
    logical = PurePosixPath(label)
    if not label or logical.is_absolute() or ".." in logical.parts:
        raise EvidenceError(f"Canonical source input has an invalid logical path: {value!r}")
    normalized = logical.as_posix()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def fingerprint_source_inputs(inputs: Iterable[str]) -> list[dict[str, Any]]:
    """Fingerprint physical inputs under explicit, stable semantic paths.

    ``LOGICAL=PHYSICAL`` mappings decouple canonical identity from temporary
    workspace layout.  A logical path of ``.`` maps a directory's contents to
    the identity root and provides a migration path for existing contracts.
    """
    records: list[dict[str, Any]] = []
    for specification in inputs:
        logical_value, separator, physical_value = specification.partition("=")
        if not separator or not physical_value.strip():
            raise EvidenceError(
                "Canonical source inputs must use LOGICAL_PATH=PHYSICAL_PATH syntax"
            )
        logical = _logical_source_label(logical_value)
        physical = Path(physical_value)
        if not physical.exists():
            raise EvidenceError(f"Canonical source input does not exist: {physical}")
        candidates = [physical] if physical.is_file() else sorted(
            (
                candidate
                for candidate in physical.rglob("*")
                if candidate.is_file()
                and "__pycache__" not in candidate.parts
                and candidate.suffix not in {".pyc", ".pyo"}
            ),
            key=lambda candidate: candidate.as_posix(),
        )
        for candidate in candidates:
            if physical.is_file():
                label = logical
            else:
                relative = candidate.relative_to(physical).as_posix()
                label = relative if logical == "." else f"{logical}/{relative}"
            records.append({
                "path": label,
                "bytes": candidate.stat().st_size,
                "sha256": file_sha256(candidate),
            })
    records.sort(key=lambda record: record["path"])
    labels = [record["path"] for record in records]
    if len(labels) != len(set(labels)):
        raise EvidenceError("Canonical source inputs contain duplicate logical paths")
    return records


def canonical_operations(scope: str, supplied: Iterable[str]) -> list[str]:
    operations = tuple(supplied)
    if scope == NORMALIZATION_SCOPE:
        if operations not in NORMALIZATION_OPERATION_CONTRACTS:
            raise EvidenceError(
                f"Canonical operation contract drift for {scope}: {operations!r}"
            )
        return list(operations)
    expected = SCOPE_OPERATION_CONTRACTS.get(scope)
    if expected is None:
        raise EvidenceError(f"Canonical Build Evidence scope has no operation contract: {scope!r}")
    if operations and operations != expected:
        raise EvidenceError(
            f"Canonical operation contract drift for {scope}: "
            f"expected {expected!r}, received {operations!r}"
        )
    return list(expected)


def fingerprint_selected_detector(
    selection: dict[str, Any] | None,
    detector_root: Path | None,
    repository_root: Path,
) -> list[dict[str, Any]]:
    """Fingerprint only the selected detector and its local detector dependencies."""
    if selection is None:
        return []
    if detector_root is None:
        raise EvidenceError("A detector implementation root is required with detector selection")
    detector_id = str(selection.get("detector") or "").strip()
    if not detector_id or not all(character.isalnum() or character == "_" for character in detector_id):
        raise EvidenceError(f"Detector selection has an invalid detector ID: {detector_id!r}")
    root = Path(detector_root).resolve()
    entrypoint = root / f"detector_{detector_id}.py"
    if not entrypoint.is_file():
        raise EvidenceError(f"Selected detector implementation does not exist: {entrypoint}")

    pending = [entrypoint]
    discovered: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in discovered:
            continue
        discovered.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            raise EvidenceError(f"Cannot inspect selected detector implementation {path}: {exc}") from exc
        relative_parent = path.relative_to(root).parent.parts
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level < 1:
                continue
            climb = node.level - 1
            if climb > len(relative_parent):
                continue
            prefix = list(relative_parent[: len(relative_parent) - climb])
            modules: list[list[str]] = []
            if node.module:
                modules.append(prefix + node.module.split("."))
            else:
                modules.extend(prefix + alias.name.split(".") for alias in node.names)
            for module_parts in modules:
                candidate = root.joinpath(*module_parts).with_suffix(".py")
                if candidate.is_file() and candidate.resolve() not in discovered:
                    pending.append(candidate)
    return fingerprint_paths(sorted(discovered), repository_root)


def runtime_identity(packages: Iterable[str]) -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in sorted(set(packages)):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise EvidenceError(f"Required runtime package is not installed: {package}") from exc
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "packages": versions,
    }


def runner_observation(args: argparse.Namespace) -> dict[str, str]:
    """Record where verification ran without making the host an effective input."""
    return {
        "runner_name": args.runner_name,
        "runner_environment": args.runner_environment,
        "runner_os": args.runner_os or platform.system(),
        "runner_arch": args.runner_arch or platform.machine(),
    }


def declared_components(values: Iterable[str]) -> dict[str, str]:
    components: dict[str, str] = {}
    for value in values:
        name, separator, version = value.partition("=")
        if not separator or not name.strip() or not version.strip():
            raise EvidenceError(f"Runtime component must use NAME=VERSION: {value!r}")
        if name in components:
            raise EvidenceError(f"Runtime component is declared more than once: {name}")
        components[name] = version
    return dict(sorted(components.items()))


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Cannot read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceError(f"{description} must contain a JSON object: {path}")
    return payload


def build_effective_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if not _is_sha256(args.source_manifest_sha256):
        raise EvidenceError("Source release-manifest SHA-256 is invalid")
    if not args.config or not args.implementation or not args.runtime_contract:
        raise EvidenceError("Configuration, implementation, and runtime contracts are all required")
    repository_root = args.repository_root.resolve()
    operations = canonical_operations(args.scope, args.operation)
    source_inputs = getattr(args, "source_input", None) or []
    source_files = (
        fingerprint_source_inputs(source_inputs)
        if source_inputs
        else fingerprint_paths([args.source_root], args.source_root)
    )
    implementation = fingerprint_paths(args.implementation, repository_root)
    configuration = fingerprint_paths(args.config, repository_root)
    runtime_contract = fingerprint_paths(args.runtime_contract, repository_root)
    selection: dict[str, Any] | None = None
    if args.selection:
        selection = canonicalize_result(_load_json_object(args.selection, "detector selection"))
    selected_detector_implementation = fingerprint_selected_detector(
        selection,
        getattr(args, "detector_implementation_root", None),
        repository_root,
    )
    return {
        "contract": {
            "name": args.scope,
            "version": args.contract_version,
            "resource_provenance_version": RESOURCE_PROVENANCE_VERSION,
            "mode": args.mode,
            "image_limit": args.image_limit,
            "operations": operations,
        },
        "source": {
            "repository": args.source_repository,
            "release": args.source_release,
            "release_manifest_sha256": args.source_manifest_sha256.lower(),
            "commit": args.source_commit,
            "files": source_files,
        },
        "configuration": configuration,
        "implementation": implementation,
        "selected_detector_implementation": selected_detector_implementation,
        "runtime_contract": runtime_contract,
        "runtime": {
            **runtime_identity(args.runtime_package),
            "declared_components": declared_components(args.runtime_component),
        },
        "detector_selection": selection,
    }


def _artifact_records(root: Path, specs: Iterable[ArtifactSpec], *, published: bool) -> list[dict[str, Any]]:
    records = []
    for spec in specs:
        relative = spec.published_path if published else spec.generated_path
        path = Path(root) / relative
        records.append({
            "logical_name": spec.logical_name,
            "published_path": spec.published_path,
            "format": spec.format,
            "canonical_sha256": canonical_file_sha256(path, spec.format),
        })
    return records


def _result_identity(artifacts: list[dict[str, Any]], pages: list[dict[str, Any]]) -> str:
    result = {
        "artifacts": artifacts,
        "pages": [
            {
                key: value
                for key, value in page.items()
                if key not in {"activity", "domain_result", "operation_identity"}
            }
            for page in pages
        ],
    }
    return canonical_hash(result)


def validate_evidence(payload: dict[str, Any], *, scope: str) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("evidence_type") != EVIDENCE_TYPE:
        raise EvidenceError("Persisted Canonical Build Evidence has an unsupported schema")
    if payload.get("scope") != scope:
        raise EvidenceError(
            f"Persisted Canonical Build Evidence scope {payload.get('scope')!r} does not match {scope!r}"
        )
    effective_inputs = payload.get("effective_inputs")
    if not isinstance(effective_inputs, dict):
        raise EvidenceError("Persisted Canonical Build Evidence has no effective_inputs object")
    if payload.get("effective_build_identity") != canonical_hash(effective_inputs):
        raise EvidenceError("Persisted effective build identity does not match its input contract")
    resource_utilization = payload.get("resource_utilization")
    if resource_utilization is not None:
        if (
            not isinstance(resource_utilization, dict)
            or resource_utilization.get("schema_version") != RESOURCE_PROVENANCE_VERSION
        ):
            raise EvidenceError("Persisted resource utilization has an unsupported schema")
        cache = resource_utilization.get("canonical_evidence_cache")
        releases = resource_utilization.get("immutable_releases")
        if not isinstance(cache, dict) or cache.get("lookup_identity") != payload.get("effective_build_identity"):
            raise EvidenceError("Persisted resource utilization has an invalid cache identity")
        if not isinstance(releases, list) or not releases:
            raise EvidenceError("Persisted resource utilization has no immutable release use")
    source = effective_inputs.get("source")
    if not isinstance(source, dict) or not _is_sha256(source.get("release_manifest_sha256")):
        raise EvidenceError("Persisted effective inputs have no valid source release-manifest SHA-256")
    source_files = source.get("files")
    if not isinstance(source_files, list) or not source_files:
        raise EvidenceError("Persisted effective inputs have no source-file identities")
    for record in source_files:
        if not isinstance(record, dict) or not _is_sha256(record.get("sha256")):
            raise EvidenceError("Persisted effective inputs contain an invalid source-file identity")
    result = payload.get("canonical_result")
    if not isinstance(result, dict):
        raise EvidenceError("Persisted Canonical Build Evidence has no canonical_result object")
    artifacts = result.get("artifacts")
    pages = result.get("pages")
    if not isinstance(artifacts, list) or not artifacts or not isinstance(pages, list):
        raise EvidenceError("Persisted canonical result is incomplete")
    if not pages and scope not in COMPACT_EVIDENCE_SCOPES:
        raise EvidenceError("Persisted canonical result contains no page evidence")
    if result.get("identity") != _result_identity(artifacts, pages):
        raise EvidenceError("Persisted canonical result identity does not match its evidence")
    expected_artifacts = [
        (spec.logical_name, spec.published_path, spec.format)
        for spec in artifact_profile(scope)
    ]
    actual_artifacts = [
        (record.get("logical_name"), record.get("published_path"), record.get("format"))
        for record in artifacts
        if isinstance(record, dict)
    ]
    if actual_artifacts != expected_artifacts:
        raise EvidenceError(f"Persisted {scope} canonical artifact contract is incomplete or unexpected")
    for record in artifacts:
        if not isinstance(record, dict) or not _is_sha256(record.get("canonical_sha256")):
            raise EvidenceError("Persisted canonical result contains an invalid artifact hash")
    ordinals: list[int] = []
    for page in pages:
        if not isinstance(page, dict):
            raise EvidenceError("Persisted canonical result contains an invalid page record")
        try:
            ordinal = int(page["global_ordinal"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceError("Persisted canonical result contains an invalid page ordinal") from exc
        required_page_hashes = (
            (
                "operation_identity",
                "evidence_record_sha256",
                "canonical_page_result_sha256",
            )
            if scope in COMPACT_EVIDENCE_SCOPES
            else
            (
                "operation_identity",
                "canonical_image_sha256",
                "analysis_derivative_sha256",
                "thumbnail_sha256",
                "canonical_page_result_sha256",
            )
            if scope == PREPROCESS_SCOPE
            else (
                (
                    "operation_identity",
                    "source_image_sha256",
                    "normalized_image_sha256",
                    "normalized_pixel_sha256",
                    "canonical_page_result_sha256",
                )
                if scope == NORMALIZATION_SCOPE
                else (
                    "operation_identity",
                    "input_pixel_sha256",
                    "output_pixel_sha256",
                    "output_image_sha256",
                    "canonical_page_result_sha256",
                )
            )
        )
        if not all(_is_sha256(page.get(field)) for field in required_page_hashes):
            raise EvidenceError(f"Persisted canonical result contains invalid hashes for page {ordinal}")
        ordinals.append(ordinal)
    if ordinals != sorted(set(ordinals)):
        raise EvidenceError("Persisted canonical page ordinals are duplicated or out of order")


def validate_published_results(payload: dict[str, Any], results_root: Path) -> None:
    validate_evidence(payload, scope=str(payload.get("scope") or ""))
    for record in payload["canonical_result"]["artifacts"]:
        relative = str(record.get("published_path") or "")
        format = str(record.get("format") or "")
        expected = str(record.get("canonical_sha256") or "")
        actual = canonical_file_sha256(Path(results_root) / relative, format)
        if actual != expected:
            raise EvidenceError(
                f"Persisted canonical result mismatch for {relative}: expected {expected}, found {actual}"
            )


def load_evidence_store(path: Path, *, scope: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "evidence_type": STORE_TYPE,
            "scope": scope,
            "records": {},
        }
    payload = _load_json_object(path, "Canonical Build Evidence store")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("evidence_type") != STORE_TYPE:
        raise EvidenceError("Persisted Canonical Build Evidence store has an unsupported schema")
    if payload.get("scope") != scope:
        raise EvidenceError("Persisted Canonical Build Evidence store has the wrong scope")
    records = payload.get("records")
    if not isinstance(records, dict):
        raise EvidenceError("Persisted Canonical Build Evidence store has no records object")
    for identity, record in records.items():
        if not isinstance(record, dict):
            raise EvidenceError(f"Persisted Canonical Build Evidence record {identity!r} is invalid")
        validate_evidence(record, scope=scope)
        if identity != record.get("effective_build_identity"):
            raise EvidenceError(f"Persisted Canonical Build Evidence record key {identity!r} is invalid")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_github_output(path: str, values: dict[str, Any]) -> None:
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")


def _append_summary(path: str, lines: list[str]) -> None:
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    effective_inputs = build_effective_inputs(args)
    identity = canonical_hash(effective_inputs)
    store = load_evidence_store(args.evidence, scope=args.scope)
    incumbent = store["records"].get(identity)
    exact = incumbent is not None

    if args.policy in {"audit", "force-verify"} and not exact:
        state = "missing" if incumbent is None else "different effective build identity"
        raise EvidenceError(f"{args.policy} requires exact persisted evidence; incumbent is {state}")
    if args.policy == "rebuild" and exact:
        raise EvidenceError("rebuild cannot replace evidence for an unchanged identity; use force-verify")

    comparison_required = exact and args.policy in {"auto", "force-verify"}
    if exact:
        assert incumbent is not None
        validate_published_results(incumbent, args.results_root)

    if args.policy == "audit":
        decision, activity, domain_result = "audit", "EVALUATED", "SKIP"
    elif args.policy == "auto" and exact and not args.artifact_required:
        decision, activity, domain_result = "reuse", "REUSED", "SKIP"
    else:
        decision, activity, domain_result = "execute", "EXECUTED", "APPLY"

    cache_lookup = "hit" if exact else "miss"
    cache_action = {
        "audit": "validated",
        "reuse": "reused",
        "execute": "verified-and-refreshed" if comparison_required else "populated",
    }[decision]
    resource_utilization = {
        "schema_version": RESOURCE_PROVENANCE_VERSION,
        "canonical_evidence_cache": {
            "scope": args.scope,
            "path": evidence_relative_path(args.scope),
            "lookup_identity": identity,
            "lookup": cache_lookup,
            "action": cache_action,
            "canonical_result_identity": (
                incumbent["canonical_result"]["identity"]
                if exact and incumbent is not None
                else None
            ),
        },
        "immutable_releases": [{
            "role": "source",
            "repository": args.source_repository,
            "release": args.source_release,
            "release_manifest_sha256": args.source_manifest_sha256.lower(),
            "commit": args.source_commit,
            "utilization": "consumed",
        }],
    }

    page_evaluations = []
    if decision in {"audit", "reuse"} and incumbent is not None:
        for page in incumbent["canonical_result"]["pages"]:
            page_evaluations.append({
                "global_ordinal": page["global_ordinal"],
                "operation_identity": page["operation_identity"],
                "activity": activity,
                "domain_result": domain_result,
            })

    plan = {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": EVIDENCE_TYPE,
        "scope": args.scope,
        "policy": args.policy,
        "decision": decision,
        "activity": activity,
        "domain_result": domain_result,
        "effective_build_identity": identity,
        "effective_inputs": effective_inputs,
        "incumbent_result_identity": (
            incumbent["canonical_result"]["identity"] if exact and incumbent is not None else None
        ),
        "comparison_required": comparison_required,
        "artifact_required": args.artifact_required,
        "resource_utilization": resource_utilization,
        "page_evaluations": page_evaluations,
        "execution": {
            "workflow_run_id": args.workflow_run_id,
            "pipeline_repository": args.pipeline_repository,
            "pipeline_commit": args.pipeline_commit,
            "runner": runner_observation(args),
            "evaluated_at_utc": utc_now(),
        },
    }
    _write_json(args.plan, plan)
    _write_github_output(args.github_output, {
        "decision": decision,
        "activity": activity,
        "domain_result": domain_result,
        "effective_build_identity": identity,
        "comparison_required": comparison_required,
        "page_count": len(page_evaluations),
    })
    evidence_relative = evidence_relative_path(args.scope)
    results_repository = getattr(args, "results_repository", "")
    # A first canonical build prepares its summary before the evidence store is
    # published. Link that pending store through the durable results branch;
    # established stores retain their immutable checked-out ref.
    evidence_ref = getattr(args, "results_ref", "main") if args.evidence.is_file() else "main"
    evidence_url = github_blob_url(results_repository, evidence_ref, evidence_relative)
    summary_lines = [
        "### Canonical Build Evidence",
        "",
        f"- Policy: `{args.policy}`",
        f"- Effective Build Identity: `{identity}`",
        f"- Activity: `{activity}`",
        f"- Domain result: `{domain_result}`",
        f"- Decision: `{decision}`",
        f"- Pages marked unnecessary: `{len(page_evaluations)}`",
        f"- Evidence cache: `{cache_lookup}` / `{cache_action}`",
        f"- Source release utilized: `{args.source_repository}@{args.source_release}`",
        f"- Evidence: {code_link(evidence_relative, evidence_url)}",
    ]
    if decision in {"audit", "reuse"} and incumbent is not None:
        persisted_elapsed = execution_elapsed_seconds(dict(incumbent.get("execution") or {}))
        if persisted_elapsed is not None:
            summary_lines.append(
                f"- Build stage time: `{format_elapsed(persisted_elapsed)}` "
                f"(`{persisted_elapsed}` seconds)"
            )
    _append_summary(args.github_summary, summary_lines)
    print(
        "[canonical-build-evidence] "
        f"policy={args.policy} "
        f"decision={decision} "
        f"activity={activity} "
        f"domain_result={domain_result} "
        f"pages_marked_unnecessary={len(page_evaluations)} "
        f"identity={identity}",
        flush=True,
    )
    return plan


def _page_results(
    output_root: Path,
    activity: str,
    domain_result: str,
    effective_build_identity: str,
) -> list[dict[str, Any]]:
    manifest = _load_json_object(output_root / "metadata/image_manifest.json", "image manifest")
    analysis = _load_json_object(output_root / "page-analysis/page-analysis.json", "page analysis")
    image_records = manifest.get("records")
    analysis_records = analysis.get("records")
    if not isinstance(image_records, list) or not isinstance(analysis_records, list):
        raise EvidenceError("Canonical page inputs/results do not contain records lists")
    analyses = {int(record["global_ordinal"]): record for record in analysis_records}
    pages = []
    for image in image_records:
        ordinal = int(image["global_ordinal"])
        result = analyses.get(ordinal)
        if result is None:
            raise EvidenceError(f"Page {ordinal} has no canonical analysis result")
        source_identity = {
            "source_docx": image.get("source_docx"),
            "source_ordinal": image.get("source_ordinal"),
            "relationship_id": image.get("relationship_id"),
            "media_path": image.get("media_path"),
            "embedded_sha256": image.get("embedded_sha256"),
            "word_crop": [
                image.get("word_crop_left"),
                image.get("word_crop_top"),
                image.get("word_crop_right"),
                image.get("word_crop_bottom"),
            ],
        }
        operation_identity = canonical_hash({
            "effective_build_identity": effective_build_identity,
            "source": source_identity,
        })
        canonical_result = canonicalize_result(result)
        page_domain_result = "ERROR" if canonical_result.get("analysis_error") else domain_result
        pages.append({
            "global_ordinal": ordinal,
            "operation_identity": operation_identity,
            "canonical_image_sha256": image.get("sha256"),
            "analysis_derivative_sha256": image.get("analysis_sha256"),
            "thumbnail_sha256": image.get("thumbnail_sha256"),
            "canonical_page_result_sha256": canonical_hash(canonical_result),
            "activity": activity,
            "domain_result": page_domain_result,
        })
    return pages


def _normalization_page_results(
    output_root: Path,
    activity: str,
    domain_result: str,
    effective_build_identity: str,
) -> list[dict[str, Any]]:
    manifest = _load_json_object(output_root / "normalization-manifest.json", "normalization manifest")
    records = manifest.get("pages")
    if not isinstance(records, list) or not records:
        raise EvidenceError("Normalization manifest does not contain page records")
    pages = []
    for record in records:
        if not isinstance(record, dict):
            raise EvidenceError("Normalization manifest contains an invalid page record")
        ordinal = int(record["global_ordinal"])
        canonical_page = {
            "global_ordinal": ordinal,
            "source_image_sha256": record.get("source_sha256"),
            "normalized_image_sha256": record.get("output_sha256"),
            "normalized_pixel_sha256": record.get("output_pixel_sha256"),
            "crop": [
                record.get("crop_left"),
                record.get("crop_top"),
                record.get("crop_right_exclusive"),
                record.get("crop_bottom_exclusive"),
            ],
            "source_dimensions": [record.get("source_width"), record.get("source_height")],
            "output_dimensions": [record.get("output_width"), record.get("output_height")],
        }
        pages.append({
            **canonical_page,
            "operation_identity": canonical_hash({
                "effective_build_identity": effective_build_identity,
                "global_ordinal": ordinal,
                "source_image_sha256": record.get("source_sha256"),
            }),
            "canonical_page_result_sha256": canonical_hash(canonical_page),
            "activity": activity,
            "domain_result": domain_result,
        })
    return pages


def _photometric_integration_page_results(
    output_root: Path,
    activity: str,
    domain_result: str,
    effective_build_identity: str,
) -> list[dict[str, Any]]:
    manifest = _load_json_object(
        output_root / "photometric-normalization-manifest.json",
        "photometric normalization manifest",
    )
    records = manifest.get("pages")
    if not isinstance(records, list) or not records:
        raise EvidenceError("Photometric normalization manifest does not contain page records")
    pages = []
    for record in records:
        if not isinstance(record, dict):
            raise EvidenceError("Photometric normalization manifest contains an invalid page record")
        ordinal = int(record["global_ordinal"])
        canonical_page = {
            "global_ordinal": ordinal,
            "route": record.get("route"),
            "pipeline_action": record.get("pipeline_action"),
            "evidence_source": record.get("evidence_source"),
            "input_pixel_sha256": record.get("input_pixel_sha256"),
            "output_pixel_sha256": record.get("output_pixel_sha256"),
            "output_image_sha256": record.get("output_sha256"),
            "output_dimensions": [record.get("output_width"), record.get("output_height")],
            "method_safe": record.get("method_safe"),
        }
        pages.append({
            **canonical_page,
            "operation_identity": canonical_hash({
                "effective_build_identity": effective_build_identity,
                "global_ordinal": ordinal,
                "input_pixel_sha256": record.get("input_pixel_sha256"),
            }),
            "canonical_page_result_sha256": canonical_hash(canonical_page),
            "activity": activity,
            "domain_result": domain_result,
        })
    return pages


def _tonal_integration_page_results(
    output_root: Path,
    activity: str,
    domain_result: str,
    effective_build_identity: str,
) -> list[dict[str, Any]]:
    manifest = _load_json_object(
        output_root / "tonal-normalization-manifest.json",
        "tonal normalization manifest",
    )
    records = manifest.get("pages")
    if not isinstance(records, list) or not records:
        raise EvidenceError("Tonal normalization manifest does not contain page records")
    pages = []
    for record in records:
        if not isinstance(record, dict):
            raise EvidenceError("Tonal normalization manifest contains an invalid page record")
        ordinal = int(record["global_ordinal"])
        canonical_page = {
            "global_ordinal": ordinal,
            "route": record.get("route"),
            "pipeline_action": record.get("pipeline_action"),
            "decision_before": record.get("decision_before"),
            "input_pixel_sha256": record.get("input_pixel_sha256"),
            "output_pixel_sha256": record.get("output_pixel_sha256"),
            "output_image_sha256": record.get("output_sha256"),
            "output_dimensions": [record.get("output_width"), record.get("output_height")],
        }
        pages.append({
            **canonical_page,
            "operation_identity": canonical_hash({
                "effective_build_identity": effective_build_identity,
                "global_ordinal": ordinal,
                "input_pixel_sha256": record.get("input_pixel_sha256"),
            }),
            "canonical_page_result_sha256": canonical_hash(canonical_page),
            "activity": activity,
            "domain_result": domain_result,
        })
    return pages


def _chromatic_integration_page_results(
    output_root: Path,
    activity: str,
    domain_result: str,
    effective_build_identity: str,
) -> list[dict[str, Any]]:
    manifest = _load_json_object(
        output_root / "chromatic-normalization-manifest.json",
        "chromatic normalization manifest",
    )
    records = manifest.get("pages")
    if not isinstance(records, list) or not records:
        raise EvidenceError("Chromatic normalization manifest does not contain page records")
    pages = []
    for record in records:
        if not isinstance(record, dict):
            raise EvidenceError("Chromatic normalization manifest contains an invalid page record")
        ordinal = int(record["global_ordinal"])
        canonical_page = {
            "global_ordinal": ordinal,
            "route": record.get("route"),
            "pipeline_action": record.get("pipeline_action"),
            "decision_before": record.get("decision_before"),
            "input_pixel_sha256": record.get("input_pixel_sha256"),
            "output_pixel_sha256": record.get("output_pixel_sha256"),
            "output_image_sha256": record.get("output_sha256"),
            "output_dimensions": [record.get("output_width"), record.get("output_height")],
        }
        pages.append({
            **canonical_page,
            "operation_identity": canonical_hash({
                "effective_build_identity": effective_build_identity,
                "global_ordinal": ordinal,
                "input_pixel_sha256": record.get("input_pixel_sha256"),
            }),
            "canonical_page_result_sha256": canonical_hash(canonical_page),
            "activity": activity,
            "domain_result": domain_result,
        })
    return pages


def _compact_evidence_page_results(
    output_root: Path,
    filename: str,
    description: str,
    activity: str,
    domain_result: str,
    effective_build_identity: str,
) -> list[dict[str, Any]]:
    payload = _load_json_object(output_root / filename, description)
    records = payload.get("pages")
    if not isinstance(records, list):
        raise EvidenceError(f"{description.capitalize()} does not contain a pages array")
    pages = []
    for record in records:
        if not isinstance(record, dict):
            raise EvidenceError(f"{description.capitalize()} contains an invalid page record")
        ordinal = int(record["global_ordinal"])
        canonical_page = {
            "global_ordinal": ordinal,
            "evidence_record_sha256": canonical_hash(canonicalize_result(record)),
        }
        pages.append({
            **canonical_page,
            "operation_identity": canonical_hash({
                "effective_build_identity": effective_build_identity,
                **canonical_page,
            }),
            "canonical_page_result_sha256": canonical_hash(canonical_page),
            "activity": activity,
            "domain_result": domain_result,
        })
    return pages


def _crop_framing_assessment_page_results(output_root, activity, domain_result, effective_build_identity):
    payload = _load_json_object(output_root / "assessment.json", "crop and framing assessment")
    records = payload.get("pages")
    algorithms = payload.get("algorithms")
    if not isinstance(records, list) or not isinstance(algorithms, dict) or not algorithms:
        raise EvidenceError("Crop and framing assessment does not contain page variants and algorithms")
    expected_algorithms = sorted(str(value) for value in algorithms)
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise EvidenceError("Crop and framing assessment contains an invalid page variant")
        try:
            ordinal = int(record["global_ordinal"])
            algorithm = str(record["algorithm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceError("Crop and framing assessment contains an invalid page variant identity") from exc
        variants = grouped.setdefault(ordinal, {})
        if not algorithm or algorithm in variants:
            raise EvidenceError(
                f"Crop and framing assessment contains duplicate algorithm evidence for page {ordinal}"
            )
        variants[algorithm] = record

    pages = []
    for ordinal, variants in sorted(grouped.items()):
        if sorted(variants) != expected_algorithms:
            raise EvidenceError(
                f"Crop and framing assessment has incomplete algorithm evidence for page {ordinal}"
            )
        evidence_records = [canonicalize_result(variants[algorithm]) for algorithm in expected_algorithms]
        canonical_page = {
            "global_ordinal": ordinal,
            "evidence_record_sha256": canonical_hash(evidence_records),
        }
        pages.append({
            **canonical_page,
            "operation_identity": canonical_hash({
                "effective_build_identity": effective_build_identity,
                **canonical_page,
            }),
            "canonical_page_result_sha256": canonical_hash(canonical_page),
            "activity": activity,
            "domain_result": domain_result,
        })
    return pages


def _orientation_deskew_assessment_page_results(output_root, activity, domain_result, effective_build_identity):
    return _compact_evidence_page_results(
        output_root, "assessment.json", "orientation and deskew assessment", activity, domain_result, effective_build_identity
    )


def _perspective_assessment_page_results(output_root, activity, domain_result, effective_build_identity):
    return _compact_evidence_page_results(
        output_root, "assessment.json", "perspective assessment", activity, domain_result, effective_build_identity
    )


def _photometric_assessment_page_results(output_root, activity, domain_result, effective_build_identity):
    return _compact_evidence_page_results(
        output_root, "assessment.json", "photometric assessment", activity, domain_result, effective_build_identity
    )


def _photometric_method_assessment_page_results(output_root, activity, domain_result, effective_build_identity):
    return _compact_evidence_page_results(
        output_root, "assessment.json", "photometric method assessment", activity, domain_result, effective_build_identity
    )


def _photometric_validation_page_results(output_root, activity, domain_result, effective_build_identity):
    return _compact_evidence_page_results(
        output_root, "validation.json", "photometric validation", activity, domain_result, effective_build_identity
    )


def _restoration_integration_page_results(
    domain: str,
    output_root: Path,
    activity: str,
    domain_result: str,
    effective_build_identity: str,
) -> list[dict[str, Any]]:
    manifest = _load_json_object(output_root / f"{domain}-normalization-manifest.json", f"{domain} normalization manifest")
    records = manifest.get("pages")
    if not isinstance(records, list) or not records:
        raise EvidenceError(f"{domain.title()} normalization manifest does not contain page records")
    pages = []
    for record in records:
        ordinal = int(record["global_ordinal"])
        canonical_page = {
            "global_ordinal": ordinal,
            "route": record.get("route"),
            "pipeline_action": record.get("pipeline_action"),
            "decision_before": record.get("decision_before"),
            "input_pixel_sha256": record.get("input_pixel_sha256"),
            "output_pixel_sha256": record.get("output_pixel_sha256"),
            "output_image_sha256": record.get("output_sha256"),
            "output_dimensions": [record.get("output_width"), record.get("output_height")],
        }
        pages.append({
            **canonical_page,
            "operation_identity": canonical_hash({"effective_build_identity": effective_build_identity, "global_ordinal": ordinal, "input_pixel_sha256": record.get("input_pixel_sha256")}),
            "canonical_page_result_sha256": canonical_hash(canonical_page),
            "activity": activity,
            "domain_result": domain_result,
        })
    return pages


def _denoising_integration_page_results(output_root, activity, domain_result, effective_build_identity):
    return _restoration_integration_page_results("denoising", output_root, activity, domain_result, effective_build_identity)


def _sharpening_integration_page_results(output_root, activity, domain_result, effective_build_identity):
    return _restoration_integration_page_results("sharpening", output_root, activity, domain_result, effective_build_identity)


def _binarization_integration_page_results(output_root, activity, domain_result, effective_build_identity):
    return _restoration_integration_page_results("binarization", output_root, activity, domain_result, effective_build_identity)


def _denoising_assessment_page_results(output_root, activity, domain_result, effective_build_identity):
    return _compact_evidence_page_results(output_root, "assessment.json", "denoising assessment", activity, domain_result, effective_build_identity)


def _denoising_method_assessment_page_results(output_root, activity, domain_result, effective_build_identity):
    return _compact_evidence_page_results(output_root, "assessment.json", "denoising method assessment", activity, domain_result, effective_build_identity)


def _denoising_validation_page_results(output_root, activity, domain_result, effective_build_identity):
    return _compact_evidence_page_results(output_root, "validation.json", "denoising validation", activity, domain_result, effective_build_identity)


def _sharpening_assessment_page_results(output_root, activity, domain_result, effective_build_identity):
    return _compact_evidence_page_results(output_root, "assessment.json", "sharpening assessment", activity, domain_result, effective_build_identity)


def _sharpening_method_assessment_page_results(output_root, activity, domain_result, effective_build_identity):
    return _compact_evidence_page_results(output_root, "assessment.json", "sharpening method assessment", activity, domain_result, effective_build_identity)


def _sharpening_validation_page_results(output_root, activity, domain_result, effective_build_identity):
    return _compact_evidence_page_results(output_root, "validation.json", "sharpening validation", activity, domain_result, effective_build_identity)


def _binarization_assessment_page_results(output_root, activity, domain_result, effective_build_identity):
    return _compact_evidence_page_results(output_root, "assessment.json", "binarization assessment", activity, domain_result, effective_build_identity)


def _binarization_method_assessment_page_results(output_root, activity, domain_result, effective_build_identity):
    return _compact_evidence_page_results(output_root, "assessment.json", "binarization method assessment", activity, domain_result, effective_build_identity)


def _binarization_validation_page_results(output_root, activity, domain_result, effective_build_identity):
    return _compact_evidence_page_results(output_root, "validation.json", "binarization validation", activity, domain_result, effective_build_identity)


def _tonal_assessment_page_results(
    output_root: Path, activity: str, domain_result: str, effective_build_identity: str
) -> list[dict[str, Any]]:
    return _compact_evidence_page_results(
        output_root, "assessment.json", "tonal assessment", activity, domain_result, effective_build_identity
    )


def _tonal_method_assessment_page_results(
    output_root: Path, activity: str, domain_result: str, effective_build_identity: str
) -> list[dict[str, Any]]:
    return _compact_evidence_page_results(
        output_root,
        "assessment.json",
        "tonal method assessment",
        activity,
        domain_result,
        effective_build_identity,
    )


def _tonal_validation_page_results(
    output_root: Path, activity: str, domain_result: str, effective_build_identity: str
) -> list[dict[str, Any]]:
    return _compact_evidence_page_results(
        output_root, "validation.json", "tonal validation", activity, domain_result, effective_build_identity
    )


def _chromatic_assessment_page_results(
    output_root: Path, activity: str, domain_result: str, effective_build_identity: str
) -> list[dict[str, Any]]:
    return _compact_evidence_page_results(
        output_root, "assessment.json", "chromatic assessment", activity, domain_result, effective_build_identity
    )


def _chromatic_method_assessment_page_results(
    output_root: Path, activity: str, domain_result: str, effective_build_identity: str
) -> list[dict[str, Any]]:
    return _compact_evidence_page_results(
        output_root,
        "assessment.json",
        "chromatic method assessment",
        activity,
        domain_result,
        effective_build_identity,
    )


def _chromatic_validation_page_results(
    output_root: Path, activity: str, domain_result: str, effective_build_identity: str
) -> list[dict[str, Any]]:
    return _compact_evidence_page_results(
        output_root, "validation.json", "chromatic validation", activity, domain_result, effective_build_identity
    )


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json_object(args.plan, "Canonical Build Evidence plan")
    if plan.get("decision") != "execute":
        raise EvidenceError("Only an executed plan can establish canonical results")
    scope = str(plan["scope"])
    artifacts = _artifact_records(args.output_root, artifact_profile(scope), published=False)
    page_builders = {
        PREPROCESS_SCOPE: _page_results,
        NORMALIZATION_SCOPE: _normalization_page_results,
        PHOTOMETRIC_INTEGRATION_SCOPE: _photometric_integration_page_results,
        TONAL_ASSESSMENT_SCOPE: _tonal_assessment_page_results,
        TONAL_METHOD_ASSESSMENT_SCOPE: _tonal_method_assessment_page_results,
        TONAL_VALIDATION_SCOPE: _tonal_validation_page_results,
        TONAL_INTEGRATION_SCOPE: _tonal_integration_page_results,
        CHROMATIC_ASSESSMENT_SCOPE: _chromatic_assessment_page_results,
        CHROMATIC_METHOD_ASSESSMENT_SCOPE: _chromatic_method_assessment_page_results,
        CHROMATIC_VALIDATION_SCOPE: _chromatic_validation_page_results,
        CHROMATIC_INTEGRATION_SCOPE: _chromatic_integration_page_results,
        DENOISING_ASSESSMENT_SCOPE: _denoising_assessment_page_results,
        DENOISING_METHOD_ASSESSMENT_SCOPE: _denoising_method_assessment_page_results,
        DENOISING_VALIDATION_SCOPE: _denoising_validation_page_results,
        DENOISING_INTEGRATION_SCOPE: _denoising_integration_page_results,
        SHARPENING_ASSESSMENT_SCOPE: _sharpening_assessment_page_results,
        SHARPENING_METHOD_ASSESSMENT_SCOPE: _sharpening_method_assessment_page_results,
        SHARPENING_VALIDATION_SCOPE: _sharpening_validation_page_results,
        SHARPENING_INTEGRATION_SCOPE: _sharpening_integration_page_results,
        BINARIZATION_ASSESSMENT_SCOPE: _binarization_assessment_page_results,
        BINARIZATION_METHOD_ASSESSMENT_SCOPE: _binarization_method_assessment_page_results,
        BINARIZATION_VALIDATION_SCOPE: _binarization_validation_page_results,
        BINARIZATION_INTEGRATION_SCOPE: _binarization_integration_page_results,
        CROP_FRAMING_ASSESSMENT_SCOPE: _crop_framing_assessment_page_results,
        ORIENTATION_DESKEW_ASSESSMENT_SCOPE: _orientation_deskew_assessment_page_results,
        PERSPECTIVE_ASSESSMENT_SCOPE: _perspective_assessment_page_results,
        PHOTOMETRIC_ASSESSMENT_SCOPE: _photometric_assessment_page_results,
        PHOTOMETRIC_METHOD_ASSESSMENT_SCOPE: _photometric_method_assessment_page_results,
        PHOTOMETRIC_VALIDATION_SCOPE: _photometric_validation_page_results,
    }
    registered = set(SCOPE_ARTIFACT_PROFILES)
    operation_scopes = set(SCOPE_OPERATION_CONTRACTS) | {NORMALIZATION_SCOPE}
    if (
        set(SCOPE_EVIDENCE_PATHS) != registered
        or set(page_builders) != registered
        or operation_scopes != registered
    ):
        raise EvidenceError("Canonical Build Evidence scope registries are inconsistent")
    page_builder = page_builders[scope]
    pages = page_builder(args.output_root, "EXECUTED", "APPLY", str(plan["effective_build_identity"]))
    result_identity = _result_identity(artifacts, pages)
    incumbent = plan.get("incumbent_result_identity")
    if plan.get("comparison_required") and incumbent != result_identity:
        raise EvidenceError(
            "Determinism verification failed for unchanged effective inputs: "
            f"expected canonical result {incumbent}, produced {result_identity}"
        )
    completed_at_utc = utc_now()
    evaluated_at_utc = str((plan.get("execution") or {}).get("evaluated_at_utc") or completed_at_utc)
    build_elapsed_seconds = elapsed_seconds(evaluated_at_utc, completed_at_utc)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": EVIDENCE_TYPE,
        "scope": plan["scope"],
        "status": "complete",
        "effective_build_identity": plan["effective_build_identity"],
        "effective_inputs": plan["effective_inputs"],
        "resource_utilization": plan["resource_utilization"],
        "canonical_result": {
            "identity": result_identity,
            "artifacts": artifacts,
            "pages": pages,
        },
        "execution": {
            **dict(plan.get("execution") or {}),
            "policy": plan["policy"],
            "activity": "EXECUTED",
            "domain_result": "APPLY",
            "completed_at_utc": completed_at_utc,
            "elapsed_seconds": build_elapsed_seconds,
            "verified_against_incumbent": bool(plan.get("comparison_required")),
        },
    }
    validate_evidence(evidence, scope=plan["scope"])
    store = load_evidence_store(args.evidence_store, scope=plan["scope"])
    store["records"][evidence["effective_build_identity"]] = evidence
    store["authoritative_identity"] = evidence["effective_build_identity"]
    _write_json(args.evidence_output, store)
    _write_github_output(args.github_output, {
        "canonical_result_identity": result_identity,
        "page_count": len(pages),
        "verified_against_incumbent": bool(plan.get("comparison_required")),
    })
    _append_summary(args.github_summary, [
        "",
        "#### Canonical result",
        "",
        f"- Canonical Result Identity: `{result_identity}`",
        f"- Pages executed: `{len(pages)}`",
        f"- Build stage time: `{format_elapsed(build_elapsed_seconds)}` "
        f"(`{build_elapsed_seconds}` seconds)",
        f"- Forced/incumbent equivalence verified: `{bool(plan.get('comparison_required'))}`",
    ])
    return evidence


def merge_stores(args: argparse.Namespace) -> dict[str, Any]:
    base = load_evidence_store(args.base, scope=args.scope)
    incoming = load_evidence_store(args.incoming, scope=args.scope)
    merged = dict(base)
    merged["records"] = dict(base["records"])
    for identity, record in incoming["records"].items():
        prior = merged["records"].get(identity)
        if prior is not None:
            prior_result = prior["canonical_result"]["identity"]
            incoming_result = record["canonical_result"]["identity"]
            if prior_result != incoming_result:
                raise EvidenceError(
                    "Concurrent Canonical Build Evidence collision for unchanged effective inputs: "
                    f"{identity} produced {prior_result} and {incoming_result}"
                )
        merged["records"][identity] = record
    incoming_authoritative = incoming.get("authoritative_identity")
    if incoming_authoritative:
        merged["authoritative_identity"] = incoming_authoritative
    _write_json(args.output, merged)
    return merged


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Create and enforce HTH Canonical Build Evidence")
    commands = root.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Resolve execute, reuse, or audit from effective inputs")
    prepare_parser.add_argument("--scope", default=PREPROCESS_SCOPE)
    prepare_parser.add_argument("--contract-version", default="1")
    prepare_parser.add_argument("--policy", choices=POLICIES, default="auto")
    prepare_parser.add_argument("--mode", required=True)
    prepare_parser.add_argument("--image-limit", type=int, default=0)
    prepare_parser.add_argument("--operation", action="append", default=[])
    prepare_parser.add_argument("--repository-root", type=Path, required=True)
    prepare_parser.add_argument("--source-root", type=Path, required=True)
    prepare_parser.add_argument(
        "--source-input",
        action="append",
        default=[],
        metavar="LOGICAL_PATH=PHYSICAL_PATH",
        help="Fingerprint a physical source under a stable semantic path",
    )
    prepare_parser.add_argument("--source-repository", required=True)
    prepare_parser.add_argument("--source-release", required=True)
    prepare_parser.add_argument("--source-manifest-sha256", required=True)
    prepare_parser.add_argument("--source-commit", required=True)
    prepare_parser.add_argument("--config", type=Path, action="append", default=[])
    prepare_parser.add_argument("--implementation", type=Path, action="append", default=[])
    prepare_parser.add_argument("--detector-implementation-root", type=Path)
    prepare_parser.add_argument("--runtime-contract", type=Path, action="append", default=[])
    prepare_parser.add_argument("--runtime-package", action="append", default=[])
    prepare_parser.add_argument("--runtime-component", action="append", default=[])
    prepare_parser.add_argument("--selection", type=Path)
    prepare_parser.add_argument("--results-root", type=Path, required=True)
    prepare_parser.add_argument("--evidence", type=Path, required=True)
    prepare_parser.add_argument("--plan", type=Path, required=True)
    prepare_parser.add_argument("--artifact-required", action="store_true")
    prepare_parser.add_argument("--pipeline-repository", default="")
    prepare_parser.add_argument("--pipeline-commit", default="")
    prepare_parser.add_argument("--results-repository", default="")
    prepare_parser.add_argument("--results-ref", default="main")
    prepare_parser.add_argument("--workflow-run-id", default="")
    prepare_parser.add_argument("--runner-name", default="")
    prepare_parser.add_argument("--runner-environment", default="")
    prepare_parser.add_argument("--runner-os", default="")
    prepare_parser.add_argument("--runner-arch", default="")
    prepare_parser.add_argument("--github-output", default="")
    prepare_parser.add_argument("--github-summary", default="")

    finalize_parser = commands.add_parser("finalize", help="Hash results and enforce unchanged-build equivalence")
    finalize_parser.add_argument("--plan", type=Path, required=True)
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser.add_argument("--evidence-store", type=Path, required=True)
    finalize_parser.add_argument("--evidence-output", type=Path, required=True)
    finalize_parser.add_argument("--github-output", default="")
    finalize_parser.add_argument("--github-summary", default="")
    merge_parser = commands.add_parser("merge", help="Merge identity-keyed stores during publication retry")
    merge_parser.add_argument("--scope", default=PREPROCESS_SCOPE)
    merge_parser.add_argument("--base", type=Path, required=True)
    merge_parser.add_argument("--incoming", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        merge_stores(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
