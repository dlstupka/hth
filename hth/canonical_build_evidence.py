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
from typing import Any, Iterable

from hth.markdown_links import code_link, github_blob_url


SCHEMA_VERSION = "1.0"
EVIDENCE_TYPE = "canonical-build-evidence"
STORE_TYPE = "canonical-build-evidence-store"
PREPROCESS_SCOPE = "hth-preprocess"
NORMALIZATION_SCOPE = "hth-normalization"
PHOTOMETRIC_INTEGRATION_SCOPE = "hth-photometric-integration"
TONAL_INTEGRATION_SCOPE = "hth-tonal-integration"
POLICIES = ("auto", "audit", "force-verify", "rebuild")

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

SCOPE_ARTIFACT_PROFILES = {
    PREPROCESS_SCOPE: PREPROCESS_ARTIFACTS,
    NORMALIZATION_SCOPE: NORMALIZATION_ARTIFACTS,
    PHOTOMETRIC_INTEGRATION_SCOPE: PHOTOMETRIC_INTEGRATION_ARTIFACTS,
    TONAL_INTEGRATION_SCOPE: TONAL_INTEGRATION_ARTIFACTS,
}

SCOPE_EVIDENCE_PATHS = {
    PREPROCESS_SCOPE: "metadata/canonical-build-evidence.json",
    NORMALIZATION_SCOPE: "normalization/canonical-build-evidence.json",
    PHOTOMETRIC_INTEGRATION_SCOPE: "normalization/photometric-integration/canonical-build-evidence.json",
    TONAL_INTEGRATION_SCOPE: "normalization/tonal-integration/canonical-build-evidence.json",
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
    if not args.operation:
        raise EvidenceError("The deterministic operation contract is empty")
    if not args.config or not args.implementation or not args.runtime_contract:
        raise EvidenceError("Configuration, implementation, and runtime contracts are all required")
    repository_root = args.repository_root.resolve()
    source_files = fingerprint_paths([args.source_root], args.source_root)
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
            "mode": args.mode,
            "image_limit": args.image_limit,
            "operations": list(args.operation),
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
    evidence_url = ""
    evidence_relative = evidence_relative_path(args.scope)
    if args.evidence.is_file():
        evidence_url = github_blob_url(
            getattr(args, "results_repository", ""),
            getattr(args, "results_ref", "main"),
            evidence_relative,
        )
    _append_summary(args.github_summary, [
        "### Canonical Build Evidence",
        "",
        f"- Policy: `{args.policy}`",
        f"- Effective Build Identity: `{identity}`",
        f"- Activity: `{activity}`",
        f"- Domain result: `{domain_result}`",
        f"- Decision: `{decision}`",
        f"- Pages marked unnecessary: `{len(page_evaluations)}`",
        f"- Evidence: {code_link(evidence_relative, evidence_url)}",
    ])
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
        TONAL_INTEGRATION_SCOPE: _tonal_integration_page_results,
    }
    registered = set(SCOPE_ARTIFACT_PROFILES)
    if set(SCOPE_EVIDENCE_PATHS) != registered or set(page_builders) != registered:
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
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": EVIDENCE_TYPE,
        "scope": plan["scope"],
        "status": "complete",
        "effective_build_identity": plan["effective_build_identity"],
        "effective_inputs": plan["effective_inputs"],
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
            "completed_at_utc": utc_now(),
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
