"""Generate detector-calibration manifests from persisted results."""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from hth.domain.calibration import authoritative_record
from hth.persistence import ResultsRepository
from hth.regression.run_semantics import legacy_run_semantics
from hth.write_regression_summary import build_combined_summary


def _golden_sha(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path is not None else None


def _matching_index_entries(
    results_root: Path,
    golden_set: Path | None = None,
    *,
    existing_only: bool = True,
    recover_persisted: bool = False,
) -> list[dict[str, Any]]:
    repository = ResultsRepository(results_root)
    if not repository.has_index("calibration-index.json"):
        raise FileNotFoundError(
            f"Missing {repository.readable_index_path('calibration-index.json')}"
        )
    return repository.calibration_entries(
        golden_set_sha256=_golden_sha(golden_set),
        existing_only=existing_only,
        # Routine readers trust the canonical index. A full durable-tree scan is
        # reserved for explicit authoritative-history and repair callers.
        recover_persisted=recover_persisted,
    )


def _selected_smoke_entries(
    results_root: Path,
    golden_set: Path | None = None,
    *,
    existing_only: bool = True,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in _matching_index_entries(
        results_root, golden_set, existing_only=existing_only
    ):
        if legacy_run_semantics(entry)[0] == "smoke":
            grouped.setdefault(str(entry["detector_id"]), []).append(entry)
    return [
        max(
            grouped[detector],
            key=lambda item: (
                str(item.get("created_at_utc") or item.get("published_at_utc") or ""),
                str(
                    (item.get("build") or {}).get("github_run_number")
                    if isinstance(item.get("build"), dict) else ""
                ),
            ),
        )
        for detector in sorted(grouped)
    ]


def smoke_record_paths(results_root: Path, golden_set: Path | None = None) -> list[str]:
    """Return the minimal durable-record checkout required by a manifest report."""
    return [
        str(entry["record_path"])
        for entry in _selected_smoke_entries(
            results_root, golden_set, existing_only=False
        )
    ]


def _materialization_record_path(entry: dict[str, Any]) -> str:
    """Return one safe record root and validate its indexed dependencies."""
    raw_record = str(entry.get("record_path") or "").strip().replace("\\", "/")
    record = PurePosixPath(raw_record)
    if not raw_record or record.is_absolute() or ".." in record.parts:
        raise ValueError(f"Unsafe persisted calibration record path: {raw_record!r}")
    normalized = record.as_posix()
    for key in ("intelligence_path", "parameter_provenance_path"):
        raw_dependency = str(entry.get(key) or "").strip().replace("\\", "/")
        if not raw_dependency:
            continue
        dependency = PurePosixPath(raw_dependency)
        dependency_path = dependency.as_posix()
        if (
            dependency.is_absolute()
            or ".." in dependency.parts
            or not dependency_path.startswith(f"{normalized}/")
        ):
            raise ValueError(
                f"Calibration {key} must remain beneath {normalized}: {raw_dependency!r}"
            )
    return normalized


def calibration_report_record_paths(
    results_root: Path,
    golden_set: Path | None = None,
) -> list[str]:
    """Return the minimal complete durable-record set for Report Writer.

    The smoke manifest and runtime schedule require the latest smoke observation
    per detector.  Best Known requires the strongest compatible calibration per
    detector independently of pipeline revision.  Both views are materialized so
    a sparse checkout has the same selection semantics as a complete results tree.
    """
    entries = _matching_index_entries(
        results_root, golden_set, existing_only=False
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(str(entry["detector_id"]), []).append(entry)

    selected = list(
        _selected_smoke_entries(results_root, golden_set, existing_only=False)
    )
    for detector in sorted(grouped):
        # Best Known needs calibration intelligence. Entries without that
        # dependency cannot participate in the rendered calibration view.
        candidates = [
            entry for entry in grouped[detector]
            if str(entry.get("intelligence_path") or "").strip()
        ]
        candidate = authoritative_record(candidates)
        if candidate is not None:
            selected.append(candidate)

    return sorted({_materialization_record_path(entry) for entry in selected})


def calibration_run_dirs(results_root: Path, golden_set: Path | None = None) -> list[Path]:
    """Resolve the best persisted calibration record per detector."""
    repository = ResultsRepository(results_root)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in _matching_index_entries(
        results_root, golden_set, recover_persisted=True
    ):
        grouped.setdefault(str(entry["detector_id"]), []).append(entry)
    selected = {
        detector: candidate
        for detector, records in grouped.items()
        if (candidate := authoritative_record(records)) is not None
    }
    if not selected:
        suffix = f" matching {golden_set}" if golden_set else ""
        raise ValueError(f"No persisted calibration records found{suffix}")
    return [repository.record_dir(selected[key]) for key in sorted(selected)]


def smoke_run_dirs(results_root: Path, golden_set: Path | None = None) -> list[Path]:
    """Resolve the latest persisted smoke observation per detector."""
    repository = ResultsRepository(results_root)
    selected = _selected_smoke_entries(results_root, golden_set)
    if not selected:
        suffix = f" matching {golden_set}" if golden_set else ""
        raise ValueError(f"No persisted smoke records found{suffix}")
    return [repository.record_dir(entry) for entry in selected]


def _normalize_persisted_run(persisted: Path, normalized: Path) -> None:
    reports = normalized / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "parameters.json", "RUN-INFO.json"):
        source = persisted / name
        if source.is_file():
            shutil.copy2(source, normalized / name)
    for name in ("summary.json", "winner-pages.json", "calibration-intelligence.json"):
        source = persisted / name
        if source.is_file():
            shutil.copy2(source, reports / name)
    required = (
        normalized / "manifest.json",
        normalized / "parameters.json",
        normalized / "RUN-INFO.json",
        reports / "summary.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Persisted calibration record {persisted} is incomplete; missing: {', '.join(missing)}"
        )


def generate_calibration_manifest(
    results_root: Path,
    output: Path,
    *,
    golden_set: Path | None,
    pipeline_repository: str,
    results_repository: str,
    results_commit: str,
    run_url: str,
) -> Path:
    repository = ResultsRepository(results_root)
    persisted_dirs = smoke_run_dirs(results_root, golden_set)
    with tempfile.TemporaryDirectory(prefix="hth-report-") as temp:
        temp_root = Path(temp)
        run_dirs: list[Path] = []
        for ordinal, persisted in enumerate(persisted_dirs, start=1):
            normalized = temp_root / f"run-{ordinal:03d}"
            _normalize_persisted_run(persisted, normalized)
            run_dirs.append(normalized)
        text = build_combined_summary(
            run_dirs,
            run_url,
            pipeline_repository=pipeline_repository,
            results_repository=results_repository,
            results_commit=results_commit,
            calibration_index=repository.readable_index_path("calibration-index.json"),
            runtime_index=repository.readable_index_path("runtime-index.json"),
            report_writer_smoke_reference=True,
            persisted_results_root=results_root,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output
