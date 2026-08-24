from __future__ import annotations

from pathlib import Path

INDEX_DIRECTORY = "indexes"
INDEX_FILENAMES = frozenset({
    "calibration-index.json",
    "multidetector-index.json",
    "optimizer-index.json",
    "orli-evidence-index.json",
    "parallelism-index.json",
    "parameter-provenance-index.json",
    "runtime-index.json",
})


def canonical_index_path(results_root: Path, filename: str) -> Path:
    if filename not in INDEX_FILENAMES:
        raise ValueError(f"Unknown HTH results index: {filename}")
    return Path(results_root) / INDEX_DIRECTORY / filename


def readable_index_path(results_root: Path, filename: str) -> Path:
    """Prefer the canonical indexes/ path, but read a legacy root index during migration."""
    canonical = canonical_index_path(results_root, filename)
    if canonical.is_file():
        return canonical
    legacy = Path(results_root) / filename
    return legacy if legacy.is_file() else canonical
