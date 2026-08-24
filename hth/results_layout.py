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


def index_results_root(index_path: Path) -> Path:
    """Return the results-repository root for a canonical or legacy index path."""
    path = Path(index_path)
    return path.parent.parent if path.parent.name == INDEX_DIRECTORY else path.parent


def resolve_index_relative_path(index_path: Path, relative_path: str | Path) -> Path:
    """Resolve a repository-relative path stored inside an HTH results index."""
    return index_results_root(index_path) / Path(relative_path)
