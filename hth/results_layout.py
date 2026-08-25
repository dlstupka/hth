"""Compatibility facade for the canonical HTH persistence boundary.

New code should import from :mod:`hth.persistence`.  This module remains so
older callers do not acquire a second definition of results-repository layout.
"""
from hth.persistence import (  # noqa: F401
    INDEX_DIRECTORY,
    INDEX_FILENAMES,
    canonical_index_path,
    readable_index_path,
    index_results_root,
    resolve_index_relative_path,
)
