"""Process-global, thread-safe native stderr redirection helpers.

File descriptor 2 belongs to the process, not to an individual Python thread.
Any detector/runtime that temporarily redirects native stderr must therefore
share one lock across the entire HTH process.  Keeping the lock here prevents
two detector-specific wrappers from each being "thread-safe" in isolation while
still racing with one another.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import tempfile
import threading
from typing import Iterator, BinaryIO

_FD2_LOCK = threading.RLock()


@contextmanager
def capture_native_stderr() -> Iterator[BinaryIO]:
    """Capture process fd 2 while holding HTH's single global stderr lock."""
    with _FD2_LOCK:
        saved_fd = os.dup(2)
        captured = tempfile.TemporaryFile(mode="w+b")
        try:
            os.dup2(captured.fileno(), 2)
            yield captured
        finally:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            captured.close()


@contextmanager
def suppress_native_stderr() -> Iterator[None]:
    """Suppress process fd 2 while holding HTH's single global stderr lock."""
    with _FD2_LOCK:
        saved_fd = os.dup(2)
        try:
            with open(os.devnull, "w") as sink:
                os.dup2(sink.fileno(), 2)
                yield
        finally:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
