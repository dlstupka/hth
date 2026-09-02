"""Shared classification for retryable remote-acquisition failures."""
from __future__ import annotations

import urllib.error


TRANSIENT_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def is_transient_network_error(
    error: BaseException,
    *,
    retry_unauthenticated_forbidden: bool = False,
) -> bool:
    """Return whether a failed remote request is safe to retry."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in TRANSIENT_HTTP_STATUS_CODES or (
            retry_unauthenticated_forbidden and error.code == 403
        )
    return isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError))
