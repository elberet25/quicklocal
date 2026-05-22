"""
Shared error classification utilities for all tools.

classify_error() maps common exception types to actionable messages and a
retryable flag so the dispatcher can decide whether to back off and retry.
"""

import socket

try:
    from googleapiclient.errors import HttpError
except ImportError:
    HttpError = None  # type: ignore[assignment,misc]

try:
    from google.auth.exceptions import RefreshError
except ImportError:
    RefreshError = None  # type: ignore[assignment,misc]

try:
    import httplib2 as _httplib2
    _HTTPLIB2_NOT_FOUND = _httplib2.ServerNotFoundError
except ImportError:
    _HTTPLIB2_NOT_FOUND = None  # type: ignore[assignment,misc]

_TOKEN_HINT = (
    "Delete the relevant token file "
    "(token.json / calendar_token.json / drive_token.json) "
    "and run the agent again to re-authorize."
)


def classify_error(error: Exception) -> dict:
    """Return {"message": str, "retryable": bool} for a caught exception."""
    if HttpError is not None and isinstance(error, HttpError):
        status = int(error.resp.status)
        if status == 429:
            return {
                "message": "Google API rate limit hit. Please try again in a moment.",
                "retryable": True,
            }
        if status >= 500:
            return {
                "message": f"Google API server error ({status}). Please try again.",
                "retryable": True,
            }
        if status == 401:
            return {
                "message": f"Google API authentication expired. {_TOKEN_HINT}",
                "retryable": False,
            }
        if status == 403:
            return {
                "message": "Google API permission denied. Check your credentials and API scopes.",
                "retryable": False,
            }
        return {"message": f"Google API error ({status}): {error}", "retryable": False}

    if RefreshError is not None and isinstance(error, RefreshError):
        return {
            "message": f"Google authentication token could not be refreshed. {_TOKEN_HINT}",
            "retryable": False,
        }

    _network_types: tuple = (ConnectionError, TimeoutError, socket.timeout)
    if _HTTPLIB2_NOT_FOUND is not None:
        _network_types = _network_types + (_HTTPLIB2_NOT_FOUND,)

    if isinstance(error, _network_types):
        return {
            "message": "Network error. Check your internet connection and try again.",
            "retryable": True,
        }

    return {"message": str(error), "retryable": False}
