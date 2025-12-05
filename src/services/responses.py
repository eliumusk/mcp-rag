"""Helpers for generating consistent JSON responses."""
from typing import Any, Dict, Optional

import json


def success_response(message: str, data: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Return a standardized success payload."""

    payload: Dict[str, Any] = {"success": True, "message": message}
    if data is not None:
        payload["data"] = data
    if metadata:
        payload["metadata"] = metadata
    return json.dumps(payload, indent=2)


def error_response(
    code: str,
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    retryable: bool = False,
) -> str:
    """Return a standardized error payload."""

    error_body: Dict[str, Any] = {"code": code, "retryable": retryable}
    if details:
        error_body["details"] = details

    payload = {
        "success": False,
        "message": message,
        "error": error_body,
    }
    return json.dumps(payload, indent=2)
