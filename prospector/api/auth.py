"""API key authentication dependency (US-005).

Keys are loaded from the PROSPECTOR_API_KEYS environment variable
(comma-separated list). All /v1/* endpoints require a valid key
via X-API-Key header or api_key query parameter.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, HTTPException, Query, Request, status


def _load_api_keys() -> set[str]:
    """Load valid API keys from environment."""
    raw = os.environ.get("PROSPECTOR_API_KEYS", "")
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    return keys


def get_api_key(
    request: Request,
    api_key: Optional[str] = Query(None, alias="api_key"),
) -> str:
    """FastAPI dependency that validates API key authentication.

    Checks X-API-Key header first, then api_key query parameter.
    Raises 401 if no valid key is provided.
    """
    key = request.headers.get("X-API-Key") or api_key

    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    valid_keys = _load_api_keys()
    if not valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    if key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    return key
