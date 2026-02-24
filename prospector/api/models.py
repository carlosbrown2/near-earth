"""Pydantic response models for the REST API.

These are separate from the internal pipeline schemas (prospector.schemas)
because the API contract may differ from internal data structures.
BLOB fields are never exposed; numeric values are native JSON numbers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    asteroids_count: int = 0
    scores_count: int = 0


class PaginatedResponse(BaseModel):
    """Base paginated response wrapper."""

    data: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0
