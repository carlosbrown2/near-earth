"""Rankings endpoint (US-001).

GET /v1/rankings — paginated ranked asteroid mining candidates
ordered by composite_score DESC.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query

from prospector.api.auth import get_api_key
from prospector.api.app import get_db
from prospector.api.models import PaginatedResponse

router = APIRouter(prefix="/v1", dependencies=[Depends(get_api_key)])

# Fields returned in ranking results (matches PRD US-001)
_RANKING_FIELDS = [
    "asteroid_id", "name", "designation",
    "composite_score", "estimated_mass_kg", "grade_estimate",
    "target_material", "unit_value", "accessibility", "score_mode",
]

_BASE_QUERY = """
    SELECT
        a.asteroid_id, a.name, a.designation,
        s.composite_score, s.estimated_mass_kg, s.grade_estimate,
        s.target_material, s.unit_value, s.accessibility, s.score_mode
    FROM scores s
    JOIN asteroids a ON s.asteroid_id = a.asteroid_id
"""

_COUNT_QUERY = """
    SELECT COUNT(*)
    FROM scores s
    JOIN asteroids a ON s.asteroid_id = a.asteroid_id
"""


def _build_where(
    mode: str, min_score: float | None, material: str | None
) -> tuple[str, list[Any]]:
    """Build WHERE clause and parameter list."""
    clauses = ["s.score_mode = ?"]
    params: list[Any] = [mode]

    if min_score is not None:
        clauses.append("s.composite_score >= ?")
        params.append(min_score)

    if material is not None:
        clauses.append("s.target_material = ?")
        params.append(material)

    return " WHERE " + " AND ".join(clauses), params


@router.get("/rankings", response_model=PaginatedResponse)
def get_rankings(
    db: sqlite3.Connection = Depends(get_db),
    mode: str = Query("earth_return", pattern="^(earth_return|in_space)$"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    min_score: Optional[float] = Query(None, ge=0),
    material: Optional[str] = Query(None),
) -> PaginatedResponse:
    """Return paginated ranked asteroid mining candidates."""
    where, params = _build_where(mode, min_score, material)

    # Total count
    total = db.execute(_COUNT_QUERY + where, params).fetchone()[0]

    # Paginated results
    query = (
        _BASE_QUERY + where
        + " ORDER BY s.composite_score DESC"
        + f" LIMIT {limit} OFFSET {offset}"
    )
    rows = db.execute(query, params).fetchall()

    data = []
    for i, row in enumerate(rows):
        record = dict(zip(_RANKING_FIELDS, row))
        record["rank"] = offset + i + 1
        data.append(record)

    return PaginatedResponse(data=data, total=total, limit=limit, offset=offset)
