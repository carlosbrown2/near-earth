"""EVOI observation priority endpoint (US-004).

GET /v1/evoi — rank uncharacterized asteroids by Expected Value
of Information, indicating which asteroids would benefit most
from new observations.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query

from prospector.api.auth import get_api_key
from prospector.api.app import get_db
from prospector.api.models import PaginatedResponse
from prospector.scoring.evoi import rank_all

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", dependencies=[Depends(get_api_key)])

# Maximum asteroids to evaluate per request (EVOI is ~2s each)
_MAX_COMPUTE = 50


@router.get("/evoi", response_model=PaginatedResponse)
def get_evoi_rankings(
    db: sqlite3.Connection = Depends(get_db),
    mode: str = Query("earth_return", pattern="^(earth_return|in_space)$"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> PaginatedResponse:
    """Return EVOI rankings for uncharacterized asteroids.

    Computes on-the-fly for asteroids without spectral taxonomy.
    Limited to a small batch per request due to computational cost.
    """
    max_asteroids = min(limit + offset, _MAX_COMPUTE)

    try:
        results = rank_all(
            db,
            mode=mode,  # type: ignore[arg-type]
            n_samples=200,
            neo_only=True,
            max_asteroids=max_asteroids,
        )
    except Exception:
        logger.exception("EVOI computation failed")
        return PaginatedResponse(data=[], total=0, limit=limit, offset=offset)

    total = len(results)

    # Apply pagination
    page = results[offset : offset + limit]

    data: list[dict[str, Any]] = []
    for r in page:
        # Look up asteroid name
        row = db.execute(
            "SELECT name FROM asteroids WHERE asteroid_id = ?",
            (r.asteroid_id,),
        ).fetchone()
        name = row[0] if row else None

        data.append({
            "asteroid_id": r.asteroid_id,
            "name": name,
            "score_mean": round(r.current_score_mean, 4),
            "score_std": round(r.current_score_std, 4),
            "evoi_vnir": round(r.evoi_vnir, 4),
            "evoi_vis": round(r.evoi_vis, 4),
            "evoi_radar": round(r.evoi_radar, 4),
            "evoi_albedo": round(r.evoi_albedo, 4),
            "best_observation": r.best_observation,
            "best_evoi": round(r.best_evoi, 4),
        })

    return PaginatedResponse(data=data, total=total, limit=limit, offset=offset)
