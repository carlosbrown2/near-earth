"""FastAPI application factory (US-005).

Creates the FastAPI app with API key authentication on /v1/* routes
and an unauthenticated /health endpoint. Database connection is
read-only from an existing prospector.db file.

Usage:
    uvicorn prospector.api.app:create_app --factory
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI

from prospector.api.auth import get_api_key
from prospector.api.models import HealthResponse
from prospector.db import get_connection


def _get_db_path() -> str:
    """Resolve database path from environment or default."""
    return os.environ.get("PROSPECTOR_DB_PATH", "data/prospector.db")


# Module-level connection, initialized during app lifespan
_db_conn: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    """FastAPI dependency providing the database connection."""
    if _db_conn is None:
        raise RuntimeError("Database not initialized")
    return _db_conn


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage database connection lifecycle."""
    global _db_conn
    db_path = _get_db_path()
    _db_conn = get_connection(db_path, create=True)
    # Allow cross-thread reads (safe for read-only, WAL mode)
    _db_conn.execute("PRAGMA query_only = ON")
    # Re-open with check_same_thread=False for async compatibility
    _db_conn.close()
    raw_conn = sqlite3.connect(db_path, check_same_thread=False)
    raw_conn.execute("PRAGMA journal_mode=WAL")
    raw_conn.execute("PRAGMA foreign_keys=ON")
    raw_conn.execute("PRAGMA query_only = ON")
    _db_conn = raw_conn
    try:
        yield
    finally:
        if _db_conn:
            _db_conn.close()
            _db_conn = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Asteroid Mining Prospector API",
        description="Ranked NEO mining candidates from public asteroid survey data",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(db: sqlite3.Connection = Depends(get_db)) -> HealthResponse:
        """Health check — unauthenticated."""
        try:
            asteroids = db.execute("SELECT COUNT(*) FROM asteroids").fetchone()[0]
        except Exception:
            asteroids = 0
        try:
            scores = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        except Exception:
            scores = 0
        return HealthResponse(
            status="ok",
            asteroids_count=asteroids,
            scores_count=scores,
        )

    # Placeholder v1 status — authenticated
    @app.get("/v1/status", dependencies=[Depends(get_api_key)])
    def v1_status() -> dict[str, str]:
        """Authenticated status endpoint."""
        return {"status": "ok", "version": "v1"}

    # Register routers
    from prospector.api.routers.rankings import router as rankings_router
    from prospector.api.routers.asteroids import router as asteroids_router
    from prospector.api.routers.evoi import router as evoi_router

    app.include_router(rankings_router)
    app.include_router(asteroids_router)
    app.include_router(evoi_router)

    return app
