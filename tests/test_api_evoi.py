"""Tests for GET /v1/evoi endpoint."""

from __future__ import annotations

import os
import sqlite3
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from prospector.api.app import create_app, get_db
from prospector.db import init_schema
from prospector.scoring.evoi import EVOIResult


def _make_evoi_result(asteroid_id: int, best_evoi: float = 5.0) -> EVOIResult:
    """Create a mock EVOIResult."""
    return EVOIResult(
        asteroid_id=asteroid_id,
        current_score_mean=50.0,
        current_score_std=10.0,
        evoi_vnir=best_evoi,
        evoi_vis=3.0,
        evoi_radar=2.0,
        evoi_albedo=1.0,
        best_observation="vnir_spectroscopy",
        best_evoi=best_evoi,
    )


@pytest.fixture
def db_conn():
    """In-memory database with schema, thread-safe for TestClient."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_schema(conn)
    # Add some asteroids
    for i in [433, 1036, 4179]:
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, 1)",
            (i, f"Asteroid-{i}"),
        )
    conn.commit()
    return conn


@pytest.fixture
def mock_results():
    """Mock EVOI results sorted by best_evoi desc."""
    return [
        _make_evoi_result(433, 8.0),
        _make_evoi_result(1036, 5.0),
        _make_evoi_result(4179, 3.0),
    ]


@pytest.fixture
def client(db_conn, mock_results):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db_conn
    with patch("prospector.api.routers.evoi.rank_all", return_value=mock_results):
        yield TestClient(app, raise_server_exceptions=False)


class TestEVOIEndpoint:
    """Test GET /v1/evoi."""

    def test_requires_auth(self, client):
        resp = client.get("/v1/evoi")
        assert resp.status_code == 401

    def test_returns_results(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/evoi", headers={"X-API-Key": "key"})
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert "total" in data
        assert data["total"] == 3

    def test_response_fields(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/evoi", headers={"X-API-Key": "key"})
        item = resp.json()["data"][0]
        required = [
            "asteroid_id", "name", "score_mean", "score_std",
            "evoi_vnir", "evoi_vis", "evoi_radar", "evoi_albedo",
            "best_observation", "best_evoi",
        ]
        for field in required:
            assert field in item, f"Missing field: {field}"

    def test_pagination_limit(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/evoi?limit=2", headers={"X-API-Key": "key"})
        data = resp.json()
        assert len(data["data"]) == 2
        assert data["limit"] == 2
        assert data["total"] == 3

    def test_pagination_offset(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/evoi?offset=2", headers={"X-API-Key": "key"})
        data = resp.json()
        assert len(data["data"]) == 1
        assert data["offset"] == 2

    def test_invalid_mode(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/evoi?mode=lunar", headers={"X-API-Key": "key"})
        assert resp.status_code == 422

    def test_asteroid_name_resolved(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/evoi", headers={"X-API-Key": "key"})
        item = resp.json()["data"][0]
        assert item["name"] == "Asteroid-433"

    def test_computation_failure_returns_empty(self, db_conn):
        """If rank_all raises, endpoint returns empty results."""
        app = create_app()
        app.dependency_overrides[get_db] = lambda: db_conn
        with patch("prospector.api.routers.evoi.rank_all", side_effect=RuntimeError("boom")):
            client = TestClient(app, raise_server_exceptions=False)
            with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
                resp = client.get("/v1/evoi", headers={"X-API-Key": "key"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"] == []
        assert data["total"] == 0

    def test_values_are_rounded(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/evoi", headers={"X-API-Key": "key"})
        item = resp.json()["data"][0]
        # Values should be numeric, not strings
        assert isinstance(item["score_mean"], (int, float))
        assert isinstance(item["best_evoi"], (int, float))
