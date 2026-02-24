"""Tests for GET /v1/rankings endpoint."""

from __future__ import annotations

import os
import sqlite3
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from prospector.api.app import create_app, get_db
from prospector.db import init_schema


@pytest.fixture
def db_conn():
    """In-memory database with schema, thread-safe for TestClient."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_schema(conn)
    return conn


@pytest.fixture
def seeded_db(db_conn):
    """Database with sample scored asteroids."""
    # Insert 5 scored asteroids
    for i in range(1, 6):
        db_conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, designation, neo, pha) VALUES (?, ?, ?, 1, 0)",
            (i, f"Asteroid-{i}", f"2024 AA{i}"),
        )
        db_conn.execute(
            "INSERT INTO scores (asteroid_id, composite_score, estimated_mass_kg, "
            "grade_estimate, target_material, unit_value, accessibility, score_mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (i, 100.0 - i * 10, 1e12 * i, 0.05, "pgm" if i <= 3 else "water", 500.0, 0.8, "earth_return"),
        )
    # Add a separate asteroid with in_space score
    db_conn.execute(
        "INSERT INTO asteroids (asteroid_id, name, designation, neo, pha) VALUES (100, 'InSpace-1', '2024 ZZ1', 1, 0)"
    )
    db_conn.execute(
        "INSERT INTO scores (asteroid_id, composite_score, estimated_mass_kg, "
        "grade_estimate, target_material, unit_value, accessibility, score_mode) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (100, 75.0, 1e12, 0.03, "water", 300.0, 0.9, "in_space"),
    )
    db_conn.commit()
    return db_conn


@pytest.fixture
def client(seeded_db):
    """Test client with seeded database and valid API key."""
    app = create_app()
    app.dependency_overrides[get_db] = lambda: seeded_db
    return TestClient(app, raise_server_exceptions=False)


class TestRankingsEndpoint:
    """Test GET /v1/rankings."""

    def test_requires_auth(self, client):
        resp = client.get("/v1/rankings")
        assert resp.status_code == 401

    def test_returns_ranked_list(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/rankings", headers={"X-API-Key": "key"})
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

    def test_default_mode_earth_return(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/rankings", headers={"X-API-Key": "key"})
        data = resp.json()
        assert data["total"] == 5
        for item in data["data"]:
            assert item["score_mode"] == "earth_return"

    def test_ordered_by_composite_score_desc(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/rankings", headers={"X-API-Key": "key"})
        items = resp.json()["data"]
        scores = [item["composite_score"] for item in items]
        assert scores == sorted(scores, reverse=True)

    def test_rank_field_present(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/rankings", headers={"X-API-Key": "key"})
        items = resp.json()["data"]
        ranks = [item["rank"] for item in items]
        assert ranks == [1, 2, 3, 4, 5]

    def test_mode_in_space(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/rankings?mode=in_space", headers={"X-API-Key": "key"})
        data = resp.json()
        assert data["total"] == 1
        assert data["data"][0]["score_mode"] == "in_space"

    def test_limit_param(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/rankings?limit=2", headers={"X-API-Key": "key"})
        data = resp.json()
        assert len(data["data"]) == 2
        assert data["total"] == 5  # total unchanged
        assert data["limit"] == 2

    def test_offset_param(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/rankings?offset=3", headers={"X-API-Key": "key"})
        data = resp.json()
        assert len(data["data"]) == 2  # 5 total, skip 3
        assert data["offset"] == 3
        assert data["data"][0]["rank"] == 4

    def test_min_score_filter(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/rankings?min_score=70", headers={"X-API-Key": "key"})
        data = resp.json()
        for item in data["data"]:
            assert item["composite_score"] >= 70
        assert data["total"] == 3  # scores 90, 80, 70

    def test_material_filter(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/rankings?material=water", headers={"X-API-Key": "key"})
        data = resp.json()
        assert data["total"] == 2
        for item in data["data"]:
            assert item["target_material"] == "water"

    def test_combined_filters(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get(
                "/v1/rankings?material=pgm&min_score=80",
                headers={"X-API-Key": "key"},
            )
        data = resp.json()
        assert data["total"] == 2  # pgm with score >= 80: scores 90, 80

    def test_invalid_mode_rejected(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/rankings?mode=lunar", headers={"X-API-Key": "key"})
        assert resp.status_code == 422

    def test_limit_max_500(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/rankings?limit=501", headers={"X-API-Key": "key"})
        assert resp.status_code == 422

    def test_empty_results(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/rankings?min_score=999", headers={"X-API-Key": "key"})
        data = resp.json()
        assert data["data"] == []
        assert data["total"] == 0

    def test_response_fields(self, client):
        """Check that all PRD-required fields are present."""
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/rankings?limit=1", headers={"X-API-Key": "key"})
        item = resp.json()["data"][0]
        required = [
            "rank", "asteroid_id", "name", "designation",
            "composite_score", "estimated_mass_kg", "grade_estimate",
            "target_material", "unit_value", "accessibility", "score_mode",
        ]
        for field in required:
            assert field in item, f"Missing field: {field}"

    def test_empty_database(self, db_conn):
        """Rankings returns empty on a database with no scores."""
        app = create_app()
        app.dependency_overrides[get_db] = lambda: db_conn
        client = TestClient(app, raise_server_exceptions=False)
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/rankings", headers={"X-API-Key": "key"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"] == []
        assert data["total"] == 0
