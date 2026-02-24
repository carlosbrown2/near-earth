"""Tests for GET /v1/asteroids (search endpoint)."""

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
    """Database with diverse asteroids for search testing."""
    asteroids = [
        (433, "Eros", "2024 ER1", 1, 0),
        (1036, "Ganymed", "2024 GA1", 1, 0),
        (4179, "Toutatis", "2024 TT1", 1, 1),
        (1, "Ceres", "2024 CE1", 0, 0),
        (25143, "Itokawa", "2024 IT1", 1, 0),
    ]
    for a in asteroids:
        db_conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, designation, neo, pha) "
            "VALUES (?, ?, ?, ?, ?)", a
        )

    orbits = [
        (433, 1.458, 0.223, 10.83, 0.149, 16.84),
        (1036, 2.665, 0.533, 26.68, 0.344, 31.7),
        (4179, 2.512, 0.634, 0.45, 0.006, 2.45),
        (1, 2.768, 0.076, 10.59, None, 939.4),
        (25143, 1.324, 0.280, 1.62, 0.013, 0.33),
    ]
    for o in orbits:
        db_conn.execute(
            "INSERT INTO orbits (asteroid_id, a, e, i, moid, diameter) "
            "VALUES (?, ?, ?, ?, ?, ?)", o
        )

    taxonomy = [
        (433, "S", 0.85),
        (1036, "S", 0.72),
        (4179, "S", 0.90),
        (1, "C", 0.95),
        (25143, "S", 0.88),
    ]
    for t in taxonomy:
        db_conn.execute(
            "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob) "
            "VALUES (?, ?, ?)", t
        )

    db_conn.commit()
    return db_conn


@pytest.fixture
def client(seeded_db):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: seeded_db
    return TestClient(app, raise_server_exceptions=False)


def _get(client, path="/v1/asteroids", **params):
    with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
        return client.get(path, params=params, headers={"X-API-Key": "key"})


class TestSearchEndpoint:
    """Test GET /v1/asteroids."""

    def test_requires_auth(self, client):
        resp = client.get("/v1/asteroids")
        assert resp.status_code == 401

    def test_returns_all(self, client):
        resp = _get(client)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["data"]) == 5

    def test_text_search_name(self, client):
        resp = _get(client, q="Ero")
        data = resp.json()
        assert data["total"] == 1
        assert data["data"][0]["name"] == "Eros"

    def test_text_search_case_insensitive(self, client):
        resp = _get(client, q="ero")
        data = resp.json()
        assert data["total"] == 1

    def test_text_search_designation(self, client):
        resp = _get(client, q="2024 IT")
        data = resp.json()
        assert data["total"] == 1
        assert data["data"][0]["name"] == "Itokawa"

    def test_neo_filter(self, client):
        resp = _get(client, neo="true")
        data = resp.json()
        assert data["total"] == 4
        for item in data["data"]:
            assert item["neo"] == 1

    def test_pha_filter(self, client):
        resp = _get(client, pha="true")
        data = resp.json()
        assert data["total"] == 1
        assert data["data"][0]["name"] == "Toutatis"

    def test_min_diameter(self, client):
        resp = _get(client, min_diameter=100)
        data = resp.json()
        # Ceres (939.4) and Ganymed (31.7) — no, just Ceres at 939.4
        for item in data["data"]:
            assert item["diameter_km"] >= 100

    def test_max_diameter(self, client):
        resp = _get(client, max_diameter=1.0)
        data = resp.json()
        assert data["total"] == 1
        assert data["data"][0]["name"] == "Itokawa"

    def test_min_moid(self, client):
        resp = _get(client, min_moid=0.1)
        data = resp.json()
        for item in data["data"]:
            assert item["moid"] >= 0.1

    def test_max_moid(self, client):
        resp = _get(client, max_moid=0.02)
        data = resp.json()
        assert data["total"] == 2  # Toutatis (0.006) and Itokawa (0.013)

    def test_taxonomy_class(self, client):
        resp = _get(client, taxonomy_class="C")
        data = resp.json()
        assert data["total"] == 1
        assert data["data"][0]["name"] == "Ceres"

    def test_combined_filters(self, client):
        resp = _get(client, neo="true", taxonomy_class="S", max_diameter=20)
        data = resp.json()
        assert data["total"] == 3  # Eros (16.84), Toutatis (2.45), Itokawa (0.33)

    def test_pagination_limit(self, client):
        resp = _get(client, limit=2)
        data = resp.json()
        assert len(data["data"]) == 2
        assert data["total"] == 5
        assert data["limit"] == 2

    def test_pagination_offset(self, client):
        resp = _get(client, limit=2, offset=3)
        data = resp.json()
        assert len(data["data"]) == 2
        assert data["offset"] == 3

    def test_empty_results(self, client):
        resp = _get(client, q="NonExistent")
        data = resp.json()
        assert data["data"] == []
        assert data["total"] == 0

    def test_response_fields(self, client):
        resp = _get(client, limit=1)
        item = resp.json()["data"][0]
        expected = ["asteroid_id", "name", "designation", "neo", "pha",
                     "a", "e", "i", "moid", "diameter_km",
                     "taxonomy_class", "taxonomy_prob"]
        for field in expected:
            assert field in item, f"Missing field: {field}"
