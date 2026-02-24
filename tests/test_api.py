"""Tests for prospector.api — FastAPI app skeleton and auth middleware."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from fastapi.testclient import TestClient

from prospector.api.app import create_app, _db_conn, get_db
from prospector.api.auth import _load_api_keys, get_api_key
from prospector.db import get_connection


@pytest.fixture
def db_conn():
    """In-memory database with schema, thread-safe for TestClient."""
    import sqlite3

    # check_same_thread=False needed because TestClient runs in worker thread
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    from prospector.db import init_schema

    init_schema(conn)
    return conn


@pytest.fixture
def app(db_conn):
    """FastAPI app with test database injected."""
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db_conn
    return app


@pytest.fixture
def client(app):
    """Test client with no auth."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def authed_client(app):
    """Test client with valid API key."""
    with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "test-key-123"}):
        return TestClient(app, raise_server_exceptions=False)


class TestLoadApiKeys:
    """Test API key loading from environment."""

    def test_no_env_var(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PROSPECTOR_API_KEYS", None)
            keys = _load_api_keys()
            assert keys == set()

    def test_single_key(self):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "my-key"}):
            keys = _load_api_keys()
            assert keys == {"my-key"}

    def test_multiple_keys(self):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key1,key2,key3"}):
            keys = _load_api_keys()
            assert keys == {"key1", "key2", "key3"}

    def test_strips_whitespace(self):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": " key1 , key2 "}):
            keys = _load_api_keys()
            assert keys == {"key1", "key2"}

    def test_ignores_empty_entries(self):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key1,,key2,"}):
            keys = _load_api_keys()
            assert keys == {"key1", "key2"}


class TestHealthEndpoint:
    """Test /health endpoint (unauthenticated)."""

    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "asteroids_count" in data
        assert "scores_count" in data

    def test_health_empty_db(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["asteroids_count"] == 0
        assert data["scores_count"] == 0

    def test_health_with_data(self, db_conn):
        db_conn.execute(
            "INSERT INTO asteroids (asteroid_id, full_name) VALUES (1, 'Ceres')"
        )
        db_conn.commit()
        app = create_app()
        app.dependency_overrides[get_db] = lambda: db_conn
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        data = resp.json()
        assert data["asteroids_count"] == 1

    def test_health_no_auth_required(self, client):
        """Health endpoint should work without any API key."""
        resp = client.get("/health")
        assert resp.status_code == 200


class TestAuthMiddleware:
    """Test API key authentication on /v1/* endpoints."""

    def test_no_key_returns_401(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "valid-key"}):
            resp = client.get("/v1/status")
            assert resp.status_code == 401
            assert resp.json()["detail"] == "Invalid or missing API key"

    def test_invalid_key_returns_401(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "valid-key"}):
            resp = client.get("/v1/status", headers={"X-API-Key": "wrong-key"})
            assert resp.status_code == 401

    def test_valid_header_key(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "valid-key"}):
            resp = client.get("/v1/status", headers={"X-API-Key": "valid-key"})
            assert resp.status_code == 200

    def test_valid_query_param_key(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "valid-key"}):
            resp = client.get("/v1/status?api_key=valid-key")
            assert resp.status_code == 200

    def test_header_takes_precedence(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "header-key"}):
            resp = client.get(
                "/v1/status?api_key=wrong-key",
                headers={"X-API-Key": "header-key"},
            )
            assert resp.status_code == 200

    def test_no_configured_keys_returns_401(self, client):
        """If no keys are configured at all, reject everything."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PROSPECTOR_API_KEYS", None)
            resp = client.get("/v1/status", headers={"X-API-Key": "any-key"})
            assert resp.status_code == 401

    def test_v1_status_response(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "test-key"}):
            resp = client.get("/v1/status", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["version"] == "v1"


class TestAppFactory:
    """Test create_app() factory."""

    def test_creates_app(self):
        app = create_app()
        assert app.title == "Asteroid Mining Prospector API"

    def test_openapi_schema_exists(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/health" in schema["paths"]

    def test_docs_endpoint(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
