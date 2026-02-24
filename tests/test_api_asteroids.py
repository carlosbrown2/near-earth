"""Tests for GET /v1/asteroids/{asteroid_id} endpoint."""

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
    """Database with a fully-populated asteroid."""
    db_conn.execute(
        "INSERT INTO asteroids (asteroid_id, name, designation, neo, pha) "
        "VALUES (433, 'Eros', '2024 ER1', 1, 0)"
    )
    db_conn.execute(
        "INSERT INTO orbits (asteroid_id, epoch, e, a, i, om, w, ma, q, H, moid, diameter, diameter_sigma) "
        "VALUES (433, 2460000.5, 0.223, 1.458, 10.83, 304.3, 178.8, 190.1, 1.133, 11.16, 0.149, 16.84, 0.06)"
    )
    db_conn.execute(
        "INSERT INTO physical_properties (asteroid_id, albedo_pv, albedo_pv_err, diameter_km, diameter_err, beaming_eta, source) "
        "VALUES (433, 0.25, 0.02, 16.84, 0.5, 1.0, 'NEOWISE')"
    )
    db_conn.execute(
        "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, classifier, input_coverage) "
        "VALUES (433, 'S', 0.85, 'classy_mahlke2022', 'vnir')"
    )
    db_conn.execute(
        "INSERT INTO band_analysis (asteroid_id, band1_center, band2_center, bar, ol_opx_ratio, fa_mol_pct, fs_mol_pct, gaffey_subtype, calibration) "
        "VALUES (433, 0.99, 1.95, 0.4, 0.65, 18.0, 15.0, 'S(IV)', 'dunn2010')"
    )
    db_conn.execute(
        "INSERT INTO rotation_properties (asteroid_id, rotation_period, period_unc, amplitude, quality_code, is_monolithic, is_binary_suspect) "
        "VALUES (433, 5.27, 0.001, 1.46, '3', 0, 0)"
    )
    db_conn.execute(
        "INSERT INTO scores (asteroid_id, composite_score, estimated_mass_kg, grade_estimate, target_material, unit_value, accessibility, score_mode) "
        "VALUES (433, 85.5, 6.7e15, 0.04, 'pgm', 420.0, 0.82, 'earth_return')"
    )
    # Second asteroid — minimal data
    db_conn.execute(
        "INSERT INTO asteroids (asteroid_id, name, designation, neo, pha) "
        "VALUES (1, 'Ceres', '2024 CE1', 0, 0)"
    )
    db_conn.commit()
    return db_conn


@pytest.fixture
def client(seeded_db):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: seeded_db
    return TestClient(app, raise_server_exceptions=False)


class TestAsteroidDetail:
    """Test GET /v1/asteroids/{asteroid_id}."""

    def test_requires_auth(self, client):
        resp = client.get("/v1/asteroids/433")
        assert resp.status_code == 401

    def test_returns_full_detail(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/asteroids/433", headers={"X-API-Key": "key"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["asteroid_id"] == 433
        assert data["name"] == "Eros"

    def test_orbit_section(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/asteroids/433", headers={"X-API-Key": "key"})
        orbit = resp.json()["orbit"]
        assert orbit is not None
        assert orbit["a"] == 1.458
        assert orbit["e"] == 0.223
        assert orbit["moid"] == 0.149

    def test_taxonomy_section(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/asteroids/433", headers={"X-API-Key": "key"})
        tax = resp.json()["taxonomy"]
        assert tax is not None
        assert tax["primary_class"] == "S"
        assert tax["primary_prob"] == 0.85

    def test_band_analysis_section(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/asteroids/433", headers={"X-API-Key": "key"})
        ba = resp.json()["band_analysis"]
        assert ba is not None
        assert ba["gaffey_subtype"] == "S(IV)"

    def test_score_section(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/asteroids/433", headers={"X-API-Key": "key"})
        score = resp.json()["score"]
        assert score is not None
        assert score["composite_score"] == 85.5

    def test_rotation_section(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/asteroids/433", headers={"X-API-Key": "key"})
        rot = resp.json()["rotation"]
        assert rot is not None
        assert rot["period"] == 5.27

    def test_404_for_unknown(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/asteroids/99999", headers={"X-API-Key": "key"})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Asteroid not found"

    def test_minimal_asteroid(self, client):
        """Asteroid with no orbit, taxonomy, scores, etc. returns null sections."""
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/asteroids/1", headers={"X-API-Key": "key"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["asteroid_id"] == 1
        assert data["name"] == "Ceres"
        assert data["orbit"] is None
        assert data["taxonomy"] is None
        assert data["score"] is None
        assert data["band_analysis"] is None

    def test_physical_properties_section(self, client):
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/asteroids/433", headers={"X-API-Key": "key"})
        pp = resp.json()["physical_properties"]
        assert pp is not None
        assert pp["albedo_pv"] == 0.25
        assert pp["source"] == "NEOWISE"

    def test_no_blob_fields(self, client):
        """Verify BLOB fields (prob_vector, wavelengths, etc.) are not exposed."""
        with patch.dict(os.environ, {"PROSPECTOR_API_KEYS": "key"}):
            resp = client.get("/v1/asteroids/433", headers={"X-API-Key": "key"})
        data = resp.json()
        # No BLOB fields should appear anywhere
        all_keys = str(data)
        assert "prob_vector" not in all_keys
        assert "wavelengths" not in all_keys
        assert "reflectance" not in all_keys
