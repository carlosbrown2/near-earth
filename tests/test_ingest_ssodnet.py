"""Tests for SsODNet supplementary data ingestion."""

import json

import pytest

from prospector.db import get_connection
from prospector.ingest.ssodnet import (
    _extract_error,
    _has_any_data,
    _safe_float,
    fetch_ssocard,
    ingest_ssodnet,
    ingest_ssodnet_json,
    parse_ssocard,
)


# --- Sample SsODNet ssoCard data (based on real API response for 433 Eros) ---

EROS_CARD = {
    "id": "Eros",
    "title": "(433) Eros",
    "name": "Eros",
    "number": 433,
    "type": "Asteroid",
    "class": "NEA>Amor",
    "parameters": {
        "physical": {
            "mass": {
                "value": 6.687e15,
                "error": {"min": -3e12, "max": 3e12},
            },
            "density": {
                "value": 2650.1266,
                "error": {"min": -148.98, "max": 148.98},
            },
            "thermal_inertia": {
                "value": 260.935,
                "error": {"min": -53.832, "max": 53.832},
            },
            "taxonomy": {
                "class": "S",
                "complex": "S",
                "technique": "Spec",
                "waverange": "VISNIR",
                "scheme": "Mahlke",
            },
        },
        "dynamical": {
            "delta_v": {
                "delta_v": 7.643,
                "transfer_time": 299.766,
            },
        },
    },
}

BENNU_CARD = {
    "id": "Bennu",
    "number": 101955,
    "parameters": {
        "physical": {
            "mass": {
                "value": 7.329e10,
                "error": {"min": -9e6, "max": 9e6},
            },
            "density": {
                "value": 1190.0,
                "error": {"min": -13.0, "max": 13.0},
            },
            "taxonomy": {
                "class": "B",
                "complex": "C",
                "scheme": "Mahlke",
                "waverange": "VISNIR",
            },
        },
        "dynamical": {},
    },
}

EMPTY_CARD = {
    "id": "Unknown",
    "number": 99999,
    "parameters": {"physical": {}, "dynamical": {}},
}


# --- Unit tests for helpers ---


class TestSafeFloat:
    def test_plain_number(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_dict_with_value(self):
        assert _safe_float({"value": 42.0}) == pytest.approx(42.0)

    def test_none(self):
        assert _safe_float(None) is None

    def test_dict_without_value(self):
        assert _safe_float({"error": 1.0}) is None

    def test_string_number(self):
        assert _safe_float("1.5") == pytest.approx(1.5)

    def test_invalid_string(self):
        assert _safe_float("not_a_number") is None

    def test_nan(self):
        assert _safe_float(float("nan")) is None


class TestExtractError:
    def test_symmetric_error(self):
        obj = {"value": 100, "error": {"min": -5, "max": 5}}
        assert _extract_error(obj) == pytest.approx(5.0)

    def test_asymmetric_error(self):
        obj = {"value": 100, "error": {"min": -3, "max": 7}}
        assert _extract_error(obj) == pytest.approx(5.0)

    def test_no_error(self):
        assert _extract_error({"value": 100}) is None

    def test_not_dict(self):
        assert _extract_error(42) is None

    def test_none(self):
        assert _extract_error(None) is None


class TestHasAnyData:
    def test_all_none(self):
        assert _has_any_data({"a": None, "b": None}) is False

    def test_some_data(self):
        assert _has_any_data({"a": None, "b": 1.0}) is True


# --- Parse tests ---


class TestParseSsocard:
    def test_eros_mass(self):
        props = parse_ssocard(EROS_CARD)
        assert props["mass_kg"] == pytest.approx(6.687e15)
        assert props["mass_unc"] == pytest.approx(3e12)

    def test_eros_density(self):
        props = parse_ssocard(EROS_CARD)
        assert props["density_kgm3"] == pytest.approx(2650.1266)
        assert props["density_unc"] == pytest.approx(148.98)

    def test_eros_thermal_inertia(self):
        props = parse_ssocard(EROS_CARD)
        assert props["thermal_inertia"] == pytest.approx(260.935)
        assert props["thermal_inertia_unc"] == pytest.approx(53.832)

    def test_eros_taxonomy(self):
        props = parse_ssocard(EROS_CARD)
        assert props["taxonomy_class"] == "S"
        assert props["taxonomy_scheme"] == "Mahlke"
        assert props["taxonomy_complex"] == "S"
        assert props["taxonomy_waverange"] == "VISNIR"

    def test_eros_delta_v(self):
        props = parse_ssocard(EROS_CARD)
        assert props["delta_v_km_s"] == pytest.approx(7.643)

    def test_bennu_no_delta_v(self):
        props = parse_ssocard(BENNU_CARD)
        assert props["delta_v_km_s"] is None
        assert props["taxonomy_class"] == "B"
        assert props["taxonomy_complex"] == "C"

    def test_empty_card(self):
        props = parse_ssocard(EMPTY_CARD)
        assert not _has_any_data(props)

    def test_missing_parameters(self):
        props = parse_ssocard({})
        assert not _has_any_data(props)


# --- Fixtures ---


@pytest.fixture
def db_conn():
    conn = get_connection(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def db_with_asteroids(db_conn):
    asteroids = [
        (433, "1898 DQ", "Eros", "433 Eros (1898 DQ)", 1, 0),
        (101955, "1999 RQ36", "Bennu", "101955 Bennu (1999 RQ36)", 1, 1),
        (25143, "1998 SF36", "Itokawa", "25143 Itokawa (1998 SF36)", 1, 0),
    ]
    db_conn.executemany(
        "INSERT INTO asteroids (asteroid_id, designation, name, full_name, neo, pha) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        asteroids,
    )
    db_conn.commit()
    return db_conn


# --- JSON file ingestion tests ---


class TestIngestSsodnetJson:
    def test_single_card(self, db_with_asteroids, tmp_path):
        json_file = tmp_path / "eros.json"
        json_file.write_text(json.dumps(EROS_CARD))
        count = ingest_ssodnet_json(json_file, db_with_asteroids)
        assert count == 1

        row = db_with_asteroids.execute(
            "SELECT mass_kg, density_kgm3, thermal_inertia, taxonomy_class, "
            "delta_v_km_s, source FROM ssodnet_properties WHERE asteroid_id = 433"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(6.687e15)
        assert row[1] == pytest.approx(2650.1266)
        assert row[2] == pytest.approx(260.935)
        assert row[3] == "S"
        assert row[4] == pytest.approx(7.643)
        assert row[5] == "SsODNet"

    def test_list_of_cards(self, db_with_asteroids, tmp_path):
        json_file = tmp_path / "batch.json"
        json_file.write_text(json.dumps([EROS_CARD, BENNU_CARD]))
        count = ingest_ssodnet_json(json_file, db_with_asteroids)
        assert count == 2

    def test_explicit_asteroid_id(self, db_with_asteroids, tmp_path):
        card = {**EROS_CARD, "number": None}  # no number in JSON
        json_file = tmp_path / "override.json"
        json_file.write_text(json.dumps(card))
        count = ingest_ssodnet_json(json_file, db_with_asteroids, asteroid_id=433)
        assert count == 1

    def test_unknown_asteroid_skipped(self, db_with_asteroids, tmp_path):
        card = {**EROS_CARD, "number": 99999}
        json_file = tmp_path / "unknown.json"
        json_file.write_text(json.dumps(card))
        count = ingest_ssodnet_json(json_file, db_with_asteroids)
        assert count == 0

    def test_empty_card_skipped(self, db_with_asteroids, tmp_path):
        card = {**EMPTY_CARD, "number": 433}
        json_file = tmp_path / "empty.json"
        json_file.write_text(json.dumps(card))
        count = ingest_ssodnet_json(json_file, db_with_asteroids)
        assert count == 0

    def test_file_not_found(self, db_with_asteroids):
        with pytest.raises(FileNotFoundError):
            ingest_ssodnet_json("/nonexistent.json", db_with_asteroids)

    def test_idempotent_reingest(self, db_with_asteroids, tmp_path):
        json_file = tmp_path / "eros.json"
        json_file.write_text(json.dumps(EROS_CARD))
        ingest_ssodnet_json(json_file, db_with_asteroids)
        ingest_ssodnet_json(json_file, db_with_asteroids)
        count = db_with_asteroids.execute(
            "SELECT COUNT(*) FROM ssodnet_properties"
        ).fetchone()[0]
        assert count == 1

    def test_no_number_no_override_skipped(self, db_with_asteroids, tmp_path):
        card = {"parameters": {"physical": {"mass": {"value": 1e15}}, "dynamical": {}}}
        json_file = tmp_path / "no_num.json"
        json_file.write_text(json.dumps(card))
        count = ingest_ssodnet_json(json_file, db_with_asteroids)
        assert count == 0

    def test_uncertainty_stored(self, db_with_asteroids, tmp_path):
        json_file = tmp_path / "eros.json"
        json_file.write_text(json.dumps(EROS_CARD))
        ingest_ssodnet_json(json_file, db_with_asteroids)
        row = db_with_asteroids.execute(
            "SELECT mass_unc, density_unc, thermal_inertia_unc "
            "FROM ssodnet_properties WHERE asteroid_id = 433"
        ).fetchone()
        assert row[0] == pytest.approx(3e12)
        assert row[1] == pytest.approx(148.98)
        assert row[2] == pytest.approx(53.832)

    def test_bennu_properties(self, db_with_asteroids, tmp_path):
        json_file = tmp_path / "bennu.json"
        json_file.write_text(json.dumps(BENNU_CARD))
        ingest_ssodnet_json(json_file, db_with_asteroids)
        row = db_with_asteroids.execute(
            "SELECT taxonomy_class, taxonomy_complex, density_kgm3, delta_v_km_s "
            "FROM ssodnet_properties WHERE asteroid_id = 101955"
        ).fetchone()
        assert row[0] == "B"
        assert row[1] == "C"
        assert row[2] == pytest.approx(1190.0)
        assert row[3] is None  # no delta-v for Bennu in our test data


# --- Live API tests (mocked) ---


class TestIngestSsodnetApi:
    def test_ingest_with_mock(self, db_with_asteroids, monkeypatch):
        """Test API-based ingestion with mocked fetch_ssocard."""
        call_log = []

        def mock_fetch(identifier, *, timeout=30):
            call_log.append(identifier)
            if identifier == 433:
                return EROS_CARD
            if identifier == 101955:
                return BENNU_CARD
            return None

        monkeypatch.setattr(
            "prospector.ingest.ssodnet.fetch_ssocard", mock_fetch
        )
        count = ingest_ssodnet([433, 101955, 25143], db_with_asteroids, delay=0)
        assert count == 2  # Eros + Bennu (Itokawa returns None from mock)
        assert 433 in call_log
        assert 101955 in call_log
        assert 25143 in call_log

    def test_unknown_ids_skipped(self, db_with_asteroids, monkeypatch):
        """Asteroid IDs not in DB are skipped without API calls."""
        call_log = []

        def mock_fetch(identifier, *, timeout=30):
            call_log.append(identifier)
            return EROS_CARD

        monkeypatch.setattr(
            "prospector.ingest.ssodnet.fetch_ssocard", mock_fetch
        )
        count = ingest_ssodnet([99999], db_with_asteroids, delay=0)
        assert count == 0
        assert len(call_log) == 0  # no API call made

    def test_api_failure_skipped(self, db_with_asteroids, monkeypatch):
        """API failures are logged and skipped gracefully."""

        def mock_fetch(identifier, *, timeout=30):
            return None

        monkeypatch.setattr(
            "prospector.ingest.ssodnet.fetch_ssocard", mock_fetch
        )
        count = ingest_ssodnet([433], db_with_asteroids, delay=0)
        assert count == 0


# --- Schema test ---


class TestSsodnetSchema:
    def test_table_exists(self, db_conn):
        tables = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ssodnet_properties'"
        ).fetchall()
        assert len(tables) == 1

    def test_columns(self, db_conn):
        cols = {
            row[1]
            for row in db_conn.execute("PRAGMA table_info(ssodnet_properties)").fetchall()
        }
        expected = {
            "asteroid_id", "mass_kg", "mass_unc", "density_kgm3", "density_unc",
            "thermal_inertia", "thermal_inertia_unc", "taxonomy_class",
            "taxonomy_scheme", "taxonomy_complex", "taxonomy_waverange",
            "delta_v_km_s", "source",
        }
        assert expected == cols
