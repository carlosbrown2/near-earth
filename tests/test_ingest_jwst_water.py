"""Tests for JWST 6μm molecular water detection ingestion."""

import json

import pytest

from prospector.db import get_connection
from prospector.ingest.jwst_water import (
    _lookup_designation,
    _parse_bool,
    _resolve_columns,
    _safe_float,
    _safe_int,
    ingest_jwst_water,
    ingest_jwst_water_json,
    DEFAULT_COLUMN_MAP,
)


# --- Unit tests for helpers ---


class TestSafeFloat:
    def test_plain_number(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_none(self):
        assert _safe_float(None) is None

    def test_string_number(self):
        assert _safe_float("1.5") == pytest.approx(1.5)

    def test_invalid_string(self):
        assert _safe_float("not_a_number") is None

    def test_nan(self):
        import math
        assert _safe_float(float("nan")) is None

    def test_int_input(self):
        assert _safe_float(42) == pytest.approx(42.0)


class TestSafeInt:
    def test_plain_int(self):
        assert _safe_int(433) == 433

    def test_float_to_int(self):
        assert _safe_int(433.0) == 433

    def test_string_int(self):
        assert _safe_int("433") == 433

    def test_none(self):
        assert _safe_int(None) is None

    def test_invalid_string(self):
        assert _safe_int("abc") is None

    def test_nan(self):
        assert _safe_int(float("nan")) is None


class TestParseBool:
    def test_true_values(self):
        for val in (True, 1, "true", "yes", "1", "y", "confirmed"):
            assert _parse_bool(val) is True, f"Expected True for {val!r}"

    def test_false_values(self):
        for val in (False, 0, "false", "no", "0", "n", "unconfirmed"):
            assert _parse_bool(val) is False, f"Expected False for {val!r}"

    def test_none(self):
        assert _parse_bool(None) is None

    def test_nan(self):
        assert _parse_bool(float("nan")) is None

    def test_unknown_string(self):
        assert _parse_bool("maybe") is None


class TestResolveColumns:
    def test_exact_match(self):
        df_cols = ["number", "designation", "detection", "water_abundance"]
        resolved = _resolve_columns(df_cols, DEFAULT_COLUMN_MAP)
        assert resolved["number"] == "number"
        assert resolved["detection"] == "detection"

    def test_alias_match(self):
        df_cols = ["num", "name", "detected", "abundance"]
        resolved = _resolve_columns(df_cols, DEFAULT_COLUMN_MAP)
        assert resolved["number"] == "num"
        assert resolved["designation"] == "name"
        assert resolved["detection"] == "detected"
        assert resolved["water_abundance"] == "abundance"

    def test_case_insensitive(self):
        df_cols = ["Number", "Detection", "WATER_ABUNDANCE"]
        resolved = _resolve_columns(df_cols, DEFAULT_COLUMN_MAP)
        assert resolved["number"] == "Number"
        assert resolved["detection"] == "Detection"

    def test_missing_column(self):
        df_cols = ["number", "detection"]
        resolved = _resolve_columns(df_cols, DEFAULT_COLUMN_MAP)
        assert resolved["water_abundance"] is None
        assert resolved["instrument"] is None


# --- Fixtures ---


@pytest.fixture
def db_conn():
    conn = get_connection(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def db_with_asteroids(db_conn):
    asteroids = [
        (7, "1958 PK", "Iris", "7 Iris", 0, 0),
        (20, "1892 CD", "Massalia", "20 Massalia", 0, 0),
        (101955, "1999 RQ36", "Bennu", "101955 Bennu (1999 RQ36)", 1, 1),
        (433, "1898 DQ", "Eros", "433 Eros (1898 DQ)", 1, 0),
    ]
    db_conn.executemany(
        "INSERT INTO asteroids (asteroid_id, designation, name, full_name, neo, pha) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        asteroids,
    )
    db_conn.commit()
    return db_conn


# --- CSV ingestion tests ---


class TestIngestJwstWaterCsv:
    def test_basic_ingest(self, db_with_asteroids, tmp_path):
        csv_file = tmp_path / "water.csv"
        csv_file.write_text(
            "number,designation,detection,water_abundance,water_abundance_unc,"
            "band_depth,band_depth_unc,instrument,reference\n"
            "7,Iris,true,450.0,90.0,0.05,0.01,JWST/MIRI,Arredondo+2024\n"
            "20,Massalia,true,350.0,70.0,0.04,0.008,JWST/MIRI,Arredondo+2024\n"
        )
        count = ingest_jwst_water(csv_file, db_with_asteroids)
        assert count == 2

        row = db_with_asteroids.execute(
            "SELECT water_abundance, water_abundance_unc, band_depth, band_depth_unc, "
            "detection, instrument, reference, source "
            "FROM jwst_water WHERE asteroid_id = 7"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(450.0)
        assert row[1] == pytest.approx(90.0)
        assert row[2] == pytest.approx(0.05)
        assert row[3] == pytest.approx(0.01)
        assert row[4] == 1  # SQLite stores booleans as int
        assert row[5] == "JWST/MIRI"
        assert row[6] == "Arredondo+2024"
        assert row[7] == "JWST"

    def test_designation_lookup(self, db_with_asteroids, tmp_path):
        """When number is missing, resolve via designation."""
        csv_file = tmp_path / "water.csv"
        csv_file.write_text(
            "designation,detection,water_abundance\n"
            "1898 DQ,true,100.0\n"
        )
        count = ingest_jwst_water(csv_file, db_with_asteroids)
        assert count == 1
        row = db_with_asteroids.execute(
            "SELECT water_abundance FROM jwst_water WHERE asteroid_id = 433"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(100.0)

    def test_unknown_asteroid_skipped(self, db_with_asteroids, tmp_path):
        csv_file = tmp_path / "water.csv"
        csv_file.write_text("number,detection\n99999,true\n")
        count = ingest_jwst_water(csv_file, db_with_asteroids)
        assert count == 0

    def test_missing_detection_skipped(self, db_with_asteroids, tmp_path):
        csv_file = tmp_path / "water.csv"
        csv_file.write_text("number,detection\n7,\n")
        count = ingest_jwst_water(csv_file, db_with_asteroids)
        assert count == 0

    def test_non_detection_stored(self, db_with_asteroids, tmp_path):
        """Non-detections (detection=false) should still be stored."""
        csv_file = tmp_path / "water.csv"
        csv_file.write_text("number,detection\n7,false\n")
        count = ingest_jwst_water(csv_file, db_with_asteroids)
        assert count == 1
        row = db_with_asteroids.execute(
            "SELECT detection FROM jwst_water WHERE asteroid_id = 7"
        ).fetchone()
        assert row[0] == 0  # False stored as 0

    def test_alias_columns(self, db_with_asteroids, tmp_path):
        """Aliases like 'num', 'detected', 'h2o_ug_g' should work."""
        csv_file = tmp_path / "water.csv"
        csv_file.write_text(
            "num,detected,h2o_ug_g\n"
            "7,yes,450.0\n"
        )
        count = ingest_jwst_water(csv_file, db_with_asteroids)
        assert count == 1

    def test_optional_columns_nullable(self, db_with_asteroids, tmp_path):
        """Ingest works when only required columns present."""
        csv_file = tmp_path / "water.csv"
        csv_file.write_text("number,detection\n7,true\n")
        count = ingest_jwst_water(csv_file, db_with_asteroids)
        assert count == 1
        row = db_with_asteroids.execute(
            "SELECT water_abundance, band_depth, instrument, reference "
            "FROM jwst_water WHERE asteroid_id = 7"
        ).fetchone()
        assert row[0] is None
        assert row[1] is None
        assert row[2] is None
        assert row[3] is None

    def test_idempotent_reingest(self, db_with_asteroids, tmp_path):
        csv_file = tmp_path / "water.csv"
        csv_file.write_text("number,detection,water_abundance\n7,true,450.0\n")
        ingest_jwst_water(csv_file, db_with_asteroids)
        ingest_jwst_water(csv_file, db_with_asteroids)
        count = db_with_asteroids.execute(
            "SELECT COUNT(*) FROM jwst_water"
        ).fetchone()[0]
        assert count == 1

    def test_file_not_found(self, db_with_asteroids):
        with pytest.raises(FileNotFoundError):
            ingest_jwst_water("/nonexistent.csv", db_with_asteroids)

    def test_missing_number_and_designation_columns(self, db_with_asteroids, tmp_path):
        csv_file = tmp_path / "water.csv"
        csv_file.write_text("detection,water_abundance\ntrue,450.0\n")
        with pytest.raises(ValueError, match="number or designation"):
            ingest_jwst_water(csv_file, db_with_asteroids)

    def test_missing_detection_column(self, db_with_asteroids, tmp_path):
        csv_file = tmp_path / "water.csv"
        csv_file.write_text("number,water_abundance\n7,450.0\n")
        with pytest.raises(ValueError, match="detection"):
            ingest_jwst_water(csv_file, db_with_asteroids)

    def test_custom_column_map(self, db_with_asteroids, tmp_path):
        csv_file = tmp_path / "water.csv"
        csv_file.write_text("ast_num,det,h2o\n7,true,450.0\n")
        custom_map = {
            **DEFAULT_COLUMN_MAP,
            "number": "ast_num",
            "detection": "det",
            "water_abundance": "h2o",
        }
        count = ingest_jwst_water(csv_file, db_with_asteroids, column_map=custom_map)
        assert count == 1


# --- JSON ingestion tests ---


class TestIngestJwstWaterJson:
    def test_single_record(self, db_with_asteroids, tmp_path):
        json_file = tmp_path / "water.json"
        record = {
            "number": 7,
            "detection": True,
            "water_abundance": 450.0,
            "water_abundance_unc": 90.0,
            "band_depth": 0.05,
            "band_depth_unc": 0.01,
            "instrument": "JWST/MIRI",
            "reference": "Arredondo+2024",
        }
        json_file.write_text(json.dumps(record))
        count = ingest_jwst_water_json(json_file, db_with_asteroids)
        assert count == 1

        row = db_with_asteroids.execute(
            "SELECT water_abundance, detection, instrument, source "
            "FROM jwst_water WHERE asteroid_id = 7"
        ).fetchone()
        assert row[0] == pytest.approx(450.0)
        assert row[1] == 1
        assert row[2] == "JWST/MIRI"
        assert row[3] == "JWST"

    def test_list_of_records(self, db_with_asteroids, tmp_path):
        json_file = tmp_path / "water.json"
        records = [
            {"number": 7, "detection": True, "water_abundance": 450.0},
            {"number": 20, "detection": True, "water_abundance": 350.0},
        ]
        json_file.write_text(json.dumps(records))
        count = ingest_jwst_water_json(json_file, db_with_asteroids)
        assert count == 2

    def test_asteroid_id_key(self, db_with_asteroids, tmp_path):
        """JSON records can use 'asteroid_id' instead of 'number'."""
        json_file = tmp_path / "water.json"
        record = {"asteroid_id": 7, "detection": True}
        json_file.write_text(json.dumps(record))
        count = ingest_jwst_water_json(json_file, db_with_asteroids)
        assert count == 1

    def test_designation_fallback(self, db_with_asteroids, tmp_path):
        json_file = tmp_path / "water.json"
        record = {"designation": "1898 DQ", "detection": True}
        json_file.write_text(json.dumps(record))
        count = ingest_jwst_water_json(json_file, db_with_asteroids)
        assert count == 1

    def test_unknown_asteroid_skipped(self, db_with_asteroids, tmp_path):
        json_file = tmp_path / "water.json"
        record = {"number": 99999, "detection": True}
        json_file.write_text(json.dumps(record))
        count = ingest_jwst_water_json(json_file, db_with_asteroids)
        assert count == 0

    def test_missing_detection_skipped(self, db_with_asteroids, tmp_path):
        json_file = tmp_path / "water.json"
        record = {"number": 7}
        json_file.write_text(json.dumps(record))
        count = ingest_jwst_water_json(json_file, db_with_asteroids)
        assert count == 0

    def test_file_not_found(self, db_with_asteroids):
        with pytest.raises(FileNotFoundError):
            ingest_jwst_water_json("/nonexistent.json", db_with_asteroids)

    def test_idempotent_reingest(self, db_with_asteroids, tmp_path):
        json_file = tmp_path / "water.json"
        record = {"number": 7, "detection": True, "water_abundance": 450.0}
        json_file.write_text(json.dumps(record))
        ingest_jwst_water_json(json_file, db_with_asteroids)
        ingest_jwst_water_json(json_file, db_with_asteroids)
        count = db_with_asteroids.execute(
            "SELECT COUNT(*) FROM jwst_water"
        ).fetchone()[0]
        assert count == 1


# --- Schema test ---


class TestJwstWaterSchema:
    def test_table_exists(self, db_conn):
        tables = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='jwst_water'"
        ).fetchall()
        assert len(tables) == 1

    def test_columns(self, db_conn):
        cols = {
            row[1]
            for row in db_conn.execute("PRAGMA table_info(jwst_water)").fetchall()
        }
        expected = {
            "asteroid_id", "water_abundance", "water_abundance_unc",
            "band_depth", "band_depth_unc", "detection",
            "instrument", "reference", "source",
        }
        assert expected == cols


# --- Lookup helper test ---


class TestLookupDesignation:
    def test_found(self, db_with_asteroids):
        assert _lookup_designation(db_with_asteroids, "1898 DQ") == 433

    def test_not_found(self, db_with_asteroids):
        assert _lookup_designation(db_with_asteroids, "2099 XX99") is None

    def test_none_input(self, db_with_asteroids):
        assert _lookup_designation(db_with_asteroids, None) is None

    def test_empty_string(self, db_with_asteroids):
        assert _lookup_designation(db_with_asteroids, "") is None
