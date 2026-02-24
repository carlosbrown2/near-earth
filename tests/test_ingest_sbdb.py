"""Tests for SBDB bulk CSV ingestion."""

import textwrap
from pathlib import Path

import pytest

from prospector.db import get_connection
from prospector.ingest.sbdb import (
    _flag_to_int,
    _safe_float,
    ingest_sbdb_csv,
    parse_full_name,
    spkid_to_asteroid_id,
)


# --- Unit tests for helper functions ---


class TestSpkidMapping:
    def test_numbered_asteroid(self):
        # SPK-ID 2004179 → asteroid 4179 (Toutatis)
        assert spkid_to_asteroid_id(2_004_179) == 4179

    def test_numbered_asteroid_one(self):
        # SPK-ID 2000001 → asteroid 1 (Ceres)
        assert spkid_to_asteroid_id(2_000_001) == 1

    def test_unnumbered_asteroid(self):
        # SPK-IDs outside 2000001-2999999 pass through unchanged
        assert spkid_to_asteroid_id(3_500_000) == 3_500_000

    def test_boundary_low(self):
        # Exactly 2000000 is NOT a valid numbered asteroid
        assert spkid_to_asteroid_id(2_000_000) == 2_000_000

    def test_boundary_high(self):
        # 2999999 is the last valid numbered asteroid
        assert spkid_to_asteroid_id(2_999_999) == 999_999


class TestParseFullName:
    def test_numbered_with_name_and_designation(self):
        result = parse_full_name("  4179 Toutatis (1989 FB)")
        assert result["number"] == "4179"
        assert result["name"] == "Toutatis"
        assert result["designation"] == "1989 FB"

    def test_numbered_with_name_only(self):
        result = parse_full_name("     1 Ceres")
        assert result["number"] == "1"
        assert result["name"] == "Ceres"
        assert result["designation"] is None

    def test_unnumbered_designation_only(self):
        result = parse_full_name("       (2024 YR4)")
        assert result["number"] is None
        assert result["name"] is None
        assert result["designation"] == "2024 YR4"

    def test_numbered_with_designation_no_name(self):
        result = parse_full_name("101955 (1999 RQ36)")
        assert result["number"] == "101955"
        assert result["designation"] == "1999 RQ36"

    def test_empty_string(self):
        result = parse_full_name("")
        assert result["number"] is None
        assert result["name"] is None
        assert result["designation"] is None

    def test_none_input(self):
        result = parse_full_name(None)
        assert result["number"] is None


class TestFlagToInt:
    def test_y(self):
        assert _flag_to_int("Y") == 1

    def test_n(self):
        assert _flag_to_int("N") == 0

    def test_none(self):
        assert _flag_to_int(None) is None

    def test_nan(self):
        assert _flag_to_int(float("nan")) is None

    def test_lowercase(self):
        assert _flag_to_int("y") == 1


class TestSafeFloat:
    def test_valid(self):
        assert _safe_float("1.234") == 1.234

    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_none(self):
        assert _safe_float(None) is None

    def test_nan(self):
        assert _safe_float(float("nan")) is None

    def test_empty_string(self):
        assert _safe_float("") is None


# --- Integration tests for CSV ingestion ---


@pytest.fixture
def sample_csv(tmp_path):
    """Create a minimal SBDB-format CSV for testing."""
    csv_content = textwrap.dedent("""\
        spkid,full_name,neo,pha,H,diameter,diameter_sigma,albedo,e,a,i,om,w,ma,epoch,q,moid
        2000433,433 Eros (1898 DQ),Y,N,11.16,16.84,0.06,0.25,0.2226,1.458,10.83,304.3,178.9,247.8,2460600.5,1.133,0.149
        2004179,4179 Toutatis (1989 FB),Y,Y,15.3,2.45,,0.13,0.6297,2.510,0.45,124.4,274.8,15.2,2460600.5,0.929,0.006
        2000001,1 Ceres,N,N,3.53,939.4,0.2,0.09,0.0758,2.769,10.59,80.3,73.6,291.4,2460600.5,2.559,
        3500001,(2024 YR4),Y,N,27.6,,,,,1.002,3.41,,,,,0.002
    """)
    csv_file = tmp_path / "test_sbdb.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def db_conn():
    """In-memory database connection for testing."""
    conn = get_connection(":memory:")
    yield conn
    conn.close()


class TestIngestSbdbCsv:
    def test_basic_ingest(self, sample_csv, db_conn):
        count = ingest_sbdb_csv(sample_csv, db_conn)
        assert count == 4

    def test_asteroids_table_populated(self, sample_csv, db_conn):
        ingest_sbdb_csv(sample_csv, db_conn)
        rows = db_conn.execute("SELECT * FROM asteroids ORDER BY asteroid_id").fetchall()
        assert len(rows) == 4

    def test_eros_identity(self, sample_csv, db_conn):
        """Verify Eros (433) is correctly parsed."""
        ingest_sbdb_csv(sample_csv, db_conn)
        row = db_conn.execute(
            "SELECT asteroid_id, name, designation, neo, pha FROM asteroids WHERE asteroid_id=433"
        ).fetchone()
        assert row is not None
        assert row[0] == 433  # asteroid_id from spkid 2000433
        assert row[1] == "Eros"  # name
        assert row[2] == "1898 DQ"  # designation
        assert row[3] == 1  # neo = Y → 1
        assert row[4] == 0  # pha = N → 0

    def test_toutatis_pha(self, sample_csv, db_conn):
        """Verify Toutatis is flagged as PHA."""
        ingest_sbdb_csv(sample_csv, db_conn)
        row = db_conn.execute(
            "SELECT neo, pha FROM asteroids WHERE asteroid_id=4179"
        ).fetchone()
        assert row == (1, 1)  # both Y

    def test_ceres_not_neo(self, sample_csv, db_conn):
        """Ceres is not a NEO."""
        ingest_sbdb_csv(sample_csv, db_conn)
        row = db_conn.execute(
            "SELECT neo, pha FROM asteroids WHERE asteroid_id=1"
        ).fetchone()
        assert row == (0, 0)

    def test_orbits_populated(self, sample_csv, db_conn):
        ingest_sbdb_csv(sample_csv, db_conn)
        rows = db_conn.execute("SELECT * FROM orbits ORDER BY asteroid_id").fetchall()
        assert len(rows) == 4

    def test_eros_orbital_elements(self, sample_csv, db_conn):
        """Verify Eros orbital elements are correctly stored."""
        ingest_sbdb_csv(sample_csv, db_conn)
        row = db_conn.execute(
            "SELECT e, a, i, H, diameter, moid FROM orbits WHERE asteroid_id=433"
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 0.2226) < 1e-4  # eccentricity
        assert abs(row[1] - 1.458) < 1e-3  # semi-major axis
        assert abs(row[2] - 10.83) < 1e-2  # inclination
        assert abs(row[3] - 11.16) < 1e-2  # H magnitude
        assert abs(row[4] - 16.84) < 1e-2  # diameter
        assert abs(row[5] - 0.149) < 1e-3  # MOID

    def test_missing_fields_null(self, sample_csv, db_conn):
        """Missing fields should be stored as NULL."""
        ingest_sbdb_csv(sample_csv, db_conn)
        # Toutatis has no diameter_sigma
        row = db_conn.execute(
            "SELECT diameter_sigma FROM orbits WHERE asteroid_id=4179"
        ).fetchone()
        assert row[0] is None

        # Ceres has no moid
        row = db_conn.execute(
            "SELECT moid FROM orbits WHERE asteroid_id=1"
        ).fetchone()
        assert row[0] is None

    def test_unnumbered_asteroid(self, sample_csv, db_conn):
        """Unnumbered object (2024 YR4) uses spkid directly as asteroid_id."""
        ingest_sbdb_csv(sample_csv, db_conn)
        row = db_conn.execute(
            "SELECT asteroid_id, designation, name FROM asteroids WHERE asteroid_id=3500001"
        ).fetchone()
        assert row is not None
        assert row[1] == "2024 YR4"  # designation
        assert row[2] is None  # no name

    def test_neo_only_filter(self, sample_csv, db_conn):
        """neo_only=True should skip Ceres."""
        count = ingest_sbdb_csv(sample_csv, db_conn, neo_only=True)
        assert count == 3  # Eros, Toutatis, 2024 YR4 — not Ceres
        row = db_conn.execute(
            "SELECT * FROM asteroids WHERE asteroid_id=1"
        ).fetchone()
        assert row is None  # Ceres not ingested

    def test_idempotent_reingest(self, sample_csv, db_conn):
        """Re-ingesting the same CSV should not create duplicates (INSERT OR REPLACE)."""
        ingest_sbdb_csv(sample_csv, db_conn)
        ingest_sbdb_csv(sample_csv, db_conn)
        count = db_conn.execute("SELECT COUNT(*) FROM asteroids").fetchone()[0]
        assert count == 4  # no duplicates

    def test_file_not_found(self, db_conn):
        with pytest.raises(FileNotFoundError):
            ingest_sbdb_csv("/nonexistent/path.csv", db_conn)

    def test_full_name_stored(self, sample_csv, db_conn):
        """full_name field should be stored as-is."""
        ingest_sbdb_csv(sample_csv, db_conn)
        row = db_conn.execute(
            "SELECT full_name FROM asteroids WHERE asteroid_id=433"
        ).fetchone()
        assert row[0] == "433 Eros (1898 DQ)"
