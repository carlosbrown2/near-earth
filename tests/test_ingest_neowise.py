"""Tests for NEOWISE diameter/albedo ingestion."""

import textwrap
from pathlib import Path

import pytest

from prospector.db import get_connection
from prospector.ingest.neowise import (
    _lookup_designation,
    _resolve_columns,
    _safe_float,
    _safe_int,
    ingest_neowise,
)


# --- Unit tests for helper functions ---


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


class TestSafeInt:
    def test_valid_int(self):
        assert _safe_int(433) == 433

    def test_valid_float(self):
        assert _safe_int(433.0) == 433

    def test_valid_string(self):
        assert _safe_int("433") == 433

    def test_none(self):
        assert _safe_int(None) is None

    def test_nan(self):
        assert _safe_int(float("nan")) is None

    def test_empty_string(self):
        assert _safe_int("") is None


class TestResolveColumns:
    def test_exact_match(self):
        cols = ["number", "desig", "diameter", "pv", "beaming"]
        result = _resolve_columns(cols, {
            "number": "number",
            "designation": "desig",
            "diameter_km": "diameter",
            "albedo_pv": "pv",
            "albedo_pv_err": "pv_err",
            "beaming_eta": "beaming",
            "diameter_err": "diameter_err",
        })
        assert result["number"] == "number"
        assert result["designation"] == "desig"
        assert result["diameter_km"] == "diameter"
        assert result["albedo_pv"] == "pv"
        assert result["beaming_eta"] == "beaming"
        # Missing columns resolve to None
        assert result["albedo_pv_err"] is None
        assert result["diameter_err"] is None

    def test_case_insensitive(self):
        cols = ["Number", "DESIG", "Diameter"]
        result = _resolve_columns(cols, {
            "number": "number",
            "designation": "desig",
            "diameter_km": "diameter",
            "albedo_pv": "pv",
            "albedo_pv_err": "pv_err",
            "beaming_eta": "beaming",
            "diameter_err": "diameter_err",
        })
        assert result["number"] == "Number"
        assert result["designation"] == "DESIG"
        assert result["diameter_km"] == "Diameter"

    def test_alias_fallback(self):
        """When mapped name isn't found, falls back to aliases."""
        cols = ["iau_number", "prov_desig", "diam_km", "albedo", "eta"]
        result = _resolve_columns(cols, {
            "number": "number",
            "designation": "desig",
            "diameter_km": "diameter",
            "albedo_pv": "pv",
            "albedo_pv_err": "pv_err",
            "beaming_eta": "beaming",
            "diameter_err": "diameter_err",
        })
        assert result["number"] == "iau_number"
        assert result["designation"] == "prov_desig"
        assert result["diameter_km"] == "diam_km"
        assert result["albedo_pv"] == "albedo"
        assert result["beaming_eta"] == "eta"


# --- Fixtures ---


@pytest.fixture
def db_conn():
    """In-memory database connection for testing."""
    conn = get_connection(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def db_with_asteroids(db_conn):
    """Database pre-populated with asteroids for cross-matching."""
    asteroids = [
        (433, "1898 DQ", "Eros", "433 Eros (1898 DQ)", 1, 0),
        (4179, "1989 FB", "Toutatis", "4179 Toutatis (1989 FB)", 1, 1),
        (101955, "1999 RQ36", "Bennu", "101955 Bennu (1999 RQ36)", 1, 1),
        (25143, "1998 SF36", "Itokawa", "25143 Itokawa (1998 SF36)", 1, 0),
        (3500001, "2024 YR4", None, "(2024 YR4)", 1, 0),
    ]
    db_conn.executemany(
        "INSERT INTO asteroids (asteroid_id, designation, name, full_name, neo, pha) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        asteroids,
    )
    db_conn.commit()
    return db_conn


@pytest.fixture
def neowise_csv(tmp_path):
    """Create a minimal NEOWISE-format CSV for testing."""
    csv_content = textwrap.dedent("""\
        number,desig,diameter,diameter_err,pv,pv_err,beaming
        433,1898 DQ,16.84,0.06,0.25,0.02,1.1
        4179,1989 FB,2.45,0.1,0.13,0.01,1.2
        101955,1999 RQ36,0.49,0.02,0.045,0.005,1.05
        ,2024 YR4,0.01,0.005,0.15,0.03,
    """)
    csv_file = tmp_path / "neowise_test.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def neowise_csv_alt_columns(tmp_path):
    """CSV with alternative column names (aliases)."""
    csv_content = textwrap.dedent("""\
        iau_number,prov_desig,diam_km,d_sig,albedo,albedo_err,eta
        433,1898 DQ,16.84,0.06,0.25,0.02,1.1
        4179,1989 FB,2.45,0.1,0.13,0.01,1.2
    """)
    csv_file = tmp_path / "neowise_alt.csv"
    csv_file.write_text(csv_content)
    return csv_file


# --- Test designation lookup ---


class TestLookupDesignation:
    def test_found(self, db_with_asteroids):
        assert _lookup_designation(db_with_asteroids, "1898 DQ") == 433

    def test_not_found(self, db_with_asteroids):
        assert _lookup_designation(db_with_asteroids, "9999 ZZ") is None

    def test_none_input(self, db_with_asteroids):
        assert _lookup_designation(db_with_asteroids, None) is None

    def test_empty_string(self, db_with_asteroids):
        assert _lookup_designation(db_with_asteroids, "") is None

    def test_whitespace_stripped(self, db_with_asteroids):
        assert _lookup_designation(db_with_asteroids, "  1898 DQ  ") == 433

    def test_unnumbered(self, db_with_asteroids):
        assert _lookup_designation(db_with_asteroids, "2024 YR4") == 3500001


# --- Integration tests for CSV ingestion ---


class TestIngestNeowise:
    def test_basic_ingest(self, neowise_csv, db_with_asteroids):
        count = ingest_neowise(neowise_csv, db_with_asteroids)
        assert count == 4  # 3 by number + 1 by designation

    def test_physical_properties_populated(self, neowise_csv, db_with_asteroids):
        ingest_neowise(neowise_csv, db_with_asteroids)
        rows = db_with_asteroids.execute(
            "SELECT * FROM physical_properties ORDER BY asteroid_id"
        ).fetchall()
        assert len(rows) == 4

    def test_eros_properties(self, neowise_csv, db_with_asteroids):
        """Verify Eros (433) physical properties are correctly stored."""
        ingest_neowise(neowise_csv, db_with_asteroids)
        row = db_with_asteroids.execute(
            "SELECT albedo_pv, albedo_pv_err, diameter_km, diameter_err, beaming_eta, source "
            "FROM physical_properties WHERE asteroid_id = 433"
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 0.25) < 1e-4     # albedo
        assert abs(row[1] - 0.02) < 1e-4     # albedo_err
        assert abs(row[2] - 16.84) < 1e-2    # diameter
        assert abs(row[3] - 0.06) < 1e-4     # diameter_err
        assert abs(row[4] - 1.1) < 1e-2      # beaming
        assert row[5] == "NEOWISE"            # source

    def test_bennu_properties(self, neowise_csv, db_with_asteroids):
        """Verify Bennu (101955) small asteroid properties."""
        ingest_neowise(neowise_csv, db_with_asteroids)
        row = db_with_asteroids.execute(
            "SELECT diameter_km, albedo_pv FROM physical_properties WHERE asteroid_id = 101955"
        ).fetchone()
        assert abs(row[0] - 0.49) < 1e-2
        assert abs(row[1] - 0.045) < 1e-4

    def test_designation_fallback(self, neowise_csv, db_with_asteroids):
        """Unnumbered asteroid matched by designation lookup."""
        ingest_neowise(neowise_csv, db_with_asteroids)
        row = db_with_asteroids.execute(
            "SELECT diameter_km, albedo_pv, beaming_eta "
            "FROM physical_properties WHERE asteroid_id = 3500001"
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 0.01) < 1e-4      # diameter
        assert abs(row[1] - 0.15) < 1e-4      # albedo
        assert row[2] is None                   # beaming is missing

    def test_source_field(self, neowise_csv, db_with_asteroids):
        """All rows should have source='NEOWISE'."""
        ingest_neowise(neowise_csv, db_with_asteroids)
        rows = db_with_asteroids.execute(
            "SELECT DISTINCT source FROM physical_properties"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "NEOWISE"

    def test_idempotent_reingest(self, neowise_csv, db_with_asteroids):
        """Re-ingesting should not create duplicates (INSERT OR REPLACE)."""
        ingest_neowise(neowise_csv, db_with_asteroids)
        ingest_neowise(neowise_csv, db_with_asteroids)
        count = db_with_asteroids.execute(
            "SELECT COUNT(*) FROM physical_properties"
        ).fetchone()[0]
        assert count == 4

    def test_file_not_found(self, db_with_asteroids):
        with pytest.raises(FileNotFoundError):
            ingest_neowise("/nonexistent/path.tab", db_with_asteroids)

    def test_alt_column_names(self, neowise_csv_alt_columns, db_with_asteroids):
        """Alternative column names should be resolved via aliases."""
        count = ingest_neowise(neowise_csv_alt_columns, db_with_asteroids)
        assert count == 2
        row = db_with_asteroids.execute(
            "SELECT diameter_km, albedo_pv FROM physical_properties WHERE asteroid_id = 433"
        ).fetchone()
        assert abs(row[0] - 16.84) < 1e-2
        assert abs(row[1] - 0.25) < 1e-4

    def test_no_matching_asteroid_skipped(self, db_conn, tmp_path):
        """Rows with no matching asteroid in DB should be skipped."""
        csv_content = textwrap.dedent("""\
            number,desig,diameter,pv,beaming
            99999,,5.0,0.1,1.0
        """)
        csv_file = tmp_path / "neowise_nomatch.csv"
        csv_file.write_text(csv_content)
        # asteroid_id=99999 doesn't exist in DB — should be skipped (FK-safe)
        count = ingest_neowise(csv_file, db_conn)
        assert count == 0

    def test_rows_with_no_data_skipped(self, db_with_asteroids, tmp_path):
        """Rows where all property values are NULL should be skipped."""
        csv_content = textwrap.dedent("""\
            number,desig,diameter,pv,beaming
            433,1898 DQ,,,
        """)
        csv_file = tmp_path / "neowise_empty.csv"
        csv_file.write_text(csv_content)
        count = ingest_neowise(csv_file, db_with_asteroids)
        assert count == 0

    def test_custom_column_map(self, db_with_asteroids, tmp_path):
        """Custom column_map should override defaults."""
        csv_content = textwrap.dedent("""\
            ast_num,des,d,pv_val
            433,1898 DQ,16.84,0.25
        """)
        csv_file = tmp_path / "neowise_custom.csv"
        csv_file.write_text(csv_content)
        count = ingest_neowise(csv_file, db_with_asteroids, column_map={
            "number": "ast_num",
            "designation": "des",
            "diameter_km": "d",
            "diameter_err": "d_err",
            "albedo_pv": "pv_val",
            "albedo_pv_err": "pv_err",
            "beaming_eta": "beaming",
        })
        assert count == 1
        row = db_with_asteroids.execute(
            "SELECT diameter_km, albedo_pv FROM physical_properties WHERE asteroid_id = 433"
        ).fetchone()
        assert abs(row[0] - 16.84) < 1e-2
