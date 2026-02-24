"""Tests for LCDB rotation period ingestion and spin-barrier filter."""

import textwrap

import pytest

from prospector.db import get_connection
from prospector.ingest.lcdb import (
    BINARY_AMPLITUDE_THRESHOLD,
    BINARY_PERIOD_THRESHOLD,
    SPIN_BARRIER_PERIOD,
    _resolve_columns,
    _safe_float,
    _safe_int,
    classify_spin,
    ingest_lcdb,
)
from prospector.scoring.scorer import compute_spin_modifier


# --- Unit tests for helpers ---


class TestSafeFloat:
    def test_valid(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_none(self):
        assert _safe_float(None) is None

    def test_nan(self):
        assert _safe_float(float("nan")) is None

    def test_empty_string(self):
        assert _safe_float("") is None


class TestSafeInt:
    def test_valid(self):
        assert _safe_int(433) == 433

    def test_float(self):
        assert _safe_int(433.0) == 433

    def test_none(self):
        assert _safe_int(None) is None


class TestResolveColumns:
    def test_exact_match(self):
        cols = ["Num", "Name", "Desig", "Per", "U"]
        result = _resolve_columns(cols, {
            "number": "Num",
            "name": "Name",
            "designation": "Desig",
            "period": "Per",
            "period_unc": "PerErr",
            "amplitude": "AmpMax",
            "amplitude_unc": "AmpErr",
            "quality_code": "U",
        })
        assert result["number"] == "Num"
        assert result["period"] == "Per"
        assert result["quality_code"] == "U"
        assert result["amplitude"] is None  # not in df

    def test_alias_fallback(self):
        cols = ["iau_number", "rotation_period", "quality"]
        result = _resolve_columns(cols, {
            "number": "Num",
            "period": "Per",
            "quality_code": "U",
        })
        assert result["number"] == "iau_number"
        assert result["period"] == "rotation_period"
        assert result["quality_code"] == "quality"


# --- Spin classification tests ---


class TestClassifySpin:
    def test_fast_rotator_monolithic(self):
        is_mono, is_binary = classify_spin(1.5, 0.3)
        assert is_mono is True
        assert is_binary is False

    def test_slow_rotator_normal(self):
        is_mono, is_binary = classify_spin(8.0, 0.3)
        assert is_mono is False
        assert is_binary is False

    def test_slow_rotator_binary_suspect(self):
        is_mono, is_binary = classify_spin(10.0, 1.2)
        assert is_mono is False
        assert is_binary is True

    def test_none_period(self):
        is_mono, is_binary = classify_spin(None, 0.5)
        assert is_mono is None
        assert is_binary is None

    def test_none_amplitude(self):
        is_mono, is_binary = classify_spin(1.5, None)
        assert is_mono is True
        assert is_binary is None

    def test_boundary_period(self):
        # Exactly at barrier — not monolithic (must be strictly less)
        is_mono, _ = classify_spin(SPIN_BARRIER_PERIOD, 0.3)
        assert is_mono is False

    def test_zero_period(self):
        is_mono, is_binary = classify_spin(0.0, 0.5)
        assert is_mono is None
        assert is_binary is None

    def test_boundary_binary(self):
        # Just above thresholds
        is_mono, is_binary = classify_spin(
            BINARY_PERIOD_THRESHOLD + 0.1,
            BINARY_AMPLITUDE_THRESHOLD + 0.1,
        )
        assert is_binary is True


# --- Spin modifier tests ---


class TestComputeSpinModifier:
    def test_monolithic_bonus(self):
        mod = compute_spin_modifier(is_monolithic=True)
        assert mod == pytest.approx(1.15)

    def test_binary_penalty(self):
        mod = compute_spin_modifier(is_binary_suspect=True)
        assert mod == pytest.approx(0.80)

    def test_monolithic_overrides_binary(self):
        # If both are True, monolithic takes priority (fast rotator can't be binary)
        mod = compute_spin_modifier(is_monolithic=True, is_binary_suspect=True)
        assert mod == pytest.approx(1.15)

    def test_no_data(self):
        mod = compute_spin_modifier()
        assert mod == pytest.approx(1.0)

    def test_false_values(self):
        mod = compute_spin_modifier(is_monolithic=False, is_binary_suspect=False)
        assert mod == pytest.approx(1.0)


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
def lcdb_csv(tmp_path):
    """Create a minimal LCDB-format CSV for testing."""
    csv_content = textwrap.dedent("""\
        Num,Name,Desig,Per,PerErr,AmpMax,AmpErr,U
        433,Eros,1898 DQ,5.27,0.001,1.45,0.02,3
        4179,Toutatis,1989 FB,176.0,0.5,2.0,0.1,3
        101955,Bennu,1999 RQ36,4.296,0.002,0.09,0.01,3
        25143,Itokawa,1998 SF36,12.132,0.001,1.1,0.05,3
        ,,,1.8,0.01,0.5,0.1,2
    """)
    csv_file = tmp_path / "lcdb_test.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def lcdb_csv_designation_match(tmp_path):
    """LCDB CSV with designation-only match (no number)."""
    csv_content = textwrap.dedent("""\
        Num,Name,Desig,Per,PerErr,AmpMax,AmpErr,U
        ,,(2024 YR4),0.5,0.01,0.3,0.05,2
    """)
    # Note: designation must match what's in DB — "2024 YR4"
    csv_content = textwrap.dedent("""\
        Num,Name,Desig,Per,PerErr,AmpMax,AmpErr,U
        ,,2024 YR4,0.5,0.01,0.3,0.05,2
    """)
    csv_file = tmp_path / "lcdb_desig.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def lcdb_csv_alt_columns(tmp_path):
    """LCDB CSV with alternative column names."""
    csv_content = textwrap.dedent("""\
        iau_number,ast_name,prov_desig,rotation_period,per_err,amplitude,amp_err,quality
        433,Eros,1898 DQ,5.27,0.001,1.45,0.02,3
    """)
    csv_file = tmp_path / "lcdb_alt.csv"
    csv_file.write_text(csv_content)
    return csv_file


# --- Integration tests ---


class TestIngestLcdb:
    def test_basic_ingest(self, lcdb_csv, db_with_asteroids):
        count = ingest_lcdb(lcdb_csv, db_with_asteroids)
        assert count == 4  # 4 with matching number, 1 row has no number/desig

    def test_rotation_properties_populated(self, lcdb_csv, db_with_asteroids):
        ingest_lcdb(lcdb_csv, db_with_asteroids)
        rows = db_with_asteroids.execute(
            "SELECT * FROM rotation_properties ORDER BY asteroid_id"
        ).fetchall()
        assert len(rows) == 4

    def test_eros_properties(self, lcdb_csv, db_with_asteroids):
        """Eros: P=5.27 hr, Amp=1.45 — slow with large amplitude → binary suspect."""
        ingest_lcdb(lcdb_csv, db_with_asteroids)
        row = db_with_asteroids.execute(
            "SELECT rotation_period, period_unc, amplitude, quality_code, "
            "is_monolithic, is_binary_suspect, source "
            "FROM rotation_properties WHERE asteroid_id = 433"
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 5.27) < 1e-4       # period
        assert abs(row[1] - 0.001) < 1e-5      # period_unc
        assert abs(row[2] - 1.45) < 1e-4       # amplitude
        assert row[3] == "3"                     # quality
        assert row[4] == 0                       # not monolithic (P > 2.2)
        assert row[5] == 0                       # not binary suspect (P < 6.0)
        assert row[6] == "LCDB"                  # source

    def test_bennu_normal_rotator(self, lcdb_csv, db_with_asteroids):
        """Bennu: P=4.296 hr, Amp=0.09 — normal rotator."""
        ingest_lcdb(lcdb_csv, db_with_asteroids)
        row = db_with_asteroids.execute(
            "SELECT is_monolithic, is_binary_suspect "
            "FROM rotation_properties WHERE asteroid_id = 101955"
        ).fetchone()
        assert row[0] == 0  # not monolithic
        assert row[1] == 0  # not binary suspect

    def test_toutatis_slow_binary_suspect(self, lcdb_csv, db_with_asteroids):
        """Toutatis: P=176 hr, Amp=2.0 — very slow with high amplitude → binary suspect."""
        ingest_lcdb(lcdb_csv, db_with_asteroids)
        row = db_with_asteroids.execute(
            "SELECT is_monolithic, is_binary_suspect "
            "FROM rotation_properties WHERE asteroid_id = 4179"
        ).fetchone()
        assert row[0] == 0  # not monolithic
        assert row[1] == 1  # binary suspect

    def test_itokawa_slow_binary_suspect(self, lcdb_csv, db_with_asteroids):
        """Itokawa: P=12.132 hr, Amp=1.1 — slow with large amplitude → binary suspect."""
        ingest_lcdb(lcdb_csv, db_with_asteroids)
        row = db_with_asteroids.execute(
            "SELECT is_monolithic, is_binary_suspect "
            "FROM rotation_properties WHERE asteroid_id = 25143"
        ).fetchone()
        assert row[0] == 0  # not monolithic
        assert row[1] == 1  # binary suspect (P>6, Amp>0.7)

    def test_designation_fallback(self, lcdb_csv_designation_match, db_with_asteroids):
        """Unnumbered asteroid matched by designation."""
        count = ingest_lcdb(lcdb_csv_designation_match, db_with_asteroids)
        assert count == 1
        row = db_with_asteroids.execute(
            "SELECT rotation_period, is_monolithic "
            "FROM rotation_properties WHERE asteroid_id = 3500001"
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 0.5) < 1e-4
        assert row[1] == 1  # P < 2.2 → monolithic

    def test_idempotent_reingest(self, lcdb_csv, db_with_asteroids):
        ingest_lcdb(lcdb_csv, db_with_asteroids)
        ingest_lcdb(lcdb_csv, db_with_asteroids)
        count = db_with_asteroids.execute(
            "SELECT COUNT(*) FROM rotation_properties"
        ).fetchone()[0]
        assert count == 4

    def test_file_not_found(self, db_with_asteroids):
        with pytest.raises(FileNotFoundError):
            ingest_lcdb("/nonexistent/path.csv", db_with_asteroids)

    def test_alt_column_names(self, lcdb_csv_alt_columns, db_with_asteroids):
        count = ingest_lcdb(lcdb_csv_alt_columns, db_with_asteroids)
        assert count == 1
        row = db_with_asteroids.execute(
            "SELECT rotation_period, amplitude FROM rotation_properties WHERE asteroid_id = 433"
        ).fetchone()
        assert abs(row[0] - 5.27) < 1e-2
        assert abs(row[1] - 1.45) < 1e-2

    def test_no_matching_asteroid_skipped(self, db_conn, tmp_path):
        csv_content = textwrap.dedent("""\
            Num,Desig,Per,AmpMax,U
            99999,,5.0,0.3,3
        """)
        csv_file = tmp_path / "lcdb_nomatch.csv"
        csv_file.write_text(csv_content)
        count = ingest_lcdb(csv_file, db_conn)
        assert count == 0

    def test_rows_without_period_skipped(self, db_with_asteroids, tmp_path):
        csv_content = textwrap.dedent("""\
            Num,Desig,Per,AmpMax,U
            433,1898 DQ,,1.0,3
        """)
        csv_file = tmp_path / "lcdb_noper.csv"
        csv_file.write_text(csv_content)
        count = ingest_lcdb(csv_file, db_with_asteroids)
        assert count == 0

    def test_quality_filter(self, lcdb_csv, db_with_asteroids, tmp_path):
        """min_quality='3' should accept U=3 and reject lower."""
        csv_content = textwrap.dedent("""\
            Num,Desig,Per,AmpMax,U
            433,1898 DQ,5.27,1.45,3
            4179,1989 FB,176.0,2.0,1
            101955,1999 RQ36,4.296,0.09,2
        """)
        csv_file = tmp_path / "lcdb_qual.csv"
        csv_file.write_text(csv_content)
        count = ingest_lcdb(csv_file, db_with_asteroids, min_quality="3")
        assert count == 1  # only Eros with U=3

    def test_quality_filter_2plus(self, db_with_asteroids, tmp_path):
        """min_quality='2+' should accept U=2+ and U=3."""
        csv_content = textwrap.dedent("""\
            Num,Desig,Per,AmpMax,U
            433,1898 DQ,5.27,1.45,3
            4179,1989 FB,176.0,2.0,2+
            101955,1999 RQ36,4.296,0.09,2
            25143,1998 SF36,12.132,1.1,1
        """)
        csv_file = tmp_path / "lcdb_qual2.csv"
        csv_file.write_text(csv_content)
        count = ingest_lcdb(csv_file, db_with_asteroids, min_quality="2+")
        assert count == 2  # Eros (3) + Toutatis (2+)

    def test_source_field(self, lcdb_csv, db_with_asteroids):
        ingest_lcdb(lcdb_csv, db_with_asteroids)
        rows = db_with_asteroids.execute(
            "SELECT DISTINCT source FROM rotation_properties"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "LCDB"


# --- Scorer integration tests ---


class TestScorerSpinModifier:
    def test_monolithic_increases_score(self):
        """Monolithic asteroid should score higher than identical non-monolithic."""
        import numpy as np
        from prospector.scoring.scorer import score_asteroid

        pv = np.zeros(17)
        pv[13] = 1.0  # S-type
        common = dict(diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
                       moid=0.05, prob_vector=pv, n_samples=100)

        rng1 = np.random.default_rng(42)
        baseline = score_asteroid(**common, rng=rng1)

        rng2 = np.random.default_rng(42)
        monolithic = score_asteroid(**common, is_monolithic=True, rng=rng2)

        assert monolithic["composite_score"] > baseline["composite_score"]
        assert monolithic["spin_modifier"] == pytest.approx(1.15)
        assert baseline["spin_modifier"] == pytest.approx(1.0)

    def test_binary_suspect_decreases_score(self):
        """Binary suspect should score lower than identical normal asteroid."""
        import numpy as np
        from prospector.scoring.scorer import score_asteroid

        pv = np.zeros(17)
        pv[13] = 1.0  # S-type
        common = dict(diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
                       moid=0.05, prob_vector=pv, n_samples=100)

        rng1 = np.random.default_rng(42)
        baseline = score_asteroid(**common, rng=rng1)

        rng2 = np.random.default_rng(42)
        binary = score_asteroid(**common, is_binary_suspect=True, rng=rng2)

        assert binary["composite_score"] < baseline["composite_score"]
        assert binary["spin_modifier"] == pytest.approx(0.80)
