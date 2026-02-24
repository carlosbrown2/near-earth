"""Tests for prospector.db — schema initialization and connection management."""

import sqlite3

import numpy as np
import pytest

from prospector.db import get_connection, init_schema, table_names

# All tables defined in the PRD §12 schema
EXPECTED_TABLES = sorted([
    "asteroids",
    "band_analysis",
    "lab_spectra",
    "orbits",
    "physical_properties",
    "scores",
    "spectra",
    "taxonomy",
])


class TestSchemaInit:
    """Schema DDL creates all expected tables."""

    def test_all_tables_created(self):
        conn = get_connection(":memory:")
        assert table_names(conn) == EXPECTED_TABLES
        conn.close()

    def test_idempotent(self):
        """Running init_schema twice does not raise or duplicate tables."""
        conn = get_connection(":memory:")
        init_schema(conn)  # second call
        assert table_names(conn) == EXPECTED_TABLES
        conn.close()


class TestAsteroidsTable:
    """Core identity table round-trip."""

    def test_insert_and_query(self):
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, designation, name, full_name, neo, pha) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (25143, "1998 SF36", "Itokawa", "25143 Itokawa (1998 SF36)", True, True),
        )
        row = conn.execute("SELECT * FROM asteroids WHERE asteroid_id = 25143").fetchone()
        assert row[0] == 25143
        assert row[2] == "Itokawa"
        assert row[4] == 1  # SQLite stores booleans as integers
        conn.close()


class TestOrbitsTable:
    """Orbital elements table with foreign key to asteroids."""

    def test_foreign_key_enforced(self):
        conn = get_connection(":memory:")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO orbits (asteroid_id, a, e, i) VALUES (?, ?, ?, ?)",
                (99999, 1.0, 0.2, 5.0),
            )
        conn.close()

    def test_insert_with_parent(self):
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, neo, pha) VALUES (?, ?, ?, ?)",
            (433, "Eros", True, True),
        )
        conn.execute(
            "INSERT INTO orbits (asteroid_id, a, e, i, H, moid) VALUES (?, ?, ?, ?, ?, ?)",
            (433, 1.458, 0.223, 10.83, 11.16, 0.149),
        )
        row = conn.execute("SELECT a, moid FROM orbits WHERE asteroid_id = 433").fetchone()
        assert abs(row[0] - 1.458) < 1e-6
        assert abs(row[1] - 0.149) < 1e-6
        conn.close()


class TestSpectraBlob:
    """Spectral BLOB columns round-trip numpy arrays."""

    def test_numpy_blob_roundtrip(self):
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, neo, pha) VALUES (?, ?, ?, ?)",
            (25143, "Itokawa", True, True),
        )

        wl = np.linspace(0.8, 2.5, 100)
        refl = np.random.default_rng(42).random(100)

        conn.execute(
            "INSERT INTO spectra (asteroid_id, survey, wavelengths, reflectance, wl_min, wl_max) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (25143, "MITHNEOS", wl.tobytes(), refl.tobytes(), float(wl.min()), float(wl.max())),
        )

        row = conn.execute(
            "SELECT wavelengths, reflectance, wl_min, wl_max FROM spectra WHERE asteroid_id = 25143"
        ).fetchone()

        wl_out = np.frombuffer(row[0], dtype=np.float64)
        refl_out = np.frombuffer(row[1], dtype=np.float64)

        np.testing.assert_array_almost_equal(wl_out, wl)
        np.testing.assert_array_almost_equal(refl_out, refl)
        assert abs(row[2] - 0.8) < 1e-6
        assert abs(row[3] - 2.5) < 1e-6
        conn.close()

    def test_multiple_spectra_per_asteroid(self):
        """One asteroid can have multiple spectral observations."""
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, neo, pha) VALUES (?, ?, ?, ?)",
            (433, "Eros", True, True),
        )
        for survey in ("MITHNEOS", "SMASS", "Gaia"):
            conn.execute(
                "INSERT INTO spectra (asteroid_id, survey, wl_min, wl_max) VALUES (?, ?, ?, ?)",
                (433, survey, 0.4, 2.5),
            )
        count = conn.execute("SELECT COUNT(*) FROM spectra WHERE asteroid_id = 433").fetchone()[0]
        assert count == 3
        conn.close()


class TestConnectionOptions:
    """get_connection configuration options."""

    def test_memory_database(self):
        conn = get_connection(":memory:")
        assert table_names(conn) == EXPECTED_TABLES
        conn.close()

    def test_file_database(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)
        assert table_names(conn) == EXPECTED_TABLES
        assert db_path.exists()
        conn.close()

    def test_create_false_skips_schema(self, tmp_path):
        db_path = tmp_path / "empty.db"
        conn = get_connection(db_path, create=False)
        assert table_names(conn) == []
        conn.close()

    def test_wal_mode_enabled(self):
        conn = get_connection(":memory:")
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        # In-memory databases may return 'memory' instead of 'wal'
        assert mode in ("wal", "memory")
        conn.close()

    def test_foreign_keys_enabled(self):
        conn = get_connection(":memory:")
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
        conn.close()

    def test_creates_parent_directories(self, tmp_path):
        db_path = tmp_path / "nested" / "dirs" / "test.db"
        conn = get_connection(db_path)
        assert db_path.exists()
        conn.close()
