"""Tests for SMASS II visible spectra ingestion."""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from prospector.db import get_connection, init_schema
from prospector.ingest.smass import (
    ingest_smass_dir,
    ingest_smass_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_spectrum(tmp_path: Path, filename: str, content: str) -> Path:
    """Write a spectrum file and return its path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content))
    return p


def _db_with_asteroids(asteroid_ids: list[int]) -> "sqlite3.Connection":
    """Create an in-memory DB pre-populated with asteroids."""
    conn = get_connection(":memory:")
    for aid in asteroid_ids:
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, 1)",
            (aid, f"Asteroid{aid}"),
        )
    conn.commit()
    return conn


# Typical SMASS II visible-range spectrum (0.44–0.92 μm) with uncertainty
SMASS_SPECTRUM_3COL = """\
    0.4400  1.020  0.012
    0.5000  1.005  0.010
    0.5500  1.000  0.009
    0.6000  0.998  0.010
    0.7000  0.985  0.011
    0.8000  0.970  0.013
    0.9200  0.955  0.015
"""


# ---------------------------------------------------------------------------
# ingest_smass_file — single file
# ---------------------------------------------------------------------------


class TestIngestSmassFile:
    def test_numbered_asteroid(self, tmp_path):
        """a004179.sp01.txt → asteroid 4179 with survey='SMASS'."""
        conn = _db_with_asteroids([4179])
        p = _write_spectrum(tmp_path, "a004179.sp01.txt", SMASS_SPECTRUM_3COL)
        assert ingest_smass_file(p, conn) is True
        conn.commit()

        row = conn.execute(
            "SELECT asteroid_id, survey, wl_min, wl_max FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()
        assert row is not None
        assert row[0] == 4179
        assert row[1] == "SMASS"
        assert abs(row[2] - 0.44) < 1e-6
        assert abs(row[3] - 0.92) < 1e-6

    def test_blob_roundtrip(self, tmp_path):
        """Wavelengths/reflectance/uncertainty survive BLOB storage."""
        conn = _db_with_asteroids([433])
        p = _write_spectrum(tmp_path, "a000433.sp01.txt", SMASS_SPECTRUM_3COL)
        ingest_smass_file(p, conn)
        conn.commit()

        row = conn.execute(
            "SELECT wavelengths, reflectance, uncertainty FROM spectra WHERE asteroid_id = 433"
        ).fetchone()
        wl = np.frombuffer(row[0], dtype=np.float64)
        refl = np.frombuffer(row[1], dtype=np.float64)
        unc = np.frombuffer(row[2], dtype=np.float64)

        assert len(wl) == 7
        np.testing.assert_allclose(wl[0], 0.44)
        np.testing.assert_allclose(wl[-1], 0.92)
        np.testing.assert_allclose(refl[0], 1.020)
        np.testing.assert_allclose(unc[0], 0.012)

    def test_two_column_file(self, tmp_path):
        """SMASS files without uncertainty column are accepted."""
        conn = _db_with_asteroids([1036])
        p = _write_spectrum(tmp_path, "a001036.sp01.txt", """\
            0.44  1.020
            0.55  1.000
            0.70  0.985
            0.92  0.955
        """)
        ingest_smass_file(p, conn)
        conn.commit()

        row = conn.execute(
            "SELECT uncertainty FROM spectra WHERE asteroid_id = 1036"
        ).fetchone()
        assert row[0] is None

    def test_skips_unknown_asteroid(self, tmp_path):
        """File referencing asteroid not in DB returns False."""
        conn = _db_with_asteroids([1])
        p = _write_spectrum(tmp_path, "a099999.sp01.txt", SMASS_SPECTRUM_3COL)
        assert ingest_smass_file(p, conn) is False
        count = conn.execute("SELECT COUNT(*) FROM spectra").fetchone()[0]
        assert count == 0

    def test_skips_unparseable_filename(self, tmp_path):
        """Non-SMASS filename is skipped."""
        conn = _db_with_asteroids([4179])
        p = _write_spectrum(tmp_path, "random_file.txt", SMASS_SPECTRUM_3COL)
        assert ingest_smass_file(p, conn) is False

    def test_multiple_spectra_per_asteroid(self, tmp_path):
        """One asteroid can have multiple SMASS spectra."""
        conn = _db_with_asteroids([4179])
        for sp in ["a004179.sp01.txt", "a004179.sp02.txt"]:
            p = _write_spectrum(tmp_path, sp, SMASS_SPECTRUM_3COL)
            ingest_smass_file(p, conn)
        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()[0]
        assert count == 2

    def test_known_ids_cache(self, tmp_path):
        """Pre-loaded known_ids set avoids per-file DB lookups."""
        conn = _db_with_asteroids([4179])
        known = {4179}
        p = _write_spectrum(tmp_path, "a004179.sp01.txt", SMASS_SPECTRUM_3COL)
        assert ingest_smass_file(p, conn, known_ids=known) is True

        p2 = _write_spectrum(tmp_path, "a000433.sp01.txt", SMASS_SPECTRUM_3COL)
        assert ingest_smass_file(p2, conn, known_ids=known) is False


# ---------------------------------------------------------------------------
# ingest_smass_dir — directory batch
# ---------------------------------------------------------------------------


class TestIngestSmassDir:
    def test_ingest_directory(self, tmp_path):
        """Batch ingest all .txt files in a directory."""
        conn = _db_with_asteroids([4179, 433, 1036])

        for name in ["a004179.sp01.txt", "a000433.sp01.txt", "a001036.sp01.txt"]:
            (tmp_path / name).write_text(textwrap.dedent(SMASS_SPECTRUM_3COL))

        count = ingest_smass_dir(tmp_path, conn)
        assert count == 3

        total = conn.execute("SELECT COUNT(*) FROM spectra").fetchone()[0]
        assert total == 3

    def test_skips_non_matching_files(self, tmp_path):
        """Non-spectrum files are skipped."""
        conn = _db_with_asteroids([4179])
        (tmp_path / "a004179.sp01.txt").write_text(
            textwrap.dedent(SMASS_SPECTRUM_3COL)
        )
        (tmp_path / "readme.txt").write_text("Not a spectrum\n")

        count = ingest_smass_dir(tmp_path, conn)
        assert count == 1

    def test_missing_dir_raises(self):
        conn = get_connection(":memory:")
        with pytest.raises(FileNotFoundError):
            ingest_smass_dir("/nonexistent/path", conn)

    def test_empty_dir(self, tmp_path):
        """Empty directory returns 0."""
        conn = _db_with_asteroids([4179])
        count = ingest_smass_dir(tmp_path, conn)
        assert count == 0

    def test_survey_field(self, tmp_path):
        """All ingested spectra have survey='SMASS'."""
        conn = _db_with_asteroids([4179])
        (tmp_path / "a004179.sp01.txt").write_text(
            textwrap.dedent(SMASS_SPECTRUM_3COL)
        )
        ingest_smass_dir(tmp_path, conn)

        surveys = [
            row[0]
            for row in conn.execute("SELECT DISTINCT survey FROM spectra").fetchall()
        ]
        assert surveys == ["SMASS"]

    def test_fk_integrity(self, tmp_path):
        """Only files matching known asteroids are ingested."""
        conn = _db_with_asteroids([4179])
        (tmp_path / "a004179.sp01.txt").write_text(
            textwrap.dedent(SMASS_SPECTRUM_3COL)
        )
        (tmp_path / "a099999.sp01.txt").write_text(
            textwrap.dedent(SMASS_SPECTRUM_3COL)
        )

        count = ingest_smass_dir(tmp_path, conn)
        assert count == 1

    def test_wl_min_max(self, tmp_path):
        """wl_min and wl_max reflect visible-range SMASS data."""
        conn = _db_with_asteroids([4179])
        (tmp_path / "a004179.sp01.txt").write_text(
            textwrap.dedent(SMASS_SPECTRUM_3COL)
        )
        ingest_smass_dir(tmp_path, conn)

        row = conn.execute(
            "SELECT wl_min, wl_max FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()
        assert abs(row[0] - 0.44) < 1e-6
        assert abs(row[1] - 0.92) < 1e-6

    def test_coexists_with_mithneos(self, tmp_path):
        """SMASS and MITHNEOS spectra for the same asteroid coexist."""
        conn = _db_with_asteroids([4179])

        # Insert a MITHNEOS spectrum first
        conn.execute(
            "INSERT INTO spectra (asteroid_id, survey, wavelengths, reflectance, wl_min, wl_max) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                4179,
                "MITHNEOS",
                np.array([0.8, 1.0, 2.5], dtype=np.float64).tobytes(),
                np.array([1.0, 0.95, 0.85], dtype=np.float64).tobytes(),
                0.8,
                2.5,
            ),
        )
        conn.commit()

        # Now ingest SMASS
        (tmp_path / "a004179.sp01.txt").write_text(
            textwrap.dedent(SMASS_SPECTRUM_3COL)
        )
        ingest_smass_dir(tmp_path, conn)

        rows = conn.execute(
            "SELECT survey FROM spectra WHERE asteroid_id = 4179 ORDER BY survey"
        ).fetchall()
        surveys = [r[0] for r in rows]
        assert "MITHNEOS" in surveys
        assert "SMASS" in surveys
        assert len(surveys) == 2
