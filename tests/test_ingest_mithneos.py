"""Tests for MITHNEOS ASCII spectra ingestion."""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from prospector.db import get_connection, init_schema
from prospector.ingest.mithneos import (
    _read_spectrum_file,
    ingest_mithneos_dir,
    ingest_mithneos_file,
)


# ---------------------------------------------------------------------------
# Helpers: create temp spectrum files
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


# ---------------------------------------------------------------------------
# _read_spectrum_file
# ---------------------------------------------------------------------------


class TestReadSpectrumFile:
    def test_two_column(self, tmp_path):
        p = _write_spectrum(tmp_path, "test.txt", """\
            0.80  1.000
            0.85  1.012
            0.90  0.998
            1.00  0.975
        """)
        wl, refl, unc = _read_spectrum_file(p)
        assert len(wl) == 4
        assert len(refl) == 4
        assert unc is None
        np.testing.assert_allclose(wl, [0.80, 0.85, 0.90, 1.00])
        np.testing.assert_allclose(refl, [1.000, 1.012, 0.998, 0.975])

    def test_three_column(self, tmp_path):
        p = _write_spectrum(tmp_path, "test.txt", """\
            0.80  1.000  0.01
            0.85  1.012  0.02
            0.90  0.998  0.015
        """)
        wl, refl, unc = _read_spectrum_file(p)
        assert len(wl) == 3
        assert unc is not None
        assert len(unc) == 3
        np.testing.assert_allclose(unc, [0.01, 0.02, 0.015])

    def test_skips_comments_and_blank_lines(self, tmp_path):
        p = _write_spectrum(tmp_path, "test.txt", """\
            # Header comment
            # Another comment

            0.80  1.000
            0.85  1.012
        """)
        wl, refl, unc = _read_spectrum_file(p)
        assert len(wl) == 2

    def test_empty_file_raises(self, tmp_path):
        p = _write_spectrum(tmp_path, "test.txt", "# only comments\n")
        with pytest.raises(ValueError, match="No valid data"):
            _read_spectrum_file(p)

    def test_dtype_is_float64(self, tmp_path):
        p = _write_spectrum(tmp_path, "test.txt", """\
            0.80  1.000
            0.85  1.012
        """)
        wl, refl, _ = _read_spectrum_file(p)
        assert wl.dtype == np.float64
        assert refl.dtype == np.float64


# ---------------------------------------------------------------------------
# ingest_mithneos_file — single file
# ---------------------------------------------------------------------------


class TestIngestMithneosFile:
    def test_numbered_asteroid(self, tmp_path):
        """a004179.sp05.txt → asteroid 4179."""
        conn = _db_with_asteroids([4179])
        p = _write_spectrum(tmp_path, "a004179.sp05.txt", """\
            0.80  1.000
            0.85  1.012
            0.90  0.998
            1.00  0.975
            2.50  0.850
        """)
        assert ingest_mithneos_file(p, conn) is True
        conn.commit()

        row = conn.execute(
            "SELECT asteroid_id, survey, wl_min, wl_max FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()
        assert row is not None
        assert row[0] == 4179
        assert row[1] == "MITHNEOS"
        assert abs(row[2] - 0.80) < 1e-6
        assert abs(row[3] - 2.50) < 1e-6

    def test_blob_roundtrip(self, tmp_path):
        """Wavelengths/reflectance survive BLOB storage."""
        conn = _db_with_asteroids([433])
        p = _write_spectrum(tmp_path, "a000433.sp01.txt", """\
            0.82  1.100
            0.95  1.050
            1.50  0.920
        """)
        ingest_mithneos_file(p, conn)
        conn.commit()

        row = conn.execute(
            "SELECT wavelengths, reflectance, uncertainty FROM spectra WHERE asteroid_id = 433"
        ).fetchone()
        wl = np.frombuffer(row[0], dtype=np.float64)
        refl = np.frombuffer(row[1], dtype=np.float64)
        np.testing.assert_allclose(wl, [0.82, 0.95, 1.50])
        np.testing.assert_allclose(refl, [1.100, 1.050, 0.920])
        assert row[2] is None  # no uncertainty column

    def test_uncertainty_stored(self, tmp_path):
        """Three-column file stores uncertainty BLOB."""
        conn = _db_with_asteroids([1036])
        p = _write_spectrum(tmp_path, "a001036.sp01.txt", """\
            0.80  1.000  0.01
            0.90  0.990  0.02
        """)
        ingest_mithneos_file(p, conn)
        conn.commit()

        row = conn.execute(
            "SELECT uncertainty FROM spectra WHERE asteroid_id = 1036"
        ).fetchone()
        assert row[0] is not None
        unc = np.frombuffer(row[0], dtype=np.float64)
        np.testing.assert_allclose(unc, [0.01, 0.02])

    def test_skips_unknown_asteroid(self, tmp_path):
        """File referencing asteroid not in DB returns False."""
        conn = _db_with_asteroids([1])  # only asteroid 1 exists
        p = _write_spectrum(tmp_path, "a099999.sp01.txt", """\
            0.80  1.000
            0.90  0.990
        """)
        assert ingest_mithneos_file(p, conn) is False
        count = conn.execute("SELECT COUNT(*) FROM spectra").fetchone()[0]
        assert count == 0

    def test_skips_unparseable_filename(self, tmp_path):
        """Non-MITHNEOS filename is skipped."""
        conn = _db_with_asteroids([4179])
        p = _write_spectrum(tmp_path, "random_file.txt", """\
            0.80  1.000
            0.90  0.990
        """)
        assert ingest_mithneos_file(p, conn) is False

    def test_multiple_spectra_per_asteroid(self, tmp_path):
        """One asteroid can have multiple spectra (different observations)."""
        conn = _db_with_asteroids([4179])
        for sp in ["a004179.sp01.txt", "a004179.sp05.txt"]:
            p = _write_spectrum(tmp_path, sp, """\
                0.80  1.000
                0.90  0.990
            """)
            ingest_mithneos_file(p, conn)
        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()[0]
        assert count == 2

    def test_known_ids_cache(self, tmp_path):
        """Pre-loaded known_ids set avoids per-file DB lookups."""
        conn = _db_with_asteroids([4179])
        known = {4179}
        p = _write_spectrum(tmp_path, "a004179.sp01.txt", """\
            0.80  1.000
            0.90  0.990
        """)
        assert ingest_mithneos_file(p, conn, known_ids=known) is True

        # Asteroid 433 not in known_ids set → skip
        p2 = _write_spectrum(tmp_path, "a000433.sp01.txt", """\
            0.80  1.000
            0.90  0.990
        """)
        assert ingest_mithneos_file(p2, conn, known_ids=known) is False


# ---------------------------------------------------------------------------
# ingest_mithneos_dir — directory batch
# ---------------------------------------------------------------------------


class TestIngestMithneosDir:
    def test_ingest_directory(self, tmp_path):
        """Batch ingest all .txt files in a directory."""
        conn = _db_with_asteroids([4179, 433, 1036])

        for name, content in [
            ("a004179.sp01.txt", "0.80 1.0\n0.90 0.99\n"),
            ("a000433.sp01.txt", "0.82 1.1\n0.95 1.05\n"),
            ("a001036.sp02.txt", "0.80 1.0\n1.50 0.9\n"),
        ]:
            (tmp_path / name).write_text(content)

        count = ingest_mithneos_dir(tmp_path, conn)
        assert count == 3

        total = conn.execute("SELECT COUNT(*) FROM spectra").fetchone()[0]
        assert total == 3

    def test_skips_non_matching_files(self, tmp_path):
        """Files that don't match MITHNEOS naming are skipped."""
        conn = _db_with_asteroids([4179])
        (tmp_path / "a004179.sp01.txt").write_text("0.80 1.0\n0.90 0.99\n")
        (tmp_path / "readme.txt").write_text("This is not a spectrum\n")

        count = ingest_mithneos_dir(tmp_path, conn)
        assert count == 1  # only the valid MITHNEOS file

    def test_missing_dir_raises(self):
        conn = get_connection(":memory:")
        with pytest.raises(FileNotFoundError):
            ingest_mithneos_dir("/nonexistent/path", conn)

    def test_empty_dir(self, tmp_path):
        """Empty directory returns 0."""
        conn = _db_with_asteroids([4179])
        count = ingest_mithneos_dir(tmp_path, conn)
        assert count == 0

    def test_survey_field(self, tmp_path):
        """All ingested spectra have survey='MITHNEOS'."""
        conn = _db_with_asteroids([4179])
        (tmp_path / "a004179.sp01.txt").write_text("0.80 1.0\n0.90 0.99\n")
        ingest_mithneos_dir(tmp_path, conn)

        surveys = [
            row[0]
            for row in conn.execute("SELECT DISTINCT survey FROM spectra").fetchall()
        ]
        assert surveys == ["MITHNEOS"]

    def test_fk_integrity(self, tmp_path):
        """Only files matching known asteroids are ingested."""
        conn = _db_with_asteroids([4179])
        (tmp_path / "a004179.sp01.txt").write_text("0.80 1.0\n0.90 0.99\n")
        (tmp_path / "a099999.sp01.txt").write_text("0.80 1.0\n0.90 0.99\n")  # not in DB

        count = ingest_mithneos_dir(tmp_path, conn)
        assert count == 1

    def test_wl_min_max(self, tmp_path):
        """wl_min and wl_max are recorded correctly."""
        conn = _db_with_asteroids([4179])
        (tmp_path / "a004179.sp01.txt").write_text("0.82 1.0\n1.25 0.95\n2.45 0.88\n")
        ingest_mithneos_dir(tmp_path, conn)

        row = conn.execute(
            "SELECT wl_min, wl_max FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()
        assert abs(row[0] - 0.82) < 1e-6
        assert abs(row[1] - 2.45) < 1e-6
