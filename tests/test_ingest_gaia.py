"""Tests for Gaia DR3 reflectance spectra ingestion."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from prospector.db import get_connection, init_schema
from prospector.ingest.gaia import (
    FLAG_COMPROMISED,
    FLAG_GOOD,
    FLAG_POOR,
    _group_spectra,
    _has_good_bands,
    _ingest_dataframe,
    _resolve_asteroid_id,
    ingest_gaia_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 16 Gaia DR3 wavelength bins (nm), evenly spaced 374–1034 nm
GAIA_WL_NM = np.linspace(374.0, 1034.0, 16)


def _make_gaia_df(
    objects: list[dict],
    *,
    n_bands: int = 16,
    add_flags: bool = True,
) -> pd.DataFrame:
    """Build a Gaia-style per-wavelength DataFrame from per-asteroid specs.

    Each object dict should have:
        source_id, number_mp (or None), denomination (or None),
        and optionally reflectance (array) and flags (array).
    """
    rows = []
    for obj in objects:
        wl_nm = GAIA_WL_NM[:n_bands]
        refl = obj.get("reflectance", np.ones(n_bands))
        err = obj.get("error", np.full(n_bands, 0.01))
        flags = obj.get("flags", np.zeros(n_bands, dtype=np.int8))

        for i in range(n_bands):
            row = {
                "source_id": obj["source_id"],
                "number_mp": obj.get("number_mp"),
                "denomination": obj.get("denomination"),
                "nb_samples": n_bands,
                "num_of_spectra": obj.get("num_of_spectra", 10),
                "wavelength": float(wl_nm[i]),
                "reflectance_spectrum": float(refl[i]),
                "reflectance_spectrum_err": float(err[i]),
            }
            if add_flags:
                row["reflectance_spectrum_flag"] = int(flags[i])
            rows.append(row)

    return pd.DataFrame(rows)


def _db_with_asteroids(asteroid_ids: list[int]) -> sqlite3.Connection:
    """Create an in-memory DB pre-populated with asteroids and orbits."""
    conn = get_connection(":memory:")
    for aid in asteroid_ids:
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, 1)",
            (aid, f"Asteroid{aid}"),
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# _group_spectra
# ---------------------------------------------------------------------------


class TestGroupSpectra:
    def test_groups_by_source_id(self):
        df = _make_gaia_df([
            {"source_id": -100, "number_mp": 4179},
            {"source_id": -200, "number_mp": 433},
        ])
        spectra = _group_spectra(df)
        assert len(spectra) == 2
        assert -100 in spectra
        assert -200 in spectra

    def test_wavelengths_sorted_and_converted_to_um(self):
        df = _make_gaia_df([{"source_id": -100, "number_mp": 4179}])
        spec = _group_spectra(df)[-100]
        wl = spec["wavelengths"]
        assert len(wl) == 16
        # Should be in μm (0.374 ... 1.034), not nm
        assert wl[0] == pytest.approx(0.374, abs=0.001)
        assert wl[-1] == pytest.approx(1.034, abs=0.001)
        # Sorted ascending
        assert np.all(np.diff(wl) > 0)

    def test_reflectance_array(self):
        refl_vals = np.linspace(0.9, 1.1, 16)
        df = _make_gaia_df([
            {"source_id": -100, "number_mp": 1, "reflectance": refl_vals}
        ])
        spec = _group_spectra(df)[-100]
        np.testing.assert_allclose(spec["reflectance"], refl_vals)

    def test_uncertainty_extracted(self):
        df = _make_gaia_df([{"source_id": -100, "number_mp": 1}])
        spec = _group_spectra(df)[-100]
        assert spec["uncertainty"] is not None
        assert len(spec["uncertainty"]) == 16
        np.testing.assert_allclose(spec["uncertainty"], 0.01)

    def test_flags_extracted(self):
        flags = np.array([0] * 14 + [1, 2], dtype=np.int8)
        df = _make_gaia_df([
            {"source_id": -100, "number_mp": 1, "flags": flags}
        ])
        spec = _group_spectra(df)[-100]
        assert spec["flags"] is not None
        assert spec["flags"][-1] == FLAG_COMPROMISED
        assert spec["flags"][-2] == FLAG_POOR

    def test_no_flags_column(self):
        df = _make_gaia_df(
            [{"source_id": -100, "number_mp": 1}],
            add_flags=False,
        )
        spec = _group_spectra(df)[-100]
        assert spec["flags"] is None

    def test_preserves_number_mp_and_denomination(self):
        df = _make_gaia_df([
            {"source_id": -100, "number_mp": 4179, "denomination": "Toutatis"},
        ])
        spec = _group_spectra(df)[-100]
        assert spec["number_mp"] == 4179
        assert spec["denomination"] == "Toutatis"


# ---------------------------------------------------------------------------
# _has_good_bands
# ---------------------------------------------------------------------------


class TestHasGoodBands:
    def test_all_good(self):
        assert _has_good_bands(np.zeros(16, dtype=np.int8)) is True

    def test_some_poor(self):
        flags = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        assert _has_good_bands(flags) is True

    def test_all_compromised(self):
        assert _has_good_bands(np.full(16, FLAG_COMPROMISED, dtype=np.int8)) is False

    def test_none_flags(self):
        assert _has_good_bands(None) is True

    def test_mixed_with_one_good(self):
        flags = np.full(16, FLAG_COMPROMISED, dtype=np.int8)
        flags[5] = FLAG_GOOD
        assert _has_good_bands(flags) is True


# ---------------------------------------------------------------------------
# _resolve_asteroid_id
# ---------------------------------------------------------------------------


class TestResolveAsteroidId:
    def test_numbered_asteroid(self):
        conn = _db_with_asteroids([4179])
        known_ids = {4179}
        aid = _resolve_asteroid_id(4179, None, conn, known_ids)
        assert aid == 4179

    def test_numbered_not_in_db(self):
        conn = _db_with_asteroids([433])
        known_ids = {433}
        aid = _resolve_asteroid_id(4179, None, conn, known_ids)
        assert aid is None

    def test_nan_number_mp_with_denomination(self):
        """Denomination fallback via entity_resolver."""
        conn = _db_with_asteroids([4179])
        # Insert designation so entity_resolver can find it
        conn.execute(
            "UPDATE asteroids SET full_name = '  4179 Toutatis (1989 AC)' WHERE asteroid_id = 4179"
        )
        conn.commit()
        known_ids = {4179}
        aid = _resolve_asteroid_id(float("nan"), "Toutatis", conn, known_ids)
        assert aid == 4179

    def test_gaia_internal_id_skipped(self):
        conn = _db_with_asteroids([1])
        known_ids = {1}
        aid = _resolve_asteroid_id(float("nan"), "Gaia-DR3SSO-12345", conn, known_ids)
        assert aid is None

    def test_both_nan(self):
        conn = _db_with_asteroids([1])
        known_ids = {1}
        aid = _resolve_asteroid_id(float("nan"), float("nan"), conn, known_ids)
        assert aid is None


# ---------------------------------------------------------------------------
# _ingest_dataframe (integration)
# ---------------------------------------------------------------------------


class TestIngestDataframe:
    def test_basic_ingest(self):
        conn = _db_with_asteroids([4179, 433])
        df = _make_gaia_df([
            {"source_id": -100, "number_mp": 4179},
            {"source_id": -200, "number_mp": 433},
        ])
        count = _ingest_dataframe(df, conn)
        assert count == 2

        rows = conn.execute("SELECT * FROM spectra WHERE survey = 'Gaia'").fetchall()
        assert len(rows) == 2

    def test_blob_roundtrip(self):
        conn = _db_with_asteroids([4179])
        refl = np.linspace(0.8, 1.2, 16)
        df = _make_gaia_df([
            {"source_id": -100, "number_mp": 4179, "reflectance": refl},
        ])
        _ingest_dataframe(df, conn)

        row = conn.execute(
            "SELECT wavelengths, reflectance, uncertainty, wl_min, wl_max "
            "FROM spectra WHERE asteroid_id = 4179 AND survey = 'Gaia'"
        ).fetchone()

        wl = np.frombuffer(row[0], dtype=np.float64)
        r = np.frombuffer(row[1], dtype=np.float64)
        unc = np.frombuffer(row[2], dtype=np.float64)

        assert len(wl) == 16
        assert len(r) == 16
        assert len(unc) == 16
        np.testing.assert_allclose(r, refl)
        assert row[3] == pytest.approx(0.374, abs=0.001)  # wl_min
        assert row[4] == pytest.approx(1.034, abs=0.001)  # wl_max

    def test_fk_validation_skips_unknown(self):
        conn = _db_with_asteroids([4179])
        df = _make_gaia_df([
            {"source_id": -100, "number_mp": 4179},
            {"source_id": -200, "number_mp": 99999},  # not in DB
        ])
        count = _ingest_dataframe(df, conn)
        assert count == 1  # only 4179 ingested

    def test_skip_compromised_spectra(self):
        conn = _db_with_asteroids([4179, 433])
        df = _make_gaia_df([
            {"source_id": -100, "number_mp": 4179,
             "flags": np.zeros(16, dtype=np.int8)},
            {"source_id": -200, "number_mp": 433,
             "flags": np.full(16, FLAG_COMPROMISED, dtype=np.int8)},
        ])
        count = _ingest_dataframe(df, conn, skip_compromised=True)
        assert count == 1  # only 4179 (good flags) ingested

    def test_allow_compromised_when_disabled(self):
        conn = _db_with_asteroids([433])
        df = _make_gaia_df([
            {"source_id": -200, "number_mp": 433,
             "flags": np.full(16, FLAG_COMPROMISED, dtype=np.int8)},
        ])
        count = _ingest_dataframe(df, conn, skip_compromised=False)
        assert count == 1  # ingested despite bad flags

    def test_empty_db_returns_zero(self):
        conn = get_connection(":memory:")
        df = _make_gaia_df([{"source_id": -100, "number_mp": 1}])
        count = _ingest_dataframe(df, conn)
        assert count == 0

    def test_survey_is_gaia(self):
        conn = _db_with_asteroids([4179])
        df = _make_gaia_df([{"source_id": -100, "number_mp": 4179}])
        _ingest_dataframe(df, conn)

        survey = conn.execute(
            "SELECT survey FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()[0]
        assert survey == "Gaia"

    def test_multiple_spectra_per_asteroid_appended(self):
        """If same asteroid appears with different source_ids (shouldn't normally
        happen, but handle gracefully by inserting all)."""
        conn = _db_with_asteroids([4179])
        df = _make_gaia_df([
            {"source_id": -100, "number_mp": 4179},
            {"source_id": -101, "number_mp": 4179},
        ])
        count = _ingest_dataframe(df, conn)
        assert count == 2
        rows = conn.execute(
            "SELECT COUNT(*) FROM spectra WHERE asteroid_id = 4179 AND survey = 'Gaia'"
        ).fetchone()[0]
        assert rows == 2

    def test_denomination_fallback_resolution(self):
        """Unnumbered object resolved via denomination."""
        conn = _db_with_asteroids([25143])
        conn.execute(
            "UPDATE asteroids SET name = 'Itokawa', full_name = ' 25143 Itokawa (1998 SF36)' "
            "WHERE asteroid_id = 25143"
        )
        conn.commit()

        df = _make_gaia_df([
            {"source_id": -300, "number_mp": None, "denomination": "Itokawa"},
        ])
        # Replace NaN explicitly for number_mp
        df["number_mp"] = df["number_mp"].where(df["number_mp"].notna(), other=float("nan"))
        count = _ingest_dataframe(df, conn)
        assert count == 1


# ---------------------------------------------------------------------------
# ingest_gaia_file (CSV file integration)
# ---------------------------------------------------------------------------


class TestIngestGaiaFile:
    def test_csv_file_ingest(self, tmp_path):
        conn = _db_with_asteroids([4179, 433])
        df = _make_gaia_df([
            {"source_id": -100, "number_mp": 4179},
            {"source_id": -200, "number_mp": 433},
        ])
        csv_path = tmp_path / "gaia_sso.csv"
        df.to_csv(csv_path, index=False)

        count = ingest_gaia_file(csv_path, conn)
        assert count == 2

    def test_file_not_found(self):
        conn = get_connection(":memory:")
        with pytest.raises(FileNotFoundError):
            ingest_gaia_file("/nonexistent/path.csv", conn)

    def test_csv_column_case_normalization(self, tmp_path):
        """Column names should be case-insensitive."""
        conn = _db_with_asteroids([4179])
        df = _make_gaia_df([{"source_id": -100, "number_mp": 4179}])
        # Uppercase column names
        df.columns = [c.upper() for c in df.columns]
        csv_path = tmp_path / "gaia_upper.csv"
        df.to_csv(csv_path, index=False)

        count = ingest_gaia_file(csv_path, conn)
        assert count == 1


# ---------------------------------------------------------------------------
# Physical sanity checks
# ---------------------------------------------------------------------------


class TestSanityChecks:
    def test_wavelength_range_is_visible(self):
        """Gaia spectra should be visible-only (0.374–1.034 μm)."""
        conn = _db_with_asteroids([4179])
        df = _make_gaia_df([{"source_id": -100, "number_mp": 4179}])
        _ingest_dataframe(df, conn)

        row = conn.execute(
            "SELECT wl_min, wl_max FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()
        assert row[0] > 0.3  # above 0.3 μm
        assert row[1] < 1.1  # below 1.1 μm
        assert row[1] - row[0] > 0.5  # reasonable coverage span

    def test_16_band_spectrum(self):
        """Each Gaia spectrum should have exactly 16 wavelength bins."""
        conn = _db_with_asteroids([4179])
        df = _make_gaia_df([{"source_id": -100, "number_mp": 4179}])
        _ingest_dataframe(df, conn)

        row = conn.execute(
            "SELECT wavelengths FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()
        wl = np.frombuffer(row[0], dtype=np.float64)
        assert len(wl) == 16

    def test_reflectance_values_physical(self):
        """Reflectance should be positive and plausible (0.01–10 range)."""
        refl = np.random.default_rng(42).uniform(0.5, 1.5, 16)
        conn = _db_with_asteroids([4179])
        df = _make_gaia_df([
            {"source_id": -100, "number_mp": 4179, "reflectance": refl},
        ])
        _ingest_dataframe(df, conn)

        row = conn.execute(
            "SELECT reflectance FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()
        r = np.frombuffer(row[0], dtype=np.float64)
        assert np.all(r > 0)
        assert np.all(r < 10)
