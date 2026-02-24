"""Tests for Stage 1B space weathering correction (Brunetto et al. 2006)."""

import numpy as np
import pytest

from prospector.db import get_connection, init_schema
from prospector.spectral.weathering import (
    CONTINUUM_REGIONS,
    CS_MAX,
    CS_MIN,
    SILICATE_CLASSES,
    _select_continuum_points,
    deweather_all,
    deweather_asteroid,
    deweather_spectrum,
    fit_cs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spectrum(wl_min=0.35, wl_max=2.50, step=0.005, cs=0.0, noise=0.0):
    """Create a synthetic spectrum with optional weathering.

    A flat continuum R=1.0 with Brunetto weathering applied:
    R_observed = R_fresh * exp(Cs/λ)
    """
    wl = np.arange(wl_min, wl_max + step / 2, step)
    # Start with flat continuum
    refl = np.ones_like(wl)
    # Apply weathering
    if cs != 0:
        refl = refl * np.exp(cs / wl)
    # Add noise
    if noise > 0:
        rng = np.random.default_rng(42)
        refl += rng.normal(0, noise, len(refl))
        refl = np.maximum(refl, 0.01)  # keep positive
    return wl, refl


def _make_silicate_spectrum(wl_min=0.35, wl_max=2.50, step=0.005, cs=-0.15):
    """Create a spectrum with silicate absorption bands + weathering.

    Adds Band I (~1 μm) and Band II (~2 μm) Gaussian absorptions
    on top of a weathered continuum.
    """
    wl = np.arange(wl_min, wl_max + step / 2, step)
    # Continuum: slightly sloped, then weathered
    refl = 1.0 + 0.05 * (wl - 0.55)  # mild positive slope
    # Add absorption bands
    refl -= 0.3 * np.exp(-((wl - 0.95) ** 2) / (2 * 0.05**2))  # Band I
    refl -= 0.2 * np.exp(-((wl - 2.00) ** 2) / (2 * 0.10**2))  # Band II
    # Apply weathering
    refl = refl * np.exp(cs / wl)
    return wl, refl


def _db_with_silicate_asteroid(
    asteroid_id=4179,
    taxonomy_class="S",
    cs=-0.2,
    normalized=True,
    wl_min=0.35,
    wl_max=2.50,
):
    """Create an in-memory DB with a silicate asteroid and normalized spectrum."""
    conn = get_connection(":memory:")

    conn.execute(
        "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, 1)",
        (asteroid_id, f"Asteroid{asteroid_id}"),
    )
    conn.execute(
        "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob) "
        "VALUES (?, ?, 0.85)",
        (asteroid_id, taxonomy_class),
    )

    wl, refl = _make_silicate_spectrum(wl_min=wl_min, wl_max=wl_max, cs=cs)

    conn.execute(
        "INSERT INTO spectra "
        "(asteroid_id, survey, wavelengths, reflectance, wl_min, wl_max, "
        "normalized, quality_flag) "
        "VALUES (?, 'MITHNEOS', ?, ?, ?, ?, ?, 'good')",
        (
            asteroid_id,
            wl.tobytes(),
            refl.tobytes(),
            float(wl.min()),
            float(wl.max()),
            normalized,
        ),
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# _select_continuum_points
# ---------------------------------------------------------------------------


class TestSelectContinuumPoints:
    def test_selects_from_defined_regions(self):
        wl, refl = _make_spectrum()
        wl_cont, refl_cont = _select_continuum_points(wl, refl)
        # All selected points should be in continuum regions
        for w in wl_cont:
            in_region = any(lo <= w <= hi for lo, hi in CONTINUUM_REGIONS)
            assert in_region, f"Point at {w:.3f} μm not in any continuum region"

    def test_excludes_nan(self):
        wl, refl = _make_spectrum()
        # Set some continuum-region reflectances to NaN
        mask_05_07 = (wl >= 0.50) & (wl <= 0.60)
        refl[mask_05_07] = np.nan
        wl_cont, refl_cont = _select_continuum_points(wl, refl)
        assert not np.any(np.isnan(refl_cont))

    def test_excludes_negative_reflectance(self):
        wl, refl = _make_spectrum()
        refl[10:15] = -0.01  # Negative values
        wl_cont, refl_cont = _select_continuum_points(wl, refl)
        assert np.all(refl_cont > 0)


# ---------------------------------------------------------------------------
# fit_cs
# ---------------------------------------------------------------------------


class TestFitCs:
    def test_flat_spectrum_cs_near_zero(self):
        """A flat spectrum (no weathering) should yield Cs ≈ 0."""
        wl, refl = _make_spectrum(cs=0.0)
        cs = fit_cs(wl, refl)
        assert cs is not None
        assert abs(cs) < 0.01

    def test_weathered_spectrum_recovers_cs(self):
        """Fit should recover the known Cs from a synthetically weathered spectrum."""
        true_cs = -0.15
        wl, refl = _make_spectrum(cs=true_cs)
        fitted_cs = fit_cs(wl, refl)
        assert fitted_cs is not None
        assert abs(fitted_cs - true_cs) < 0.02

    def test_strong_weathering(self):
        """Test with stronger weathering effect."""
        true_cs = -0.40
        wl, refl = _make_spectrum(cs=true_cs)
        fitted_cs = fit_cs(wl, refl)
        assert fitted_cs is not None
        assert fitted_cs < -0.3  # Should be significantly negative

    def test_insufficient_continuum_returns_none(self):
        """Spectrum with only absorption-band wavelengths should fail."""
        wl = np.arange(0.85, 1.15, 0.005)  # Only Band I region
        refl = np.ones_like(wl) * 0.8
        cs = fit_cs(wl, refl)
        assert cs is None

    def test_cs_bounds_sanity(self):
        """Fitted Cs should be within plausible physical range."""
        wl, refl = _make_spectrum(cs=-0.2)
        cs = fit_cs(wl, refl)
        assert cs is not None
        assert CS_MIN <= cs <= CS_MAX

    def test_noisy_spectrum(self):
        """Fit should still work with moderate noise."""
        wl, refl = _make_spectrum(cs=-0.15, noise=0.01)
        cs = fit_cs(wl, refl)
        assert cs is not None
        assert abs(cs - (-0.15)) < 0.10  # Wider tolerance for noise

    def test_silicate_spectrum_cs_recovery(self):
        """Fit should work on a spectrum with absorption bands."""
        wl, refl = _make_silicate_spectrum(cs=-0.20)
        fitted_cs = fit_cs(wl, refl)
        assert fitted_cs is not None
        # Less precise because absorption bands affect continuum,
        # but should still be in the right ballpark
        assert fitted_cs < -0.05


# ---------------------------------------------------------------------------
# deweather_spectrum
# ---------------------------------------------------------------------------


class TestDeweatherSpectrum:
    def test_roundtrip_recovery(self):
        """Applying weathering then deweathering should recover the original."""
        wl, refl_fresh = _make_spectrum(cs=0.0)
        cs = -0.20
        refl_weathered = refl_fresh * np.exp(cs / wl)
        refl_recovered = deweather_spectrum(wl, refl_weathered, cs)
        np.testing.assert_allclose(refl_recovered, refl_fresh, atol=1e-10)

    def test_cs_zero_no_change(self):
        """Cs=0 should leave the spectrum unchanged."""
        wl, refl = _make_spectrum()
        result = deweather_spectrum(wl, refl, 0.0)
        np.testing.assert_allclose(result, refl)

    def test_negative_cs_increases_blue_end(self):
        """Deweathering (removing reddening) should increase the blue end."""
        wl, refl = _make_spectrum(cs=-0.20)
        result = deweather_spectrum(wl, refl, -0.20)
        # Blue end should be brighter after removing reddening
        blue_mask = wl < 0.6
        assert np.mean(result[blue_mask]) >= np.mean(refl[blue_mask])

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        wl, refl = _make_spectrum()
        result = deweather_spectrum(wl, refl, -0.15)
        assert result.shape == refl.shape

    def test_handles_nan(self):
        """NaN values in reflectance should propagate through."""
        wl, refl = _make_spectrum()
        refl[10:15] = np.nan
        result = deweather_spectrum(wl, refl, -0.15)
        assert np.all(np.isnan(result[10:15]))
        assert not np.any(np.isnan(result[:10]))


# ---------------------------------------------------------------------------
# deweather_asteroid (DB integration)
# ---------------------------------------------------------------------------


class TestDeweatherAsteroid:
    def test_basic_correction(self):
        conn = _db_with_silicate_asteroid(cs=-0.20)
        assert deweather_asteroid(4179, conn)
        conn.commit()

        row = conn.execute(
            "SELECT weathering_cs, deweathered FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()
        assert row[0] is not None  # Cs stored
        assert row[0] < 0  # Should be negative (reddening)
        assert row[1] is not None  # Deweathered BLOB stored

        # BLOB roundtrip
        deweathered = np.frombuffer(row[1], dtype=np.float64)
        wl_blob = conn.execute(
            "SELECT wavelengths FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()[0]
        wl = np.frombuffer(wl_blob, dtype=np.float64)
        assert len(deweathered) == len(wl)

    def test_skips_non_silicate(self):
        conn = _db_with_silicate_asteroid(taxonomy_class="C")
        assert not deweather_asteroid(4179, conn)

    def test_skips_without_taxonomy(self):
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (1, 'Test', 1)"
        )
        wl, refl = _make_spectrum()
        conn.execute(
            "INSERT INTO spectra "
            "(asteroid_id, survey, wavelengths, reflectance, wl_min, wl_max, normalized) "
            "VALUES (1, 'MITHNEOS', ?, ?, 0.35, 2.50, TRUE)",
            (wl.tobytes(), refl.tobytes()),
        )
        conn.commit()
        assert not deweather_asteroid(1, conn)

    def test_skips_unnormalized(self):
        conn = _db_with_silicate_asteroid(normalized=False)
        # The spectrum won't match normalized=TRUE filter
        assert not deweather_asteroid(4179, conn)

    def test_skips_already_corrected(self):
        conn = _db_with_silicate_asteroid(cs=-0.20)
        # First correction
        assert deweather_asteroid(4179, conn)
        conn.commit()
        # Second attempt should skip (deweathered IS NOT NULL)
        assert not deweather_asteroid(4179, conn)

    def test_preserves_original_spectrum(self):
        """Original reflectance should remain unchanged."""
        conn = _db_with_silicate_asteroid(cs=-0.20)
        # Save original
        orig = conn.execute(
            "SELECT reflectance FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()[0]
        orig_arr = np.frombuffer(orig, dtype=np.float64).copy()

        deweather_asteroid(4179, conn)
        conn.commit()

        # Check original unchanged
        after = conn.execute(
            "SELECT reflectance FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()[0]
        after_arr = np.frombuffer(after, dtype=np.float64)
        np.testing.assert_array_equal(orig_arr, after_arr)

    def test_all_silicate_classes_eligible(self):
        """All silicate classes should be processed."""
        for cls in SILICATE_CLASSES:
            conn = _db_with_silicate_asteroid(
                asteroid_id=hash(cls) % 100000,
                taxonomy_class=cls,
                cs=-0.15,
            )
            result = deweather_asteroid(hash(cls) % 100000, conn)
            assert result, f"Class {cls} should be eligible"

    def test_non_silicate_classes_skipped(self):
        """Non-silicate classes should be skipped."""
        for cls in ["C", "B", "M", "X", "D", "E", "P"]:
            conn = _db_with_silicate_asteroid(
                asteroid_id=hash(cls) % 100000 + 100000,
                taxonomy_class=cls,
                cs=-0.15,
            )
            result = deweather_asteroid(hash(cls) % 100000 + 100000, conn)
            assert not result, f"Class {cls} should be skipped"


# ---------------------------------------------------------------------------
# deweather_all (batch)
# ---------------------------------------------------------------------------


class TestDeweatherAll:
    def test_batch_processing(self):
        conn = get_connection(":memory:")
        # Create 3 silicate asteroids
        for aid, cls in [(100, "S"), (200, "Q"), (300, "V")]:
            conn.execute(
                "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, 1)",
                (aid, f"Ast{aid}"),
            )
            conn.execute(
                "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob) "
                "VALUES (?, ?, 0.8)",
                (aid, cls),
            )
            wl, refl = _make_silicate_spectrum(cs=-0.15)
            conn.execute(
                "INSERT INTO spectra "
                "(asteroid_id, survey, wavelengths, reflectance, wl_min, wl_max, "
                "normalized, quality_flag) "
                "VALUES (?, 'MITHNEOS', ?, ?, ?, ?, TRUE, 'good')",
                (aid, wl.tobytes(), refl.tobytes(), float(wl.min()), float(wl.max())),
            )
        conn.commit()

        count = deweather_all(conn)
        assert count == 3

        # All should have corrections
        rows = conn.execute(
            "SELECT COUNT(*) FROM spectra WHERE weathering_cs IS NOT NULL"
        ).fetchone()[0]
        assert rows == 3

    def test_skips_non_silicate_in_batch(self):
        conn = get_connection(":memory:")
        # 1 silicate + 1 C-type
        for aid, cls in [(100, "S"), (200, "C")]:
            conn.execute(
                "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, 1)",
                (aid, f"Ast{aid}"),
            )
            conn.execute(
                "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob) "
                "VALUES (?, ?, 0.8)",
                (aid, cls),
            )
            wl, refl = _make_silicate_spectrum(cs=-0.15)
            conn.execute(
                "INSERT INTO spectra "
                "(asteroid_id, survey, wavelengths, reflectance, wl_min, wl_max, "
                "normalized, quality_flag) "
                "VALUES (?, 'MITHNEOS', ?, ?, ?, ?, TRUE, 'good')",
                (aid, wl.tobytes(), refl.tobytes(), float(wl.min()), float(wl.max())),
            )
        conn.commit()

        count = deweather_all(conn)
        assert count == 1  # Only S-type

    def test_idempotent(self):
        conn = _db_with_silicate_asteroid(cs=-0.20)
        first = deweather_all(conn)
        second = deweather_all(conn)
        assert first == 1
        assert second == 0  # Already corrected

    def test_empty_db(self):
        conn = get_connection(":memory:")
        assert deweather_all(conn) == 0


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_weathering_columns_exist(self):
        conn = get_connection(":memory:")
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(spectra)").fetchall()
        }
        assert "weathering_cs" in cols
        assert "deweathered" in cols

    def test_migration_idempotent(self):
        """Running init_schema twice should not fail."""
        conn = get_connection(":memory:")
        init_schema(conn)  # Second call
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(spectra)").fetchall()
        }
        assert "weathering_cs" in cols
        assert "deweathered" in cols


# ---------------------------------------------------------------------------
# Physical sanity checks
# ---------------------------------------------------------------------------


class TestPhysicalSanity:
    def test_deweathered_less_red(self):
        """De-weathered spectrum should be less red (flatter slope) than original."""
        wl, refl_weathered = _make_spectrum(cs=-0.25)
        cs = fit_cs(wl, refl_weathered)
        refl_deweathered = deweather_spectrum(wl, refl_weathered, cs)

        # Compute spectral slope (red/blue ratio)
        blue = (wl >= 0.5) & (wl <= 0.6)
        red = (wl >= 2.2) & (wl <= 2.4)
        slope_orig = np.mean(refl_weathered[red]) / np.mean(refl_weathered[blue])
        slope_dw = np.mean(refl_deweathered[red]) / np.mean(refl_deweathered[blue])
        assert slope_dw < slope_orig

    def test_weathering_makes_spectrum_redder(self):
        """Sanity: applying negative Cs should redden the spectrum."""
        wl = np.arange(0.35, 2.55, 0.005)
        refl_flat = np.ones_like(wl)
        cs = -0.2
        refl_weathered = refl_flat * np.exp(cs / wl)
        # Red end should be brighter than blue end
        assert np.mean(refl_weathered[wl > 2.0]) > np.mean(refl_weathered[wl < 0.5])
