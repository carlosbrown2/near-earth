"""Tests for prospector.spectral.preprocessing (Stage 0)."""

import numpy as np
import pytest

from prospector.db import get_connection, init_schema
from prospector.spectral.preprocessing import (
    DEFAULT_GRID,
    GROUND_BASED_SURVEYS,
    NORM_WAVELENGTH,
    SNR_GOOD_THRESHOLD,
    TELLURIC_REGIONS,
    assess_quality,
    estimate_snr,
    mask_telluric,
    normalize,
    preprocess_all,
    preprocess_spectrum,
    resample,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    """In-memory DB with schema initialized."""
    c = get_connection(":memory:")
    return c


def _insert_asteroid(conn, asteroid_id=4179, name="Toutatis"):
    """Insert a test asteroid."""
    conn.execute(
        "INSERT OR IGNORE INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, 1)",
        (asteroid_id, name),
    )


def _insert_spectrum(
    conn,
    asteroid_id=4179,
    survey="MITHNEOS",
    wavelengths=None,
    reflectance=None,
    uncertainty=None,
):
    """Insert a test spectrum and return its spectrum_id."""
    if wavelengths is None:
        wavelengths = np.linspace(0.8, 2.5, 200)
    if reflectance is None:
        # Simulated S-type spectrum: gentle slope with Band I minimum
        reflectance = 0.8 + 0.3 * (wavelengths - 0.8) - 0.15 * np.exp(
            -((wavelengths - 1.0) ** 2) / (2 * 0.05**2)
        )

    wl_blob = wavelengths.astype(np.float64).tobytes()
    refl_blob = reflectance.astype(np.float64).tobytes()
    unc_blob = uncertainty.astype(np.float64).tobytes() if uncertainty is not None else None

    cur = conn.execute(
        "INSERT INTO spectra "
        "(asteroid_id, survey, wavelengths, reflectance, uncertainty, wl_min, wl_max) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            asteroid_id,
            survey,
            wl_blob,
            refl_blob,
            unc_blob,
            float(wavelengths.min()),
            float(wavelengths.max()),
        ),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------


class TestResample:
    def test_within_range_interpolated(self):
        wl = np.array([1.0, 1.5, 2.0])
        refl = np.array([0.5, 1.0, 0.8])
        grid = np.array([1.0, 1.25, 1.5, 1.75, 2.0])

        g, r, u = resample(wl, refl, None, grid)

        np.testing.assert_array_equal(g, grid)
        assert r[0] == pytest.approx(0.5)
        assert r[2] == pytest.approx(1.0)
        assert r[4] == pytest.approx(0.8)
        # Interpolated midpoints
        assert r[1] == pytest.approx(0.75)
        assert r[3] == pytest.approx(0.9)
        assert u is None

    def test_nan_outside_range(self):
        wl = np.array([1.0, 1.5, 2.0])
        refl = np.array([0.5, 1.0, 0.8])
        grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

        _, r, _ = resample(wl, refl, None, grid)

        assert np.isnan(r[0])  # 0.5 < 1.0
        assert not np.isnan(r[1])  # 1.0 in range
        assert not np.isnan(r[3])  # 2.0 in range
        assert np.isnan(r[4])  # 2.5 > 2.0

    def test_uncertainty_resampled(self):
        wl = np.array([1.0, 2.0])
        refl = np.array([1.0, 2.0])
        unc = np.array([0.1, 0.2])
        grid = np.array([1.0, 1.5, 2.0])

        _, _, u = resample(wl, refl, unc, grid)

        assert u is not None
        assert u[0] == pytest.approx(0.1)
        assert u[1] == pytest.approx(0.15)
        assert u[2] == pytest.approx(0.2)

    def test_default_grid_used(self):
        wl = np.array([0.8, 2.5])
        refl = np.array([1.0, 1.0])

        g, r, _ = resample(wl, refl, None)

        assert len(g) == len(DEFAULT_GRID)
        # Points within [0.8, 2.5] should be valid
        in_range = (g >= 0.8) & (g <= 2.5)
        assert not np.any(np.isnan(r[in_range]))
        # Points outside should be NaN
        out_range = (g < 0.8) | (g > 2.5)
        assert np.all(np.isnan(r[out_range]))

    def test_returns_copy_of_grid(self):
        wl = np.array([1.0, 2.0])
        refl = np.array([1.0, 1.0])
        grid = np.array([1.0, 1.5, 2.0])

        g, _, _ = resample(wl, refl, None, grid)

        # Modification should not affect original
        g[0] = 999.0
        assert grid[0] == 1.0


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_at_550nm(self):
        wl = np.array([0.5, 0.55, 0.6])
        refl = np.array([0.4, 0.5, 0.6])

        r, _ = normalize(wl, refl, None)

        # At 550nm, value was 0.5 → normalized to 1.0
        assert r[1] == pytest.approx(1.0)
        assert r[0] == pytest.approx(0.8)  # 0.4/0.5
        assert r[2] == pytest.approx(1.2)  # 0.6/0.5

    def test_interpolated_ref(self):
        wl = np.array([0.50, 0.60])
        refl = np.array([1.0, 2.0])

        r, _ = normalize(wl, refl, None)

        # ref at 0.55 → interp gives 1.5
        assert r[0] == pytest.approx(1.0 / 1.5)
        assert r[1] == pytest.approx(2.0 / 1.5)

    def test_fallback_nearest_when_out_of_range(self):
        # NIR-only spectrum (no 550nm coverage)
        wl = np.array([0.8, 1.0, 1.5, 2.0])
        refl = np.array([0.6, 0.8, 1.0, 0.9])

        r, _ = normalize(wl, refl, None)

        # Nearest to 550nm is 0.8μm → value 0.6
        assert r[0] == pytest.approx(1.0)  # 0.6/0.6
        assert r[1] == pytest.approx(0.8 / 0.6)

    def test_uncertainty_scaled(self):
        wl = np.array([0.5, 0.55, 0.6])
        refl = np.array([0.4, 0.5, 0.6])
        unc = np.array([0.01, 0.02, 0.03])

        _, u = normalize(wl, refl, unc)

        assert u is not None
        assert u[1] == pytest.approx(0.02 / 0.5)

    def test_all_nan_returns_copy(self):
        wl = np.array([0.5, 0.55, 0.6])
        refl = np.array([np.nan, np.nan, np.nan])

        r, _ = normalize(wl, refl, None)

        assert np.all(np.isnan(r))

    def test_zero_at_ref_returns_copy(self):
        wl = np.array([0.5, 0.55, 0.6])
        refl = np.array([0.0, 0.0, 0.0])

        r, _ = normalize(wl, refl, None)

        # Cannot normalize by zero — returns copy
        np.testing.assert_array_equal(r, refl)


# ---------------------------------------------------------------------------
# mask_telluric
# ---------------------------------------------------------------------------


class TestMaskTelluric:
    def test_masks_default_regions(self):
        wl = np.arange(0.8, 2.5, 0.01)
        refl = np.ones_like(wl)

        r, _ = mask_telluric(wl, refl, None)

        for lo, hi in TELLURIC_REGIONS:
            band = (wl >= lo) & (wl <= hi)
            assert np.all(np.isnan(r[band])), f"Not masked: {lo}-{hi}"

        # Outside telluric regions should be unmasked
        outside = np.ones(len(wl), dtype=bool)
        for lo, hi in TELLURIC_REGIONS:
            outside &= ~((wl >= lo) & (wl <= hi))
        assert not np.any(np.isnan(r[outside]))

    def test_custom_regions(self):
        wl = np.array([1.0, 1.1, 1.2, 1.3])
        refl = np.array([1.0, 1.0, 1.0, 1.0])

        r, _ = mask_telluric(wl, refl, None, regions=[(1.05, 1.15)])

        assert not np.isnan(r[0])
        assert np.isnan(r[1])
        assert not np.isnan(r[2])
        assert not np.isnan(r[3])

    def test_uncertainty_also_masked(self):
        wl = np.array([1.35, 1.40, 1.45, 1.50])
        refl = np.array([1.0, 1.0, 1.0, 1.0])
        unc = np.array([0.1, 0.1, 0.1, 0.1])

        r, u = mask_telluric(wl, refl, unc)

        assert np.isnan(r[0]) and np.isnan(r[1]) and np.isnan(r[2])
        assert u is not None
        assert np.isnan(u[0]) and np.isnan(u[1]) and np.isnan(u[2])
        assert not np.isnan(r[3]) and not np.isnan(u[3])

    def test_does_not_modify_originals(self):
        wl = np.array([1.35, 1.40, 1.50])
        refl = np.array([1.0, 1.0, 1.0])

        r, _ = mask_telluric(wl, refl, None)

        assert np.isnan(r[0])
        assert refl[0] == 1.0  # original untouched


# ---------------------------------------------------------------------------
# estimate_snr
# ---------------------------------------------------------------------------


class TestEstimateSNR:
    def test_from_uncertainty(self):
        refl = np.ones(50)
        unc = np.full(50, 0.01)

        snr = estimate_snr(refl, unc)

        assert snr == pytest.approx(100.0)  # 1.0 / 0.01

    def test_from_smoothness_heuristic(self):
        # Clean spectrum → high SNR
        refl = np.linspace(0.8, 1.2, 100)

        snr = estimate_snr(refl, None)

        assert snr > 50  # smooth spectrum should have high SNR

    def test_noisy_spectrum_lower_snr(self):
        rng = np.random.default_rng(42)
        refl = np.ones(100) + rng.normal(0, 0.1, 100)

        snr = estimate_snr(refl, None)

        # Noisy spectrum should have lower SNR
        assert snr < 50

    def test_too_few_channels(self):
        refl = np.array([1.0, 1.0])

        snr = estimate_snr(refl, None)

        assert np.isnan(snr)

    def test_nan_channels_excluded(self):
        refl = np.concatenate([np.ones(30), np.full(10, np.nan), np.ones(30)])
        unc = np.concatenate([np.full(30, 0.01), np.full(10, np.nan), np.full(30, 0.01)])

        snr = estimate_snr(refl, unc)

        assert snr == pytest.approx(100.0)

    def test_fallback_to_smoothness_when_unc_all_nan(self):
        refl = np.ones(50)
        unc = np.full(50, np.nan)

        snr = estimate_snr(refl, unc)

        # Should fall back to smoothness (flat spectrum → high SNR)
        assert not np.isnan(snr)
        assert snr > 10


# ---------------------------------------------------------------------------
# assess_quality
# ---------------------------------------------------------------------------


class TestAssessQuality:
    def test_good(self):
        assert assess_quality(50.0, 0.8, 2.5) == "good"

    def test_low_snr(self):
        assert assess_quality(10.0, 0.8, 2.5) == "low_snr"

    def test_nan_snr(self):
        assert assess_quality(float("nan"), 0.8, 2.5) == "low_snr"

    def test_partial_coverage(self):
        # Good SNR but narrow wavelength range
        assert assess_quality(50.0, 0.8, 1.0) == "partial"

    def test_snr_threshold_exact(self):
        # At threshold → good
        assert assess_quality(SNR_GOOD_THRESHOLD, 0.8, 2.5) == "good"

    def test_just_below_threshold(self):
        assert assess_quality(SNR_GOOD_THRESHOLD - 0.1, 0.8, 2.5) == "low_snr"


# ---------------------------------------------------------------------------
# preprocess_spectrum (integration)
# ---------------------------------------------------------------------------


class TestPreprocessSpectrum:
    def test_basic_preprocessing(self, conn):
        _insert_asteroid(conn)
        sid = _insert_spectrum(conn)

        result = preprocess_spectrum(sid, conn)

        assert result is True

        row = conn.execute(
            "SELECT normalized, snr_estimate, quality_flag, wavelengths, reflectance "
            "FROM spectra WHERE spectrum_id = ?",
            (sid,),
        ).fetchone()

        assert row[0] == 1  # normalized = TRUE (sqlite boolean)
        assert row[1] is not None  # snr_estimate set
        assert row[2] in ("good", "low_snr", "partial")
        # Wavelengths should now be the grid size
        wl = np.frombuffer(row[3], dtype=np.float64)
        assert len(wl) == len(DEFAULT_GRID)

    def test_normalization_applied(self, conn):
        _insert_asteroid(conn)
        # Simple flat spectrum at reflectance=2.0
        wl = np.linspace(0.45, 2.5, 200)
        refl = np.full_like(wl, 2.0)
        sid = _insert_spectrum(conn, wavelengths=wl, reflectance=refl)

        preprocess_spectrum(sid, conn)

        row = conn.execute(
            "SELECT wavelengths, reflectance FROM spectra WHERE spectrum_id = ?",
            (sid,),
        ).fetchone()
        wl_out = np.frombuffer(row[0], dtype=np.float64)
        refl_out = np.frombuffer(row[1], dtype=np.float64)

        # At the 550nm normalization point, reflectance should be 1.0
        idx_550 = np.argmin(np.abs(wl_out - NORM_WAVELENGTH))
        valid = ~np.isnan(refl_out)
        if valid[idx_550]:
            assert refl_out[idx_550] == pytest.approx(1.0, abs=0.05)

    def test_telluric_masked_for_ground_based(self, conn):
        _insert_asteroid(conn)
        wl = np.linspace(0.8, 2.5, 500)
        refl = np.ones_like(wl)
        sid = _insert_spectrum(conn, survey="MITHNEOS", wavelengths=wl, reflectance=refl)

        preprocess_spectrum(sid, conn)

        row = conn.execute(
            "SELECT wavelengths, reflectance FROM spectra WHERE spectrum_id = ?",
            (sid,),
        ).fetchone()
        wl_out = np.frombuffer(row[0], dtype=np.float64)
        refl_out = np.frombuffer(row[1], dtype=np.float64)

        # Channels in telluric regions should be NaN
        for lo, hi in TELLURIC_REGIONS:
            band = (wl_out >= lo) & (wl_out <= hi)
            if band.any():
                assert np.all(np.isnan(refl_out[band]))

    def test_telluric_not_masked_for_space(self, conn):
        _insert_asteroid(conn)
        wl = np.linspace(0.374, 1.034, 16)
        refl = np.ones_like(wl)
        sid = _insert_spectrum(conn, survey="Gaia", wavelengths=wl, reflectance=refl)

        preprocess_spectrum(sid, conn)

        row = conn.execute(
            "SELECT reflectance FROM spectra WHERE spectrum_id = ?",
            (sid,),
        ).fetchone()
        refl_out = np.frombuffer(row[0], dtype=np.float64)

        # No telluric masking for space-based — valid channels should not be NaN
        valid = ~np.isnan(refl_out)
        assert valid.sum() > 0

    def test_not_found_returns_false(self, conn):
        assert preprocess_spectrum(9999, conn) is False

    def test_with_uncertainty(self, conn):
        _insert_asteroid(conn)
        wl = np.linspace(0.8, 2.5, 200)
        refl = np.ones_like(wl)
        unc = np.full_like(wl, 0.01)
        sid = _insert_spectrum(conn, wavelengths=wl, reflectance=refl, uncertainty=unc)

        preprocess_spectrum(sid, conn)

        row = conn.execute(
            "SELECT uncertainty, snr_estimate FROM spectra WHERE spectrum_id = ?",
            (sid,),
        ).fetchone()

        assert row[0] is not None  # uncertainty blob preserved
        assert row[1] is not None and row[1] > 0

    def test_quality_flag_set(self, conn):
        _insert_asteroid(conn)
        # High-quality wide-coverage spectrum with low noise
        wl = np.linspace(0.8, 2.5, 200)
        refl = np.ones_like(wl)
        unc = np.full_like(wl, 0.01)
        sid = _insert_spectrum(conn, wavelengths=wl, reflectance=refl, uncertainty=unc)

        preprocess_spectrum(sid, conn)

        row = conn.execute(
            "SELECT quality_flag FROM spectra WHERE spectrum_id = ?",
            (sid,),
        ).fetchone()

        assert row[0] == "good"

    def test_partial_coverage_flagged(self, conn):
        _insert_asteroid(conn)
        # Narrow-band spectrum
        wl = np.linspace(0.8, 1.0, 50)
        refl = np.ones_like(wl)
        unc = np.full_like(wl, 0.01)
        sid = _insert_spectrum(conn, wavelengths=wl, reflectance=refl, uncertainty=unc)

        preprocess_spectrum(sid, conn)

        row = conn.execute(
            "SELECT quality_flag FROM spectra WHERE spectrum_id = ?",
            (sid,),
        ).fetchone()

        assert row[0] == "partial"


# ---------------------------------------------------------------------------
# preprocess_all (batch)
# ---------------------------------------------------------------------------


class TestPreprocessAll:
    def test_processes_unnormalized(self, conn):
        _insert_asteroid(conn)
        sid1 = _insert_spectrum(conn)
        sid2 = _insert_spectrum(conn)

        count = preprocess_all(conn)

        assert count == 2

        for sid in (sid1, sid2):
            row = conn.execute(
                "SELECT normalized FROM spectra WHERE spectrum_id = ?",
                (sid,),
            ).fetchone()
            assert row[0] == 1

    def test_skips_already_normalized(self, conn):
        _insert_asteroid(conn)
        sid = _insert_spectrum(conn)

        # Manually mark as normalized
        conn.execute("UPDATE spectra SET normalized = TRUE WHERE spectrum_id = ?", (sid,))
        conn.commit()

        count = preprocess_all(conn)

        assert count == 0

    def test_returns_zero_when_no_spectra(self, conn):
        count = preprocess_all(conn)
        assert count == 0

    def test_mixed_surveys(self, conn):
        _insert_asteroid(conn, 4179, "Toutatis")
        _insert_asteroid(conn, 433, "Eros")

        wl_nir = np.linspace(0.8, 2.5, 200)
        wl_vis = np.linspace(0.374, 1.034, 16)

        _insert_spectrum(conn, asteroid_id=4179, survey="MITHNEOS",
                         wavelengths=wl_nir, reflectance=np.ones_like(wl_nir))
        _insert_spectrum(conn, asteroid_id=433, survey="Gaia",
                         wavelengths=wl_vis, reflectance=np.ones_like(wl_vis))

        count = preprocess_all(conn)

        assert count == 2

        # Both should now be normalized
        rows = conn.execute(
            "SELECT survey, normalized, quality_flag FROM spectra ORDER BY survey"
        ).fetchall()
        assert all(r[1] == 1 for r in rows)
