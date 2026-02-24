"""Tests for prospector.spectral.band_analysis — Stage 2 band parameter extraction."""

import numpy as np
import pytest

from prospector.db import get_connection, init_schema
from prospector.spectral.band_analysis import (
    BAND1_SEARCH,
    BAND2_SEARCH,
    LAB_TEMPERATURE,
    SILICATE_CLASSES,
    analyze_all,
    analyze_asteroid,
    analyze_spectrum,
    classify_gaffey_subtype,
    compute_band_area,
    dunn_calibration,
    estimate_temperature,
    find_band_center,
    find_peak,
    linear_continuum,
    lindsay_bar_correction,
    remove_continuum,
    temperature_correct_band_centers,
)


# ---- Helpers ----

def make_s_type_spectrum(bic=0.95, biic=1.95, depth1=0.30, depth2=0.15):
    """Create a synthetic S-type asteroid spectrum with known band positions."""
    wl = np.arange(0.45, 2.50, 0.005)
    # Gently rising continuum
    continuum = 0.9 + 0.15 * (wl - 0.45) / 2.0
    # Band I: Gaussian absorption
    band1 = depth1 * np.exp(-((wl - bic) ** 2) / (2 * 0.07 ** 2))
    # Band II: broader Gaussian absorption
    band2 = depth2 * np.exp(-((wl - biic) ** 2) / (2 * 0.12 ** 2))
    refl = continuum - band1 - band2
    return wl, refl


def make_olivine_spectrum():
    """Create a synthetic olivine-dominated spectrum (no significant Band II)."""
    wl = np.arange(0.45, 2.50, 0.005)
    continuum = 0.9 + 0.10 * (wl - 0.45) / 2.0
    # Strong Band I centered at 1.06 um (olivine)
    band1 = 0.25 * np.exp(-((wl - 1.06) ** 2) / (2 * 0.08 ** 2))
    # Very weak/no Band II
    refl = continuum - band1
    return wl, refl


def setup_test_db():
    """Create an in-memory DB with schema and a test S-type asteroid."""
    conn = get_connection(":memory:")
    conn.execute(
        "INSERT INTO asteroids (asteroid_id, designation, neo) VALUES (?, ?, ?)",
        (4179, "1989 FB", 1),
    )
    conn.execute(
        "INSERT INTO orbits (asteroid_id, a, e, i) VALUES (?, ?, ?, ?)",
        (4179, 2.51, 0.63, 0.45),
    )
    conn.execute(
        "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, classifier, input_coverage) "
        "VALUES (?, ?, ?, ?, ?)",
        (4179, "S", 0.85, "classy_mahlke2022", "vnir"),
    )
    # Insert normalized spectrum with NIR coverage
    wl, refl = make_s_type_spectrum()
    conn.execute(
        "INSERT INTO spectra (asteroid_id, survey, wavelengths, reflectance, "
        "wl_min, wl_max, normalized, quality_flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (4179, "MITHNEOS", wl.tobytes(), refl.tobytes(), 0.45, 2.495, True, "good"),
    )
    conn.commit()
    return conn


# ---- Unit tests: find_peak ----

class TestFindPeak:
    def test_finds_maximum(self):
        wl = np.arange(0.5, 1.0, 0.01)
        refl = np.sin((wl - 0.5) * np.pi / 0.5)  # peak at 0.75
        peak = find_peak(wl, refl, (0.5, 1.0))
        assert abs(peak - 0.75) < 0.02

    def test_handles_nan(self):
        wl = np.arange(0.6, 0.9, 0.01)
        refl = np.ones_like(wl)
        refl[10] = 1.5  # peak at index 10
        refl[5:8] = np.nan
        peak = find_peak(wl, refl, (0.6, 0.9))
        assert peak == pytest.approx(wl[10], abs=0.01)

    def test_returns_none_insufficient_data(self):
        wl = np.array([0.7, 0.75])
        refl = np.array([1.0, 1.1])
        assert find_peak(wl, refl, (0.6, 0.8)) is None

    def test_region_filtering(self):
        wl = np.arange(0.5, 1.5, 0.01)
        refl = np.ones_like(wl)
        refl[20] = 1.5  # peak at 0.7
        refl[80] = 2.0  # higher peak at 1.3 (outside region)
        peak = find_peak(wl, refl, (0.6, 0.8))
        assert abs(peak - 0.7) < 0.02


# ---- Unit tests: continuum ----

class TestContinuum:
    def test_linear_continuum(self):
        wl = np.arange(0.5, 1.5, 0.01)
        refl = 0.8 + 0.2 * (wl - 0.5)  # linear from 0.8 to 1.0
        cont = linear_continuum(wl, refl, 0.5, 1.49)
        np.testing.assert_allclose(cont, refl, atol=0.01)

    def test_remove_continuum_flat(self):
        wl = np.arange(0.5, 1.5, 0.01)
        refl = np.ones_like(wl)
        cr = remove_continuum(wl, refl, 0.5, 1.49)
        valid = ~np.isnan(cr)
        np.testing.assert_allclose(cr[valid], 1.0, atol=1e-10)

    def test_remove_continuum_absorption(self):
        wl = np.arange(0.5, 1.5, 0.01)
        refl = np.ones_like(wl) - 0.3 * np.exp(-((wl - 1.0) ** 2) / (2 * 0.05 ** 2))
        cr = remove_continuum(wl, refl, 0.5, 1.49)
        # Minimum of continuum-removed should be < 1.0
        valid = ~np.isnan(cr)
        assert cr[valid].min() < 0.8


# ---- Unit tests: find_band_center ----

class TestFindBandCenter:
    def test_known_center(self):
        wl = np.arange(0.8, 1.2, 0.005)
        # Absorption centered at 0.95
        cr = 1.0 - 0.3 * np.exp(-((wl - 0.95) ** 2) / (2 * 0.04 ** 2))
        center = find_band_center(wl, cr, (0.82, 1.15))
        assert center == pytest.approx(0.95, abs=0.01)

    def test_offset_center(self):
        wl = np.arange(0.8, 1.2, 0.005)
        cr = 1.0 - 0.25 * np.exp(-((wl - 1.02) ** 2) / (2 * 0.04 ** 2))
        center = find_band_center(wl, cr, (0.82, 1.15))
        assert center == pytest.approx(1.02, abs=0.015)

    def test_no_absorption(self):
        wl = np.arange(0.8, 1.2, 0.005)
        cr = np.ones_like(wl)  # flat, no absorption
        center = find_band_center(wl, cr, (0.82, 1.15))
        assert center is None

    def test_handles_nan_gap(self):
        wl = np.arange(0.8, 1.2, 0.005)
        cr = 1.0 - 0.3 * np.exp(-((wl - 0.95) ** 2) / (2 * 0.04 ** 2))
        cr[(wl > 0.93) & (wl < 0.97)] = np.nan  # gap right at center
        center = find_band_center(wl, cr, (0.82, 1.15))
        # Polynomial should interpolate through gap
        assert center is not None
        assert center == pytest.approx(0.95, abs=0.03)


# ---- Unit tests: compute_band_area ----

class TestComputeBandArea:
    def test_positive_area(self):
        wl = np.arange(0.7, 1.5, 0.005)
        cr = 1.0 - 0.3 * np.exp(-((wl - 1.0) ** 2) / (2 * 0.06 ** 2))
        area = compute_band_area(wl, cr, 0.7, 1.5)
        assert area is not None
        assert area > 0

    def test_deeper_band_larger_area(self):
        wl = np.arange(0.7, 1.5, 0.005)
        cr_shallow = 1.0 - 0.1 * np.exp(-((wl - 1.0) ** 2) / (2 * 0.06 ** 2))
        cr_deep = 1.0 - 0.4 * np.exp(-((wl - 1.0) ** 2) / (2 * 0.06 ** 2))
        area_shallow = compute_band_area(wl, cr_shallow, 0.7, 1.5)
        area_deep = compute_band_area(wl, cr_deep, 0.7, 1.5)
        assert area_deep > area_shallow

    def test_no_absorption_returns_none(self):
        wl = np.arange(0.7, 1.5, 0.005)
        cr = np.ones_like(wl) + 0.1  # above 1.0, no absorption
        area = compute_band_area(wl, cr, 0.7, 1.5)
        assert area is None


# ---- Unit tests: temperature ----

class TestTemperature:
    def test_estimate_1au(self):
        t = estimate_temperature(1.0)
        assert t == pytest.approx(280.0, abs=1.0)

    def test_estimate_2au(self):
        t = estimate_temperature(2.0)
        assert t == pytest.approx(280.0 / np.sqrt(2.0), abs=1.0)

    def test_none_returns_lab(self):
        assert estimate_temperature(None) == LAB_TEMPERATURE

    def test_correction_at_lab_temp(self):
        bic_c, biic_c = temperature_correct_band_centers(0.95, 1.95, LAB_TEMPERATURE)
        assert bic_c == pytest.approx(0.95)
        assert biic_c == pytest.approx(1.95)

    def test_correction_hotter(self):
        # Hotter asteroid → observed bands shift to longer wavelengths
        # Correction should shift them back (shorter)
        bic_c, biic_c = temperature_correct_band_centers(0.95, 1.95, 350.0)
        assert bic_c < 0.95
        assert biic_c < 1.95

    def test_correction_cooler(self):
        bic_c, biic_c = temperature_correct_band_centers(0.95, 1.95, 250.0)
        assert bic_c > 0.95
        assert biic_c > 1.95


# ---- Unit tests: Lindsay red-edge correction ----

class TestLindsayCorrection:
    def test_no_correction_at_standard(self):
        assert lindsay_bar_correction(1.0, 2.44) == 1.0

    def test_longer_red_edge_reduces_bar(self):
        bar_corr = lindsay_bar_correction(1.0, 2.50)
        assert bar_corr < 1.0

    def test_shorter_red_edge_increases_bar(self):
        bar_corr = lindsay_bar_correction(1.0, 2.38)
        assert bar_corr > 1.0


# ---- Unit tests: Gaffey classification ----

class TestGaffeySubtype:
    def test_s1_pure_olivine(self):
        assert classify_gaffey_subtype(1.06, 0.05) == "S(I)"

    def test_s2_olivine_pyroxene(self):
        assert classify_gaffey_subtype(1.03, 0.25) == "S(II)"

    def test_s3_transitional(self):
        assert classify_gaffey_subtype(0.96, 0.50) == "S(III)"

    def test_s4_ordinary_chondrite(self):
        assert classify_gaffey_subtype(0.95, 1.0) == "S(IV)"

    def test_s5_pyroxene_rich(self):
        assert classify_gaffey_subtype(0.91, 0.80) == "S(V)"

    def test_s6_opx_dominant(self):
        assert classify_gaffey_subtype(0.91, 1.30) == "S(VI)"

    def test_s7_basaltic(self):
        assert classify_gaffey_subtype(0.93, 2.00) == "S(VII)"


# ---- Unit tests: Dunn calibration ----

class TestDunnCalibration:
    def test_ol_ratio_at_bar_1(self):
        result = dunn_calibration(1.0, 0.95, 1.95)
        # ol/(ol+px) = -0.242 * 1.0 + 0.728 = 0.486
        assert result["ol_opx_ratio"] == pytest.approx(0.486, abs=0.001)

    def test_ol_ratio_clamped_low(self):
        result = dunn_calibration(5.0, 0.95, 1.95)
        assert result["ol_opx_ratio"] == 0.0

    def test_ol_ratio_clamped_high(self):
        result = dunn_calibration(0.0, 0.95, 1.95)
        assert result["ol_opx_ratio"] == pytest.approx(0.728, abs=0.001)

    def test_fa_nonnegative(self):
        result = dunn_calibration(1.0, 0.95, 1.95)
        assert result["fa_mol_pct"] >= 0.0

    def test_fs_none_without_biic(self):
        result = dunn_calibration(1.0, 0.95, None)
        assert result["fs_mol_pct"] is None

    def test_fs_nonnegative(self):
        result = dunn_calibration(1.0, 0.95, 1.95)
        assert result["fs_mol_pct"] is not None
        assert result["fs_mol_pct"] >= 0.0


# ---- Integration tests: analyze_spectrum ----

class TestAnalyzeSpectrum:
    def test_synthetic_s_type(self):
        wl, refl = make_s_type_spectrum(bic=0.95, biic=1.95)
        result = analyze_spectrum(wl, refl)
        assert result is not None
        assert result["band1_center"] == pytest.approx(0.95, abs=0.03)
        assert result["band2_center"] is not None
        assert result["bar"] is not None
        assert result["bar"] > 0

    def test_synthetic_s_type_gaffey_subtype(self):
        wl, refl = make_s_type_spectrum(bic=0.95, biic=1.95)
        result = analyze_spectrum(wl, refl)
        assert result is not None
        assert result["gaffey_subtype"] is not None
        assert result["gaffey_subtype"].startswith("S(")

    def test_temperature_correction_applied(self):
        wl, refl = make_s_type_spectrum(bic=0.95, biic=1.95)
        # NEO at 1 AU: T=280K < 300K → correction shifts centers up
        result = analyze_spectrum(wl, refl, semi_major_axis=1.0)
        assert result is not None
        assert result["band1_center"] > 0.94  # shifted from raw

    def test_olivine_spectrum(self):
        wl, refl = make_olivine_spectrum()
        result = analyze_spectrum(wl, refl)
        assert result is not None
        assert result["band1_center"] == pytest.approx(1.06, abs=0.03)
        # Very low or no BAR for pure olivine
        if result["bar"] is not None:
            assert result["bar"] < 0.20

    def test_telluric_masked_spectrum(self):
        wl, refl = make_s_type_spectrum(bic=0.95, biic=1.95)
        # Mask telluric regions
        refl[(wl >= 1.35) & (wl <= 1.45)] = np.nan
        refl[(wl >= 1.80) & (wl <= 2.00)] = np.nan
        result = analyze_spectrum(wl, refl)
        assert result is not None
        assert result["band1_center"] == pytest.approx(0.95, abs=0.04)

    def test_flat_spectrum_returns_none(self):
        wl = np.arange(0.45, 2.50, 0.005)
        refl = np.ones_like(wl)
        result = analyze_spectrum(wl, refl)
        assert result is None

    def test_insufficient_coverage_returns_none(self):
        # Visible-only spectrum (no NIR)
        wl = np.arange(0.45, 0.90, 0.005)
        refl = np.ones_like(wl)
        result = analyze_spectrum(wl, refl)
        assert result is None


# ---- Integration tests: DB ----

class TestAnalyzeAsteroid:
    def test_analyze_s_type(self):
        conn = setup_test_db()
        success = analyze_asteroid(4179, conn)
        assert success

        row = conn.execute(
            "SELECT band1_center, bar, gaffey_subtype, calibration "
            "FROM band_analysis WHERE asteroid_id = 4179"
        ).fetchone()
        assert row is not None
        band1_center, bar, gaffey, calibration = row
        assert 0.85 <= band1_center <= 1.10
        assert bar is not None
        assert gaffey is not None
        assert calibration in ("dunn2010", "gaffey1993_zone")

    def test_skips_non_silicate(self):
        conn = setup_test_db()
        # Change taxonomy to C-type
        conn.execute("UPDATE taxonomy SET primary_class = 'C' WHERE asteroid_id = 4179")
        assert not analyze_asteroid(4179, conn)

    def test_skips_no_nir(self):
        conn = setup_test_db()
        # Change spectrum to visible-only
        conn.execute("UPDATE spectra SET wl_max = 0.9 WHERE asteroid_id = 4179")
        assert not analyze_asteroid(4179, conn)

    def test_skips_no_taxonomy(self):
        conn = setup_test_db()
        conn.execute("DELETE FROM taxonomy WHERE asteroid_id = 4179")
        assert not analyze_asteroid(4179, conn)

    def test_idempotent(self):
        conn = setup_test_db()
        analyze_asteroid(4179, conn)
        conn.commit()
        # Running again should replace (INSERT OR REPLACE)
        analyze_asteroid(4179, conn)
        count = conn.execute("SELECT COUNT(*) FROM band_analysis").fetchone()[0]
        assert count == 1


class TestAnalyzeAll:
    def test_batch(self):
        conn = setup_test_db()
        # Add a second asteroid
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, designation, neo) VALUES (?, ?, ?)",
            (25143, "1998 SF36", 1),
        )
        conn.execute(
            "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, "
            "classifier, input_coverage) VALUES (?, ?, ?, ?, ?)",
            (25143, "Q", 0.90, "classy_mahlke2022", "vnir"),
        )
        wl, refl = make_s_type_spectrum(bic=0.97, biic=1.98)
        conn.execute(
            "INSERT INTO spectra (asteroid_id, survey, wavelengths, reflectance, "
            "wl_min, wl_max, normalized, quality_flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (25143, "MITHNEOS", wl.tobytes(), refl.tobytes(), 0.45, 2.495, True, "good"),
        )
        conn.commit()

        count = analyze_all(conn)
        assert count == 2

        rows = conn.execute("SELECT asteroid_id FROM band_analysis ORDER BY asteroid_id").fetchall()
        assert [r[0] for r in rows] == [4179, 25143]

    def test_skips_already_analyzed(self):
        conn = setup_test_db()
        analyze_all(conn)
        # Second call should find nothing new
        count = analyze_all(conn)
        assert count == 0

    def test_empty_db(self):
        conn = get_connection(":memory:")
        count = analyze_all(conn)
        assert count == 0
