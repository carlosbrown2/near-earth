"""Tests for prospector.spectral.cnn_mineral — Stage 2B CNN mineral quantification."""

import sqlite3

import numpy as np
import pytest

from prospector.db import get_connection, init_schema
from prospector.spectral.cnn_mineral import (
    CNN_GRID,
    CNN_N_CHANNELS,
    CNN_WL_MAX,
    CNN_WL_MIN,
    FAIR_AGREEMENT,
    GOOD_AGREEMENT,
    MIN_VALID_FRACTION,
    SILICATE_CLASSES,
    _fill_nan_channels,
    assess_agreement,
    predict_asteroid,
    predict_all,
    predict_spectrum,
    resample_to_cnn_grid,
)


# ---- Helpers ----

def make_s_type_spectrum(bic=0.95, biic=1.95, depth1=0.30, depth2=0.15):
    """Create a synthetic S-type asteroid spectrum with known band positions."""
    wl = np.arange(0.45, 2.50, 0.005)
    continuum = 0.9 + 0.15 * (wl - 0.45) / 2.0
    band1 = depth1 * np.exp(-((wl - bic) ** 2) / (2 * 0.07 ** 2))
    band2 = depth2 * np.exp(-((wl - biic) ** 2) / (2 * 0.12 ** 2))
    refl = continuum - band1 - band2
    return wl, refl


def make_nir_only_spectrum():
    """Create NIR-only spectrum (0.8-2.5 μm)."""
    wl = np.arange(0.80, 2.50, 0.005)
    refl = 0.9 + 0.1 * (wl - 0.8) / 1.7
    return wl, refl


def setup_test_db(include_band_analysis=False):
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
        "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, "
        "prob_vector, classifier, input_coverage) VALUES (?, ?, ?, ?, ?, ?)",
        (4179, "S", 0.85, np.zeros(17).tobytes(), "classy_mahlke2022", "vnir"),
    )
    wl, refl = make_s_type_spectrum()
    conn.execute(
        "INSERT INTO spectra (asteroid_id, survey, wavelengths, reflectance, "
        "wl_min, wl_max, normalized, quality_flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (4179, "MITHNEOS", wl.tobytes(), refl.tobytes(), 0.45, 2.495, True, "good"),
    )
    if include_band_analysis:
        conn.execute(
            "INSERT INTO band_analysis (asteroid_id, band1_center, band2_center, "
            "bar, ol_opx_ratio, gaffey_subtype, calibration) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (4179, 0.95, 1.95, 1.0, 0.50, "S(IV)", "dunn2010"),
        )
    conn.commit()
    return conn


def make_mock_prediction(ol=45.0, opx=35.0, cpx=20.0, fa=18.0, fs=16.0, wo=5.0):
    """Create a mock CNN prediction result."""
    return {
        "ol_pct": ol,
        "opx_pct": opx,
        "cpx_pct": cpx,
        "fa_mol_pct": fa,
        "fs_mol_pct": fs,
        "wo_mol_pct": wo,
        "ol_unc": 3.0,
        "opx_unc": 3.5,
        "cpx_unc": 2.0,
        "fa_unc": 2.5,
        "fs_unc": 2.0,
        "wo_unc": 1.5,
        "method": "korda2023",
    }


# ---- Resampling tests ----

class TestResampleToCnnGrid:
    """Tests for resample_to_cnn_grid()."""

    def test_full_vnir_spectrum(self):
        wl, refl = make_s_type_spectrum()
        result = resample_to_cnn_grid(wl, refl)
        assert result is not None
        assert len(result) == CNN_N_CHANNELS

    def test_nan_outside_range(self):
        """Values outside original range should be NaN."""
        wl = np.arange(0.50, 2.40, 0.005)
        refl = np.ones_like(wl) * 0.9
        result = resample_to_cnn_grid(wl, refl)
        assert result is not None
        # Points below 0.50 should be NaN
        assert np.isnan(result[0])
        # Points within range should be valid
        mid_idx = CNN_N_CHANNELS // 2
        assert not np.isnan(result[mid_idx])

    def test_narrow_spectrum_rejected(self):
        """Spectrum with < 1.0 μm coverage should be rejected."""
        wl = np.arange(0.80, 1.20, 0.005)
        refl = np.ones_like(wl) * 0.9
        result = resample_to_cnn_grid(wl, refl)
        assert result is None

    def test_few_valid_points_rejected(self):
        """Spectrum with < 10 valid points should be rejected."""
        wl = np.arange(0.45, 2.50, 0.005)
        refl = np.full_like(wl, np.nan)
        refl[:5] = 1.0
        result = resample_to_cnn_grid(wl, refl)
        assert result is None

    def test_all_nan_rejected(self):
        wl = np.arange(0.45, 2.50, 0.005)
        refl = np.full_like(wl, np.nan)
        result = resample_to_cnn_grid(wl, refl)
        assert result is None

    def test_output_on_cnn_grid(self):
        """Resampled output should be on the standard CNN grid."""
        wl, refl = make_s_type_spectrum()
        result = resample_to_cnn_grid(wl, refl)
        assert result is not None
        assert len(result) == len(CNN_GRID)

    def test_telluric_gaps_handled(self):
        """Telluric NaN gaps are interpolated through by interp1d."""
        wl, refl = make_s_type_spectrum()
        telluric = (wl >= 1.35) & (wl <= 1.45)
        refl[telluric] = np.nan
        result = resample_to_cnn_grid(wl, refl)
        assert result is not None
        # interp1d skips NaN inputs but interpolates the output grid
        # through the gap — valid result in telluric region
        cnn_telluric = (CNN_GRID >= 1.36) & (CNN_GRID <= 1.44)
        # Result should still be finite (interpolated through gap)
        assert np.isfinite(result[cnn_telluric]).all()

    def test_min_valid_fraction(self):
        """Spectrum with too few valid channels after resampling is rejected."""
        wl = np.arange(0.45, 2.50, 0.005)
        refl = np.ones_like(wl) * 0.9
        # NaN out most of the spectrum (keep only a small chunk)
        refl[:50] = np.nan
        refl[100:] = np.nan
        result = resample_to_cnn_grid(wl, refl)
        # Only ~50 channels valid out of 401 = 12.5% < 80%
        assert result is None


# ---- NaN gap filling tests ----

class TestFillNanChannels:
    """Tests for _fill_nan_channels()."""

    def test_no_nans_unchanged(self):
        refl = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        result = _fill_nan_channels(refl)
        np.testing.assert_array_equal(result, refl)

    def test_interior_nans_interpolated(self):
        refl = np.array([1.0, np.nan, np.nan, 0.7, 0.6])
        result = _fill_nan_channels(refl)
        assert not np.isnan(result[1])
        assert not np.isnan(result[2])
        # Linear interpolation: 1.0 → 0.7 over 3 steps
        assert abs(result[1] - 0.9) < 0.01
        assert abs(result[2] - 0.8) < 0.01

    def test_returns_copy(self):
        refl = np.array([1.0, np.nan, 0.8])
        result = _fill_nan_channels(refl)
        assert result is not refl

    def test_all_nans_unchanged(self):
        refl = np.full(5, np.nan)
        result = _fill_nan_channels(refl)
        assert np.isnan(result).all()

    def test_single_valid(self):
        refl = np.full(5, np.nan)
        refl[2] = 1.0
        result = _fill_nan_channels(refl)
        # Only one valid point — can't interpolate
        assert np.isnan(result).sum() == 4


# ---- Agreement assessment tests ----

class TestAssessAgreement:
    """Tests for assess_agreement()."""

    def test_good_agreement(self):
        cnn = make_mock_prediction(ol=55.0, opx=45.0)  # ratio=0.55
        classical = {"ol_opx_ratio": 0.50}  # diff=5pp → good
        assert assess_agreement(cnn, classical) == "good"

    def test_fair_agreement(self):
        cnn = make_mock_prediction(ol=70.0, opx=30.0)  # ratio=0.70
        classical = {"ol_opx_ratio": 0.50}  # diff=20pp → fair
        assert assess_agreement(cnn, classical) == "fair"

    def test_poor_agreement(self):
        cnn = make_mock_prediction(ol=90.0, opx=10.0)  # ratio=0.90
        classical = {"ol_opx_ratio": 0.40}  # diff=50pp → poor
        assert assess_agreement(cnn, classical) == "poor"

    def test_no_classical_result(self):
        cnn = make_mock_prediction()
        assert assess_agreement(cnn, None) == "no_classical"

    def test_no_classical_ol_opx(self):
        cnn = make_mock_prediction()
        assert assess_agreement(cnn, {"ol_opx_ratio": None}) == "no_classical"

    def test_zero_ol_and_opx(self):
        cnn = make_mock_prediction(ol=0.0, opx=0.0, cpx=100.0)
        classical = {"ol_opx_ratio": 0.5}
        assert assess_agreement(cnn, classical) == "poor"


# ---- Domain check tests ----

class TestDomainChecks:
    """Tests for S*-complex domain enforcement."""

    def test_silicate_classes_match_band_analysis(self):
        """CNN SILICATE_CLASSES should match band_analysis SILICATE_CLASSES."""
        from prospector.spectral.band_analysis import SILICATE_CLASSES as BA_CLASSES
        assert SILICATE_CLASSES == BA_CLASSES

    def test_c_type_rejected(self):
        """C-type asteroid should not get CNN prediction."""
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, designation, neo) VALUES (?, ?, ?)",
            (101955, "Bennu", 1),
        )
        conn.execute(
            "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, "
            "prob_vector, classifier, input_coverage) VALUES (?, ?, ?, ?, ?, ?)",
            (101955, "C", 0.90, np.zeros(17).tobytes(), "classy_mahlke2022", "vnir"),
        )
        wl, refl = make_s_type_spectrum()
        conn.execute(
            "INSERT INTO spectra (asteroid_id, survey, wavelengths, reflectance, "
            "wl_min, wl_max, normalized, quality_flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (101955, "MITHNEOS", wl.tobytes(), refl.tobytes(), 0.45, 2.495, True, "good"),
        )
        conn.commit()
        assert predict_asteroid(101955, conn) is False

    def test_m_type_rejected(self):
        """M-type asteroid should not get CNN prediction."""
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, designation, neo) VALUES (?, ?, ?)",
            (16, "Psyche", 0),
        )
        conn.execute(
            "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, "
            "prob_vector, classifier, input_coverage) VALUES (?, ?, ?, ?, ?, ?)",
            (16, "M", 0.92, np.zeros(17).tobytes(), "classy_mahlke2022", "vnir"),
        )
        conn.commit()
        assert predict_asteroid(16, conn) is False

    def test_no_taxonomy_rejected(self):
        """Asteroid without taxonomy entry should be skipped."""
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, designation, neo) VALUES (?, ?, ?)",
            (4179, "1989 FB", 1),
        )
        conn.commit()
        assert predict_asteroid(4179, conn) is False

    def test_no_spectrum_rejected(self):
        """S-type asteroid without VNIR spectrum should be skipped."""
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, designation, neo) VALUES (?, ?, ?)",
            (4179, "1989 FB", 1),
        )
        conn.execute(
            "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, "
            "prob_vector, classifier, input_coverage) VALUES (?, ?, ?, ?, ?, ?)",
            (4179, "S", 0.85, np.zeros(17).tobytes(), "classy_mahlke2022", "vnir"),
        )
        conn.commit()
        assert predict_asteroid(4179, conn) is False


# ---- CNN grid constants tests ----

class TestCnnConstants:
    """Verify CNN grid configuration matches Korda 2023 spec."""

    def test_grid_range(self):
        assert CNN_WL_MIN == 0.45
        assert CNN_WL_MAX == 2.45

    def test_grid_spacing(self):
        diffs = np.diff(CNN_GRID)
        np.testing.assert_allclose(diffs, 0.005, atol=1e-10)

    def test_grid_length(self):
        expected = int((CNN_WL_MAX - CNN_WL_MIN) / 0.005) + 1
        assert CNN_N_CHANNELS == expected

    def test_min_valid_fraction(self):
        assert 0.5 <= MIN_VALID_FRACTION <= 0.95


# ---- Schema tests ----

class TestCnnMineralTable:
    """Tests for the cnn_mineral table in the schema."""

    def test_table_exists(self):
        conn = get_connection(":memory:")
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "cnn_mineral" in tables

    def test_insert_and_retrieve(self):
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, designation, neo) VALUES (?, ?, ?)",
            (4179, "1989 FB", 1),
        )
        conn.execute(
            "INSERT INTO cnn_mineral "
            "(asteroid_id, ol_pct, opx_pct, cpx_pct, fa_mol_pct, fs_mol_pct, wo_mol_pct, "
            "ol_unc, opx_unc, cpx_unc, fa_unc, fs_unc, wo_unc, "
            "classical_agreement, method) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (4179, 45.0, 35.0, 20.0, 18.0, 16.0, 5.0,
             3.0, 3.5, 2.0, 2.5, 2.0, 1.5, "good", "korda2023"),
        )
        conn.commit()

        row = conn.execute(
            "SELECT ol_pct, opx_pct, cpx_pct, fa_mol_pct, fs_mol_pct, wo_mol_pct, "
            "classical_agreement, method FROM cnn_mineral WHERE asteroid_id = ?",
            (4179,),
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(45.0)
        assert row[1] == pytest.approx(35.0)
        assert row[2] == pytest.approx(20.0)
        assert row[6] == "good"
        assert row[7] == "korda2023"

    def test_fk_constraint(self):
        """FK to asteroids table should be enforced."""
        conn = get_connection(":memory:")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO cnn_mineral (asteroid_id, ol_pct) VALUES (?, ?)",
                (99999, 45.0),
            )

    def test_insert_or_replace(self):
        """INSERT OR REPLACE should update existing entries."""
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, designation, neo) VALUES (?, ?, ?)",
            (4179, "1989 FB", 1),
        )
        conn.execute(
            "INSERT INTO cnn_mineral (asteroid_id, ol_pct, method) VALUES (?, ?, ?)",
            (4179, 45.0, "korda2023"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO cnn_mineral (asteroid_id, ol_pct, method) VALUES (?, ?, ?)",
            (4179, 55.0, "korda2023"),
        )
        conn.commit()
        row = conn.execute(
            "SELECT ol_pct FROM cnn_mineral WHERE asteroid_id = ?",
            (4179,),
        ).fetchone()
        assert row[0] == pytest.approx(55.0)


# ---- predict_spectrum tests (mocked torch) ----

class TestPredictSpectrum:
    """Tests for predict_spectrum() with mocked model."""

    def test_insufficient_coverage_returns_none(self):
        """Short spectrum should be rejected before hitting model."""
        wl = np.arange(0.80, 1.20, 0.005)
        refl = np.ones_like(wl) * 0.9
        result = predict_spectrum(wl, refl, _model="unused")
        assert result is None

    def test_all_nan_returns_none(self):
        wl = np.arange(0.45, 2.50, 0.005)
        refl = np.full_like(wl, np.nan)
        result = predict_spectrum(wl, refl, _model="unused")
        assert result is None

    def test_with_mock_model(self):
        """Test prediction pipeline with a mock model that returns fixed values."""
        # Skip if torch not available — the mock approach simulates it
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        class MockModel:
            def train(self):
                pass
            def __call__(self, x):
                # Return [OL%, OPX%, CPX%, Fa, Fs, Wo]
                return torch.tensor([[45.0, 35.0, 20.0, 18.0, 16.0, 5.0]])
            def parameters(self):
                return []

        wl, refl = make_s_type_spectrum()
        result = predict_spectrum(wl, refl, n_mc=5, _model=MockModel())
        assert result is not None
        assert "ol_pct" in result
        assert "opx_pct" in result
        assert "cpx_pct" in result
        assert "fa_mol_pct" in result
        assert "fs_mol_pct" in result
        assert "wo_mol_pct" in result
        assert result["method"] == "korda2023"
        # Modal abundances should sum to ~100%
        total = result["ol_pct"] + result["opx_pct"] + result["cpx_pct"]
        assert abs(total - 100.0) < 0.01

    def test_modal_normalization(self):
        """Modal abundances should be normalized to sum to 100%."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        class MockModel:
            def train(self):
                pass
            def __call__(self, x):
                # Raw values that don't sum to 100
                return torch.tensor([[30.0, 20.0, 10.0, 18.0, 16.0, 5.0]])

        wl, refl = make_s_type_spectrum()
        result = predict_spectrum(wl, refl, n_mc=3, _model=MockModel())
        assert result is not None
        total = result["ol_pct"] + result["opx_pct"] + result["cpx_pct"]
        assert abs(total - 100.0) < 0.01
        # Proportions preserved: 30/60=0.5, 20/60=0.333, 10/60=0.167
        assert result["ol_pct"] == pytest.approx(50.0, abs=0.1)
        assert result["opx_pct"] == pytest.approx(100.0 / 3, abs=0.1)


# ---- predict_all query tests ----

class TestPredictAllQuery:
    """Test the SQL query logic of predict_all (without actual model)."""

    def test_eligible_asteroid_found(self):
        """predict_all should find S-type asteroids with VNIR spectra."""
        conn = setup_test_db()
        rows = conn.execute(
            "SELECT DISTINCT s.asteroid_id "
            "FROM spectra s "
            "JOIN taxonomy t ON s.asteroid_id = t.asteroid_id "
            "LEFT JOIN cnn_mineral c ON s.asteroid_id = c.asteroid_id "
            "WHERE s.normalized = TRUE AND s.wl_max >= 2.0 "
            "AND t.primary_class IN ('S','Q','K','A','L','O','R','V') "
            "AND c.asteroid_id IS NULL",
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 4179

    def test_already_predicted_skipped(self):
        """Asteroids already in cnn_mineral should be skipped."""
        conn = setup_test_db()
        conn.execute(
            "INSERT INTO cnn_mineral (asteroid_id, ol_pct, method) VALUES (?, ?, ?)",
            (4179, 45.0, "korda2023"),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT DISTINCT s.asteroid_id "
            "FROM spectra s "
            "JOIN taxonomy t ON s.asteroid_id = t.asteroid_id "
            "LEFT JOIN cnn_mineral c ON s.asteroid_id = c.asteroid_id "
            "WHERE s.normalized = TRUE AND s.wl_max >= 2.0 "
            "AND t.primary_class IN ('S','Q','K','A','L','O','R','V') "
            "AND c.asteroid_id IS NULL",
        ).fetchall()
        assert len(rows) == 0

    def test_c_type_not_in_query(self):
        """C-type asteroids should not appear in eligible query."""
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, designation, neo) VALUES (?, ?, ?)",
            (101955, "Bennu", 1),
        )
        conn.execute(
            "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, "
            "prob_vector, classifier, input_coverage) VALUES (?, ?, ?, ?, ?, ?)",
            (101955, "C", 0.90, np.zeros(17).tobytes(), "classy_mahlke2022", "vnir"),
        )
        wl, refl = make_s_type_spectrum()
        conn.execute(
            "INSERT INTO spectra (asteroid_id, survey, wavelengths, reflectance, "
            "wl_min, wl_max, normalized, quality_flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (101955, "MITHNEOS", wl.tobytes(), refl.tobytes(), 0.45, 2.495, True, "good"),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT DISTINCT s.asteroid_id "
            "FROM spectra s "
            "JOIN taxonomy t ON s.asteroid_id = t.asteroid_id "
            "LEFT JOIN cnn_mineral c ON s.asteroid_id = c.asteroid_id "
            "WHERE s.normalized = TRUE AND s.wl_max >= 2.0 "
            "AND t.primary_class IN ('S','Q','K','A','L','O','R','V') "
            "AND c.asteroid_id IS NULL",
        ).fetchall()
        assert len(rows) == 0


# ---- Integration: predict_asteroid with classical comparison ----

class TestPredictAsteroidIntegration:
    """Integration tests for predict_asteroid with mocked model."""

    def test_stores_with_classical_agreement(self):
        """predict_asteroid should store results and assess agreement."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        from unittest.mock import patch

        class MockModel:
            def train(self):
                pass
            def __call__(self, x):
                # OL=50, OPX=50 → ratio=0.5 → matches classical 0.50 → good
                return torch.tensor([[50.0, 50.0, 0.0, 18.0, 16.0, 5.0]])
            def parameters(self):
                return []

        conn = setup_test_db(include_band_analysis=True)

        with patch("prospector.spectral.cnn_mineral._check_dependencies", return_value=True):
            result = predict_asteroid(4179, conn, n_mc=3, _model=MockModel())

        assert result is True
        conn.commit()

        row = conn.execute(
            "SELECT ol_pct, opx_pct, cpx_pct, classical_agreement, method "
            "FROM cnn_mineral WHERE asteroid_id = ?",
            (4179,),
        ).fetchone()
        assert row is not None
        assert row[3] == "good"  # classical_agreement
        assert row[4] == "korda2023"

    def test_stores_no_classical_when_missing(self):
        """Without band_analysis entry, agreement should be 'no_classical'."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        from unittest.mock import patch

        class MockModel:
            def train(self):
                pass
            def __call__(self, x):
                return torch.tensor([[45.0, 35.0, 20.0, 18.0, 16.0, 5.0]])

        conn = setup_test_db(include_band_analysis=False)

        with patch("prospector.spectral.cnn_mineral._check_dependencies", return_value=True):
            result = predict_asteroid(4179, conn, n_mc=3, _model=MockModel())

        assert result is True
        conn.commit()

        row = conn.execute(
            "SELECT classical_agreement FROM cnn_mineral WHERE asteroid_id = ?",
            (4179,),
        ).fetchone()
        assert row[0] == "no_classical"


# ---- Physical sanity tests ----

class TestPhysicalSanity:
    """Sanity checks on the module's physical constraints."""

    def test_agreement_thresholds_reasonable(self):
        """Good/fair thresholds should be physically meaningful."""
        assert GOOD_AGREEMENT == 10.0   # 10pp matches ~10% Dunn RMSE
        assert FAIR_AGREEMENT == 20.0

    def test_cnn_grid_covers_diagnostic_bands(self):
        """CNN grid must cover both 1 μm and 2 μm absorption bands."""
        assert CNN_WL_MIN <= 0.80  # Before Band I onset
        assert CNN_WL_MAX >= 2.30  # After Band II

    def test_silicate_classes_complete(self):
        """All expected silicate classes should be in the set."""
        expected = {"S", "Q", "K", "A", "L", "O", "R", "V"}
        assert SILICATE_CLASSES == expected
