"""Tests for prospector.spectral.phase_curve — H, G1, G2 phase curve fitting."""

import math

import numpy as np
import pytest

from prospector.db import get_connection, init_schema
from prospector.spectral.phase_curve import (
    MIN_OBS,
    _phi1,
    _phi2,
    _phi3,
    fit_all,
    fit_phase_curve,
    geometric_albedo,
    h_g1g2_model,
    phase_integral,
    reduce_magnitude,
    taxonomy_hint,
)


# --- Basis function tests ---

class TestBasisFunctions:
    """Verify the cubic-spline basis functions match tabulated values."""

    def test_phi1_at_zero(self):
        assert _phi1(np.array([0.0]))[0] == pytest.approx(1.0, abs=1e-6)

    def test_phi2_at_zero(self):
        assert _phi2(np.array([0.0]))[0] == pytest.approx(1.0, abs=1e-6)

    def test_phi3_at_zero(self):
        assert _phi3(np.array([0.0]))[0] == pytest.approx(1.0, abs=1e-6)

    def test_phi1_at_30(self):
        """Phi_1(30°) ≈ 0.0135 per Muinonen et al. (2010)."""
        assert _phi1(np.array([30.0]))[0] == pytest.approx(0.0135, abs=1e-4)

    def test_phi2_at_30(self):
        """Phi_2(30°) ≈ 0.0226."""
        assert _phi2(np.array([30.0]))[0] == pytest.approx(0.0226, abs=1e-4)

    def test_phi3_at_30(self):
        """Phi_3(30°) ≈ 0.3918."""
        assert _phi3(np.array([30.0]))[0] == pytest.approx(0.3918, abs=1e-4)

    def test_phi1_at_90(self):
        """Phi_1(90°) ≈ 2.4e-4."""
        assert _phi1(np.array([90.0]))[0] == pytest.approx(2.4e-4, abs=1e-4)

    def test_phi_monotonically_decreasing(self):
        """All basis functions should decrease with phase angle (on tabulated grid)."""
        # Test at tabulated points where monotonicity is exact
        alpha = np.array([0, 0.3, 1, 2, 4, 8, 12, 20, 30, 60, 90, 120, 150.0])
        for phi_fn in (_phi1, _phi2, _phi3):
            vals = phi_fn(alpha)
            assert np.all(np.diff(vals) <= 1e-10)

    def test_phi_clipped_nonnegative(self):
        """Basis functions clipped to [0, 1]."""
        alpha = np.array([0.0, 50.0, 100.0, 150.0])
        for phi_fn in (_phi1, _phi2, _phi3):
            vals = phi_fn(alpha)
            assert np.all(vals >= 0.0)
            assert np.all(vals <= 1.0)


# --- Forward model tests ---

class TestForwardModel:
    def test_at_zero_phase(self):
        """At alpha=0, all Phi=1, so V(0) = H - 2.5*log10(1) = H."""
        H, G1, G2 = 20.0, 0.3, 0.2
        mag = h_g1g2_model(np.array([0.0]), H, G1, G2)
        assert mag[0] == pytest.approx(H, abs=1e-6)

    def test_magnitude_increases_with_phase(self):
        """Brightness decreases (magnitude increases) with phase angle."""
        alpha = np.linspace(0, 120, 50)
        mag = h_g1g2_model(alpha, 18.0, 0.3, 0.2)
        assert np.all(np.diff(mag) >= -1e-10)

    def test_scalar_input(self):
        """Scalar phase angle should work."""
        mag = h_g1g2_model(np.array([45.0]), 18.0, 0.3, 0.2)
        assert np.isfinite(mag[0])

    def test_g1_g2_affect_slope(self):
        """Higher G1+G2 (less G3) → faster brightness drop at moderate angles.

        Phi_1 and Phi_2 drop faster than Phi_3, so high G1+G2 means
        fainter (larger magnitude) at moderate phase angles.
        """
        alpha = np.array([30.0])
        mag_high_g3 = h_g1g2_model(alpha, 18.0, 0.05, 0.05)  # G3=0.9, slow drop
        mag_low_g3 = h_g1g2_model(alpha, 18.0, 0.4, 0.3)     # G3=0.3, fast drop
        # Low G3 → dominated by fast-dropping Phi1/Phi2 → fainter → larger mag
        assert mag_low_g3[0] > mag_high_g3[0]


# --- Phase integral ---

class TestPhaseIntegral:
    def test_typical_values(self):
        """Phase integral for S-type (G1≈0.25, G2≈0.21)."""
        q = phase_integral(0.25, 0.21)
        assert 0.2 < q < 0.5  # reasonable range

    def test_formula(self):
        """Verify the linear formula."""
        q = phase_integral(0.3, 0.2)
        expected = 0.009082 + 0.4061 * 0.3 + 0.8768 * 0.2
        assert q == pytest.approx(expected, abs=1e-8)


# --- Geometric albedo ---

class TestGeometricAlbedo:
    def test_known_asteroid(self):
        """Vesta: H≈3.2, D≈525 km → p_V ≈ 0.42."""
        pv = geometric_albedo(3.2, 525.0)
        assert 0.3 < pv < 0.6

    def test_zero_diameter(self):
        assert math.isnan(geometric_albedo(18.0, 0.0))

    def test_negative_diameter(self):
        assert math.isnan(geometric_albedo(18.0, -1.0))


# --- Taxonomy hint ---

class TestTaxonomyHint:
    def test_s_type(self):
        assert taxonomy_hint(0.25, 0.21) == 'S'

    def test_c_type(self):
        assert taxonomy_hint(0.82, 0.02) == 'C'

    def test_x_type(self):
        assert taxonomy_hint(0.58, 0.10) == 'X'

    def test_b_type(self):
        assert taxonomy_hint(0.09, 0.55) == 'B'

    def test_no_match(self):
        """Far from any centroid → None."""
        assert taxonomy_hint(0.5, 0.5) is None


# --- Reduce magnitude ---

class TestReduceMagnitude:
    def test_unit_distances(self):
        """At r=delta=1 AU, reduced mag = apparent mag."""
        assert reduce_magnitude(15.0, 1.0, 1.0) == pytest.approx(15.0, abs=1e-8)

    def test_typical_correction(self):
        """At r=1.5, delta=0.8 → correction = -5*log10(1.2) ≈ -0.40."""
        reduced = reduce_magnitude(15.0, 1.5, 0.8)
        expected = 15.0 - 5.0 * np.log10(1.5 * 0.8)
        assert reduced == pytest.approx(expected, abs=1e-8)


# --- Full fitting tests ---

class TestFitPhaseCurve:
    def _generate_synthetic(self, H=18.0, G1=0.30, G2=0.20, n=40, noise=0.05, seed=42):
        """Generate synthetic phase curve observations.

        Includes the opposition surge region (alpha < 5 deg) which is
        critical for discriminating G1 from G2.
        """
        rng = np.random.default_rng(seed)
        # Include small phase angles for opposition surge discrimination
        alpha_small = rng.uniform(0.3, 5.0, size=n // 4)
        alpha_large = rng.uniform(5.0, 100.0, size=n - n // 4)
        alpha = np.concatenate([alpha_small, alpha_large])
        alpha.sort()
        mag = h_g1g2_model(alpha, H, G1, G2)
        mag += rng.normal(0, noise, size=len(alpha))
        return alpha, mag

    def test_recovers_parameters(self):
        """Fit should recover true H, G1, G2 from clean synthetic data."""
        H_true, G1_true, G2_true = 18.0, 0.30, 0.20
        alpha, mag = self._generate_synthetic(H_true, G1_true, G2_true, noise=0.02)

        result = fit_phase_curve(alpha, mag)
        assert result is not None
        assert result['H'] == pytest.approx(H_true, abs=0.15)
        assert result['G1'] == pytest.approx(G1_true, abs=0.15)
        assert result['G2'] == pytest.approx(G2_true, abs=0.15)

    def test_c_type_parameters(self):
        """Recover C-type phase curve parameters."""
        H_true, G1_true, G2_true = 22.0, 0.80, 0.03
        alpha, mag = self._generate_synthetic(H_true, G1_true, G2_true, noise=0.02)

        result = fit_phase_curve(alpha, mag)
        assert result is not None
        assert result['G1'] == pytest.approx(G1_true, abs=0.20)
        assert result['taxonomy_hint'] in ('C', 'D')  # high G1 cluster

    def test_returns_none_with_few_obs(self):
        """Fewer than MIN_OBS observations → None."""
        alpha = np.array([5.0, 10.0, 20.0])
        mag = np.array([18.5, 19.0, 19.8])
        assert fit_phase_curve(alpha, mag) is None

    def test_handles_nan_observations(self):
        """NaN observations are filtered out."""
        alpha, mag = self._generate_synthetic(n=20)
        # Insert NaNs
        mag[3] = np.nan
        alpha[7] = np.nan

        result = fit_phase_curve(alpha, mag)
        assert result is not None
        assert result['n_obs'] == 18

    def test_weighted_fit(self):
        """Weighted fitting with uncertainties."""
        alpha, mag = self._generate_synthetic(noise=0.05)
        unc = np.full_like(mag, 0.05)

        result = fit_phase_curve(alpha, mag, mag_unc=unc)
        assert result is not None
        assert result['rms_residual'] < 0.2

    def test_output_fields(self):
        """All expected keys are present in the output."""
        alpha, mag = self._generate_synthetic()
        result = fit_phase_curve(alpha, mag)

        expected_keys = {
            'H', 'G1', 'G2', 'h_unc', 'g1_unc', 'g2_unc',
            'phase_integral', 'rms_residual', 'n_obs',
            'alpha_min', 'alpha_max', 'taxonomy_hint',
        }
        assert expected_keys == set(result.keys())

    def test_g1_g2_constraint(self):
        """G1 + G2 <= 1 is enforced."""
        alpha, mag = self._generate_synthetic()
        result = fit_phase_curve(alpha, mag)
        assert result['G1'] + result['G2'] <= 1.0 + 1e-10

    def test_phase_integral_computed(self):
        """Phase integral should be in output and match formula."""
        alpha, mag = self._generate_synthetic()
        result = fit_phase_curve(alpha, mag)
        expected_q = 0.009082 + 0.4061 * result['G1'] + 0.8768 * result['G2']
        assert result['phase_integral'] == pytest.approx(expected_q, abs=1e-8)


# --- Database integration ---

class TestFitAll:
    def _setup_db(self):
        """Create an in-memory DB with photometry data."""
        conn = get_connection(":memory:")

        # Create the photometry table (not part of core schema)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mpc_photometry (
                obs_id INTEGER PRIMARY KEY AUTOINCREMENT,
                asteroid_id INTEGER REFERENCES asteroids(asteroid_id),
                phase_angle_deg REAL,
                reduced_mag REAL,
                mag_unc REAL
            )
        """)

        # Insert test asteroids
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, designation, neo) VALUES (433, 'Eros', 1)"
        )
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, designation, neo) VALUES (1036, 'Ganymed', 1)"
        )

        # Generate noiseless synthetic observations for Eros (S-type: G1≈0.25, G2≈0.21)
        # Include small phase angles and use no noise to test DB integration cleanly
        alphas = np.concatenate([
            np.linspace(0.3, 5.0, 8),   # opposition surge region
            np.linspace(5.0, 90.0, 17),  # moderate to large phase angles
        ])
        for alpha in alphas:
            alpha = float(alpha)
            mag = float(h_g1g2_model(np.array([alpha]), 11.16, 0.25, 0.21)[0])
            conn.execute(
                "INSERT INTO mpc_photometry (asteroid_id, phase_angle_deg, reduced_mag, mag_unc) "
                "VALUES (433, ?, ?, 0.03)",
                (alpha, mag),
            )

        # Only 3 obs for Ganymed → should be skipped
        for alpha in [5.0, 15.0, 30.0]:
            mag = float(h_g1g2_model(np.array([alpha]), 9.45, 0.30, 0.18)[0])
            conn.execute(
                "INSERT INTO mpc_photometry (asteroid_id, phase_angle_deg, reduced_mag) "
                "VALUES (1036, ?, ?)",
                (alpha, mag),
            )

        conn.commit()
        return conn

    def test_fit_all_basic(self):
        conn = self._setup_db()
        count = fit_all(conn)
        # Only Eros should be fitted (Ganymed has too few obs)
        assert count == 1

        row = conn.execute(
            "SELECT h_fit, g1, g2, n_obs, taxonomy_hint FROM phase_curve WHERE asteroid_id = 433"
        ).fetchone()
        assert row is not None
        h_fit, g1, g2, n_obs, tax_hint = row
        assert h_fit == pytest.approx(11.16, abs=0.3)
        assert n_obs == 25
        assert tax_hint == 'S'

    def test_fit_all_no_table(self):
        """Missing photometry table returns 0."""
        conn = get_connection(":memory:")
        count = fit_all(conn)
        assert count == 0

    def test_fit_all_idempotent(self):
        """Running fit_all twice replaces previous results."""
        conn = self._setup_db()
        fit_all(conn)
        fit_all(conn)
        rows = conn.execute("SELECT COUNT(*) FROM phase_curve").fetchone()
        assert rows[0] == 1  # still just Eros
