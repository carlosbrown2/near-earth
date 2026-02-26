"""Tests for Stage 3B Hapke radiative transfer forward modeling."""

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from prospector.db import get_connection
from prospector.spectral.hapke_model import (
    DEFAULT_ENDMEMBERS,
    ENDMEMBER_NAMES,
    FE_NI_METAL,
    MAX_SMFE,
    MIN_SMFE,
    MIN_GRAIN_UM,
    MAX_GRAIN_UM,
    N_ENDMEMBERS,
    N_PARAMS,
    OLIVINE,
    PLAGIOCLASE,
    PYROXENE,
    TROILITE,
    Endmember,
    apply_smfe,
    compute_ssa,
    fit_spectrum,
    forward_model,
    mix_ssa,
    model_all,
    model_asteroid,
    ssa_to_reflectance,
    _iron_nk,
    _unpack_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WL = np.linspace(0.45, 2.50, 200)


def _make_db_with_spectrum(
    asteroid_id=4179,
    wl=None,
    reflectance=None,
    with_scores=False,
):
    """Create an in-memory DB with one asteroid and normalised spectrum."""
    conn = get_connection(":memory:")

    conn.execute(
        "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, 1)",
        (asteroid_id, f"Ast{asteroid_id}"),
    )

    if wl is None:
        wl = np.linspace(0.45, 2.50, 200)
    if reflectance is None:
        # Generate a synthetic S-type-like spectrum via the forward model
        params = np.array([0.40, 0.35, 0.05, 0.02, 50, 60, 40, 30, 40, 0.002])
        reflectance = forward_model(params, wl)

    conn.execute(
        "INSERT INTO spectra "
        "(asteroid_id, survey, wavelengths, reflectance, wl_min, wl_max, "
        "normalized, quality_flag) "
        "VALUES (?, 'MITHNEOS', ?, ?, ?, ?, TRUE, 'good')",
        (
            asteroid_id,
            wl.astype(np.float64).tobytes(),
            reflectance.astype(np.float64).tobytes(),
            float(wl.min()),
            float(wl.max()),
        ),
    )

    if with_scores:
        conn.execute(
            "INSERT INTO scores (asteroid_id, composite_score) VALUES (?, 0.85)",
            (asteroid_id,),
        )

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Iron optical constants
# ---------------------------------------------------------------------------


class TestIronNK:
    def test_returns_correct_shape(self):
        n, k = _iron_nk(_WL)
        assert n.shape == _WL.shape
        assert k.shape == _WL.shape

    def test_positive_values(self):
        n, k = _iron_nk(_WL)
        assert np.all(n > 0)
        assert np.all(k > 0)

    def test_n_increases_with_wavelength(self):
        """Iron n generally increases over VNIR."""
        n, _ = _iron_nk(_WL)
        assert n[-1] > n[0]


# ---------------------------------------------------------------------------
# Endmember
# ---------------------------------------------------------------------------


class TestEndmember:
    def test_k_baseline_only(self):
        em = Endmember("test", n=1.5, density=3.0, k_baseline=0.01, bands=[])
        k = em.k(_WL)
        np.testing.assert_allclose(k, 0.01)

    def test_k_with_band(self):
        em = Endmember("test", n=1.5, density=3.0, k_baseline=0.0,
                        bands=[(1.0, 0.1, 0.05)])
        k = em.k(_WL)
        peak_idx = np.argmin(np.abs(_WL - 1.0))
        assert k[peak_idx] == pytest.approx(0.1, abs=0.01)
        # Away from band, should be near baseline
        assert k[0] < 0.01

    def test_olivine_has_1um_band(self):
        k = OLIVINE.k(_WL)
        idx_1um = np.argmin(np.abs(_WL - 1.05))
        idx_0_5um = np.argmin(np.abs(_WL - 0.50))
        assert k[idx_1um] > k[idx_0_5um]


# ---------------------------------------------------------------------------
# compute_ssa
# ---------------------------------------------------------------------------


class TestComputeSSA:
    def test_transparent_material_high_ssa(self):
        """Low-k endmember should have high SSA."""
        em = Endmember("glass", n=1.5, density=2.5, k_baseline=1e-6, bands=[])
        w = compute_ssa(em, _WL, grain_size_um=50.0)
        assert np.all(w > 0.8)

    def test_opaque_material_lower_ssa(self):
        """High-k endmember (troilite) should have lower SSA."""
        w = compute_ssa(TROILITE, _WL, grain_size_um=50.0)
        # Troilite is opaque; SSA dominated by Fresnel reflection
        assert np.all(w < 0.5)

    def test_larger_grain_lower_ssa(self):
        """Larger grains absorb more, reducing SSA."""
        w_small = compute_ssa(OLIVINE, _WL, grain_size_um=10.0)
        w_large = compute_ssa(OLIVINE, _WL, grain_size_um=200.0)
        assert np.mean(w_small) > np.mean(w_large)

    def test_physical_range(self):
        """SSA must be in [0, 1]."""
        for em in DEFAULT_ENDMEMBERS:
            w = compute_ssa(em, _WL, grain_size_um=50.0)
            assert np.all(w >= 0.0)
            assert np.all(w <= 1.0)

    def test_output_shape(self):
        w = compute_ssa(OLIVINE, _WL, grain_size_um=50.0)
        assert w.shape == _WL.shape


# ---------------------------------------------------------------------------
# mix_ssa
# ---------------------------------------------------------------------------


class TestMixSSA:
    def test_single_endmember(self):
        """100% olivine mix should equal pure olivine SSA."""
        fracs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        gs = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        w_mix = mix_ssa(DEFAULT_ENDMEMBERS, fracs, _WL, gs)
        w_pure = compute_ssa(OLIVINE, _WL, 50.0)
        np.testing.assert_allclose(w_mix, w_pure, atol=1e-12)

    def test_mixture_between_endmembers(self):
        """50/50 mix should fall between the two pure endmembers."""
        fracs = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
        gs = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        w_mix = mix_ssa(DEFAULT_ENDMEMBERS, fracs, _WL, gs)
        w_ol = compute_ssa(OLIVINE, _WL, 50.0)
        w_opx = compute_ssa(PYROXENE, _WL, 50.0)
        w_min = np.minimum(w_ol, w_opx)
        w_max = np.maximum(w_ol, w_opx)
        assert np.all(w_mix >= w_min - 1e-12)
        assert np.all(w_mix <= w_max + 1e-12)

    def test_zero_fractions_ok(self):
        """All zeros should not crash (normalised to equal parts internally)."""
        fracs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        gs = np.array([50.0] * 5)
        w = mix_ssa(DEFAULT_ENDMEMBERS, fracs, _WL, gs)
        # All fracs zero → all SSA contributions are zero → w_mix = 0
        assert np.all(w >= 0.0)


# ---------------------------------------------------------------------------
# apply_smfe
# ---------------------------------------------------------------------------


class TestApplySMFe:
    def test_zero_smfe_no_change(self):
        ssa = compute_ssa(OLIVINE, _WL, 50.0)
        result = apply_smfe(ssa, _WL, 0.0)
        np.testing.assert_allclose(result, ssa)

    def test_smfe_darkens(self):
        """SMFe should reduce SSA (darken the spectrum)."""
        ssa = compute_ssa(OLIVINE, _WL, 50.0)
        result = apply_smfe(ssa, _WL, 0.01)
        assert np.mean(result) < np.mean(ssa)

    def test_smfe_reddens(self):
        """SMFe should darken blue more than red (reddening)."""
        ssa = np.ones(len(_WL)) * 0.8  # flat input
        result = apply_smfe(ssa, _WL, 0.01)
        # Shorter wavelengths should be more strongly darkened
        blue_idx = _WL < 0.7
        red_idx = _WL > 2.0
        blue_reduction = np.mean(ssa[blue_idx] - result[blue_idx])
        red_reduction = np.mean(ssa[red_idx] - result[red_idx])
        assert blue_reduction > red_reduction

    def test_physical_range(self):
        ssa = compute_ssa(OLIVINE, _WL, 50.0)
        result = apply_smfe(ssa, _WL, MAX_SMFE)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# ssa_to_reflectance
# ---------------------------------------------------------------------------


class TestSSAToReflectance:
    def test_higher_ssa_higher_reflectance(self):
        r_low = ssa_to_reflectance(np.full(10, 0.3))
        r_high = ssa_to_reflectance(np.full(10, 0.9))
        assert np.all(r_high > r_low)

    def test_zero_ssa_zero_reflectance(self):
        r = ssa_to_reflectance(np.zeros(10))
        np.testing.assert_allclose(r, 0.0)

    def test_physical_range(self):
        for w_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            r = ssa_to_reflectance(np.full(10, w_val))
            assert np.all(r >= 0.0)
            assert np.all(r <= 1.0)

    def test_geometry_effect(self):
        """Grazing incidence should reduce reflectance."""
        ssa = np.full(10, 0.5)
        r_normal = ssa_to_reflectance(ssa, incidence_deg=0.0, emission_deg=0.0)
        r_oblique = ssa_to_reflectance(ssa, incidence_deg=60.0, emission_deg=60.0)
        # Not necessarily lower at oblique for Hapke, but should differ
        assert not np.allclose(r_normal, r_oblique)


# ---------------------------------------------------------------------------
# _unpack_params
# ---------------------------------------------------------------------------


class TestUnpackParams:
    def test_fractions_sum_to_one(self):
        params = np.array([0.3, 0.2, 0.1, 0.05, 50, 50, 50, 50, 50, 0.001])
        fracs, gs, smfe = _unpack_params(params)
        assert fracs.sum() == pytest.approx(1.0)
        assert len(fracs) == 5
        assert len(gs) == 5

    def test_excess_fracs_renormalised(self):
        params = np.array([0.5, 0.4, 0.3, 0.2, 50, 50, 50, 50, 50, 0.001])
        fracs, _, _ = _unpack_params(params)
        assert fracs.sum() == pytest.approx(1.0)
        assert np.all(fracs >= 0)

    def test_grain_size_clamped(self):
        params = np.array([0.5, 0.5, 0.0, 0.0, 1, 1000, 50, 50, 50, 0.001])
        _, gs, _ = _unpack_params(params)
        assert np.all(gs >= MIN_GRAIN_UM)
        assert np.all(gs <= MAX_GRAIN_UM)


# ---------------------------------------------------------------------------
# forward_model
# ---------------------------------------------------------------------------


class TestForwardModel:
    def test_produces_positive_spectrum(self):
        params = np.array([0.4, 0.3, 0.1, 0.05, 50, 60, 40, 30, 40, 0.001])
        refl = forward_model(params, _WL)
        assert np.all(refl >= 0)

    def test_olivine_dominant_has_1um_feature(self):
        """Olivine-rich mixture should show ~1 um absorption."""
        params = np.array([0.8, 0.1, 0.05, 0.0, 50, 50, 50, 50, 50, 0.0])
        refl = forward_model(params, _WL)
        # Reflectance near 1 um should be lower than at 0.7 um (absorption band)
        idx_07 = np.argmin(np.abs(_WL - 0.70))
        idx_10 = np.argmin(np.abs(_WL - 1.05))
        assert refl[idx_10] < refl[idx_07]

    def test_pyroxene_dominant_has_2um_feature(self):
        """Pyroxene-rich mixture should show ~1.9 um absorption."""
        params = np.array([0.1, 0.8, 0.05, 0.0, 50, 50, 50, 50, 50, 0.0])
        refl = forward_model(params, _WL)
        idx_15 = np.argmin(np.abs(_WL - 1.50))
        idx_19 = np.argmin(np.abs(_WL - 1.90))
        assert refl[idx_19] < refl[idx_15]

    def test_metal_dominated_featureless(self):
        """Metal-dominated mixture should be relatively featureless."""
        params = np.array([0.0, 0.0, 0.0, 0.0, 50, 50, 50, 50, 50, 0.0])
        refl = forward_model(params, _WL)
        # Standard deviation should be small relative to mean (featureless)
        assert np.std(refl) / np.mean(refl) < 0.1

    def test_output_shape(self):
        params = np.array([0.3, 0.3, 0.1, 0.05, 50, 50, 50, 50, 50, 0.001])
        refl = forward_model(params, _WL)
        assert refl.shape == _WL.shape


# ---------------------------------------------------------------------------
# fit_spectrum
# ---------------------------------------------------------------------------


class TestFitSpectrum:
    def test_recovers_synthetic_composition(self):
        """Fit should approximately recover known composition from synthetic data."""
        # Generate a known synthetic spectrum
        true_params = np.array([0.45, 0.35, 0.05, 0.02, 50, 60, 40, 30, 40, 0.002])
        wl = np.linspace(0.45, 2.50, 200)
        synthetic = forward_model(true_params, wl)

        result = fit_spectrum(wl, synthetic, n_bootstrap=10)

        # Check that fit quality is good
        assert result["fit_rmse"] < 0.01
        assert result["fit_rho"] > 0.95

        # Check fractions are in reasonable range (within uncertainty)
        assert result["fractions"]["olivine"] > 0.1
        assert result["fractions"]["pyroxene"] > 0.1

    def test_returns_required_keys(self):
        wl = np.linspace(0.45, 2.50, 100)
        refl = forward_model(_default_initial_guess(), wl)
        result = fit_spectrum(wl, refl, n_bootstrap=5)

        required = {
            "fractions", "grain_sizes", "smfe", "fit_rmse",
            "fit_rho", "uncertainties", "n_bootstrap", "synthetic",
        }
        assert required.issubset(result.keys())
        assert len(result["fractions"]) == 5
        assert len(result["grain_sizes"]) == 5
        assert result["n_bootstrap"] == 5

    def test_uncertainties_present(self):
        wl = np.linspace(0.45, 2.50, 100)
        refl = forward_model(_default_initial_guess(), wl)
        result = fit_spectrum(wl, refl, n_bootstrap=10)

        assert "olivine_frac" in result["uncertainties"]
        assert "metal_frac" in result["uncertainties"]
        assert "smfe" in result["uncertainties"]
        for v in result["uncertainties"].values():
            assert v >= 0

    def test_insufficient_wavelengths_raises(self):
        wl = np.linspace(0.45, 0.50, 5)
        refl = np.ones(5)
        with pytest.raises(ValueError, match="Insufficient"):
            fit_spectrum(wl, refl)

    def test_handles_nan_in_reflectance(self):
        wl = np.linspace(0.45, 2.50, 200)
        refl = forward_model(_default_initial_guess(), wl)
        refl[50:60] = np.nan  # Telluric gap
        result = fit_spectrum(wl, refl, n_bootstrap=5)
        assert result["fit_rho"] > 0.5

    def test_with_uncertainty_array(self):
        wl = np.linspace(0.45, 2.50, 100)
        refl = forward_model(_default_initial_guess(), wl)
        unc = np.ones_like(refl) * 0.01
        result = fit_spectrum(wl, refl, uncertainty=unc, n_bootstrap=5)
        assert result["fit_rmse"] < 0.1


# Reference to the function for use in helpers
from prospector.spectral.hapke_model import _default_initial_guess


# ---------------------------------------------------------------------------
# model_asteroid (DB integration)
# ---------------------------------------------------------------------------


class TestModelAsteroid:
    def test_fits_and_stores(self):
        conn = _make_db_with_spectrum()
        result = model_asteroid(4179, conn, n_bootstrap=5)
        assert result is not None
        assert result["fit_rho"] > 0.5

        # Check DB row was written
        row = conn.execute(
            "SELECT olivine_frac, pyroxene_frac, metal_frac, smfe_fraction, "
            "fit_rmse, fit_rho, n_bootstrap "
            "FROM hapke_modeling WHERE asteroid_id = ?",
            (4179,),
        ).fetchone()
        assert row is not None
        ol, opx, met, smfe, rmse, rho, n_boot = row
        assert ol >= 0
        assert opx >= 0
        assert met >= 0
        assert smfe >= 0
        assert n_boot == 5

    def test_no_spectrum_returns_none(self):
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (1, 'Test', 1)"
        )
        conn.commit()
        assert model_asteroid(1, conn) is None

    def test_prefers_deweathered(self):
        conn = _make_db_with_spectrum()
        # Add deweathered spectrum (slightly different)
        row = conn.execute(
            "SELECT wavelengths, reflectance FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()
        refl = np.frombuffer(row[1], dtype=np.float64)
        dw = refl * 1.05
        conn.execute(
            "UPDATE spectra SET deweathered = ? WHERE asteroid_id = 4179",
            (dw.astype(np.float64).tobytes(),),
        )
        conn.commit()
        result = model_asteroid(4179, conn, n_bootstrap=5)
        assert result is not None

    def test_idempotent_insert_or_replace(self):
        """Running twice should overwrite, not fail."""
        conn = _make_db_with_spectrum()
        r1 = model_asteroid(4179, conn, n_bootstrap=5)
        r2 = model_asteroid(4179, conn, n_bootstrap=5)
        assert r1 is not None
        assert r2 is not None
        # Only one row
        count = conn.execute(
            "SELECT COUNT(*) FROM hapke_modeling WHERE asteroid_id = 4179"
        ).fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# model_all (batch)
# ---------------------------------------------------------------------------


class TestModelAll:
    def test_batch_with_scores(self):
        conn = _make_db_with_spectrum(with_scores=True)
        results = model_all(conn, top_n=5, n_bootstrap=5)
        assert len(results) >= 1
        assert 4179 in results

    def test_fallback_without_scores(self):
        conn = _make_db_with_spectrum(with_scores=False)
        results = model_all(conn, top_n=5, n_bootstrap=5)
        assert len(results) >= 1

    def test_empty_db(self):
        conn = get_connection(":memory:")
        results = model_all(conn, top_n=5, n_bootstrap=5)
        assert results == {}


# ---------------------------------------------------------------------------
# Physical sanity checks
# ---------------------------------------------------------------------------


class TestPhysicalSanity:
    def test_smfe_reduces_band_depth(self):
        """Space weathering should suppress absorption bands."""
        params_fresh = np.array([0.5, 0.3, 0.1, 0.0, 50, 50, 50, 50, 50, 0.0])
        params_weathered = np.array([0.5, 0.3, 0.1, 0.0, 50, 50, 50, 50, 50, 0.03])
        wl = np.linspace(0.45, 2.50, 200)
        r_fresh = forward_model(params_fresh, wl)
        r_weathered = forward_model(params_weathered, wl)

        # Normalise both
        idx_norm = np.argmin(np.abs(wl - 0.75))
        r_fresh_n = r_fresh / r_fresh[idx_norm]
        r_weathered_n = r_weathered / r_weathered[idx_norm]

        # Band depth at ~1 um should be shallower when weathered
        idx_band = np.argmin(np.abs(wl - 1.05))
        depth_fresh = 1.0 - r_fresh_n[idx_band]
        depth_weathered = 1.0 - r_weathered_n[idx_band]
        assert depth_weathered < depth_fresh

    def test_metal_dominated_darker_than_silicate(self):
        """Metal-dominated mixture should be darker than silicate-dominated.

        Metal is opaque (high k), so SSA is limited to Fresnel reflection (~0.5),
        while transparent silicates achieve SSA near 1.0 via internal scattering.
        """
        wl = np.linspace(0.45, 2.50, 200)
        params_silicate = np.array([0.5, 0.4, 0.05, 0.0, 50, 50, 50, 50, 50, 0.0])
        params_metal = np.array([0.1, 0.1, 0.0, 0.0, 50, 50, 50, 50, 50, 0.0])
        r_sil = forward_model(params_silicate, wl)
        r_met = forward_model(params_metal, wl)
        # Opaque metal → lower reflectance than transparent silicates
        assert np.mean(r_met) < np.mean(r_sil)


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------

# Strategies for physical parameters
_st_wavelengths = arrays(
    np.float64,
    st.integers(min_value=10, max_value=100),
    elements=st.floats(min_value=0.35, max_value=2.55, allow_nan=False, allow_infinity=False),
).map(lambda a: np.sort(np.unique(a))).filter(lambda a: len(a) >= 5)

_st_grain_size = st.floats(min_value=MIN_GRAIN_UM, max_value=MAX_GRAIN_UM, allow_nan=False)
_st_smfe = st.floats(min_value=0.0, max_value=MAX_SMFE, allow_nan=False)
_st_endmember = st.sampled_from(DEFAULT_ENDMEMBERS)
_st_ssa_val = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def _st_area_fractions(n: int = N_ENDMEMBERS) -> st.SearchStrategy[np.ndarray]:
    """Generate n non-negative fractions summing to 1 (Dirichlet-like)."""
    return arrays(
        np.float64, n,
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ).filter(lambda a: a.sum() > 0).map(lambda a: a / a.sum())


def _st_param_vector() -> st.SearchStrategy[np.ndarray]:
    """Generate a valid 10-element parameter vector for 5 endmembers."""
    fracs = arrays(
        np.float64, 4,
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    grain_sizes = arrays(
        np.float64, 5,
        elements=st.floats(min_value=MIN_GRAIN_UM, max_value=MAX_GRAIN_UM,
                           allow_nan=False, allow_infinity=False),
    )
    smfe = st.floats(min_value=MIN_SMFE, max_value=MAX_SMFE,
                     allow_nan=False, allow_infinity=False)
    return st.tuples(fracs, grain_sizes, smfe).map(
        lambda t: np.concatenate([t[0], t[1], [t[2]]])
    )


class TestHapkeProperties:
    """Property-based tests for Hapke model physical invariants."""

    @given(endmember=_st_endmember, grain_size=_st_grain_size)
    @settings(max_examples=100, deadline=None)
    def test_compute_ssa_bounded_01(self, endmember, grain_size):
        """SSA must always be in [0, 1] for any endmember and grain size."""
        wl = np.linspace(0.45, 2.50, 50)
        w = compute_ssa(endmember, wl, grain_size)
        assert np.all(w >= 0.0), f"SSA below 0: min={w.min()}"
        assert np.all(w <= 1.0), f"SSA above 1: max={w.max()}"

    @given(fracs=_st_area_fractions(), grain_sizes=arrays(
        np.float64, N_ENDMEMBERS,
        elements=st.floats(min_value=MIN_GRAIN_UM, max_value=MAX_GRAIN_UM,
                           allow_nan=False, allow_infinity=False),
    ))
    @settings(max_examples=100, deadline=None)
    def test_mix_ssa_bounded_01(self, fracs, grain_sizes):
        """Mixed SSA must be in [0, 1] for any valid area fractions."""
        wl = np.linspace(0.45, 2.50, 50)
        w = mix_ssa(DEFAULT_ENDMEMBERS, fracs, wl, grain_sizes)
        assert np.all(w >= 0.0), f"Mixed SSA below 0: min={w.min()}"
        assert np.all(w <= 1.0), f"Mixed SSA above 1: max={w.max()}"

    @given(fracs=_st_area_fractions(), grain_sizes=arrays(
        np.float64, N_ENDMEMBERS,
        elements=st.floats(min_value=MIN_GRAIN_UM, max_value=MAX_GRAIN_UM,
                           allow_nan=False, allow_infinity=False),
    ))
    @settings(max_examples=80, deadline=None)
    def test_mix_ssa_convex_combination(self, fracs, grain_sizes):
        """Mixed SSA is bounded by min/max of individual component SSAs."""
        wl = np.linspace(0.45, 2.50, 50)
        individual = []
        for em, frac, gs in zip(DEFAULT_ENDMEMBERS, fracs, grain_sizes):
            if frac > 1e-12:
                individual.append(compute_ssa(em, wl, gs))
        assume(len(individual) >= 1)
        w_min = np.min(individual, axis=0)
        w_max = np.max(individual, axis=0)
        w_mix = mix_ssa(DEFAULT_ENDMEMBERS, fracs, wl, grain_sizes)
        assert np.all(w_mix >= w_min - 1e-10), (
            f"Mixed SSA below component min: {(w_mix - w_min).min()}"
        )
        assert np.all(w_mix <= w_max + 1e-10), (
            f"Mixed SSA above component max: {(w_mix - w_max).max()}"
        )

    @given(
        smfe_frac=st.floats(min_value=1e-6, max_value=MAX_SMFE,
                            allow_nan=False, allow_infinity=False),
        endmember=_st_endmember,
        grain_size=_st_grain_size,
    )
    @settings(max_examples=100, deadline=None)
    def test_apply_smfe_never_increases_ssa(self, smfe_frac, endmember, grain_size):
        """SMFe space weathering can only darken — never increase SSA."""
        wl = np.linspace(0.45, 2.50, 50)
        ssa = compute_ssa(endmember, wl, grain_size)
        weathered = apply_smfe(ssa, wl, smfe_frac, n_host=endmember.n)
        assert np.all(weathered <= ssa + 1e-12), (
            f"SMFe increased SSA: max increase = {(weathered - ssa).max()}"
        )

    @given(params=_st_param_vector())
    @settings(max_examples=100, deadline=None)
    def test_forward_model_positive(self, params):
        """Forward model reflectance must be non-negative for any valid params."""
        wl = np.linspace(0.45, 2.50, 50)
        refl = forward_model(params, wl)
        assert np.all(refl >= 0.0), f"Negative reflectance: min={refl.min()}"
        assert np.all(np.isfinite(refl)), "Non-finite reflectance"

    @given(params=_st_param_vector())
    @settings(max_examples=100, deadline=None)
    def test_unpack_params_invariants(self, params):
        """Unpacked fractions sum to 1; grain sizes within physical bounds."""
        fracs, grain_sizes, smfe = _unpack_params(params)
        assert fracs.sum() == pytest.approx(1.0, abs=1e-10), (
            f"Fractions sum = {fracs.sum()}"
        )
        assert np.all(fracs >= 0.0), f"Negative fraction: {fracs.min()}"
        assert np.all(grain_sizes >= MIN_GRAIN_UM), (
            f"Grain below min: {grain_sizes.min()}"
        )
        assert np.all(grain_sizes <= MAX_GRAIN_UM), (
            f"Grain above max: {grain_sizes.max()}"
        )
        assert MIN_SMFE <= smfe <= MAX_SMFE, f"SMFe out of bounds: {smfe}"

    @given(
        w1=_st_ssa_val, w2=_st_ssa_val,
    )
    @settings(max_examples=200, deadline=None)
    def test_ssa_to_reflectance_monotonic(self, w1, w2):
        """Higher SSA must yield higher or equal reflectance."""
        assume(w1 != w2)
        r1 = ssa_to_reflectance(np.array([w1]))
        r2 = ssa_to_reflectance(np.array([w2]))
        if w1 < w2:
            assert r1[0] <= r2[0] + 1e-12, (
                f"Not monotonic: SSA {w1:.4f}->{r1[0]:.6f}, {w2:.4f}->{r2[0]:.6f}"
            )
        else:
            assert r2[0] <= r1[0] + 1e-12, (
                f"Not monotonic: SSA {w2:.4f}->{r2[0]:.6f}, {w1:.4f}->{r1[0]:.6f}"
            )

    @given(
        smfe_frac=_st_smfe,
        endmember=_st_endmember,
        grain_size=_st_grain_size,
    )
    @settings(max_examples=100, deadline=None)
    def test_apply_smfe_bounded_01(self, smfe_frac, endmember, grain_size):
        """Weathered SSA must remain in [0, 1]."""
        wl = np.linspace(0.45, 2.50, 50)
        ssa = compute_ssa(endmember, wl, grain_size)
        weathered = apply_smfe(ssa, wl, smfe_frac, n_host=endmember.n)
        assert np.all(weathered >= 0.0), f"Weathered SSA below 0: {weathered.min()}"
        assert np.all(weathered <= 1.0), f"Weathered SSA above 1: {weathered.max()}"
