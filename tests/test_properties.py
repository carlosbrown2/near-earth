"""Property-based tests using Hypothesis.

These tests explore input spaces no human would think to cover. They verify
mathematical invariants, physical constraints, and monotonicity properties
that must hold for ALL inputs — not just the examples Ralph happened to pick.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from prospector.scoring.granvik_prior import (
    MAHLKE_CLASSES,
    SOURCE_REGIONS,
    source_region_probabilities,
    taxonomy_prior,
    tisserand,
)
from prospector.scoring.evoi import (
    _make_albedo_posterior,
    _make_complex_posterior,
    _make_peaked_posterior,
    _make_radar_posterior,
    compute_evoi,
)
from prospector.scoring.scorer import (
    compute_accessibility,
    compute_confidence,
    compute_spin_modifier,
    compute_thermal_depletion_factor,
    estimate_mass_kg,
    load_config,
    sample_from_dist,
    score_asteroid,
)
from prospector.spectral.band_analysis import (
    classify_gaffey_subtype,
    dunn_calibration,
    estimate_temperature,
    lindsay_bar_correction,
    temperature_correct_band_centers,
)
from prospector.schemas import Distribution, ScoringResult, BandAnalysisResult


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

# Physically reasonable orbital elements
orbital_a = st.floats(0.5, 6.0, allow_nan=False, allow_infinity=False)
orbital_e = st.floats(0.0, 0.99, allow_nan=False, allow_infinity=False)
orbital_i = st.floats(0.0, 90.0, allow_nan=False, allow_infinity=False)
moid_au = st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False)
diameter_km = st.floats(0.001, 500.0, allow_nan=False, allow_infinity=False)
density_gcm3 = st.floats(0.1, 8.0, allow_nan=False, allow_infinity=False)
perihelion_au = st.floats(0.05, 5.0, allow_nan=False, allow_infinity=False)

# Distribution dict strategy
dist_strategy = st.fixed_dictionaries({
    "mean": st.floats(0.1, 100.0, allow_nan=False, allow_infinity=False),
    "std": st.floats(0.01, 10.0, allow_nan=False, allow_infinity=False),
    "min": st.floats(0.0, 50.0, allow_nan=False, allow_infinity=False),
    "max": st.floats(0.1, 200.0, allow_nan=False, allow_infinity=False),
}).filter(lambda d: d["min"] < d["max"] and d["min"] <= d["mean"] <= d["max"])


# ---------------------------------------------------------------------------
# Mass estimation
# ---------------------------------------------------------------------------

class TestMassProperties:
    @given(d=diameter_km, rho=density_gcm3)
    def test_mass_always_non_negative(self, d, rho):
        assert estimate_mass_kg(d, rho) >= 0

    @given(d=diameter_km, rho=density_gcm3)
    def test_mass_scales_with_diameter_cubed(self, d, rho):
        m1 = estimate_mass_kg(d, rho)
        m2 = estimate_mass_kg(2 * d, rho)
        if m1 > 0:
            ratio = m2 / m1
            assert abs(ratio - 8.0) < 1e-6

    @given(d=diameter_km, rho=density_gcm3)
    def test_mass_scales_linearly_with_density(self, d, rho):
        m1 = estimate_mass_kg(d, rho)
        m2 = estimate_mass_kg(d, 2 * rho)
        if m1 > 0:
            ratio = m2 / m1
            assert abs(ratio - 2.0) < 1e-6

    @given(d=diameter_km, rho=density_gcm3)
    def test_mass_equals_sphere_formula(self, d, rho):
        mass = estimate_mass_kg(d, rho)
        r_m = d * 1000.0 / 2.0
        expected = (4.0 / 3.0) * math.pi * r_m**3 * rho * 1000.0
        assert mass == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Accessibility
# ---------------------------------------------------------------------------

class TestAccessibilityProperties:
    @given(m=moid_au)
    def test_bounded_with_moid(self, m):
        score = compute_accessibility(moid_au=m)
        assert 0.01 <= score <= 1.0

    @given(a=orbital_a, e=orbital_e)
    def test_bounded_with_fallback(self, a, e):
        score = compute_accessibility(a=a, e=e)
        assert 0.01 <= score <= 1.0

    @given(
        m1=st.floats(0.0, 2.0, allow_nan=False, allow_infinity=False),
        m2=st.floats(0.0, 2.0, allow_nan=False, allow_infinity=False),
    )
    def test_closer_moid_higher_score(self, m1, m2):
        """Monotonicity: lower MOID → higher or equal accessibility."""
        assume(m1 < m2)
        s1 = compute_accessibility(moid_au=m1)
        s2 = compute_accessibility(moid_au=m2)
        assert s1 >= s2

    def test_no_data_returns_minimum(self):
        assert compute_accessibility() == 0.01


# ---------------------------------------------------------------------------
# Confidence (entropy-based)
# ---------------------------------------------------------------------------

class TestConfidenceProperties:
    @given(st.integers(0, 16))
    def test_peaked_distribution_high_confidence(self, idx):
        """A single-class distribution should give confidence = 1.0."""
        pv = np.zeros(17)
        pv[idx] = 1.0
        conf = compute_confidence(pv)
        assert conf == pytest.approx(1.0)

    def test_uniform_distribution_low_confidence(self):
        pv = np.ones(17) / 17.0
        conf = compute_confidence(pv)
        assert conf < 0.05

    @given(
        data=st.lists(
            st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
            min_size=17, max_size=17,
        )
    )
    def test_always_bounded(self, data):
        pv = np.array(data)
        conf = compute_confidence(pv)
        assert 0.01 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Thermal depletion
# ---------------------------------------------------------------------------

class TestThermalDepletionProperties:
    @given(q=perihelion_au)
    def test_always_in_unit_interval(self, q):
        factor = compute_thermal_depletion_factor(q)
        assert 0.0 <= factor <= 1.0

    def test_none_returns_one(self):
        assert compute_thermal_depletion_factor(None) == 1.0

    @given(q=st.floats(1.0, 10.0, allow_nan=False, allow_infinity=False))
    def test_safe_perihelion_no_penalty(self, q):
        assert compute_thermal_depletion_factor(q) == 1.0

    @given(
        q1=st.floats(0.05, 3.0, allow_nan=False, allow_infinity=False),
        q2=st.floats(0.05, 3.0, allow_nan=False, allow_infinity=False),
    )
    def test_monotonically_increasing(self, q1, q2):
        """Farther perihelion → equal or more water retained."""
        assume(q1 < q2)
        f1 = compute_thermal_depletion_factor(q1)
        f2 = compute_thermal_depletion_factor(q2)
        assert f1 <= f2 + 1e-10


# ---------------------------------------------------------------------------
# Spin modifier
# ---------------------------------------------------------------------------

class TestSpinModifierProperties:
    def test_monolithic_bonus(self):
        mod = compute_spin_modifier(is_monolithic=True)
        assert mod > 1.0

    def test_binary_penalty(self):
        mod = compute_spin_modifier(is_binary_suspect=True)
        assert mod < 1.0

    def test_no_data_neutral(self):
        assert compute_spin_modifier() == 1.0


# ---------------------------------------------------------------------------
# sample_from_dist
# ---------------------------------------------------------------------------

class TestSampleFromDistProperties:
    @given(d=dist_strategy)
    @settings(max_examples=200)
    def test_always_within_bounds(self, d):
        rng = np.random.default_rng(42)
        for _ in range(50):
            val = sample_from_dist(d, rng)
            assert d["min"] <= val <= d["max"]


# ---------------------------------------------------------------------------
# Taxonomy prior (Granvik)
# ---------------------------------------------------------------------------

class TestTaxonomyPriorProperties:
    @given(a=orbital_a, e=orbital_e, i=orbital_i)
    def test_sums_to_one(self, a, e, i):
        pv = taxonomy_prior(a, e, i)
        assert pv.shape == (17,)
        assert abs(pv.sum() - 1.0) < 1e-10

    @given(a=orbital_a, e=orbital_e, i=orbital_i)
    def test_all_non_negative(self, a, e, i):
        pv = taxonomy_prior(a, e, i)
        assert (pv >= 0).all()

    @given(a=orbital_a, e=orbital_e, i=orbital_i)
    def test_source_region_probs_sum_to_one(self, a, e, i):
        probs = source_region_probabilities(a, e, i)
        assert set(probs.keys()) == set(SOURCE_REGIONS)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-10
        assert all(v >= 0 for v in probs.values())


# ---------------------------------------------------------------------------
# Tisserand parameter
# ---------------------------------------------------------------------------

class TestTisserandProperties:
    @given(a=orbital_a, e=orbital_e, i=orbital_i)
    def test_finite_for_valid_orbits(self, a, e, i):
        t = tisserand(a, e, i)
        assert math.isfinite(t)

    def test_zero_a_returns_inf(self):
        assert tisserand(0.0, 0.5, 10.0) == float("inf")


# ---------------------------------------------------------------------------
# Band analysis helpers
# ---------------------------------------------------------------------------

class TestTemperatureProperties:
    @given(a=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False))
    def test_temperature_positive(self, a):
        t = estimate_temperature(a)
        assert t > 0

    @given(
        a1=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False),
        a2=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False),
    )
    def test_temperature_decreases_with_distance(self, a1, a2):
        assume(a1 < a2)
        assert estimate_temperature(a1) >= estimate_temperature(a2)


class TestGaffeyProperties:
    @given(
        bic=st.floats(0.85, 1.10, allow_nan=False, allow_infinity=False),
        bar=st.floats(0.0, 3.0, allow_nan=False, allow_infinity=False),
    )
    def test_always_returns_valid_subtype(self, bic, bar):
        subtype = classify_gaffey_subtype(bic, bar)
        assert subtype.startswith("S(")
        assert subtype.endswith(")")
        roman = subtype[2:-1]
        assert roman in ("I", "II", "III", "IV", "V", "VI", "VII")


class TestDunnCalibrationProperties:
    @given(
        bar=st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False),
        bic=st.floats(0.85, 1.10, allow_nan=False, allow_infinity=False),
        biic=st.floats(1.70, 2.30, allow_nan=False, allow_infinity=False),
    )
    def test_ol_ratio_bounded(self, bar, bic, biic):
        result = dunn_calibration(bar, bic, biic)
        assert 0.0 <= result["ol_opx_ratio"] <= 1.0

    @given(
        bar=st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False),
        bic=st.floats(0.85, 1.10, allow_nan=False, allow_infinity=False),
    )
    def test_fa_non_negative(self, bar, bic):
        result = dunn_calibration(bar, bic, None)
        assert result["fa_mol_pct"] >= 0
        assert result["fs_mol_pct"] is None


class TestLindsayCorrectionProperties:
    @given(
        bar=st.floats(0.01, 5.0, allow_nan=False, allow_infinity=False),
        red_edge=st.floats(2.30, 2.55, allow_nan=False, allow_infinity=False),
    )
    def test_correction_preserves_sign(self, bar, red_edge):
        corrected = lindsay_bar_correction(bar, red_edge)
        assert corrected > 0


# ---------------------------------------------------------------------------
# Pydantic schema validation
# ---------------------------------------------------------------------------

class TestDistributionSchema:
    def test_valid(self):
        d = Distribution(mean=3.0, std=1.0, min=1.0, max=5.0)
        assert d.mean == 3.0

    def test_min_gt_max_rejected(self):
        with pytest.raises(Exception):
            Distribution(mean=3.0, std=1.0, min=5.0, max=1.0)

    def test_mean_outside_bounds_rejected(self):
        with pytest.raises(Exception):
            Distribution(mean=10.0, std=1.0, min=1.0, max=5.0)

    def test_negative_std_rejected(self):
        with pytest.raises(Exception):
            Distribution(mean=3.0, std=-1.0, min=1.0, max=5.0)


# ---------------------------------------------------------------------------
# Metamorphic relations (physical invariants across full pipeline)
# ---------------------------------------------------------------------------

class TestMetamorphicRelations:
    """Metamorphic relation tests verifying physical monotonicity and
    conservation laws across the full scoring and EVOI input space."""

    _config = load_config()

    # --- Score monotonicity (MR2: diameter) ---

    @given(
        d1=st.floats(0.01, 100.0, allow_nan=False, allow_infinity=False),
        d2=st.floats(0.01, 100.0, allow_nan=False, allow_infinity=False),
        a=st.floats(0.8, 4.0, allow_nan=False, allow_infinity=False),
        e=st.floats(0.0, 0.8, allow_nan=False, allow_infinity=False),
        i=st.floats(0.0, 60.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=200)
    def test_score_monotonic_in_diameter(self, d1, d2, a, e, i, seed):
        """Larger asteroid always scores ≥ smaller (same composition/orbit)."""
        assume(d1 < d2)
        r_small = score_asteroid(
            diameter_km=d1, a=a, e=e, i_deg=i,
            config=self._config, n_samples=200,
            rng=np.random.default_rng(seed),
        )
        r_large = score_asteroid(
            diameter_km=d2, a=a, e=e, i_deg=i,
            config=self._config, n_samples=200,
            rng=np.random.default_rng(seed),
        )
        assert r_large["composite_score"] >= r_small["composite_score"]

    # --- Score monotonicity (MR4: MOID → accessibility → score) ---

    @given(
        m1=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        m2=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        a=st.floats(0.8, 4.0, allow_nan=False, allow_infinity=False),
        e=st.floats(0.0, 0.8, allow_nan=False, allow_infinity=False),
        i=st.floats(0.0, 60.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=200)
    def test_score_monotonic_in_moid(self, m1, m2, a, e, i, seed):
        """Closer MOID → higher or equal composite score."""
        assume(m1 < m2)
        r_close = score_asteroid(
            diameter_km=1.0, a=a, e=e, i_deg=i, moid=m1,
            config=self._config, n_samples=200,
            rng=np.random.default_rng(seed),
        )
        r_far = score_asteroid(
            diameter_km=1.0, a=a, e=e, i_deg=i, moid=m2,
            config=self._config, n_samples=200,
            rng=np.random.default_rng(seed),
        )
        assert r_close["composite_score"] >= r_far["composite_score"]

    # --- MR7: Score non-negative ---

    @given(
        a=st.floats(0.8, 4.0, allow_nan=False, allow_infinity=False),
        e=st.floats(0.0, 0.8, allow_nan=False, allow_infinity=False),
        i=st.floats(0.0, 60.0, allow_nan=False, allow_infinity=False),
        d=st.floats(0.01, 50.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=200)
    def test_score_always_non_negative(self, a, e, i, d, seed):
        """score_asteroid always returns composite_score ≥ 0."""
        result = score_asteroid(
            diameter_km=d, a=a, e=e, i_deg=i,
            config=self._config, n_samples=200,
            rng=np.random.default_rng(seed),
        )
        assert result["composite_score"] >= 0

    # --- MR5: Thermal depletion monotonic at score level ---

    @given(
        q1=st.floats(0.05, 2.0, allow_nan=False, allow_infinity=False),
        q2=st.floats(0.05, 2.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=200)
    def test_thermal_depletion_monotonic_at_score_level(self, q1, q2, seed):
        """Lower perihelion → lower water score for C-type in in_space mode."""
        assume(q1 < q2)
        pv = np.zeros(17)
        pv[2] = 1.0  # Pure C-type (water-only revenue)
        r_hot = score_asteroid(
            diameter_km=1.0, a=2.0, e=0.3, i_deg=5.0,
            prob_vector=pv, config=self._config, n_samples=200,
            mode="in_space", rng=np.random.default_rng(seed), q_au=q1,
        )
        r_cool = score_asteroid(
            diameter_km=1.0, a=2.0, e=0.3, i_deg=5.0,
            prob_vector=pv, config=self._config, n_samples=200,
            mode="in_space", rng=np.random.default_rng(seed), q_au=q2,
        )
        assert r_hot["composite_score"] <= r_cool["composite_score"] + 1e-6

    # --- MR6: Posterior validity (all _make_*_posterior functions) ---

    @given(
        cls_idx=st.integers(0, 16),
        peak_prob=st.floats(0.1, 0.99, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_peaked_posterior_valid(self, cls_idx, peak_prob):
        """_make_peaked_posterior output sums to 1 and is non-negative."""
        cls = MAHLKE_CLASSES[cls_idx]
        post = _make_peaked_posterior(cls, peak_prob)
        assert post.shape == (17,)
        assert (post >= 0).all()
        assert abs(post.sum() - 1.0) < 1e-10

    @given(
        cls_idx=st.integers(0, 16),
        peak_prob=st.floats(0.1, 0.99, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_complex_posterior_valid(self, cls_idx, peak_prob):
        """_make_complex_posterior output sums to 1 and is non-negative."""
        cls = MAHLKE_CLASSES[cls_idx]
        post = _make_complex_posterior(cls, peak_prob)
        assert post.shape == (17,)
        assert (post >= 0).all()
        assert abs(post.sum() - 1.0) < 1e-10

    @given(
        data=st.lists(
            st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
            min_size=17, max_size=17,
        ),
        is_metallic=st.booleans(),
    )
    @settings(max_examples=200)
    def test_radar_posterior_valid(self, data, is_metallic):
        """_make_radar_posterior output sums to 1 and is non-negative."""
        prior = np.array(data)
        total = prior.sum()
        assume(total > 0)
        prior /= total
        post = _make_radar_posterior(prior, is_metallic)
        assert post.shape == (17,)
        assert (post >= 0).all()
        assert abs(post.sum() - 1.0) < 1e-10

    @given(
        data=st.lists(
            st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
            min_size=17, max_size=17,
        ),
        albedo=st.floats(0.01, 0.60, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_albedo_posterior_valid(self, data, albedo):
        """_make_albedo_posterior output sums to 1 and is non-negative."""
        prior = np.array(data)
        total = prior.sum()
        assume(total > 0)
        prior /= total
        post = _make_albedo_posterior(prior, albedo)
        assert post.shape == (17,)
        assert (post >= 0).all()
        assert abs(post.sum() - 1.0) < 1e-10

    # --- MR1: Prior sharpening reduces uncertainty (prerequisite for EVOI decrease) ---

    @given(
        cls_idx=st.integers(0, 16),
        peak=st.floats(0.5, 0.999, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_peaked_prior_higher_confidence(self, cls_idx, peak):
        """More concentrated prior → higher confidence (lower entropy).
        This is the mathematical prerequisite for EVOI decreasing as prior
        sharpens: a peaked prior has less scoring uncertainty to reduce."""
        uniform = np.ones(17) / 17.0
        peaked = np.full(17, (1.0 - peak) / 16.0)
        peaked[cls_idx] = peak
        peaked /= peaked.sum()
        assert compute_confidence(peaked) >= compute_confidence(uniform)

    # --- MR3: VNIR posterior more concentrated than vis posterior ---

    @given(cls_idx=st.integers(0, 16))
    @settings(max_examples=200)
    def test_vnir_posterior_more_concentrated(self, cls_idx):
        """VNIR resolves single class (85% peaked), vis resolves complex
        (~70% spread across complex members).  VNIR posterior always has
        higher confidence, which is why VNIR EVOI ≥ vis EVOI."""
        cls = MAHLKE_CLASSES[cls_idx]
        vnir_post = _make_peaked_posterior(cls)
        vis_post = _make_complex_posterior(cls)
        conf_vnir = compute_confidence(vnir_post)
        conf_vis = compute_confidence(vis_post)
        assert conf_vnir >= conf_vis

    # --- MR8: EVOI bounded by prior std ---

    @given(
        a=st.floats(0.8, 4.0, allow_nan=False, allow_infinity=False),
        e=st.floats(0.0, 0.8, allow_nan=False, allow_infinity=False),
        i=st.floats(0.0, 60.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=100, deadline=None)
    def test_evoi_bounded_by_prior_std(self, a, e, i, seed):
        """best_evoi ≤ score_std (can't reduce uncertainty below zero).
        Guaranteed by the max(0, ...) clamp in compute_evoi."""
        result = compute_evoi(
            diameter_km=1.0, a=a, e=e, i_deg=i,
            config=self._config, n_samples=30,
            rng=np.random.default_rng(seed),
        )
        assert result["best_evoi"] <= result["score_std"] + 1e-6
