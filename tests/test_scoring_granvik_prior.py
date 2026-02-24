"""Tests for the Granvik 2018 debiased population prior."""

import math

import numpy as np
import pytest

from prospector.scoring.granvik_prior import (
    MAHLKE_CLASSES,
    SOURCE_REGIONS,
    _SOURCE_VECTORS,
    source_region_probabilities,
    taxonomy_prior,
    taxonomy_prior_dict,
    tisserand,
)


# ---------------------------------------------------------------------------
# Tisserand parameter
# ---------------------------------------------------------------------------

class TestTisserand:
    def test_earth_like_orbit(self):
        """Earth-like orbit (a=1, e=0.017, i=0) → T_J ≈ 6.2."""
        t = tisserand(1.0, 0.017, 0.0)
        assert 6.0 < t < 6.5

    def test_jupiter_family_comet(self):
        """Typical JFC orbit → 2 < T_J < 3."""
        # 67P/Churyumov-Gerasimenko-like: a≈3.46, e≈0.64, i≈7.04°
        t = tisserand(3.46, 0.64, 7.04)
        assert 2.0 < t < 3.0

    def test_halley_type_comet(self):
        """Halley-type orbit → T_J < 2."""
        # High i, moderate a, high e
        t = tisserand(17.8, 0.967, 162.0)
        assert t < 2.0

    def test_main_belt_asteroid(self):
        """Typical main belt asteroid → T_J > 3."""
        # Ceres-like: a≈2.77, e≈0.076, i≈10.6°
        t = tisserand(2.77, 0.076, 10.6)
        assert t > 3.0

    def test_zero_a_returns_inf(self):
        assert tisserand(0.0, 0.5, 10.0) == float("inf")

    def test_circular_zero_inclination(self):
        """Circular face-on orbit simplifies: T_J = a_J/a + 2·sqrt(a/a_J)."""
        a = 2.5
        expected = 5.2044 / a + 2.0 * math.sqrt(a / 5.2044)
        assert abs(tisserand(a, 0.0, 0.0) - expected) < 1e-10


# ---------------------------------------------------------------------------
# Source region probabilities
# ---------------------------------------------------------------------------

class TestSourceRegionProbabilities:
    def test_returns_all_seven_regions(self):
        probs = source_region_probabilities(1.5, 0.5, 10.0)
        assert set(probs.keys()) == set(SOURCE_REGIONS)

    def test_sums_to_one(self):
        probs = source_region_probabilities(1.5, 0.5, 10.0)
        assert abs(sum(probs.values()) - 1.0) < 1e-10

    def test_all_non_negative(self):
        probs = source_region_probabilities(1.5, 0.5, 10.0)
        assert all(v >= 0 for v in probs.values())

    def test_inner_belt_neo_favors_nu6(self):
        """NEO with inner-belt-like origin should favour ν6."""
        # a≈1.5 (low), moderate e, low i → typical ν6 delivery
        probs = source_region_probabilities(1.5, 0.3, 6.0)
        # ν6 should be top or near-top source
        assert probs["nu6"] > 0.2

    def test_jfc_orbit_favors_jfc(self):
        """JFC-like orbit (2 < T_J < 3) should boost JFC probability."""
        # 67P-like orbit: a≈3.46, e≈0.64, i≈7.04° (T_J ≈ 2.75)
        probs = source_region_probabilities(3.46, 0.64, 7.04)
        # JFC should get a significant share
        assert probs["jfc"] > 0.05

    def test_high_inclination_favors_hungaria_or_phocaea(self):
        """High inclination (i≈22°) near 1.9 AU should favour Hungaria."""
        probs = source_region_probabilities(1.9, 0.10, 22.0)
        # Hungaria should be the dominant source
        assert probs["hungaria"] > probs["nu6"]

    def test_extreme_orbit_returns_uniform(self):
        """Extreme orbit far from all resonances → fallback uniform prior."""
        # Very large a, nearly parabolic — all Gaussians effectively zero
        probs = source_region_probabilities(50.0, 0.99, 80.0)
        expected = 1.0 / len(SOURCE_REGIONS)
        for v in probs.values():
            assert abs(v - expected) < 1e-10

    def test_toutatis_typical_neo(self):
        """4179 Toutatis (a=2.51, e=0.63, i=0.45°) — well-studied NEO."""
        probs = source_region_probabilities(2.51, 0.63, 0.45)
        assert abs(sum(probs.values()) - 1.0) < 1e-10
        # Should have non-trivial contributions from multiple sources
        non_trivial = sum(1 for v in probs.values() if v > 0.05)
        assert non_trivial >= 2

    def test_bennu_carbonaceous(self):
        """101955 Bennu (a=1.126, e=0.204, i=6.035°) — ν6 region."""
        probs = source_region_probabilities(1.126, 0.204, 6.035)
        assert probs["nu6"] > 0.15


# ---------------------------------------------------------------------------
# Source taxonomy vectors (internal consistency)
# ---------------------------------------------------------------------------

class TestSourceVectors:
    def test_all_regions_have_vectors(self):
        assert set(_SOURCE_VECTORS.keys()) == set(SOURCE_REGIONS)

    def test_vectors_sum_to_one(self):
        for region, vec in _SOURCE_VECTORS.items():
            assert abs(vec.sum() - 1.0) < 1e-10, f"{region} vector doesn't sum to 1"

    def test_vectors_non_negative(self):
        for region, vec in _SOURCE_VECTORS.items():
            assert np.all(vec >= 0), f"{region} has negative entries"

    def test_vector_length(self):
        for vec in _SOURCE_VECTORS.values():
            assert len(vec) == len(MAHLKE_CLASSES) == 17

    def test_nu6_s_dominated(self):
        """Inner belt (ν6) should be S-complex dominated."""
        idx_s = MAHLKE_CLASSES.index("S")
        assert _SOURCE_VECTORS["nu6"][idx_s] > 0.4

    def test_2_1_c_dominated(self):
        """Outer belt (2:1) should be C-complex dominated."""
        idx_c = MAHLKE_CLASSES.index("C")
        idx_ch = MAHLKE_CLASSES.index("Ch")
        idx_b = MAHLKE_CLASSES.index("B")
        c_total = (
            _SOURCE_VECTORS["2:1"][idx_c]
            + _SOURCE_VECTORS["2:1"][idx_ch]
            + _SOURCE_VECTORS["2:1"][idx_b]
        )
        assert c_total > 0.35

    def test_hungaria_e_dominated(self):
        """Hungaria region should be E-type dominated."""
        idx_e = MAHLKE_CLASSES.index("E")
        assert _SOURCE_VECTORS["hungaria"][idx_e] > 0.3

    def test_jfc_d_dominated(self):
        """JFC source should be D-type dominated."""
        idx_d = MAHLKE_CLASSES.index("D")
        assert _SOURCE_VECTORS["jfc"][idx_d] > 0.3


# ---------------------------------------------------------------------------
# Taxonomy prior (end-to-end)
# ---------------------------------------------------------------------------

class TestTaxonomyPrior:
    def test_returns_17_elements(self):
        prior = taxonomy_prior(1.5, 0.5, 10.0)
        assert prior.shape == (17,)

    def test_sums_to_one(self):
        prior = taxonomy_prior(1.5, 0.5, 10.0)
        assert abs(prior.sum() - 1.0) < 1e-10

    def test_all_non_negative(self):
        prior = taxonomy_prior(1.5, 0.5, 10.0)
        assert np.all(prior >= 0)

    def test_dtype_float64(self):
        prior = taxonomy_prior(1.5, 0.5, 10.0)
        assert prior.dtype == np.float64

    def test_inner_neo_s_type_dominant(self):
        """Inner-belt NEO should have S as highest-probability class."""
        prior = taxonomy_prior(1.3, 0.3, 5.0)
        idx_s = MAHLKE_CLASSES.index("S")
        assert prior[idx_s] == prior.max()

    def test_outer_neo_c_type_elevated(self):
        """NEO from outer belt should have elevated C-complex probability."""
        prior = taxonomy_prior(2.8, 0.5, 12.0)
        idx_c = MAHLKE_CLASSES.index("C")
        idx_ch = MAHLKE_CLASSES.index("Ch")
        c_total = prior[idx_c] + prior[idx_ch]
        assert c_total > 0.15

    def test_jfc_orbit_d_type_elevated(self):
        """JFC orbit should elevate D-type probability."""
        # 67P-like orbit: a≈3.46, e≈0.64, i≈7.04° (T_J ≈ 2.75)
        prior = taxonomy_prior(3.46, 0.64, 7.04)
        idx_d = MAHLKE_CLASSES.index("D")
        # D should be meaningfully present for cometary orbit
        assert prior[idx_d] > 0.05

    def test_hungaria_orbit_e_type_elevated(self):
        """Hungaria-region orbit should elevate E-type probability."""
        prior = taxonomy_prior(1.9, 0.10, 22.0)
        idx_e = MAHLKE_CLASSES.index("E")
        assert prior[idx_e] > 0.15

    def test_known_neo_itokawa(self):
        """25143 Itokawa (S-type, a=1.324, e=0.280, i=1.62°)."""
        prior = taxonomy_prior(1.324, 0.280, 1.62)
        idx_s = MAHLKE_CLASSES.index("S")
        # S-type should be high-probability for this inner-belt-origin NEO
        assert prior[idx_s] > 0.3

    def test_known_neo_bennu(self):
        """101955 Bennu (B-type, a=1.126, e=0.204, i=6.035°)."""
        prior = taxonomy_prior(1.126, 0.204, 6.035)
        # Prior should give non-trivial probability to B/C/Ch
        idx_b = MAHLKE_CLASSES.index("B")
        idx_c = MAHLKE_CLASSES.index("C")
        idx_ch = MAHLKE_CLASSES.index("Ch")
        carbonaceous = prior[idx_b] + prior[idx_c] + prior[idx_ch]
        # Not necessarily dominant (this is a prior, not truth), but non-trivial
        assert carbonaceous > 0.05

    def test_different_orbits_give_different_priors(self):
        """Two very different orbits should produce different priors."""
        prior_inner = taxonomy_prior(1.2, 0.2, 3.0)
        prior_outer = taxonomy_prior(2.8, 0.6, 15.0)
        # At least some classes should differ substantially
        diff = np.abs(prior_inner - prior_outer)
        assert diff.max() > 0.05


# ---------------------------------------------------------------------------
# taxonomy_prior_dict
# ---------------------------------------------------------------------------

class TestTaxonomyPriorDict:
    def test_returns_all_classes(self):
        d = taxonomy_prior_dict(1.5, 0.5, 10.0)
        assert set(d.keys()) == set(MAHLKE_CLASSES)

    def test_values_match_array(self):
        a, e, i = 1.5, 0.5, 10.0
        arr = taxonomy_prior(a, e, i)
        d = taxonomy_prior_dict(a, e, i)
        for idx, cls in enumerate(MAHLKE_CLASSES):
            assert abs(d[cls] - arr[idx]) < 1e-15

    def test_values_are_float(self):
        d = taxonomy_prior_dict(1.5, 0.5, 10.0)
        assert all(isinstance(v, float) for v in d.values())

    def test_sums_to_one(self):
        d = taxonomy_prior_dict(1.5, 0.5, 10.0)
        assert abs(sum(d.values()) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Blob round-trip (compatibility with taxonomy.prob_vector format)
# ---------------------------------------------------------------------------

class TestBlobRoundTrip:
    def test_tobytes_frombuffer(self):
        """Prior vector survives numpy BLOB serialisation (scorer interop)."""
        prior = taxonomy_prior(1.5, 0.5, 10.0)
        blob = prior.tobytes()
        recovered = np.frombuffer(blob, dtype=np.float64)
        np.testing.assert_array_equal(prior, recovered)

    def test_blob_length(self):
        """BLOB should be 17 × 8 = 136 bytes."""
        prior = taxonomy_prior(1.5, 0.5, 10.0)
        assert len(prior.tobytes()) == 17 * 8
