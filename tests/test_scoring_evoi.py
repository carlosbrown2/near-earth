"""Tests for prospector.scoring.evoi — Expected Value of Information ranking."""

import math

import numpy as np
import pytest

from prospector.scoring.evoi import (
    EVOIResult,
    _ALBEDO_BINS,
    _ALBEDO_PROBS,
    _CLASS_IDX,
    _COMPLEXES,
    _class_to_complex,
    _make_albedo_posterior,
    _make_complex_posterior,
    _make_peaked_posterior,
    _make_radar_posterior,
    _score_distribution,
    compute_evoi,
    compute_evoi_for_asteroid,
    rank_all,
)
from prospector.scoring.granvik_prior import MAHLKE_CLASSES
from prospector.scoring.scorer import load_config
from prospector.db import get_connection


# --- Helpers ---

def _uniform_prior():
    """Uniform 17-class prior (maximum entropy)."""
    return np.ones(len(MAHLKE_CLASSES)) / len(MAHLKE_CLASSES)


def _peaked_prior(cls="S", prob=0.9):
    """Prior peaked on one class."""
    p = np.full(len(MAHLKE_CLASSES), (1.0 - prob) / (len(MAHLKE_CLASSES) - 1))
    p[_CLASS_IDX[cls]] = prob
    return p


def _setup_uncharacterised_neo(conn, asteroid_id=99001, diameter=0.5,
                                a=1.3, e=0.2, i=10.0, moid=0.05):
    """Insert an NEO with orbital elements but NO taxonomy."""
    conn.execute(
        "INSERT INTO asteroids (asteroid_id, name, neo, pha) VALUES (?, ?, ?, ?)",
        (asteroid_id, f"Test{asteroid_id}", True, True),
    )
    conn.execute(
        "INSERT INTO orbits (asteroid_id, a, e, i, moid, diameter) VALUES (?, ?, ?, ?, ?, ?)",
        (asteroid_id, a, e, i, moid, diameter),
    )
    return asteroid_id


def _setup_characterised_neo(conn, asteroid_id=99002):
    """Insert an NEO with taxonomy (already characterised)."""
    conn.execute(
        "INSERT INTO asteroids (asteroid_id, name, neo, pha) VALUES (?, ?, ?, ?)",
        (asteroid_id, f"Test{asteroid_id}", True, True),
    )
    conn.execute(
        "INSERT INTO orbits (asteroid_id, a, e, i, moid, diameter) VALUES (?, ?, ?, ?, ?, ?)",
        (asteroid_id, 1.5, 0.3, 15.0, 0.08, 1.0),
    )
    prob_vec = _peaked_prior("S", 0.85)
    conn.execute(
        "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, prob_vector, classifier) "
        "VALUES (?, ?, ?, ?, ?)",
        (asteroid_id, "S", 0.85, prob_vec.tobytes(), "test"),
    )
    return asteroid_id


# --- Posterior Construction Tests ---

class TestMakePeakedPosterior:
    """Tests for _make_peaked_posterior."""

    def test_sums_to_one(self):
        post = _make_peaked_posterior("S")
        assert abs(post.sum() - 1.0) < 1e-10

    def test_peak_class_has_highest_prob(self):
        post = _make_peaked_posterior("M")
        idx = _CLASS_IDX["M"]
        assert post[idx] == post.max()

    def test_peak_probability_approximately_correct(self):
        post = _make_peaked_posterior("C", peak_prob=0.90)
        idx = _CLASS_IDX["C"]
        assert post[idx] > 0.85

    def test_all_classes_represented(self):
        post = _make_peaked_posterior("S")
        assert all(p > 0 for p in post)

    def test_length(self):
        post = _make_peaked_posterior("V")
        assert len(post) == len(MAHLKE_CLASSES)


class TestMakeComplexPosterior:
    """Tests for _make_complex_posterior."""

    def test_sums_to_one(self):
        post = _make_complex_posterior("S")
        assert abs(post.sum() - 1.0) < 1e-10

    def test_s_complex_boosted(self):
        post = _make_complex_posterior("S")
        s_members = _COMPLEXES["S"]
        s_prob = sum(post[_CLASS_IDX[c]] for c in s_members)
        # S-complex should hold ~70% of probability
        assert s_prob > 0.60

    def test_c_complex_boosted_for_c_type(self):
        post = _make_complex_posterior("C")
        c_members = _COMPLEXES["C"]
        c_prob = sum(post[_CLASS_IDX[c]] for c in c_members)
        assert c_prob > 0.60

    def test_less_peaked_than_vnir(self):
        """Visible-only should be less informative than VNIR."""
        vnir = _make_peaked_posterior("S")
        vis = _make_complex_posterior("S")
        # VNIR should have lower entropy (more peaked)
        h_vnir = -np.sum(vnir[vnir > 0] * np.log2(vnir[vnir > 0]))
        h_vis = -np.sum(vis[vis > 0] * np.log2(vis[vis > 0]))
        assert h_vnir < h_vis


class TestMakeRadarPosterior:
    """Tests for _make_radar_posterior."""

    def test_sums_to_one(self):
        prior = _uniform_prior()
        post = _make_radar_posterior(prior, True)
        assert abs(post.sum() - 1.0) < 1e-10

    def test_metallic_boosts_m_type(self):
        prior = _uniform_prior()
        post = _make_radar_posterior(prior, is_metallic=True)
        assert post[_CLASS_IDX["M"]] > prior[_CLASS_IDX["M"]]

    def test_nonmetallic_suppresses_m_type(self):
        prior = _uniform_prior()
        post = _make_radar_posterior(prior, is_metallic=False)
        assert post[_CLASS_IDX["M"]] < prior[_CLASS_IDX["M"]]

    def test_preserves_length(self):
        prior = _uniform_prior()
        post = _make_radar_posterior(prior, True)
        assert len(post) == len(MAHLKE_CLASSES)


class TestMakeAlbedoPosterior:
    """Tests for _make_albedo_posterior."""

    def test_sums_to_one(self):
        prior = _uniform_prior()
        for albedo in [0.03, 0.15, 0.45]:
            post = _make_albedo_posterior(prior, albedo)
            assert abs(post.sum() - 1.0) < 1e-10

    def test_low_albedo_boosts_c_type(self):
        prior = _uniform_prior()
        post = _make_albedo_posterior(prior, 0.03)
        assert post[_CLASS_IDX["C"]] > prior[_CLASS_IDX["C"]]

    def test_high_albedo_boosts_e_type(self):
        prior = _uniform_prior()
        post = _make_albedo_posterior(prior, 0.45)
        assert post[_CLASS_IDX["E"]] > prior[_CLASS_IDX["E"]]

    def test_moderate_albedo_boosts_s_type(self):
        prior = _uniform_prior()
        post = _make_albedo_posterior(prior, 0.20)
        assert post[_CLASS_IDX["S"]] > prior[_CLASS_IDX["S"]]


class TestClassToComplex:
    """Tests for _class_to_complex."""

    def test_s_complex_members(self):
        for cls in ["S", "Q", "A", "K", "L", "O", "R", "V"]:
            assert _class_to_complex(cls) == "S"

    def test_c_complex_members(self):
        for cls in ["C", "Ch", "B", "D", "P"]:
            assert _class_to_complex(cls) == "C"

    def test_x_complex_members(self):
        for cls in ["X", "M", "E"]:
            assert _class_to_complex(cls) == "X"

    def test_z_is_end_member(self):
        assert _class_to_complex("Z") == "end"


# --- Score Distribution Tests ---

class TestScoreDistribution:
    """Tests for _score_distribution."""

    def test_returns_array(self):
        config = load_config()
        rng = np.random.default_rng(42)
        prior = _peaked_prior("S")
        scores = _score_distribution(
            1.0, 1.3, 0.2, 10.0, 0.05, prior, config, "earth_return", 50, rng
        )
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 50

    def test_nonnegative_scores(self):
        config = load_config()
        rng = np.random.default_rng(42)
        prior = _peaked_prior("M")
        scores = _score_distribution(
            1.0, 1.3, 0.2, 10.0, 0.05, prior, config, "earth_return", 100, rng
        )
        assert all(s >= 0 for s in scores)

    def test_peaked_prior_less_variance_than_uniform(self):
        """A peaked prior should produce less score variance (coefficient of variation)."""
        config = load_config()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        scores_uniform = _score_distribution(
            1.0, 1.3, 0.2, 10.0, 0.05, _uniform_prior(), config, "earth_return", 1000, rng1
        )
        scores_peaked = _score_distribution(
            1.0, 1.3, 0.2, 10.0, 0.05, _peaked_prior("S", 0.99), config, "earth_return", 1000, rng2
        )
        # Use coefficient of variation (std/mean) — peaked has less relative uncertainty
        cv_uniform = np.std(scores_uniform) / max(np.mean(scores_uniform), 1e-30)
        cv_peaked = np.std(scores_peaked) / max(np.mean(scores_peaked), 1e-30)
        assert cv_peaked < cv_uniform


# --- EVOI Computation Tests ---

class TestComputeEVOI:
    """Tests for compute_evoi."""

    def test_returns_expected_keys(self):
        result = compute_evoi(
            1.0, 1.3, 0.2, 10.0, moid=0.05,
            prior=_uniform_prior(), n_samples=50,
            rng=np.random.default_rng(42),
        )
        expected_keys = {
            "score_mean", "score_std", "evoi_vnir", "evoi_vis",
            "evoi_radar", "evoi_albedo", "best_observation", "best_evoi",
        }
        assert set(result.keys()) == expected_keys

    def test_evoi_nonnegative(self):
        result = compute_evoi(
            1.0, 1.3, 0.2, 10.0, moid=0.05,
            prior=_uniform_prior(), n_samples=50,
            rng=np.random.default_rng(42),
        )
        assert result["evoi_vnir"] >= 0
        assert result["evoi_vis"] >= 0
        assert result["evoi_radar"] >= 0
        assert result["evoi_albedo"] >= 0

    def test_vnir_evoi_greater_than_vis(self):
        """VNIR spectroscopy is more informative than visible-only."""
        result = compute_evoi(
            1.0, 1.3, 0.2, 10.0, moid=0.05,
            prior=_uniform_prior(), n_samples=200,
            rng=np.random.default_rng(42),
        )
        assert result["evoi_vnir"] >= result["evoi_vis"]

    def test_peaked_prior_low_evoi(self):
        """Already well-known asteroid should have low EVOI."""
        result = compute_evoi(
            1.0, 1.3, 0.2, 10.0, moid=0.05,
            prior=_peaked_prior("S", 0.99), n_samples=100,
            rng=np.random.default_rng(42),
        )
        # EVOI should be small when prior is already peaked
        assert result["best_evoi"] < result["score_std"] * 0.5

    def test_uniform_prior_higher_evoi(self):
        """Uniform prior (unknown) should have higher EVOI than peaked."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        result_uniform = compute_evoi(
            1.0, 1.3, 0.2, 10.0, moid=0.05,
            prior=_uniform_prior(), n_samples=500, rng=rng1,
        )
        result_peaked = compute_evoi(
            1.0, 1.3, 0.2, 10.0, moid=0.05,
            prior=_peaked_prior("S", 0.99), n_samples=500, rng=rng2,
        )
        assert result_uniform["evoi_vnir"] >= result_peaked["evoi_vnir"]

    def test_best_observation_is_valid_type(self):
        result = compute_evoi(
            1.0, 1.3, 0.2, 10.0, moid=0.05,
            prior=_uniform_prior(), n_samples=50,
            rng=np.random.default_rng(42),
        )
        assert result["best_observation"] in {
            "vnir_spectroscopy", "vis_spectroscopy", "radar", "albedo"
        }

    def test_best_evoi_matches_best_observation(self):
        result = compute_evoi(
            1.0, 1.3, 0.2, 10.0, moid=0.05,
            prior=_uniform_prior(), n_samples=50,
            rng=np.random.default_rng(42),
        )
        evoi_map = {
            "vnir_spectroscopy": result["evoi_vnir"],
            "vis_spectroscopy": result["evoi_vis"],
            "radar": result["evoi_radar"],
            "albedo": result["evoi_albedo"],
        }
        assert result["best_evoi"] == evoi_map[result["best_observation"]]

    def test_uses_granvik_prior_when_none(self):
        """Should default to Granvik prior when no prior is given."""
        result = compute_evoi(
            1.0, 1.3, 0.2, 10.0, moid=0.05,
            prior=None, n_samples=50,
            rng=np.random.default_rng(42),
        )
        assert result["score_mean"] > 0


# --- EVOIResult Dataclass Tests ---

class TestEVOIResult:
    """Tests for EVOIResult dataclass."""

    def test_evoi_by_type(self):
        r = EVOIResult(
            asteroid_id=1,
            current_score_mean=100.0,
            current_score_std=50.0,
            evoi_vnir=20.0,
            evoi_vis=10.0,
            evoi_radar=5.0,
            evoi_albedo=8.0,
            best_observation="vnir_spectroscopy",
            best_evoi=20.0,
        )
        assert r.evoi_by_type == {
            "vnir_spectroscopy": 20.0,
            "vis_spectroscopy": 10.0,
            "radar": 5.0,
            "albedo": 8.0,
        }


# --- Database Integration Tests ---

class TestComputeEVOIForAsteroid:
    """Tests for compute_evoi_for_asteroid."""

    def test_returns_result_for_uncharacterised(self):
        conn = get_connection(":memory:")
        asteroid_id = _setup_uncharacterised_neo(conn)
        result = compute_evoi_for_asteroid(
            asteroid_id, conn, n_samples=50,
            rng=np.random.default_rng(42),
        )
        assert result is not None
        assert isinstance(result, EVOIResult)
        assert result.asteroid_id == asteroid_id

    def test_returns_result_for_characterised(self):
        """Even characterised asteroids get EVOI (may benefit from more observations)."""
        conn = get_connection(":memory:")
        asteroid_id = _setup_characterised_neo(conn)
        result = compute_evoi_for_asteroid(
            asteroid_id, conn, n_samples=50,
            rng=np.random.default_rng(42),
        )
        assert result is not None

    def test_returns_none_for_missing_asteroid(self):
        conn = get_connection(":memory:")
        result = compute_evoi_for_asteroid(
            99999, conn, n_samples=50,
            rng=np.random.default_rng(42),
        )
        assert result is None

    def test_returns_none_for_no_diameter(self):
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, neo, pha) VALUES (?, ?, ?, ?)",
            (99003, "NoDiam", True, True),
        )
        conn.execute(
            "INSERT INTO orbits (asteroid_id, a, e, i) VALUES (?, ?, ?, ?)",
            (99003, 1.3, 0.2, 10.0),
        )
        result = compute_evoi_for_asteroid(
            99003, conn, n_samples=50,
            rng=np.random.default_rng(42),
        )
        assert result is None


class TestRankAll:
    """Tests for rank_all."""

    def test_ranks_uncharacterised_neos(self):
        conn = get_connection(":memory:")
        _setup_uncharacterised_neo(conn, 99001, diameter=0.5, a=1.3, e=0.2, i=10.0)
        _setup_uncharacterised_neo(conn, 99002, diameter=2.0, a=1.5, e=0.3, i=15.0)
        _setup_uncharacterised_neo(conn, 99003, diameter=0.1, a=2.0, e=0.5, i=25.0)
        results = rank_all(conn, n_samples=50)
        assert len(results) == 3
        # Should be sorted by best_evoi descending
        for i in range(len(results) - 1):
            assert results[i].best_evoi >= results[i + 1].best_evoi

    def test_excludes_characterised_neos(self):
        conn = get_connection(":memory:")
        _setup_uncharacterised_neo(conn, 99001)
        _setup_characterised_neo(conn, 99002)
        results = rank_all(conn, n_samples=50)
        ids = [r.asteroid_id for r in results]
        assert 99001 in ids
        assert 99002 not in ids

    def test_max_asteroids_limits_results(self):
        conn = get_connection(":memory:")
        for i in range(5):
            _setup_uncharacterised_neo(conn, 99001 + i, diameter=float(i + 1))
        results = rank_all(conn, n_samples=50, max_asteroids=2)
        assert len(results) <= 2

    def test_neo_only_filter(self):
        conn = get_connection(":memory:")
        # Insert an MBA (neo=False)
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, neo, pha) VALUES (?, ?, ?, ?)",
            (99010, "MBA", False, False),
        )
        conn.execute(
            "INSERT INTO orbits (asteroid_id, a, e, i, diameter) VALUES (?, ?, ?, ?, ?)",
            (99010, 2.5, 0.1, 5.0, 10.0),
        )
        _setup_uncharacterised_neo(conn, 99001)
        results = rank_all(conn, n_samples=50, neo_only=True)
        ids = [r.asteroid_id for r in results]
        assert 99010 not in ids
        assert 99001 in ids

    def test_empty_database(self):
        conn = get_connection(":memory:")
        results = rank_all(conn, n_samples=50)
        assert results == []


# --- Physical Sanity Tests ---

class TestPhysicalSanity:
    """Physical sanity checks for EVOI results."""

    def test_larger_asteroid_higher_score(self):
        """Larger asteroids should have higher absolute scores."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        prior = _uniform_prior()

        r_small = compute_evoi(
            0.1, 1.3, 0.2, 10.0, moid=0.05, prior=prior, n_samples=100, rng=rng1
        )
        r_large = compute_evoi(
            5.0, 1.3, 0.2, 10.0, moid=0.05, prior=prior, n_samples=100, rng=rng2
        )
        assert r_large["score_mean"] > r_small["score_mean"]

    def test_albedo_bins_sum_to_one(self):
        """Albedo probability bins should be a valid distribution."""
        assert abs(sum(_ALBEDO_PROBS) - 1.0) < 1e-10

    def test_all_observation_types_produce_evoi(self):
        """Every observation type should have a defined EVOI >= 0."""
        result = compute_evoi(
            1.0, 1.3, 0.2, 10.0, moid=0.05,
            prior=_uniform_prior(), n_samples=50,
            rng=np.random.default_rng(42),
        )
        for key in ["evoi_vnir", "evoi_vis", "evoi_radar", "evoi_albedo"]:
            assert result[key] >= 0, f"{key} should be non-negative"

    def test_score_std_positive_for_uniform_prior(self):
        """A uniform prior should produce nonzero score variance."""
        result = compute_evoi(
            1.0, 1.3, 0.2, 10.0, moid=0.05,
            prior=_uniform_prior(), n_samples=100,
            rng=np.random.default_rng(42),
        )
        assert result["score_std"] > 0
