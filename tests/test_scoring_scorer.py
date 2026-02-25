"""Tests for prospector.scoring.scorer — Monte Carlo composite mining scorer."""

import math

import numpy as np
import pytest

from prospector.db import get_connection, init_schema
from prospector.scoring.granvik_prior import MAHLKE_CLASSES
from prospector.scoring.scorer import (
    DEFAULT_MATERIAL_VALUES,
    DEFAULT_RECOVERABILITY,
    SCORED_MATERIALS,
    ConfigArrays,
    _grade_to_fraction,
    build_config_arrays,
    compute_accessibility,
    compute_confidence,
    compute_thermal_depletion_factor,
    estimate_mass_kg,
    load_config,
    sample_from_dist,
    score_all,
    score_asteroid,
)


# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def config():
    """Load the actual grade_density_priors.yaml config."""
    return load_config()


@pytest.fixture
def rng():
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(12345)


@pytest.fixture
def db():
    """In-memory database with schema initialized."""
    conn = get_connection(":memory:")
    return conn


@pytest.fixture
def db_with_asteroids(db):
    """DB with a few sample asteroids for scoring tests."""
    # Itokawa (S-type, well-known NEO)
    db.execute(
        "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (25143, 'Itokawa', 1)"
    )
    db.execute(
        """INSERT INTO orbits (asteroid_id, a, e, i, moid, diameter, H)
           VALUES (25143, 1.324, 0.280, 1.62, 0.013, 0.33, 19.2)"""
    )
    # Bennu (C-type, carbonaceous NEO)
    db.execute(
        "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (101955, 'Bennu', 1)"
    )
    db.execute(
        """INSERT INTO orbits (asteroid_id, a, e, i, moid, diameter, H)
           VALUES (101955, 1.126, 0.204, 6.03, 0.003, 0.49, 20.2)"""
    )
    # Psyche (M-type, large metallic MBA — not NEO)
    db.execute(
        "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (16, 'Psyche', 0)"
    )
    db.execute(
        """INSERT INTO orbits (asteroid_id, a, e, i, moid, diameter)
           VALUES (16, 2.921, 0.134, 3.10, 1.60, 226.0)"""
    )
    # Asteroid with NEOWISE diameter but no SBDB diameter
    db.execute(
        "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (4179, 'Toutatis', 1)"
    )
    db.execute(
        """INSERT INTO orbits (asteroid_id, a, e, i, moid)
           VALUES (4179, 2.510, 0.634, 0.45, 0.006)"""
    )
    db.execute(
        """INSERT INTO physical_properties (asteroid_id, diameter_km, source)
           VALUES (4179, 2.45, 'NEOWISE')"""
    )

    db.commit()
    return db


# ---- ConfigArrays tests -----------------------------------------------------


class TestConfigArrays:
    """Tests for ConfigArrays dataclass and build_config_arrays()."""

    def test_returns_config_arrays_instance(self, config):
        ca = build_config_arrays(config)
        assert isinstance(ca, ConfigArrays)

    def test_density_arrays_shape(self, config):
        ca = build_config_arrays(config)
        assert ca.density_mean.shape == (17,)
        assert ca.density_std.shape == (17,)
        assert ca.density_min.shape == (17,)
        assert ca.density_max.shape == (17,)

    def test_density_all_classes_populated(self, config):
        """Every class should have a density entry (YAML covers all 17 + fallback)."""
        ca = build_config_arrays(config)
        for i, cls in enumerate(MAHLKE_CLASSES):
            assert ca.density_mean[i] > 0, f"{cls} density mean should be positive"
            assert ca.density_std[i] > 0, f"{cls} density std should be positive"
            assert ca.density_max[i] > ca.density_min[i], f"{cls} max > min"

    def test_density_matches_yaml(self, config):
        """Spot-check: M-type density mean should be 3.86 from Carry (2012)."""
        ca = build_config_arrays(config)
        m_idx = MAHLKE_CLASSES.index("M")
        assert ca.density_mean[m_idx] == pytest.approx(3.86)
        assert ca.density_std[m_idx] == pytest.approx(1.54)

    def test_grade_arrays_per_material(self, config):
        ca = build_config_arrays(config)
        for material in SCORED_MATERIALS:
            assert material in ca.grade_mean
            assert ca.grade_mean[material].shape == (17,)
            assert ca.grade_std[material].shape == (17,)
            assert ca.grade_min[material].shape == (17,)
            assert ca.grade_max[material].shape == (17,)

    def test_grade_zeros_for_missing_classes(self, config):
        """PGM has no entry for B-type → should be zero."""
        ca = build_config_arrays(config)
        b_idx = MAHLKE_CLASSES.index("B")
        assert ca.grade_mean["pgm"][b_idx] == 0.0
        assert ca.grade_std["pgm"][b_idx] == 0.0

    def test_grade_matches_yaml_pgm_m(self, config):
        """Spot-check: PGM M-type mean should be 20.0 ppm (Cannon 2023)."""
        ca = build_config_arrays(config)
        m_idx = MAHLKE_CLASSES.index("M")
        assert ca.grade_mean["pgm"][m_idx] == pytest.approx(20.0)
        assert ca.grade_std["pgm"][m_idx] == pytest.approx(10.0)

    def test_grade_matches_yaml_water_c(self, config):
        """Spot-check: Water C-type mean should be 10.0 wt% (Alexander 2012)."""
        ca = build_config_arrays(config)
        c_idx = MAHLKE_CLASSES.index("C")
        assert ca.grade_mean["water"][c_idx] == pytest.approx(10.0)

    def test_unit_factors(self, config):
        ca = build_config_arrays(config)
        # PGM is in ppm → factor 1e-6
        assert ca.unit_factors["pgm"] == pytest.approx(1e-6)
        # Water is in wt_pct → factor 0.01
        assert ca.unit_factors["water"] == pytest.approx(0.01)
        # Iron is in wt_pct → factor 0.01
        assert ca.unit_factors["iron"] == pytest.approx(0.01)

    def test_recoverability_arrays_shape(self, config):
        ca = build_config_arrays(config)
        for material in SCORED_MATERIALS:
            assert material in ca.recoverability
            assert ca.recoverability[material].shape == (17,)

    def test_recoverability_matches_defaults(self, config):
        """Without config overrides, should match DEFAULT_RECOVERABILITY."""
        ca = build_config_arrays(config)
        m_idx = MAHLKE_CLASSES.index("M")
        assert ca.recoverability["pgm"][m_idx] == pytest.approx(0.30)
        assert ca.recoverability["iron"][m_idx] == pytest.approx(0.80)

    def test_recoverability_zeros_for_missing(self, config):
        """B-type has no PGM recoverability → should be 0."""
        ca = build_config_arrays(config)
        b_idx = MAHLKE_CLASSES.index("B")
        assert ca.recoverability["pgm"][b_idx] == 0.0

    def test_frozen_dataclass(self, config):
        """ConfigArrays should be immutable (frozen=True)."""
        ca = build_config_arrays(config)
        with pytest.raises(AttributeError):
            ca.density_mean = np.zeros(17)

    def test_both_modes_same_structure(self, config):
        """earth_return and in_space should produce same array shapes."""
        ca_er = build_config_arrays(config, mode="earth_return")
        ca_is = build_config_arrays(config, mode="in_space")
        assert ca_er.density_mean.shape == ca_is.density_mean.shape
        # Density arrays are mode-independent
        np.testing.assert_array_equal(ca_er.density_mean, ca_is.density_mean)
        # Grade arrays are mode-independent
        for m in SCORED_MATERIALS:
            np.testing.assert_array_equal(ca_er.grade_mean[m], ca_is.grade_mean[m])

    def test_all_density_values_non_negative(self, config):
        ca = build_config_arrays(config)
        assert np.all(ca.density_mean >= 0)
        assert np.all(ca.density_std >= 0)
        assert np.all(ca.density_min >= 0)
        assert np.all(ca.density_max >= 0)

    def test_all_recoverability_in_range(self, config):
        ca = build_config_arrays(config)
        for material in SCORED_MATERIALS:
            assert np.all(ca.recoverability[material] >= 0.0)
            assert np.all(ca.recoverability[material] <= 1.0)


# ---- load_config tests ------------------------------------------------------


class TestLoadConfig:
    def test_loads_default_config(self, config):
        assert "density_priors" in config
        assert "grade_estimates" in config
        assert "metadata" in config

    def test_all_mahlke_classes_have_density(self, config):
        classes = config["density_priors"]["classes"]
        for cls in MAHLKE_CLASSES:
            assert cls in classes, f"Missing density prior for {cls}"
            d = classes[cls]
            assert "mean" in d and "std" in d and "min" in d and "max" in d

    def test_density_values_positive(self, config):
        for cls, d in config["density_priors"]["classes"].items():
            assert d["mean"] > 0, f"{cls} mean density must be positive"
            assert d["min"] >= 0, f"{cls} min density must be non-negative"
            assert d["max"] > d["min"], f"{cls} max must exceed min"


# ---- sample_from_dist tests -------------------------------------------------


class TestSampleFromDist:
    def test_within_bounds(self, rng):
        dist = {"mean": 3.0, "std": 1.0, "min": 1.0, "max": 5.0}
        for _ in range(500):
            val = sample_from_dist(dist, rng)
            assert 1.0 <= val <= 5.0

    def test_near_mean(self, rng):
        dist = {"mean": 10.0, "std": 0.001, "min": 0.0, "max": 100.0}
        val = sample_from_dist(dist, rng)
        assert abs(val - 10.0) < 0.1

    def test_uses_default_rng(self):
        dist = {"mean": 5.0, "std": 1.0, "min": 0.0, "max": 10.0}
        val = sample_from_dist(dist)
        assert 0.0 <= val <= 10.0


# ---- estimate_mass_kg tests -------------------------------------------------


class TestEstimateMass:
    def test_unit_sphere(self):
        # 1 km diameter, 1 g/cm³ = 1000 kg/m³
        mass = estimate_mass_kg(1.0, 1.0)
        expected = (4 / 3) * math.pi * (500.0) ** 3 * 1000.0
        assert abs(mass - expected) < 1.0

    def test_itokawa_mass(self):
        # Itokawa: ~0.33 km, ~1.9 g/cm³ → ~3.6e10 kg (published ~3.51e10 kg)
        mass = estimate_mass_kg(0.33, 1.9)
        assert 1e10 < mass < 1e11

    def test_psyche_mass(self):
        # Psyche: ~226 km, ~3.86 g/cm³ → ~2.3e19 kg (published ~2.72e19 kg)
        mass = estimate_mass_kg(226.0, 3.86)
        assert 1e19 < mass < 5e19

    def test_zero_diameter(self):
        assert estimate_mass_kg(0.0, 3.0) == 0.0


# ---- compute_accessibility tests --------------------------------------------


class TestComputeAccessibility:
    def test_low_moid_high_score(self):
        # MOID 0.01 AU → exp(-0.1) ≈ 0.90
        score = compute_accessibility(moid_au=0.01)
        assert score > 0.8

    def test_high_moid_low_score(self):
        # MOID 1.0 AU → exp(-10) ≈ 4.5e-5 → clips to 0.01
        score = compute_accessibility(moid_au=1.0)
        assert score == pytest.approx(0.01, abs=0.001)

    def test_zero_moid_perfect_score(self):
        score = compute_accessibility(moid_au=0.0)
        assert score == pytest.approx(1.0)

    def test_fallback_perihelion(self):
        # a=1.0, e=0.0 → q=1.0, score = 1.0 - 0 = 1.0
        score = compute_accessibility(a=1.0, e=0.0)
        assert score == pytest.approx(1.0)

    def test_fallback_far_perihelion(self):
        # a=3.0, e=0.0 → q=3.0, score = max(0, 1 - 2/0.5) = max(0, -3) → clips to 0.01
        score = compute_accessibility(a=3.0, e=0.0)
        assert score == pytest.approx(0.01, abs=0.001)

    def test_no_data_returns_minimum(self):
        score = compute_accessibility()
        assert score == 0.01

    def test_moid_preferred_over_fallback(self):
        # Should use MOID when both provided
        score_moid = compute_accessibility(moid_au=0.01, a=3.0, e=0.0)
        assert score_moid > 0.5  # MOID gives high score, fallback would give 0.01


# ---- compute_confidence tests ------------------------------------------------


class TestComputeConfidence:
    def test_peaked_distribution(self):
        # All mass on one class → entropy 0 → confidence 1.0
        pv = np.zeros(17)
        pv[13] = 1.0  # S-type
        conf = compute_confidence(pv)
        assert conf == pytest.approx(1.0)

    def test_uniform_distribution(self):
        # Uniform → max entropy → confidence near 0
        pv = np.ones(17) / 17.0
        conf = compute_confidence(pv)
        assert conf < 0.05

    def test_two_class_distribution(self):
        # Split between two classes → intermediate confidence
        pv = np.zeros(17)
        pv[13] = 0.5  # S
        pv[2] = 0.5  # C
        conf = compute_confidence(pv)
        assert 0.5 < conf < 1.0

    def test_empty_vector(self):
        pv = np.zeros(17)
        conf = compute_confidence(pv)
        assert conf == 0.01

    def test_minimum_clamp(self):
        # Very spread distribution still returns >= 0.01
        pv = np.ones(17) / 17.0
        conf = compute_confidence(pv)
        assert conf >= 0.01


# ---- _grade_to_fraction tests -----------------------------------------------


class TestGradeToFraction:
    def test_ppm(self):
        assert _grade_to_fraction(20.0, "ppm") == pytest.approx(20e-6)

    def test_wt_pct(self):
        assert _grade_to_fraction(10.0, "wt_pct") == pytest.approx(0.10)

    def test_passthrough(self):
        assert _grade_to_fraction(0.5, "fraction") == pytest.approx(0.5)


# ---- score_asteroid tests ---------------------------------------------------


class TestScoreAsteroid:
    def test_returns_required_keys(self, config, rng):
        result = score_asteroid(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            config=config, n_samples=100, rng=rng,
        )
        required_keys = {
            "composite_score", "estimated_mass_kg", "grade_estimate",
            "target_material", "unit_value", "accessibility",
            "confidence", "spin_modifier", "thermal_depletion_factor",
            "jwst_water_boost", "score_mode", "material_contributions",
        }
        assert required_keys == set(result.keys())

    def test_score_positive(self, config, rng):
        result = score_asteroid(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            config=config, n_samples=200, rng=rng,
        )
        assert result["composite_score"] > 0
        assert result["estimated_mass_kg"] > 0

    def test_larger_asteroid_higher_score(self, config):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        small = score_asteroid(
            diameter_km=0.1, a=1.5, e=0.3, i_deg=10.0,
            config=config, n_samples=500, rng=rng1,
        )
        large = score_asteroid(
            diameter_km=10.0, a=1.5, e=0.3, i_deg=10.0,
            config=config, n_samples=500, rng=rng2,
        )
        assert large["composite_score"] > small["composite_score"]

    def test_earth_return_vs_in_space(self, config, rng):
        # For an inner-belt S-type orbit, earth_return should favor PGMs,
        # in_space should value water more (but S-type has no water so still PGM)
        result_er = score_asteroid(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            config=config, n_samples=500, mode="earth_return",
            rng=np.random.default_rng(42),
        )
        result_is = score_asteroid(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            config=config, n_samples=500, mode="in_space",
            rng=np.random.default_rng(42),
        )
        assert result_er["score_mode"] == "earth_return"
        assert result_is["score_mode"] == "in_space"
        # In-space mode values water and structural materials more
        assert result_is["composite_score"] != result_er["composite_score"]

    def test_c_type_water_score(self, config):
        # A C-type peaked asteroid should score well for water
        pv = np.zeros(17)
        pv[2] = 0.9  # C-type
        pv[3] = 0.1  # Ch-type
        result = score_asteroid(
            diameter_km=1.0, a=2.5, e=0.3, i_deg=5.0,
            prob_vector=pv, config=config, n_samples=500,
            mode="in_space", rng=np.random.default_rng(42),
        )
        # Water should be a top contributor in in-space mode
        assert result["material_contributions"]["water"] > 0

    def test_m_type_pgm_dominates_earth_return(self, config):
        # M-type asteroid → PGM should dominate in earth_return mode
        pv = np.zeros(17)
        pv[8] = 1.0  # M-type
        result = score_asteroid(
            diameter_km=1.0, a=2.5, e=0.3, i_deg=5.0,
            prob_vector=pv, config=config, n_samples=500,
            mode="earth_return", rng=np.random.default_rng(42),
        )
        assert result["target_material"] == "pgm"
        assert result["material_contributions"]["pgm"] > 0

    def test_uses_granvik_prior_when_no_prob_vector(self, config, rng):
        result = score_asteroid(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            prob_vector=None, config=config, n_samples=100, rng=rng,
        )
        # Should still produce a valid score using the Granvik prior
        assert result["composite_score"] > 0
        assert result["confidence"] > 0

    def test_accessibility_in_result(self, config, rng):
        result = score_asteroid(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            moid=0.01, config=config, n_samples=100, rng=rng,
        )
        assert result["accessibility"] > 0.5

    def test_confidence_peaked_taxonomy(self, config, rng):
        pv = np.zeros(17)
        pv[13] = 1.0  # 100% S-type
        result = score_asteroid(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            prob_vector=pv, config=config, n_samples=100, rng=rng,
        )
        assert result["confidence"] == pytest.approx(1.0)

    def test_material_contributions_non_negative(self, config, rng):
        result = score_asteroid(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            config=config, n_samples=200, rng=rng,
        )
        for m, v in result["material_contributions"].items():
            assert v >= 0, f"{m} contribution should be non-negative"

    def test_deterministic_with_same_seed(self, config):
        kwargs = dict(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            config=config, n_samples=500,
        )
        r1 = score_asteroid(**kwargs, rng=np.random.default_rng(99))
        r2 = score_asteroid(**kwargs, rng=np.random.default_rng(99))
        assert r1["composite_score"] == pytest.approx(r2["composite_score"])

    def test_custom_material_values(self, config, rng):
        custom_values = {"pgm": 100_000, "water": 0, "iron": 0, "olivine": 0, "pyroxene": 0}
        result = score_asteroid(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            config=config, n_samples=200,
            material_values=custom_values, rng=rng,
        )
        # Only PGM should contribute
        assert result["material_contributions"]["water"] == 0
        assert result["material_contributions"]["iron"] == 0

    def test_custom_recoverability(self, config, rng):
        # Set all recoverability to zero → score should be zero
        zero_recover = {m: {} for m in SCORED_MATERIALS}
        result = score_asteroid(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            config=config, n_samples=100,
            recoverability=zero_recover, rng=rng,
        )
        assert result["composite_score"] == pytest.approx(0.0)


# ---- score_all tests (DB integration) ----------------------------------------


class TestScoreAll:
    def test_scores_asteroids_with_diameters(self, db_with_asteroids):
        count = score_all(db_with_asteroids, n_samples=100, mode="earth_return")
        # Should score all 4 asteroids (3 with SBDB diameters + 1 with NEOWISE)
        assert count == 4

    def test_scores_written_to_db(self, db_with_asteroids):
        score_all(db_with_asteroids, n_samples=100)
        rows = db_with_asteroids.execute("SELECT * FROM scores").fetchall()
        assert len(rows) == 4

    def test_score_columns_populated(self, db_with_asteroids):
        score_all(db_with_asteroids, n_samples=100)
        row = db_with_asteroids.execute(
            "SELECT estimated_mass_kg, grade_estimate, target_material, "
            "unit_value, accessibility, composite_score, score_mode "
            "FROM scores WHERE asteroid_id = 25143"
        ).fetchone()
        assert row is not None
        mass, grade, material, uv, access, score, mode = row
        assert mass > 0
        assert grade > 0
        assert material in SCORED_MATERIALS
        assert uv > 0
        assert 0 < access <= 1.0
        assert score > 0
        assert mode == "earth_return"

    def test_neowise_diameter_fallback(self, db_with_asteroids):
        """Toutatis has NEOWISE diameter but no SBDB diameter — should still be scored."""
        score_all(db_with_asteroids, n_samples=100)
        row = db_with_asteroids.execute(
            "SELECT composite_score FROM scores WHERE asteroid_id = 4179"
        ).fetchone()
        assert row is not None
        assert row[0] > 0

    def test_uses_spectral_taxonomy_when_available(self, db_with_asteroids):
        # Add spectral taxonomy for Itokawa (S-type)
        pv = np.zeros(17, dtype=np.float64)
        pv[13] = 0.85  # S
        pv[11] = 0.10  # Q
        pv[6] = 0.05  # K
        db_with_asteroids.execute(
            """INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob,
               prob_vector, classifier, input_coverage)
               VALUES (25143, 'S', 0.85, ?, 'classy_mahlke2022', 'vnir')""",
            (pv.tobytes(),),
        )
        db_with_asteroids.commit()

        score_all(db_with_asteroids, n_samples=200)
        row = db_with_asteroids.execute(
            "SELECT composite_score FROM scores WHERE asteroid_id = 25143"
        ).fetchone()
        assert row is not None
        assert row[0] > 0

    def test_in_space_mode(self, db_with_asteroids):
        count = score_all(db_with_asteroids, n_samples=100, mode="in_space")
        assert count == 4
        row = db_with_asteroids.execute(
            "SELECT score_mode FROM scores WHERE asteroid_id = 101955"
        ).fetchone()
        assert row[0] == "in_space"

    def test_idempotent_rerun(self, db_with_asteroids):
        """Running score_all twice should update (not duplicate) scores."""
        score_all(db_with_asteroids, n_samples=100)
        score_all(db_with_asteroids, n_samples=100)
        count = db_with_asteroids.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        assert count == 4

    def test_skips_asteroids_without_diameter(self, db):
        """Asteroids with no diameter from either source should not be scored."""
        db.execute(
            "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (99999, 'NoDiam', 1)"
        )
        db.execute(
            "INSERT INTO orbits (asteroid_id, a, e, i) VALUES (99999, 1.5, 0.3, 10.0)"
        )
        db.commit()
        count = score_all(db, n_samples=100)
        assert count == 0

    def test_skips_asteroids_without_orbital_elements(self, db):
        """Asteroids missing a, e, or i should not be scored."""
        db.execute(
            "INSERT INTO asteroids (asteroid_id, name) VALUES (99998, 'NoOrbit')"
        )
        db.execute(
            "INSERT INTO orbits (asteroid_id, diameter) VALUES (99998, 1.0)"
        )
        db.commit()
        count = score_all(db, n_samples=100)
        assert count == 0


# ---- compute_thermal_depletion_factor tests ----------------------------------


class TestComputeThermalDepletionFactor:
    def test_no_data_returns_one(self):
        assert compute_thermal_depletion_factor(None) == 1.0

    def test_safe_perihelion(self):
        # q >= 1.0 AU → no penalty
        assert compute_thermal_depletion_factor(1.0) == 1.0
        assert compute_thermal_depletion_factor(1.5) == 1.0
        assert compute_thermal_depletion_factor(3.0) == 1.0

    def test_low_perihelion_severe_penalty(self):
        # q ≈ 0.2 AU → near-complete depletion
        factor = compute_thermal_depletion_factor(0.2)
        assert factor < 0.10

    def test_moderate_perihelion_partial_penalty(self):
        # q ≈ 0.6 AU → ~50% retention (sigmoid center)
        factor = compute_thermal_depletion_factor(0.6)
        assert 0.4 < factor < 0.6

    def test_monotonically_increasing(self):
        """Higher perihelion → higher water retention."""
        q_values = [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
        factors = [compute_thermal_depletion_factor(q) for q in q_values]
        for i in range(len(factors) - 1):
            assert factors[i] <= factors[i + 1], (
                f"Factor should increase: q={q_values[i]}→{factors[i]} vs "
                f"q={q_values[i+1]}→{factors[i+1]}"
            )

    def test_output_range(self):
        """Factor always in [0, 1]."""
        for q in [0.01, 0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0]:
            factor = compute_thermal_depletion_factor(q)
            assert 0.0 <= factor <= 1.0, f"q={q}: factor={factor} out of range"


class TestThermalDepletionIntegration:
    def test_low_q_reduces_water_score(self, config):
        """C-type asteroid at low perihelion should have reduced water score."""
        pv = np.zeros(17)
        pv[2] = 1.0  # C-type (water-bearing)

        # Safe orbit (q ≈ 1.7 AU)
        safe = score_asteroid(
            diameter_km=1.0, a=2.0, e=0.15, i_deg=5.0,
            prob_vector=pv, config=config, n_samples=500,
            mode="in_space", rng=np.random.default_rng(42),
        )
        # Low-perihelion orbit (q ≈ 0.3 AU)
        hot = score_asteroid(
            diameter_km=1.0, a=2.0, e=0.85, i_deg=5.0,
            prob_vector=pv, config=config, n_samples=500,
            mode="in_space", rng=np.random.default_rng(42),
        )
        assert hot["material_contributions"]["water"] < safe["material_contributions"]["water"]
        assert hot["thermal_depletion_factor"] < 0.2
        assert safe["thermal_depletion_factor"] == 1.0

    def test_low_q_does_not_affect_pgm(self, config):
        """M-type PGM score should be unaffected by thermal depletion."""
        pv = np.zeros(17)
        pv[8] = 1.0  # M-type (metal-rich)

        safe = score_asteroid(
            diameter_km=1.0, a=2.0, e=0.15, i_deg=5.0,
            prob_vector=pv, config=config, n_samples=500,
            mode="earth_return", rng=np.random.default_rng(42),
        )
        hot = score_asteroid(
            diameter_km=1.0, a=2.0, e=0.85, i_deg=5.0,
            prob_vector=pv, config=config, n_samples=500,
            mode="earth_return", rng=np.random.default_rng(42),
        )
        # PGM contributions should be similar (only accessibility differs)
        # The thermal depletion factor does NOT affect PGMs
        assert hot["thermal_depletion_factor"] < 0.2
        # PGM contribution is unaffected by thermal depletion (only accessibility differs)
        # Normalise out accessibility difference to compare material contributions
        hot_pgm_norm = hot["material_contributions"]["pgm"] / hot["accessibility"]
        safe_pgm_norm = safe["material_contributions"]["pgm"] / safe["accessibility"]
        assert hot_pgm_norm == pytest.approx(safe_pgm_norm, rel=0.15)

    def test_explicit_q_au_overrides_derived(self, config):
        """Explicit q_au parameter should override a*(1-e)."""
        pv = np.zeros(17)
        pv[2] = 1.0  # C-type

        # Orbital elements give q = 2.0 * 0.85 = 0.3 AU (hot), but override with safe q
        result = score_asteroid(
            diameter_km=1.0, a=2.0, e=0.85, i_deg=5.0,
            prob_vector=pv, config=config, n_samples=100,
            mode="in_space", rng=np.random.default_rng(42),
            q_au=1.5,  # override: safe perihelion
        )
        assert result["thermal_depletion_factor"] == 1.0

    def test_thermal_depletion_in_result_dict(self, config, rng):
        result = score_asteroid(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            config=config, n_samples=100, rng=rng,
        )
        assert "thermal_depletion_factor" in result
        assert 0.0 <= result["thermal_depletion_factor"] <= 1.0


# ---- Scoring sanity checks ---------------------------------------------------


class TestScoringSanity:
    """Physical plausibility checks on scoring results."""

    def test_close_accessible_neo_beats_distant_mba(self, config):
        """A close NEO should outscore a distant MBA of similar size."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        neo = score_asteroid(
            diameter_km=1.0, a=1.3, e=0.2, i_deg=5.0,
            moid=0.01, config=config, n_samples=500, rng=rng1,
        )
        mba = score_asteroid(
            diameter_km=1.0, a=2.8, e=0.1, i_deg=10.0,
            moid=1.5, config=config, n_samples=500, rng=rng2,
        )
        assert neo["composite_score"] > mba["composite_score"]

    def test_metallic_higher_pgm_than_carbonaceous(self, config):
        """M-type should have higher PGM contribution than C-type."""
        pv_m = np.zeros(17)
        pv_m[8] = 1.0  # M
        pv_c = np.zeros(17)
        pv_c[2] = 1.0  # C

        m_result = score_asteroid(
            diameter_km=1.0, a=2.0, e=0.3, i_deg=5.0,
            prob_vector=pv_m, config=config, n_samples=500,
            mode="earth_return", rng=np.random.default_rng(42),
        )
        c_result = score_asteroid(
            diameter_km=1.0, a=2.0, e=0.3, i_deg=5.0,
            prob_vector=pv_c, config=config, n_samples=500,
            mode="earth_return", rng=np.random.default_rng(42),
        )
        assert m_result["material_contributions"]["pgm"] > c_result["material_contributions"]["pgm"]

    def test_carbonaceous_higher_water_than_silicate(self, config):
        """C-type should have higher water contribution than S-type."""
        pv_c = np.zeros(17)
        pv_c[2] = 1.0  # C
        pv_s = np.zeros(17)
        pv_s[13] = 1.0  # S

        c_result = score_asteroid(
            diameter_km=1.0, a=2.0, e=0.3, i_deg=5.0,
            prob_vector=pv_c, config=config, n_samples=500,
            mode="in_space", rng=np.random.default_rng(42),
        )
        s_result = score_asteroid(
            diameter_km=1.0, a=2.0, e=0.3, i_deg=5.0,
            prob_vector=pv_s, config=config, n_samples=500,
            mode="in_space", rng=np.random.default_rng(42),
        )
        assert c_result["material_contributions"]["water"] > s_result["material_contributions"]["water"]

    def test_mass_scales_with_diameter_cubed(self, config):
        """Mass should scale as diameter³ for same density."""
        pv = np.zeros(17)
        pv[13] = 1.0  # S-type (fixed taxonomy to fix density draws)
        r1 = score_asteroid(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            prob_vector=pv, config=config, n_samples=1000,
            rng=np.random.default_rng(42),
        )
        r2 = score_asteroid(
            diameter_km=2.0, a=1.5, e=0.3, i_deg=10.0,
            prob_vector=pv, config=config, n_samples=1000,
            rng=np.random.default_rng(42),
        )
        ratio = r2["estimated_mass_kg"] / r1["estimated_mass_kg"]
        assert ratio == pytest.approx(8.0, rel=0.1)  # 2³ = 8


# ---- Default constants tests -------------------------------------------------


class TestDefaults:
    def test_scored_materials_list(self):
        assert "pgm" in SCORED_MATERIALS
        assert "water" in SCORED_MATERIALS
        assert "iron" in SCORED_MATERIALS

    def test_material_values_both_modes(self):
        assert "earth_return" in DEFAULT_MATERIAL_VALUES
        assert "in_space" in DEFAULT_MATERIAL_VALUES
        for mode in ["earth_return", "in_space"]:
            for m in SCORED_MATERIALS:
                assert m in DEFAULT_MATERIAL_VALUES[mode]

    def test_water_cheaper_on_earth(self):
        assert DEFAULT_MATERIAL_VALUES["earth_return"]["water"] < DEFAULT_MATERIAL_VALUES["in_space"]["water"]

    def test_recoverability_values_in_range(self):
        for material, classes in DEFAULT_RECOVERABILITY.items():
            for cls, factor in classes.items():
                assert 0.0 <= factor <= 1.0, f"{material}/{cls} factor out of range"
