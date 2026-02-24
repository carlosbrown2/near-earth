"""Tests for scoring/grade_density_priors.yaml.

Validates structure, distribution format, value ranges, and literature-derived
sanity checks for the Monte Carlo scoring config.
"""

import math
from pathlib import Path

import pytest
import yaml

YAML_PATH = Path(__file__).parent.parent / "scoring" / "grade_density_priors.yaml"

MAHLKE_CLASSES = ["A", "B", "C", "Ch", "D", "E", "K", "L", "M", "O", "P", "Q", "R", "S", "V", "X", "Z"]

DISTRIBUTION_KEYS = {"mean", "std", "min", "max"}


@pytest.fixture(scope="module")
def config():
    """Load the YAML config once for all tests."""
    assert YAML_PATH.exists(), f"Config file not found: {YAML_PATH}"
    with open(YAML_PATH) as f:
        return yaml.safe_load(f)


# --- Structural tests -------------------------------------------------------


def test_yaml_loads(config):
    """YAML is parseable and non-empty."""
    assert config is not None
    assert isinstance(config, dict)


def test_top_level_keys(config):
    """Config has required top-level sections."""
    for key in ["metadata", "density_priors", "grade_estimates", "references"]:
        assert key in config, f"Missing top-level key: {key}"


def test_all_mahlke_classes_have_density(config):
    """Every Mahlke 2022 taxonomy class has a density prior."""
    density_classes = set(config["density_priors"]["classes"].keys())
    for cls in MAHLKE_CLASSES:
        assert cls in density_classes, f"Missing density prior for class {cls}"


def test_density_unit(config):
    """Density unit is g/cm3."""
    assert config["density_priors"]["unit"] == "g/cm3"


# --- Distribution format tests ----------------------------------------------


def _check_distribution(dist, label):
    """Validate a single distribution dict."""
    for key in DISTRIBUTION_KEYS:
        assert key in dist, f"{label}: missing key '{key}'"
        assert isinstance(dist[key], (int, float)), f"{label}: {key} must be numeric"

    assert dist["min"] <= dist["mean"] <= dist["max"], (
        f"{label}: mean ({dist['mean']}) not in [{dist['min']}, {dist['max']}]"
    )
    assert dist["std"] >= 0, f"{label}: std must be non-negative"


def test_density_distributions(config):
    """All density priors have valid distribution format."""
    for cls, dist in config["density_priors"]["classes"].items():
        _check_distribution(dist, f"density_priors.{cls}")


def test_grade_distributions(config):
    """All grade estimates have valid distribution format."""
    for material, mat_data in config["grade_estimates"].items():
        if "classes" in mat_data:
            for cls, dist in mat_data["classes"].items():
                _check_distribution(dist, f"grade_estimates.{material}.{cls}")
        elif "subgroups" in mat_data:
            for sg, dist in mat_data["subgroups"].items():
                _check_distribution(dist, f"grade_estimates.{material}.{sg}")


# --- Physical plausibility tests --------------------------------------------


def test_density_positive(config):
    """All density means are positive."""
    for cls, dist in config["density_priors"]["classes"].items():
        assert dist["mean"] > 0, f"density_priors.{cls}: mean must be positive"
        assert dist["min"] >= 0, f"density_priors.{cls}: min must be non-negative"


def test_density_ranges_physical(config):
    """Densities within physical bounds (0-8 g/cm³ for asteroids)."""
    for cls, dist in config["density_priors"]["classes"].items():
        assert dist["max"] <= 8.0, (
            f"density_priors.{cls}: max density {dist['max']} exceeds iron meteorite grain density (~7.8)"
        )


def test_m_type_densest(config):
    """M-type should have the highest mean density (metallic)."""
    densities = config["density_priors"]["classes"]
    m_mean = densities["M"]["mean"]
    for cls, dist in densities.items():
        if cls in ("X", "Z"):
            continue  # ambiguous / unclassifiable
        if cls != "M":
            assert dist["mean"] <= m_mean, (
                f"{cls} mean density ({dist['mean']}) exceeds M-type ({m_mean})"
            )


def test_carbonaceous_less_dense_than_silicate(config):
    """C-complex should be less dense than S-complex."""
    densities = config["density_priors"]["classes"]
    assert densities["C"]["mean"] < densities["S"]["mean"]
    assert densities["B"]["mean"] < densities["S"]["mean"]
    assert densities["D"]["mean"] < densities["S"]["mean"]


# --- Material-specific value checks ------------------------------------------


def test_pgm_cannon_not_kargel(config):
    """PGM M-type uses Cannon (2023) values, not Kargel (1994).

    Cannon revised PGM estimates downward from Kargel's ~30-65 ppm
    to ~20 ppm median due to non-chondritic PGM ratios at high Ir.
    """
    pgm_m = config["grade_estimates"]["pgm"]["classes"]["M"]
    # Cannon2023 median ~20 ppm; Kargel1994 median ~47 ppm
    assert pgm_m["mean"] < 30, (
        f"PGM M-type mean ({pgm_m['mean']}) looks like Kargel values; should use Cannon2023"
    )
    assert "Cannon" in pgm_m.get("source", ""), "PGM M-type source should cite Cannon"


def test_pgm_m_type_higher_than_s_type(config):
    """M-type PGMs should be much higher than S-type (iron vs OC analog)."""
    pgm = config["grade_estimates"]["pgm"]["classes"]
    assert pgm["M"]["mean"] > 5 * pgm["S"]["mean"]


def test_water_c_types_significant(config):
    """C-complex water content should be substantial (>5 wt%)."""
    water = config["grade_estimates"]["water"]["classes"]
    for cls in ["C", "Ch"]:
        assert water[cls]["mean"] >= 5.0, f"Water {cls} mean too low for carbonaceous analog"


def test_water_ch_higher_than_c(config):
    """Ch (hydrated C) should have equal or higher water than generic C."""
    water = config["grade_estimates"]["water"]["classes"]
    assert water["Ch"]["mean"] >= water["C"]["mean"]


def test_iron_m_type_dominant(config):
    """M-type iron content should be > 50 wt% (metallic body)."""
    iron_m = config["grade_estimates"]["iron"]["classes"]["M"]
    assert iron_m["mean"] > 50.0


def test_olivine_a_type_dominant(config):
    """A-type olivine should be highest (olivine-dominated)."""
    olivine = config["grade_estimates"]["olivine"]["classes"]
    a_mean = olivine["A"]["mean"]
    for cls, dist in olivine.items():
        if cls != "A":
            assert dist["mean"] <= a_mean, (
                f"Olivine {cls} ({dist['mean']}) exceeds A-type ({a_mean})"
            )


def test_pyroxene_v_type_high(config):
    """V-type pyroxene should be high (HED achondrite)."""
    pyroxene_v = config["grade_estimates"]["pyroxene"]["classes"]["V"]
    assert pyroxene_v["mean"] >= 40.0


def test_s_type_metal_subgroups(config):
    """S-type metal fractions: H > L > LL."""
    metal = config["grade_estimates"]["s_type_metal_fraction"]["subgroups"]
    assert metal["H"]["mean"] > metal["L"]["mean"] > metal["LL"]["mean"]


# --- Sanity check: 16 Psyche mass estimate ----------------------------------


def test_psyche_mass_sanity(config):
    """Estimate mass of 16 Psyche using M-type density; compare to published.

    16 Psyche: D ~ 226 km, published mass ~ 2.72e19 kg.
    Our M-type bulk density (Carry 2012) should give a mass within ~50%
    (accounting for Psyche being denser than the population mean).
    """
    d_km = 226.0
    d_m = d_km * 1000
    r_m = d_m / 2

    rho_m = config["density_priors"]["classes"]["M"]
    rho_mean = rho_m["mean"] * 1000  # g/cm³ → kg/m³

    volume = (4 / 3) * math.pi * r_m ** 3
    mass_est = volume * rho_mean

    published_mass = 2.72e19  # kg
    ratio = mass_est / published_mass

    # Should be within factor of 2 (density uncertainty is large for M-type)
    assert 0.5 < ratio < 2.0, (
        f"Psyche mass estimate {mass_est:.2e} kg vs published {published_mass:.2e} kg "
        f"(ratio {ratio:.2f}) — outside 50-200% range"
    )


# --- References section tests -----------------------------------------------


def test_cannon_reference_exists(config):
    """Cannon et al. (2023) must be in references (critical for PGM values)."""
    assert "Cannon2023" in config["references"]


def test_carry_reference_exists(config):
    """Carry (2012) must be in references (critical for density priors)."""
    assert "Carry2012" in config["references"]


def test_alexander_reference_exists(config):
    """Alexander et al. (2012) must be in references (critical for water values)."""
    assert "Alexander2012" in config["references"]


def test_kargel_documented(config):
    """Kargel (1994) should be documented as superseded reference."""
    assert "Kargel1994" in config["references"]
    kargel = config["references"]["Kargel1994"]
    assert "supersed" in kargel.get("used_for", "").lower() or "comparison" in kargel.get("used_for", "").lower()
