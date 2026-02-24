"""Tests for recoverability/accessibility scoring config (YAML)."""

import math
from pathlib import Path

import yaml
import pytest

# Path to the YAML config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "scoring" / "recoverability_accessibility.yaml"

MATERIALS = ["pgm", "water", "iron", "olivine", "pyroxene"]

MAHLKE_CLASSES = [
    "A", "B", "C", "Ch", "D", "E", "K", "L", "M",
    "O", "P", "Q", "R", "S", "V", "X", "Z",
]


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


class TestConfigStructure:
    def test_file_exists(self):
        assert CONFIG_PATH.exists()

    def test_top_level_keys(self, config):
        assert "metadata" in config
        assert "recoverability" in config
        assert "accessibility" in config

    def test_all_materials_present(self, config):
        for mat in MATERIALS:
            assert mat in config["recoverability"], f"Missing material: {mat}"

    def test_each_material_has_classes(self, config):
        for mat in MATERIALS:
            section = config["recoverability"][mat]
            assert "classes" in section, f"{mat} missing 'classes'"
            classes = section["classes"]
            assert len(classes) > 0, f"{mat} has no classes"

    def test_each_material_has_description(self, config):
        for mat in MATERIALS:
            section = config["recoverability"][mat]
            assert "description" in section, f"{mat} missing description"

    def test_accessibility_has_function_form(self, config):
        acc = config["accessibility"]
        assert "function_form" in acc
        assert acc["function_form"] == "exponential_decay"

    def test_accessibility_has_dv_scale(self, config):
        dv = config["accessibility"]["delta_v"]
        assert "dv_scale" in dv
        assert "value" in dv["dv_scale"]
        assert dv["dv_scale"]["unit"] == "km/s"

    def test_accessibility_has_moid_proxy(self, config):
        moid = config["accessibility"]["moid_proxy"]
        assert "moid_scale" in moid
        assert moid["moid_scale"]["value"] == 0.1
        assert moid["moid_scale"]["unit"] == "AU"

    def test_accessibility_has_tiers(self, config):
        tiers = config["accessibility"]["tiers"]
        assert len(tiers) >= 3

    def test_verification_examples_present(self, config):
        examples = config["verification_examples"]
        assert len(examples) >= 4
        names = [e["asteroid"] for e in examples]
        assert "16 Psyche" in names
        assert "101955 Bennu" in names


# ---------------------------------------------------------------------------
# Value range tests
# ---------------------------------------------------------------------------


class TestRecoverabilityValues:
    def test_all_factors_0_to_1(self, config):
        for mat in MATERIALS:
            classes = config["recoverability"][mat]["classes"]
            for cls, factor in classes.items():
                assert 0.0 <= factor <= 1.0, (
                    f"{mat}/{cls} factor {factor} out of range [0,1]"
                )

    def test_all_class_keys_are_valid_mahlke(self, config):
        for mat in MATERIALS:
            classes = config["recoverability"][mat]["classes"]
            for cls in classes:
                assert cls in MAHLKE_CLASSES, (
                    f"{mat} has invalid class key: {cls}"
                )


# ---------------------------------------------------------------------------
# Physical plausibility tests
# ---------------------------------------------------------------------------


class TestPhysicalPlausibility:
    def test_m_type_highest_pgm_recoverability(self, config):
        """M-type should have highest PGM recoverability (bulk metal body)."""
        pgm = config["recoverability"]["pgm"]["classes"]
        m_val = pgm["M"]
        for cls, val in pgm.items():
            if cls != "M":
                assert val <= m_val, f"PGM: {cls} ({val}) > M ({m_val})"

    def test_s_type_pgm_lower_than_m(self, config):
        """S-type PGM requires two-stage extraction → lower factor."""
        pgm = config["recoverability"]["pgm"]["classes"]
        assert pgm["S"] < pgm["M"]

    def test_c_type_highest_water(self, config):
        """C-type (CI/CM) should have among the highest water recoverability."""
        water = config["recoverability"]["water"]["classes"]
        assert water["C"] >= 0.6
        assert water["Ch"] >= 0.6

    def test_s_type_water_zero(self, config):
        """S-type water (~450 ug/g) is below extraction threshold → factor 0."""
        water = config["recoverability"]["water"]["classes"]
        assert water.get("S", 0) == 0.0

    def test_m_type_iron_highest(self, config):
        """M-type should have highest iron recoverability (bulk metal)."""
        iron = config["recoverability"]["iron"]["classes"]
        m_val = iron["M"]
        assert m_val >= 0.7
        for cls, val in iron.items():
            if cls != "M":
                assert val <= m_val

    def test_a_type_olivine_high(self, config):
        """A-type (olivine-dominated) should have high olivine recoverability."""
        olivine = config["recoverability"]["olivine"]["classes"]
        assert olivine["A"] >= 0.7

    def test_v_type_pyroxene_high(self, config):
        """V-type (HED, pyroxene-rich) should have high pyroxene recoverability."""
        pyroxene = config["recoverability"]["pyroxene"]["classes"]
        assert pyroxene["V"] >= 0.5

    def test_nonzero_rankings_diverse(self, config):
        """At least 3 classes should have nonzero recoverability per material."""
        for mat in MATERIALS:
            classes = config["recoverability"][mat]["classes"]
            nonzero = sum(1 for v in classes.values() if v > 0)
            assert nonzero >= 3, (
                f"{mat} has only {nonzero} nonzero classes; ranking may be degenerate"
            )


# ---------------------------------------------------------------------------
# Accessibility function tests
# ---------------------------------------------------------------------------


class TestAccessibilityFunction:
    def test_dv_scale_in_range(self, config):
        """dv_scale should be 3-15 km/s (physically motivated)."""
        scale = config["accessibility"]["delta_v"]["dv_scale"]["value"]
        assert 3.0 <= scale <= 15.0

    def test_ero_high_accessibility(self, config):
        """ERO-class asteroids (<0.5 km/s) should score > 0.9."""
        scale = config["accessibility"]["delta_v"]["dv_scale"]["value"]
        score = math.exp(-0.5 / scale)
        assert score > 0.9

    def test_nhats_moderate_accessibility(self, config):
        """NHATS limit (12 km/s) should score > 0.1."""
        scale = config["accessibility"]["delta_v"]["dv_scale"]["value"]
        score = math.exp(-12.0 / scale)
        assert score > 0.1

    def test_very_high_dv_low_accessibility(self, config):
        """Very high delta-v (>20 km/s) should score < 0.1."""
        scale = config["accessibility"]["delta_v"]["dv_scale"]["value"]
        score = math.exp(-20.0 / scale)
        assert score < 0.1

    def test_moid_scale_matches_phase1(self, config):
        """MOID scale should match Phase 1 scorer default (0.1 AU)."""
        scale = config["accessibility"]["moid_proxy"]["moid_scale"]["value"]
        assert scale == 0.1

    def test_monotonically_decreasing(self, config):
        """Higher delta-v should give lower accessibility."""
        scale = config["accessibility"]["delta_v"]["dv_scale"]["value"]
        dvs = [1.0, 3.0, 5.0, 7.0, 10.0, 15.0]
        scores = [math.exp(-dv / scale) for dv in dvs]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]


# ---------------------------------------------------------------------------
# Cross-validation with verification examples
# ---------------------------------------------------------------------------


class TestVerificationExamples:
    def test_psyche_iron_high_recoverability(self, config):
        """16 Psyche (M-type): high iron recoverability."""
        iron = config["recoverability"]["iron"]["classes"]
        assert iron["M"] >= 0.7

    def test_bennu_water_moderate(self, config):
        """Bennu (B-type): moderate water recoverability."""
        water = config["recoverability"]["water"]["classes"]
        assert water["B"] >= 0.3

    def test_itokawa_olivine_moderate(self, config):
        """Itokawa (S-type): moderate olivine recoverability."""
        olivine = config["recoverability"]["olivine"]["classes"]
        assert olivine["S"] >= 0.3

    def test_apophis_better_accessibility_than_psyche(self, config):
        """Apophis (~3.5 km/s) should be more accessible than Psyche (~8.5 km/s)."""
        scale = config["accessibility"]["delta_v"]["dv_scale"]["value"]
        apophis_score = math.exp(-3.5 / scale)
        psyche_score = math.exp(-8.5 / scale)
        assert apophis_score > psyche_score
