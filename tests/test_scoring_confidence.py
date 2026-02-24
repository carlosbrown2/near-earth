"""Tests for confidence scoring configuration (scoring/confidence_config.yaml).

Validates structure, weight sums, normalization bounds, concordance maps,
and worked examples for the 8-factor confidence formula.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "scoring" / "confidence_config.yaml"


@pytest.fixture(scope="module")
def config():
    """Load confidence config."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

class TestConfigStructure:

    def test_file_exists(self):
        assert CONFIG_PATH.exists(), "scoring/confidence_config.yaml must exist"

    def test_has_metadata(self, config):
        assert "metadata" in config
        assert "formula_type" in config["metadata"]
        assert config["metadata"]["formula_type"] == "weighted_geometric_mean"

    def test_has_floor(self, config):
        floor = config["metadata"]["floor"]
        assert 0 < floor < 0.5, f"Floor {floor} should be small positive"

    def test_has_factors(self, config):
        assert "factors" in config
        assert len(config["factors"]) == 8, "Should have exactly 8 factors"

    def test_all_factors_present(self, config):
        expected = {
            "taxonomy_entropy",
            "wavelength_coverage",
            "snr",
            "n_observations",
            "taxonomy_analog_concordance",
            "method_agreement",
            "supplementary_data",
            "pgm_convergence",
        }
        actual = set(config["factors"].keys())
        assert actual == expected

    def test_each_factor_has_required_keys(self, config):
        required = {"weight", "normalization", "description"}
        for name, factor in config["factors"].items():
            for key in required:
                assert key in factor, f"Factor '{name}' missing key '{key}'"

    def test_has_implementation_spec(self, config):
        assert "implementation" in config
        assert "function_signature" in config["implementation"]
        assert "input_sources" in config["implementation"]

    def test_has_worked_examples(self, config):
        assert "worked_examples" in config
        assert len(config["worked_examples"]) >= 5


# ---------------------------------------------------------------------------
# Weight tests
# ---------------------------------------------------------------------------

class TestWeights:

    def test_weights_sum_to_one(self, config):
        total = sum(f["weight"] for f in config["factors"].values())
        assert total == pytest.approx(1.0, abs=0.001), (
            f"Weights sum to {total}, should be 1.0"
        )

    def test_all_weights_positive(self, config):
        for name, factor in config["factors"].items():
            assert factor["weight"] > 0, f"Weight for '{name}' must be positive"

    def test_wavelength_coverage_is_largest(self, config):
        """Wavelength coverage should be the most heavily weighted factor."""
        wl_weight = config["factors"]["wavelength_coverage"]["weight"]
        for name, factor in config["factors"].items():
            if name != "wavelength_coverage":
                assert wl_weight >= factor["weight"], (
                    f"Wavelength coverage weight ({wl_weight}) should be >= "
                    f"{name} weight ({factor['weight']})"
                )

    def test_taxonomy_entropy_is_second(self, config):
        """Taxonomy entropy should be the second most weighted."""
        weights = {n: f["weight"] for n, f in config["factors"].items()}
        sorted_weights = sorted(weights.items(), key=lambda x: -x[1])
        assert sorted_weights[1][0] == "taxonomy_entropy"


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestNormalization:

    def test_taxonomy_entropy_h_max(self, config):
        """H_max should be log2(17)."""
        h_max = config["factors"]["taxonomy_entropy"]["H_max"]
        assert h_max == pytest.approx(np.log2(17), abs=0.01)

    def test_wavelength_coverage_lookup_values(self, config):
        lookup = config["factors"]["wavelength_coverage"]["lookup"]
        # Best case = 1.0
        assert lookup["vnir_albedo"] == 1.0
        # Worst case = floor
        assert lookup["none"] <= 0.1
        # NIR is worth more than visible-only
        assert lookup["nir_only"] > lookup["vis_only"]
        # Albedo improves every coverage level
        assert lookup["vnir_albedo"] > lookup["vnir"]
        assert lookup["nir_albedo"] > lookup["nir_only"]
        assert lookup["vis_albedo"] > lookup["vis_only"]

    def test_snr_good_is_reasonable(self, config):
        """SNR_good should be >= quality threshold (25) from preprocessing."""
        snr_good = config["factors"]["snr"]["snr_good"]
        assert snr_good >= 25, "SNR_good should be >= preprocessing threshold"
        assert snr_good <= 200, "SNR_good should not be unreasonably high"

    def test_n_observations_saturate(self, config):
        n_sat = config["factors"]["n_observations"]["n_saturate"]
        assert 2 <= n_sat <= 10, f"n_saturate={n_sat} should be reasonable"

    def test_method_agreement_max_disagreement(self, config):
        max_dis = config["factors"]["method_agreement"]["max_disagreement"]
        assert 0.2 <= max_dis <= 1.0, (
            f"max_disagreement={max_dis} should be in [0.2, 1.0]"
        )

    def test_pgm_max_signals(self, config):
        max_sig = config["factors"]["pgm_convergence"]["max_signals"]
        assert max_sig >= 4, "Should allow at least 4 PGM signals"


# ---------------------------------------------------------------------------
# Concordance map tests
# ---------------------------------------------------------------------------

class TestConcordanceMap:

    def test_all_mahlke_classes_covered(self, config):
        """Concordance map should cover all 17 Mahlke classes."""
        mahlke = ["A", "B", "C", "Ch", "D", "E", "K", "L", "M",
                   "O", "P", "Q", "R", "S", "V", "X", "Z"]
        conc_map = config["factors"]["taxonomy_analog_concordance"]["concordance_map"]
        for cls in mahlke:
            assert cls in conc_map, f"Class '{cls}' missing from concordance map"

    def test_s_type_includes_oc(self, config):
        conc = config["factors"]["taxonomy_analog_concordance"]["concordance_map"]
        assert "Ordinary Chondrite" in conc["S"]

    def test_m_type_includes_iron(self, config):
        conc = config["factors"]["taxonomy_analog_concordance"]["concordance_map"]
        assert "Iron" in conc["M"]

    def test_c_type_includes_cc(self, config):
        conc = config["factors"]["taxonomy_analog_concordance"]["concordance_map"]
        assert "Carbonaceous Chondrite" in conc["C"]

    def test_concordance_scores(self, config):
        """Concordance scoring should follow concordant > partial > discordant."""
        scores = config["factors"]["taxonomy_analog_concordance"]["scores"]
        assert scores["concordant"] > scores["partial"]
        assert scores["partial"] > scores["discordant"]
        assert scores["discordant"] > 0

    def test_z_class_accepts_anything(self, config):
        conc = config["factors"]["taxonomy_analog_concordance"]["concordance_map"]
        assert conc["Z"] == [], "Z class should accept any match (empty list)"


# ---------------------------------------------------------------------------
# Supplementary data tests
# ---------------------------------------------------------------------------

class TestSupplementaryData:

    def test_radar_is_highest_weighted(self, config):
        """Radar albedo should be the most informative supplementary signal."""
        signals = config["factors"]["supplementary_data"]["signals"]
        radar_weight = signals["radar_albedo"]["weight"]
        for name, sig in signals.items():
            if name != "radar_albedo":
                assert radar_weight >= sig["weight"], (
                    f"Radar weight ({radar_weight}) should be >= "
                    f"{name} weight ({sig['weight']})"
                )

    def test_all_signal_weights_positive(self, config):
        signals = config["factors"]["supplementary_data"]["signals"]
        for name, sig in signals.items():
            assert sig["weight"] > 0, f"Signal '{name}' weight must be positive"


# ---------------------------------------------------------------------------
# Worked example tests
# ---------------------------------------------------------------------------

class TestWorkedExamples:

    def _compute_confidence(self, config, factor_values):
        """Compute confidence using the formula from config."""
        floor = config["metadata"]["floor"]
        factors = config["factors"]
        confidence = 1.0
        for name, factor in factors.items():
            weight = factor["weight"]
            value = factor_values.get(name, factor.get("default_when_missing", 0.5))
            bounded = max(floor, min(1.0, value))
            confidence *= bounded ** weight
        return confidence

    def test_itokawa_highest(self, config):
        """Itokawa (sample return) should score highest."""
        examples = config["worked_examples"]
        itokawa = self._compute_confidence(
            config, examples["itokawa"]["factor_values"]
        )
        for name, ex in examples.items():
            if name != "itokawa":
                other = self._compute_confidence(config, ex["factor_values"])
                assert itokawa >= other * 0.95, (
                    f"Itokawa ({itokawa:.3f}) should score >= {name} ({other:.3f})"
                )

    def test_sample_return_beats_gaia_only(self, config):
        """Sample-return targets should substantially outrank Gaia-only."""
        examples = config["worked_examples"]
        itokawa = self._compute_confidence(
            config, examples["itokawa"]["factor_values"]
        )
        gaia = self._compute_confidence(
            config, examples["gaia_only_mba"]["factor_values"]
        )
        assert itokawa > gaia * 1.5, (
            f"Itokawa ({itokawa:.3f}) should be >1.5x Gaia-only ({gaia:.3f})"
        )

    def test_uncharacterized_scores_lowest(self, config):
        """Uncharacterized NEO should score lowest."""
        examples = config["worked_examples"]
        unchar = self._compute_confidence(
            config, examples["uncharacterized_neo"]["factor_values"]
        )
        for name, ex in examples.items():
            if name != "uncharacterized_neo":
                other = self._compute_confidence(config, ex["factor_values"])
                assert other > unchar, (
                    f"{name} ({other:.3f}) should outrank "
                    f"uncharacterized ({unchar:.3f})"
                )

    def test_ordering_is_sensible(self, config):
        """Full ordering: Itokawa >= Bennu > Eros > Psyche > Gaia > Unchar."""
        examples = config["worked_examples"]
        scores = {}
        for name, ex in examples.items():
            scores[name] = self._compute_confidence(
                config, ex["factor_values"]
            )

        assert scores["itokawa"] >= scores["bennu"] * 0.95
        assert scores["bennu"] > scores["eros"] * 0.95
        assert scores["eros"] > scores["psyche"] * 0.95
        assert scores["psyche"] > scores["gaia_only_mba"]
        assert scores["gaia_only_mba"] > scores["uncharacterized_neo"]

    def test_all_confidences_in_bounds(self, config):
        """All computed confidences should be in [0, 1]."""
        examples = config["worked_examples"]
        for name, ex in examples.items():
            conf = self._compute_confidence(config, ex["factor_values"])
            assert 0.0 <= conf <= 1.0, (
                f"{name} confidence {conf:.3f} out of bounds"
            )

    def test_expected_confidence_matches(self, config):
        """Computed confidence should approximately match expected values."""
        examples = config["worked_examples"]
        for name, ex in examples.items():
            computed = self._compute_confidence(config, ex["factor_values"])
            expected = ex["expected_confidence"]
            assert computed == pytest.approx(expected, abs=0.15), (
                f"{name}: computed {computed:.3f} != expected {expected:.3f}"
            )


# ---------------------------------------------------------------------------
# Formula computation tests
# ---------------------------------------------------------------------------

class TestFormulaProperties:

    def test_perfect_score(self, config):
        """All factors at 1.0 should give confidence = 1.0."""
        floor = config["metadata"]["floor"]
        factors = config["factors"]
        all_ones = {name: 1.0 for name in factors}
        confidence = 1.0
        for name, factor in factors.items():
            confidence *= max(floor, all_ones[name]) ** factor["weight"]
        assert confidence == pytest.approx(1.0, abs=0.001)

    def test_floor_prevents_zero(self, config):
        """All factors at 0.0 should give confidence = floor^1.0, not 0."""
        floor = config["metadata"]["floor"]
        factors = config["factors"]
        all_zeros = {name: 0.0 for name in factors}
        confidence = 1.0
        for name, factor in factors.items():
            value = max(floor, all_zeros[name])
            confidence *= value ** factor["weight"]
        assert confidence > 0, "Floor should prevent zero confidence"
        assert confidence == pytest.approx(floor, abs=0.01)

    def test_single_poor_factor_penalizes(self, config):
        """One factor at floor with rest at 1.0 should reduce score."""
        floor = config["metadata"]["floor"]
        factors = config["factors"]
        # All perfect except wavelength_coverage at floor
        values = {name: 1.0 for name in factors}
        values["wavelength_coverage"] = floor
        confidence = 1.0
        for name, factor in factors.items():
            confidence *= max(floor, values[name]) ** factor["weight"]
        # With weight=0.25, floor^0.25 = 0.05^0.25 ≈ 0.47
        assert confidence < 0.60, (
            f"Poor wavelength coverage should penalize score to < 0.60, got {confidence}"
        )

    def test_monotonicity(self, config):
        """Improving any factor should not decrease confidence."""
        floor = config["metadata"]["floor"]
        factors = config["factors"]
        base = {name: 0.5 for name in factors}

        def compute(vals):
            conf = 1.0
            for n, f in factors.items():
                conf *= max(floor, vals[n]) ** f["weight"]
            return conf

        base_conf = compute(base)
        for name in factors:
            improved = dict(base)
            improved[name] = 0.8
            assert compute(improved) >= base_conf, (
                f"Improving '{name}' should not decrease confidence"
            )


# ---------------------------------------------------------------------------
# Default value tests
# ---------------------------------------------------------------------------

class TestDefaults:

    def test_all_factors_have_defaults(self, config):
        """Each factor should specify a default for missing data."""
        for name, factor in config["factors"].items():
            assert "default_when_missing" in factor, (
                f"Factor '{name}' needs default_when_missing"
            )

    def test_defaults_are_bounded(self, config):
        floor = config["metadata"]["floor"]
        for name, factor in config["factors"].items():
            default = factor["default_when_missing"]
            if default is not None:
                assert floor <= default <= 1.0, (
                    f"Default for '{name}' ({default}) should be in [{floor}, 1.0]"
                )
