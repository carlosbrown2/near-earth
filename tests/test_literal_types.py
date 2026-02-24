"""Tests for Literal type constraints on mode and observation strings."""

import numpy as np
import pytest
from pydantic import ValidationError

from prospector.schemas import (
    EVOIResultSchema,
    ObservationType,
    ScoringMode,
    ScoringResult,
)


class TestScoringModeLiteral:
    """Verify ScoringMode Literal rejects invalid strings at Pydantic validation."""

    def test_valid_earth_return(self):
        result = ScoringResult(
            composite_score=1.0,
            estimated_mass_kg=1e10,
            grade_estimate=0.01,
            target_material="pgm",
            unit_value=50000,
            accessibility=0.5,
            confidence=0.5,
            spin_modifier=1.0,
            thermal_depletion_factor=1.0,
            jwst_water_boost=1.0,
            score_mode="earth_return",
            material_contributions={"pgm": 1.0},
        )
        assert result.score_mode == "earth_return"

    def test_valid_in_space(self):
        result = ScoringResult(
            composite_score=1.0,
            estimated_mass_kg=1e10,
            grade_estimate=0.01,
            target_material="water",
            unit_value=500,
            accessibility=0.5,
            confidence=0.5,
            spin_modifier=1.0,
            thermal_depletion_factor=1.0,
            jwst_water_boost=1.0,
            score_mode="in_space",
            material_contributions={"water": 1.0},
        )
        assert result.score_mode == "in_space"

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValidationError, match="score_mode"):
            ScoringResult(
                composite_score=1.0,
                estimated_mass_kg=1e10,
                grade_estimate=0.01,
                target_material="pgm",
                unit_value=50000,
                accessibility=0.5,
                confidence=0.5,
                spin_modifier=1.0,
                thermal_depletion_factor=1.0,
                jwst_water_boost=1.0,
                score_mode="lunar_surface",  # type: ignore[arg-type]
                material_contributions={"pgm": 1.0},
            )


class TestObservationTypeLiteral:
    """Verify ObservationType Literal rejects invalid strings at Pydantic validation."""

    def test_valid_observation_types(self):
        # best_evoi must equal the max of evoi values (validator checks this)
        for obs_type in ("vnir_spectroscopy", "vis_spectroscopy", "radar", "albedo"):
            result = EVOIResultSchema(
                score_mean=100.0,
                score_std=50.0,
                evoi_vnir=20.0,
                evoi_vis=10.0,
                evoi_radar=5.0,
                evoi_albedo=8.0,
                best_observation=obs_type,
                best_evoi=20.0,  # must match max(evoi_*) per model validator
            )
            assert result.best_observation == obs_type

    def test_invalid_observation_rejected(self):
        with pytest.raises(ValidationError, match="best_observation"):
            EVOIResultSchema(
                score_mean=100.0,
                score_std=50.0,
                evoi_vnir=20.0,
                evoi_vis=10.0,
                evoi_radar=5.0,
                evoi_albedo=8.0,
                best_observation="infrared_photometry",  # type: ignore[arg-type]
                best_evoi=20.0,
            )


class TestScoreAsteroidLiteralIntegration:
    """Verify score_asteroid output passes Literal-typed Pydantic validation."""

    def test_output_validated_with_literal_mode(self):
        from prospector.scoring.scorer import score_asteroid, load_config

        config = load_config()
        rng = np.random.default_rng(42)
        for mode in ("earth_return", "in_space"):
            result = score_asteroid(
                diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
                config=config, n_samples=50, rng=rng, mode=mode,
            )
            assert result["score_mode"] == mode
            # Pydantic validation happens inside score_asteroid — if it
            # passed, the Literal constraint was satisfied.


class TestComputeEVOILiteralIntegration:
    """Verify compute_evoi output passes Literal-typed Pydantic validation."""

    def test_output_validated_with_literal_observation(self):
        from prospector.scoring.evoi import compute_evoi

        prior = np.ones(17) / 17.0
        rng = np.random.default_rng(42)
        result = compute_evoi(
            1.0, 1.3, 0.2, 10.0, moid=0.05,
            prior=prior, n_samples=50, rng=rng,
        )
        assert result["best_observation"] in {
            "vnir_spectroscopy", "vis_spectroscopy", "radar", "albedo"
        }
