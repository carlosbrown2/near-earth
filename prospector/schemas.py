"""Pydantic schema contracts for pipeline stage boundaries.

Validates shapes, types, and physical ranges at every handoff point.
If agent-generated code produces a result that violates these contracts,
it fails fast with a clear error — stronger backpressure than unit tests.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ScoringMode = Literal["earth_return", "in_space"]
ObservationType = Literal["vnir_spectroscopy", "vis_spectroscopy", "radar", "albedo"]


class Distribution(BaseModel):
    """A bounded normal distribution for Monte Carlo sampling."""

    mean: float
    std: float = Field(ge=0)
    min: float
    max: float

    @model_validator(mode="after")
    def bounds_consistent(self) -> "Distribution":
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) > max ({self.max})")
        if not (self.min <= self.mean <= self.max):
            raise ValueError(
                f"mean ({self.mean}) outside [{self.min}, {self.max}]"
            )
        return self


class ScoringResult(BaseModel):
    """Contract for score_asteroid() output."""

    composite_score: float = Field(ge=0)
    estimated_mass_kg: float = Field(ge=0)
    grade_estimate: float = Field(ge=0)
    target_material: str
    unit_value: float = Field(ge=0)
    accessibility: float = Field(ge=0.01, le=1.0)
    confidence: float = Field(ge=0.01, le=1.0)
    spin_modifier: float = Field(ge=0.5, le=1.5)
    thermal_depletion_factor: float = Field(ge=0, le=1.0)
    jwst_water_boost: float = Field(default=1.0, ge=1.0)
    score_mode: ScoringMode
    material_contributions: dict[str, float]

    @model_validator(mode="after")
    def contributions_non_negative(self) -> "ScoringResult":
        for mat, val in self.material_contributions.items():
            if val < 0:
                raise ValueError(
                    f"material_contributions[{mat!r}] = {val} < 0"
                )
        return self


class BandAnalysisResult(BaseModel):
    """Contract for analyze_spectrum() output."""

    band1_center: float = Field(ge=0.85, le=1.10)
    band2_center: float | None = Field(default=None, ge=1.70, le=2.30)
    bar: float | None = Field(default=None, ge=0)
    ol_opx_ratio: float | None = Field(default=None, ge=0, le=1.0)
    fa_mol_pct: float | None = Field(default=None, ge=0)
    fs_mol_pct: float | None = Field(default=None, ge=0)
    gaffey_subtype: str | None = None
    calibration: str | None = None

    @model_validator(mode="after")
    def dunn_requires_s4(self) -> "BandAnalysisResult":
        if self.calibration == "dunn2010" and self.gaffey_subtype != "S(IV)":
            raise ValueError(
                f"Dunn calibration only valid for S(IV), got {self.gaffey_subtype}"
            )
        return self


class EVOIResultSchema(BaseModel):
    """Contract for EVOI computation output."""

    score_mean: float = Field(ge=0)
    score_std: float = Field(ge=0)
    evoi_vnir: float = Field(ge=0)
    evoi_vis: float = Field(ge=0)
    evoi_radar: float = Field(ge=0)
    evoi_albedo: float = Field(ge=0)
    best_observation: ObservationType
    best_evoi: float = Field(ge=0)

    @model_validator(mode="after")
    def best_matches_max(self) -> "EVOIResultSchema":
        evoi_map = {
            "vnir_spectroscopy": self.evoi_vnir,
            "vis_spectroscopy": self.evoi_vis,
            "radar": self.evoi_radar,
            "albedo": self.evoi_albedo,
        }
        actual_max = max(evoi_map.values())
        if abs(self.best_evoi - actual_max) > 1e-10:
            raise ValueError(
                f"best_evoi ({self.best_evoi}) != max of evoi values ({actual_max})"
            )
        return self
