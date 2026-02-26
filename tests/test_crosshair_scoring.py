"""CrossHair SMT-backed property tests for scoring and band-analysis functions.

These tests re-verify key properties from test_properties.py using the CrossHair
backend, which uses Z3 SMT solving instead of random sampling.  This finds
boundary-condition bugs that random testing misses.

Only pure-Python functions (no numpy/scipy internals) are tested here — CrossHair
cannot symbolically execute through C extensions.

Marked ``crosshair`` so CI can run them nightly rather than on every commit.
"""

import math

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from prospector.scoring.scorer import (
    compute_thermal_depletion_factor,
    estimate_mass_kg,
    compute_spin_modifier,
)
from prospector.spectral.band_analysis import (
    classify_gaffey_subtype,
    dunn_calibration,
    lindsay_bar_correction,
    temperature_correct_band_centers,
    DBIC_DT,
    DBIIC_DT,
    LAB_TEMPERATURE,
)

# All tests in this module use CrossHair SMT backend and nightly marker.
pytestmark = pytest.mark.crosshair

# CrossHair settings: low max_examples (SMT is exhaustive, not statistical),
# generous deadline, suppress too_slow health check.
_ch = settings(
    backend="crosshair",
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# 1. estimate_mass_kg — sphere formula invariants
# ---------------------------------------------------------------------------

class TestMassKgCrossHair:
    @given(
        d=st.floats(0.001, 500.0, allow_nan=False, allow_infinity=False),
        rho=st.floats(0.1, 8.0, allow_nan=False, allow_infinity=False),
    )
    @_ch
    def test_mass_non_negative(self, d: float, rho: float) -> None:
        assert estimate_mass_kg(d, rho) >= 0

    @given(
        d=st.floats(0.001, 200.0, allow_nan=False, allow_infinity=False),
        rho=st.floats(0.1, 8.0, allow_nan=False, allow_infinity=False),
    )
    @_ch
    def test_mass_scales_with_diameter_cubed(self, d: float, rho: float) -> None:
        """Doubling diameter → 8× mass (volume ~ d³)."""
        m1 = estimate_mass_kg(d, rho)
        m2 = estimate_mass_kg(2.0 * d, rho)
        if m1 > 0:
            ratio = m2 / m1
            assert abs(ratio - 8.0) < 1e-6


# ---------------------------------------------------------------------------
# 2. compute_thermal_depletion_factor — monotonicity + bounds
# ---------------------------------------------------------------------------

class TestThermalDepletionCrossHair:
    @given(q=st.floats(0.05, 5.0, allow_nan=False, allow_infinity=False))
    @_ch
    def test_always_in_unit_interval(self, q: float) -> None:
        factor = compute_thermal_depletion_factor(q)
        assert 0.0 <= factor <= 1.0

    @given(
        q1=st.floats(0.05, 3.0, allow_nan=False, allow_infinity=False),
        q2=st.floats(0.05, 3.0, allow_nan=False, allow_infinity=False),
    )
    @_ch
    def test_monotonically_increasing(self, q1: float, q2: float) -> None:
        """Farther perihelion → equal or more water retained."""
        assume(q1 < q2)
        f1 = compute_thermal_depletion_factor(q1)
        f2 = compute_thermal_depletion_factor(q2)
        assert f1 <= f2 + 1e-10


# ---------------------------------------------------------------------------
# 3. classify_gaffey_subtype — exhaustive zone coverage
# ---------------------------------------------------------------------------

class TestGaffeyCrossHair:
    @given(
        bic=st.floats(0.85, 1.10, allow_nan=False, allow_infinity=False),
        bar=st.floats(0.0, 3.0, allow_nan=False, allow_infinity=False),
    )
    @_ch
    def test_always_returns_valid_subtype(self, bic: float, bar: float) -> None:
        subtype = classify_gaffey_subtype(bic, bar)
        assert subtype.startswith("S(")
        assert subtype.endswith(")")
        roman = subtype[2:-1]
        assert roman in ("I", "II", "III", "IV", "V", "VI", "VII")


# ---------------------------------------------------------------------------
# 4. dunn_calibration — mineral chemistry bounds
# ---------------------------------------------------------------------------

class TestDunnCalibrationCrossHair:
    @given(
        bar=st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False),
        bic=st.floats(0.85, 1.10, allow_nan=False, allow_infinity=False),
        biic=st.floats(1.70, 2.30, allow_nan=False, allow_infinity=False),
    )
    @_ch
    def test_ol_ratio_bounded(self, bar: float, bic: float, biic: float) -> None:
        result = dunn_calibration(bar, bic, biic)
        assert 0.0 <= result["ol_opx_ratio"] <= 1.0

    @given(
        bar=st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False),
        bic=st.floats(0.85, 1.10, allow_nan=False, allow_infinity=False),
    )
    @_ch
    def test_fa_non_negative(self, bar: float, bic: float) -> None:
        result = dunn_calibration(bar, bic, None)
        assert result["fa_mol_pct"] >= 0
        assert result["fs_mol_pct"] is None


# ---------------------------------------------------------------------------
# 5. lindsay_bar_correction — sign preservation
# ---------------------------------------------------------------------------

class TestLindsayCorrectionCrossHair:
    @given(
        bar=st.floats(0.01, 5.0, allow_nan=False, allow_infinity=False),
        red_edge=st.floats(2.30, 2.55, allow_nan=False, allow_infinity=False),
    )
    @_ch
    def test_correction_preserves_sign(self, bar: float, red_edge: float) -> None:
        corrected = lindsay_bar_correction(bar, red_edge)
        assert corrected > 0


# ---------------------------------------------------------------------------
# 6. temperature_correct_band_centers — round-trip
# ---------------------------------------------------------------------------

class TestTempCorrectionCrossHair:
    @given(
        bic=st.floats(0.85, 1.10, allow_nan=False, allow_infinity=False),
        biic=st.floats(1.70, 2.30, allow_nan=False, allow_infinity=False),
        temp=st.floats(100.0, 500.0, allow_nan=False, allow_infinity=False),
    )
    @_ch
    def test_round_trip(self, bic: float, biic: float, temp: float) -> None:
        """Correct to lab temp, then un-correct back — should recover originals."""
        bic_corr, biic_corr = temperature_correct_band_centers(bic, biic, temp)
        # Reverse: apply correction with lab temperature as "observed" temp
        bic_back, biic_back = temperature_correct_band_centers(
            bic_corr, biic_corr, 2.0 * LAB_TEMPERATURE - temp
        )
        assert abs(bic_back - bic) < 1e-10
        assert biic_back is not None
        assert abs(biic_back - biic) < 1e-10


# ---------------------------------------------------------------------------
# 7. compute_spin_modifier — exhaustive boolean coverage
# ---------------------------------------------------------------------------

class TestSpinModifierCrossHair:
    def test_monolithic_bonus(self) -> None:
        assert compute_spin_modifier(is_monolithic=True) > 1.0

    def test_binary_penalty(self) -> None:
        assert compute_spin_modifier(is_binary_suspect=True) < 1.0

    def test_no_data_neutral(self) -> None:
        assert compute_spin_modifier() == 1.0

    def test_monolithic_overrides_binary(self) -> None:
        """Monolithic takes priority when both flags are set."""
        assert compute_spin_modifier(is_monolithic=True, is_binary_suspect=True) > 1.0
