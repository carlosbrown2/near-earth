"""Pytest integration for fuzz_preprocessing harness.

Runs each fuzz target with a corpus of known-tricky inputs AND random
payloads to verify they handle edge cases without crashing. This tests
the harness portably (no Atheris required).

For full coverage-guided fuzzing, run the harness directly:
    python tests/fuzz_preprocessing.py -max_total_time=60
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tests.fuzz_preprocessing import (
    fuzz_combined,
    fuzz_estimate_snr,
    fuzz_mask_telluric,
    fuzz_normalize,
    fuzz_resample,
)


def _pack_floats(*vals: float) -> bytes:
    return struct.pack(f"<{len(vals)}d", *vals)


# ---------------------------------------------------------------------------
# Seed corpus: hand-crafted edge-case payloads
# ---------------------------------------------------------------------------


class TestFuzzResampleCorpus:
    """Seed corpus for resample() fuzz target."""

    def test_empty(self) -> None:
        fuzz_resample(b"")

    def test_single_byte_sweep(self) -> None:
        for b in range(256):
            fuzz_resample(bytes([b]))

    def test_zeros(self) -> None:
        fuzz_resample(b"\x00" * 500)

    def test_ones(self) -> None:
        fuzz_resample(b"\xff" * 500)

    def test_nan_heavy(self) -> None:
        payload = _pack_floats(float("nan"), float("nan"), 0.5, 1.0, float("nan"))
        fuzz_resample(payload)

    def test_inf_values(self) -> None:
        payload = _pack_floats(float("inf"), float("-inf"), 1e308, -1e308)
        fuzz_resample(payload)

    def test_subnormal_floats(self) -> None:
        # Smallest positive subnormal: 5e-324
        payload = _pack_floats(5e-324, 5e-324, 1e-300, 1e-300)
        fuzz_resample(payload)


class TestFuzzNormalizeCorpus:
    """Seed corpus for normalize() fuzz target."""

    def test_empty(self) -> None:
        fuzz_normalize(b"")

    def test_all_zero(self) -> None:
        fuzz_normalize(_pack_floats(0.0, 0.0, 0.0, 0.0))

    def test_all_nan(self) -> None:
        fuzz_normalize(_pack_floats(*([float("nan")] * 10)))

    def test_negative_ref_wl(self) -> None:
        # Payload with negative reference wavelength
        fuzz_normalize(b"\xff" * 200)

    def test_zeros_and_nans(self) -> None:
        fuzz_normalize(_pack_floats(0.0, float("nan"), 0.0, float("nan")))


class TestFuzzMaskTelluricCorpus:
    """Seed corpus for mask_telluric() fuzz target."""

    def test_empty(self) -> None:
        fuzz_mask_telluric(b"")

    def test_all_ones(self) -> None:
        fuzz_mask_telluric(b"\xff" * 200)

    def test_all_zeros(self) -> None:
        fuzz_mask_telluric(b"\x00" * 200)


class TestFuzzEstimateSnrCorpus:
    """Seed corpus for estimate_snr() fuzz target."""

    def test_empty(self) -> None:
        fuzz_estimate_snr(b"")

    def test_all_nan(self) -> None:
        fuzz_estimate_snr(_pack_floats(*([float("nan")] * 20)))

    def test_all_zero(self) -> None:
        fuzz_estimate_snr(_pack_floats(*([0.0] * 20)))

    def test_single_value(self) -> None:
        fuzz_estimate_snr(_pack_floats(1.0))

    def test_inf_uncertainty(self) -> None:
        fuzz_estimate_snr(_pack_floats(float("inf"), float("-inf"), 1.0, 0.0))


# ---------------------------------------------------------------------------
# Random fuzzing: high-volume random payloads via numpy RNG
# ---------------------------------------------------------------------------


class TestFuzzRandom:
    """Random payloads — the pytest equivalent of a short fuzz run."""

    @pytest.mark.parametrize("seed", range(20))
    def test_resample_random(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        fuzz_resample(rng.bytes(rng.integers(0, 4096)))

    @pytest.mark.parametrize("seed", range(20))
    def test_normalize_random(self, seed: int) -> None:
        rng = np.random.default_rng(1000 + seed)
        fuzz_normalize(rng.bytes(rng.integers(0, 4096)))

    @pytest.mark.parametrize("seed", range(20))
    def test_mask_telluric_random(self, seed: int) -> None:
        rng = np.random.default_rng(2000 + seed)
        fuzz_mask_telluric(rng.bytes(rng.integers(0, 4096)))

    @pytest.mark.parametrize("seed", range(20))
    def test_estimate_snr_random(self, seed: int) -> None:
        rng = np.random.default_rng(3000 + seed)
        fuzz_estimate_snr(rng.bytes(rng.integers(0, 4096)))

    @pytest.mark.parametrize("seed", range(20))
    def test_combined_random(self, seed: int) -> None:
        rng = np.random.default_rng(4000 + seed)
        fuzz_combined(rng.bytes(rng.integers(2, 4096)))


class TestFuzzBulk:
    """Bulk random fuzzing — 10k iterations, ~10s equivalent."""

    def test_bulk_random_10k(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10_000):
            size = int(rng.integers(0, 2048))
            data = rng.bytes(size)
            fuzz_combined(data)
