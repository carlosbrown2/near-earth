"""Coverage-guided fuzz testing for spectral preprocessing functions.

Uses Atheris (Google's Python fuzzing engine) when available, with a
pure-Python random fallback for environments where Atheris can't build
(macOS ARM, missing libFuzzer, etc.).

Run with Atheris (if installed):
    python tests/fuzz_preprocessing.py                    # 60s default
    python tests/fuzz_preprocessing.py -max_total_time=120

Run with fallback (no Atheris):
    python tests/fuzz_preprocessing.py                    # 60s random fuzzing

The harness focuses on crash detection only — NaN/Inf in outputs is
expected and acceptable (resample intentionally produces NaN outside range).
"""

from __future__ import annotations

import os
import struct
import sys
import time

# Ensure project root is on sys.path when running as a script
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import deal
import numpy as np

from prospector.spectral.preprocessing import (
    estimate_snr,
    mask_telluric,
    normalize,
    resample,
)

try:
    import atheris  # type: ignore[import-not-found]

    HAS_ATHERIS = True
except ImportError:
    HAS_ATHERIS = False


class FuzzedDataProvider:
    """Minimal pure-Python reimplementation of atheris.FuzzedDataProvider.

    Consumes bytes sequentially, exactly like the Atheris version, so that
    the same harness code works with or without the C extension.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def remaining_bytes(self) -> int:
        return max(0, len(self._data) - self._pos)

    def ConsumeBytes(self, n: int) -> bytes:
        chunk = self._data[self._pos : self._pos + n]
        self._pos += n
        return chunk

    def ConsumeIntInRange(self, lo: int, hi: int) -> int:
        if lo == hi:
            return lo
        raw = self.ConsumeBytes(4)
        if len(raw) < 4:
            return lo
        val = int.from_bytes(raw, "little", signed=False)
        return lo + val % (hi - lo + 1)

    def ConsumeBool(self) -> bool:
        raw = self.ConsumeBytes(1)
        return bool(raw[0] & 1) if raw else False

    def ConsumeFloat(self) -> float:
        raw = self.ConsumeBytes(8)
        if len(raw) < 8:
            return 0.0
        return float(struct.unpack("<d", raw)[0])


def _get_fdp(data: bytes) -> FuzzedDataProvider:
    """Return an atheris or fallback FuzzedDataProvider."""
    if HAS_ATHERIS:
        return atheris.FuzzedDataProvider(data)  # type: ignore[no-any-return]
    return FuzzedDataProvider(data)


def _consume_float_array(fdp: FuzzedDataProvider, max_len: int = 200) -> np.ndarray:  # type: ignore[type-arg]
    """Build a numpy float64 array from fuzz data."""
    n = fdp.ConsumeIntInRange(0, max_len)
    raw = fdp.ConsumeBytes(n * 8)
    if len(raw) < 8:
        return np.array([], dtype=np.float64)
    count = len(raw) // 8
    values = struct.unpack(f"<{count}d", raw[: count * 8])
    return np.array(values, dtype=np.float64)


def _consume_sorted_positive_array(
    fdp: FuzzedDataProvider, max_len: int = 200
) -> np.ndarray:  # type: ignore[type-arg]
    """Build a sorted, finite, positive float64 array (mimics wavelengths)."""
    arr = _consume_float_array(fdp, max_len)
    if len(arr) == 0:
        return arr
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.array([], dtype=np.float64)
    arr = np.abs(arr) + 1e-10  # ensure positive
    arr = np.sort(arr)
    return arr


def fuzz_resample(data: bytes) -> None:
    fdp = _get_fdp(data)

    wavelengths = _consume_sorted_positive_array(fdp)
    n = len(wavelengths)
    if n < 2:
        return  # precondition: need >= 2 points

    reflectance = _consume_float_array(fdp, max_len=n)
    if len(reflectance) != n:
        reflectance = np.resize(reflectance, n) if len(reflectance) > 0 else np.zeros(n)

    has_unc = fdp.ConsumeBool()
    uncertainty = None
    if has_unc:
        uncertainty = _consume_float_array(fdp, max_len=n)
        if len(uncertainty) != n:
            uncertainty = (
                np.resize(uncertainty, n) if len(uncertainty) > 0 else np.zeros(n)
            )

    use_custom_grid = fdp.ConsumeBool()
    grid = None
    if use_custom_grid:
        grid = _consume_sorted_positive_array(fdp, max_len=100)
        if len(grid) == 0:
            grid = None

    try:
        resample(wavelengths, reflectance, uncertainty, grid)
    except (ValueError, TypeError, deal.PreContractError):
        pass


def fuzz_normalize(data: bytes) -> None:
    fdp = _get_fdp(data)

    wavelengths = _consume_float_array(fdp)
    n = len(wavelengths)
    if n == 0:
        return

    reflectance = _consume_float_array(fdp, max_len=n)
    if len(reflectance) != n:
        reflectance = np.resize(reflectance, n) if len(reflectance) > 0 else np.zeros(n)

    has_unc = fdp.ConsumeBool()
    uncertainty = None
    if has_unc:
        uncertainty = _consume_float_array(fdp, max_len=n)
        if len(uncertainty) != n:
            uncertainty = (
                np.resize(uncertainty, n) if len(uncertainty) > 0 else np.zeros(n)
            )

    ref_wl = fdp.ConsumeFloat()

    try:
        normalize(wavelengths, reflectance, uncertainty, ref_wl=ref_wl)
    except (ValueError, TypeError, deal.PreContractError):
        pass


def fuzz_mask_telluric(data: bytes) -> None:
    fdp = _get_fdp(data)

    wavelengths = _consume_float_array(fdp)
    n = len(wavelengths)
    if n == 0:
        return

    reflectance = _consume_float_array(fdp, max_len=n)
    if len(reflectance) != n:
        reflectance = np.resize(reflectance, n) if len(reflectance) > 0 else np.zeros(n)

    has_unc = fdp.ConsumeBool()
    uncertainty = None
    if has_unc:
        uncertainty = _consume_float_array(fdp, max_len=n)
        if len(uncertainty) != n:
            uncertainty = (
                np.resize(uncertainty, n) if len(uncertainty) > 0 else np.zeros(n)
            )

    n_regions = fdp.ConsumeIntInRange(0, 5)
    regions: list[tuple[float, float]] = []
    for _ in range(n_regions):
        lo = fdp.ConsumeFloat()
        hi = fdp.ConsumeFloat()
        regions.append((lo, hi))

    try:
        mask_telluric(
            wavelengths, reflectance, uncertainty, regions=regions if regions else None
        )
    except (ValueError, TypeError):
        pass


def fuzz_estimate_snr(data: bytes) -> None:
    fdp = _get_fdp(data)

    reflectance = _consume_float_array(fdp)

    has_unc = fdp.ConsumeBool()
    uncertainty = None
    if has_unc:
        n = len(reflectance)
        uncertainty = _consume_float_array(fdp, max_len=n)
        if len(uncertainty) != n:
            uncertainty = (
                np.resize(uncertainty, n) if len(uncertainty) > 0 else np.zeros(n)
            )

    try:
        estimate_snr(reflectance, uncertainty)
    except (ValueError, TypeError):
        pass


_TARGETS = [fuzz_resample, fuzz_normalize, fuzz_mask_telluric, fuzz_estimate_snr]


def fuzz_combined(data: bytes) -> None:
    """Combined harness: randomly dispatches to one of the 4 fuzz targets."""
    if len(data) < 2:
        return
    fdp = _get_fdp(data)
    choice = fdp.ConsumeIntInRange(0, 3)
    remaining = fdp.ConsumeBytes(fdp.remaining_bytes())
    _TARGETS[choice](remaining)


def _run_fallback_fuzzer(duration_seconds: int = 60) -> None:
    """Pure-Python random fuzzer for environments without Atheris."""
    rng = np.random.default_rng(seed=int(time.time()))
    runs = 0
    start = time.monotonic()

    print(f"Atheris not available — running pure-Python random fuzzer for {duration_seconds}s")

    while time.monotonic() - start < duration_seconds:
        size = int(rng.integers(0, 4096))
        data = rng.bytes(size)
        fuzz_combined(data)
        runs += 1
        if runs % 5000 == 0:
            elapsed = time.monotonic() - start
            print(f"  {runs} runs, {elapsed:.1f}s elapsed")

    elapsed = time.monotonic() - start
    print(f"Completed {runs} runs in {elapsed:.1f}s — no crashes")


def main() -> None:
    if HAS_ATHERIS:
        argv = sys.argv
        if not any(arg.startswith("-max_total_time") for arg in argv):
            argv = argv + ["-max_total_time=60"]

        with atheris.instrument_imports():
            pass  # targets already imported at module level

        atheris.Setup(argv, fuzz_combined)
        atheris.Fuzz()
    else:
        duration = 60
        for arg in sys.argv[1:]:
            if arg.startswith("-max_total_time="):
                duration = int(arg.split("=")[1])
        _run_fallback_fuzzer(duration)


if __name__ == "__main__":
    main()
