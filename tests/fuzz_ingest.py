"""Coverage-guided fuzz testing for ingest parser functions.

Fuzzes DB-free parsing functions from sbdb, entity_resolver, and mithneos
modules that process untrusted external data.

Run:
    python tests/fuzz_ingest.py                       # 60s default
    python tests/fuzz_ingest.py -max_total_time=120

Uses Atheris when available, pure-Python random fallback otherwise.
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

from prospector.ingest.entity_resolver import (
    normalize_designation,
    pack_designation,
    parse_mithneos_filename,
    resolve,
    unpack_designation,
)
from prospector.ingest.mithneos import _read_spectrum_file
from prospector.ingest.sbdb import (
    _flag_to_int,
    _safe_float,
    parse_full_name,
    spkid_to_asteroid_id,
)

try:
    import atheris  # type: ignore[import-not-found]

    HAS_ATHERIS = True
except ImportError:
    HAS_ATHERIS = False


class FuzzedDataProvider:
    """Minimal pure-Python reimplementation of atheris.FuzzedDataProvider."""

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

    def ConsumeUnicodeNoSurrogates(self, n: int) -> str:
        raw = self.ConsumeBytes(n)
        return raw.decode("utf-8", errors="replace")

    def ConsumeInt(self, n_bytes: int) -> int:
        raw = self.ConsumeBytes(n_bytes)
        if len(raw) < n_bytes:
            return 0
        return int.from_bytes(raw, "little", signed=True)


def _get_fdp(data: bytes) -> FuzzedDataProvider:
    if HAS_ATHERIS:
        return atheris.FuzzedDataProvider(data)  # type: ignore[no-any-return]
    return FuzzedDataProvider(data)


# ---- Fuzz targets: sbdb.py ----


def fuzz_parse_full_name(data: bytes) -> None:
    """Fuzz parse_full_name with random strings."""
    fdp = _get_fdp(data)
    s = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 500))
    try:
        result = parse_full_name(s)
        assert isinstance(result, dict)
    except (ValueError, TypeError):
        pass


def fuzz_spkid(data: bytes) -> None:
    """Fuzz spkid_to_asteroid_id with random integers."""
    fdp = _get_fdp(data)
    spkid = fdp.ConsumeInt(8)
    try:
        result = spkid_to_asteroid_id(spkid)
        assert isinstance(result, int)
    except (ValueError, TypeError, OverflowError):
        pass


def fuzz_safe_float(data: bytes) -> None:
    """Fuzz _safe_float with various types."""
    fdp = _get_fdp(data)
    choice = fdp.ConsumeIntInRange(0, 4)
    val: object
    if choice == 0:
        val = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 50))
    elif choice == 1:
        raw = fdp.ConsumeBytes(8)
        val = struct.unpack("<d", raw)[0] if len(raw) == 8 else None
    elif choice == 2:
        val = None
    elif choice == 3:
        val = float("nan")
    else:
        val = fdp.ConsumeInt(4)

    try:
        result = _safe_float(val)
        assert result is None or isinstance(result, float)
    except (ValueError, TypeError):
        pass


def fuzz_flag_to_int(data: bytes) -> None:
    """Fuzz _flag_to_int with random values."""
    fdp = _get_fdp(data)
    choice = fdp.ConsumeIntInRange(0, 3)
    val: object
    if choice == 0:
        val = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 20))
    elif choice == 1:
        val = None
    elif choice == 2:
        val = float("nan")
    else:
        raw = fdp.ConsumeBytes(8)
        val = struct.unpack("<d", raw)[0] if len(raw) == 8 else 0.0

    try:
        result = _flag_to_int(val)
        assert result is None or isinstance(result, int)
    except (ValueError, TypeError):
        pass


# ---- Fuzz targets: entity_resolver.py ----


def fuzz_pack_designation(data: bytes) -> None:
    """Fuzz pack_designation with random strings."""
    fdp = _get_fdp(data)
    s = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 100))
    try:
        result = pack_designation(s)
        assert isinstance(result, str)
    except (ValueError, TypeError, KeyError, IndexError):
        pass


def fuzz_unpack_designation(data: bytes) -> None:
    """Fuzz unpack_designation with random strings."""
    fdp = _get_fdp(data)
    s = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 100))
    try:
        result = unpack_designation(s)
        assert isinstance(result, str)
    except (ValueError, TypeError, KeyError, IndexError):
        pass


def fuzz_normalize_designation(data: bytes) -> None:
    """Fuzz normalize_designation with random strings."""
    fdp = _get_fdp(data)
    s = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 100))
    try:
        result = normalize_designation(s)
        assert isinstance(result, str)
    except (ValueError, TypeError):
        pass


def fuzz_parse_mithneos_filename(data: bytes) -> None:
    """Fuzz parse_mithneos_filename with random strings."""
    fdp = _get_fdp(data)
    s = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 200))
    try:
        result = parse_mithneos_filename(s)
        assert result is None or isinstance(result, (int, str))
    except (ValueError, TypeError):
        pass


def fuzz_resolve_no_db(data: bytes) -> None:
    """Fuzz resolve() without DB connection (pure parsing path)."""
    fdp = _get_fdp(data)
    s = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 200))
    try:
        result = resolve(s, conn=None)
        # result is ResolvedID or None
    except (ValueError, TypeError):
        pass


# ---- Fuzz target: mithneos _read_spectrum_file ----


def fuzz_read_spectrum_file(data: bytes) -> None:
    """Fuzz _read_spectrum_file by writing random bytes to a temp file."""
    fdp = _get_fdp(data)
    content = fdp.ConsumeBytes(fdp.remaining_bytes())

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".txt", delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        _read_spectrum_file(Path(tmp_path))
    except (ValueError, UnicodeDecodeError, OSError):
        pass  # expected for malformed data
    finally:
        os.unlink(tmp_path)


# ---- Combined harness ----

_TARGETS = [
    fuzz_parse_full_name,
    fuzz_spkid,
    fuzz_safe_float,
    fuzz_flag_to_int,
    fuzz_pack_designation,
    fuzz_unpack_designation,
    fuzz_normalize_designation,
    fuzz_parse_mithneos_filename,
    fuzz_resolve_no_db,
    fuzz_read_spectrum_file,
]


def fuzz_combined(data: bytes) -> None:
    """Combined harness: randomly dispatches to one of the fuzz targets."""
    if len(data) < 2:
        return
    fdp = _get_fdp(data)
    choice = fdp.ConsumeIntInRange(0, len(_TARGETS) - 1)
    remaining = fdp.ConsumeBytes(fdp.remaining_bytes())
    _TARGETS[choice](remaining)


def _run_fallback_fuzzer(duration_seconds: int = 60) -> None:
    """Pure-Python random fuzzer for environments without Atheris."""
    rng = np.random.default_rng(seed=int(time.time()))
    runs = 0
    start = time.monotonic()

    print(
        f"Atheris not available — running pure-Python random fuzzer for {duration_seconds}s"
    )

    while time.monotonic() - start < duration_seconds:
        size = int(rng.integers(0, 4096))
        data = rng.bytes(size)
        fuzz_combined(data)
        runs += 1
        if runs % 2000 == 0:
            elapsed = time.monotonic() - start
            print(f"  {runs} runs, {elapsed:.1f}s elapsed")

    elapsed = time.monotonic() - start
    print(f"Completed {runs} runs in {elapsed:.1f}s — no crashes")


def main() -> None:
    if HAS_ATHERIS:
        argv = sys.argv
        if not any(arg.startswith("-max_total_time") for arg in argv):
            argv = argv + ["-max_total_time=60"]

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
