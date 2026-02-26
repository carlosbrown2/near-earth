"""Pytest integration for fuzz_ingest harness.

Runs each fuzz target with seed corpus and random payloads. Works without
Atheris (pure-Python FuzzedDataProvider fallback).

For full coverage-guided fuzzing:
    python tests/fuzz_ingest.py -max_total_time=60
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.fuzz_ingest import (
    fuzz_combined,
    fuzz_flag_to_int,
    fuzz_normalize_designation,
    fuzz_pack_designation,
    fuzz_parse_full_name,
    fuzz_parse_mithneos_filename,
    fuzz_read_spectrum_file,
    fuzz_resolve_no_db,
    fuzz_safe_float,
    fuzz_spkid,
    fuzz_unpack_designation,
)


# ---------------------------------------------------------------------------
# Seed corpus: sbdb.py
# ---------------------------------------------------------------------------


class TestFuzzParseFullName:
    def test_empty(self) -> None:
        fuzz_parse_full_name(b"")

    def test_typical_names(self) -> None:
        for name in [
            b"  4179 Toutatis (1989 FB)",
            b"     1 Ceres",
            b"       (2024 YR4)",
            b"()",
            b"0 ()",
            b"99999999 LongName (2099 ZZ999)",
        ]:
            fuzz_parse_full_name(name)

    def test_unicode(self) -> None:
        fuzz_parse_full_name("Ωmega αsteroid 42 (日本語)".encode("utf-8"))

    def test_nulls_and_control(self) -> None:
        fuzz_parse_full_name(b"\x00\x01\x02\x03\x04\x05")

    @pytest.mark.parametrize("seed", range(20))
    def test_random(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        fuzz_parse_full_name(rng.bytes(rng.integers(0, 2048)))


class TestFuzzSpkid:
    def test_edge_values(self) -> None:
        for val in [b"\x00" * 12, b"\xff" * 12, b"\x80" * 12]:
            fuzz_spkid(val)

    @pytest.mark.parametrize("seed", range(10))
    def test_random(self, seed: int) -> None:
        rng = np.random.default_rng(100 + seed)
        fuzz_spkid(rng.bytes(rng.integers(0, 64)))


class TestFuzzSafeFloat:
    def test_edge_types(self) -> None:
        for payload in [b"\x00" * 16, b"\xff" * 16, b"", b"\x02" + b"nan", b"\x01"]:
            fuzz_safe_float(payload)

    @pytest.mark.parametrize("seed", range(10))
    def test_random(self, seed: int) -> None:
        rng = np.random.default_rng(200 + seed)
        fuzz_safe_float(rng.bytes(rng.integers(0, 256)))


class TestFuzzFlagToInt:
    def test_known_flags(self) -> None:
        for payload in [b"\x00Y", b"\x00N", b"\x00\x00", b"\x02nan"]:
            fuzz_flag_to_int(payload)

    @pytest.mark.parametrize("seed", range(10))
    def test_random(self, seed: int) -> None:
        rng = np.random.default_rng(300 + seed)
        fuzz_flag_to_int(rng.bytes(rng.integers(0, 256)))


# ---------------------------------------------------------------------------
# Seed corpus: entity_resolver.py
# ---------------------------------------------------------------------------


class TestFuzzPackDesignation:
    def test_valid_designations(self) -> None:
        for desig in [b"1998 SF36", b"1989 FB", b"2024 YR4", b"2000 AA"]:
            fuzz_pack_designation(desig)

    def test_invalid(self) -> None:
        for payload in [b"", b"garbage", b"12345", b"\xff" * 50]:
            fuzz_pack_designation(payload)

    @pytest.mark.parametrize("seed", range(10))
    def test_random(self, seed: int) -> None:
        rng = np.random.default_rng(400 + seed)
        fuzz_pack_designation(rng.bytes(rng.integers(0, 512)))


class TestFuzzUnpackDesignation:
    def test_valid(self) -> None:
        for packed in [b"J98S36F", b"J89F00B", b"K24Y04R"]:
            fuzz_unpack_designation(packed)

    def test_invalid(self) -> None:
        for payload in [b"", b"XXXXXXX", b"\x00" * 20]:
            fuzz_unpack_designation(payload)

    @pytest.mark.parametrize("seed", range(10))
    def test_random(self, seed: int) -> None:
        rng = np.random.default_rng(500 + seed)
        fuzz_unpack_designation(rng.bytes(rng.integers(0, 512)))


class TestFuzzNormalizeDesignation:
    def test_mixed(self) -> None:
        for payload in [b"1998 SF36", b"J98S36F", b"garbage", b""]:
            fuzz_normalize_designation(payload)

    @pytest.mark.parametrize("seed", range(10))
    def test_random(self, seed: int) -> None:
        rng = np.random.default_rng(600 + seed)
        fuzz_normalize_designation(rng.bytes(rng.integers(0, 512)))


class TestFuzzParseMithneosFilename:
    def test_valid_filenames(self) -> None:
        for name in [
            b"a004179.sp05.txt",
            b"a000001.sp01.txt",
            b"au2024yr4.sp01.txt",
        ]:
            fuzz_parse_mithneos_filename(name)

    def test_invalid(self) -> None:
        for payload in [b"", b"random.txt", b"\xff" * 100]:
            fuzz_parse_mithneos_filename(payload)

    @pytest.mark.parametrize("seed", range(10))
    def test_random(self, seed: int) -> None:
        rng = np.random.default_rng(700 + seed)
        fuzz_parse_mithneos_filename(rng.bytes(rng.integers(0, 512)))


class TestFuzzResolveNoDb:
    def test_numeric_ids(self) -> None:
        for payload in [b"25143", b"1", b"0", b"2000001", b"9999999"]:
            fuzz_resolve_no_db(payload)

    def test_designations(self) -> None:
        for payload in [b"1998 SF36", b"J98S36F", b"Ceres"]:
            fuzz_resolve_no_db(payload)

    def test_garbage(self) -> None:
        fuzz_resolve_no_db(b"\x00" * 100)
        fuzz_resolve_no_db(b"\xff" * 200)

    @pytest.mark.parametrize("seed", range(10))
    def test_random(self, seed: int) -> None:
        rng = np.random.default_rng(800 + seed)
        fuzz_resolve_no_db(rng.bytes(rng.integers(0, 512)))


# ---------------------------------------------------------------------------
# Seed corpus: mithneos _read_spectrum_file
# ---------------------------------------------------------------------------


class TestFuzzReadSpectrumFile:
    def test_valid_spectrum(self) -> None:
        content = b"# comment\n0.8 1.0 0.01\n0.9 0.95 0.02\n1.0 0.90 0.015\n"
        fuzz_read_spectrum_file(content)

    def test_two_column(self) -> None:
        content = b"0.8 1.0\n0.9 0.95\n1.0 0.90\n"
        fuzz_read_spectrum_file(content)

    def test_empty_file(self) -> None:
        fuzz_read_spectrum_file(b"")

    def test_all_comments(self) -> None:
        fuzz_read_spectrum_file(b"# comment\n# more\n")

    def test_binary_garbage(self) -> None:
        fuzz_read_spectrum_file(b"\x00\xff\xfe\xfd" * 100)

    def test_truncated_lines(self) -> None:
        fuzz_read_spectrum_file(b"0.8\n\n\n0.9 0.95\n")

    @pytest.mark.parametrize("seed", range(10))
    def test_random(self, seed: int) -> None:
        rng = np.random.default_rng(900 + seed)
        fuzz_read_spectrum_file(rng.bytes(rng.integers(0, 2048)))


# ---------------------------------------------------------------------------
# Bulk random fuzzing
# ---------------------------------------------------------------------------


class TestFuzzBulkIngest:
    """Bulk random fuzzing — 5k iterations combined harness."""

    def test_bulk_random_5k(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(5_000):
            size = int(rng.integers(2, 2048))
            data = rng.bytes(size)
            fuzz_combined(data)
