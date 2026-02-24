"""Tests for cross-survey entity resolution (prospector.ingest.entity_resolver).

Tests MPC packed/unpacked designation conversion, identifier normalization,
and DB-backed resolution against 50+ known multi-survey asteroids.
"""

import sqlite3

import pytest

from prospector.db import get_connection, init_schema
from prospector.ingest.entity_resolver import (
    ResolvedID,
    _decode_cycle,
    _encode_cycle,
    normalize_designation,
    pack_designation,
    parse_mithneos_filename,
    resolve,
    spkid_to_asteroid_id,
    unpack_designation,
)


# ---------------------------------------------------------------------------
# Known multi-survey asteroids for cross-validation.
# Each entry: (IAU number, name, unpacked designation, packed designation, SPK-ID)
# ---------------------------------------------------------------------------
KNOWN_ASTEROIDS = [
    # Well-characterized NEOs
    (25143, "Itokawa", "1998 SF36", "J98S36F", 2025143),
    (101955, "Bennu", "1999 RQ36", "J99R36Q", 2101955),
    (4179, "Toutatis", "1989 AC", "J89A00C", 2004179),
    (433, "Eros", "1898 DQ", "I98D00Q", 2000433),
    (1566, "Icarus", "1949 MA", "J49M00A", 2001566),
    (3200, "Phaethon", "1983 TB", "J83T00B", 2003200),
    (1862, "Apollo", "1932 HA", "J32H00A", 2001862),
    (2062, "Aten", "1976 AA", "J76A00A", 2002062),
    (99942, "Apophis", "2004 MN4", "K04M04N", 2099942),
    (162173, "Ryugu", "1999 JU3", "J99J03U", 2162173),
    # Potentially hazardous asteroids
    (29075, "1950 DA", "1950 DA", "J50D00A", 2029075),
    (4660, "Nereus", "1982 DB", "J82D00B", 2004660),
    (3361, "Orpheus", "1982 HR", "J82H00R", 2003361),
    (2063, "Bacchus", "1977 HB", "J77H00B", 2002063),
    (65803, "Didymos", "1996 GT", "J96G00T", 2065803),
    (6489, "Golevka", "1991 JX", "J91J00X", 2006489),
    (4769, "Castalia", "1989 PB", "J89P00B", 2004769),
    (1685, "Toro", "1948 OA", "J48O00A", 2001685),
    (1620, "Geographos", "1951 RA", "J51R00A", 2001620),
    (1627, "Ivar", "1929 SH", "J29S00H", 2001627),
    # M-type / metallic asteroids (mining candidates)
    (16, "Psyche", "1852 FA", "I52F00A", 2000016),
    (21, "Lutetia", "1852 OA", "I52O00A", 2000021),
    (22, "Kalliope", "1852 PA", "I52P00A", 2000022),
    (216, "Kleopatra", "1880 CA", "I80C00A", 2000216),
    (758, "Mancunia", "1912 PE", "J12P00E", 2000758),
    # S-type / silicate asteroids
    (243, "Ida", "1884 OA", "I84O00A", 2000243),
    (951, "Gaspra", "1916 SB", "J16S00B", 2000951),
    (1036, "Ganymed", "1924 TD", "J24T00D", 2001036),
    (433, "Eros", "1898 DQ", "I98D00Q", 2000433),
    # C-type / carbonaceous asteroids
    (1, "Ceres", None, None, 2000001),
    (2, "Pallas", "1802 FA", "I02F00A", 2000002),
    (10, "Hygiea", "1849 GA", "I49G00A", 2000010),
    (52, "Europa", "1858 CA", "I58C00A", 2000052),
    (704, "Interamnia", "1910 KN", "J10K00N", 2000704),
    # Recent discoveries with high cycle numbers
    (363599, None, "2004 FG11", "K04F11G", 2363599),
    (153201, None, "2000 WO107", "K00WA7O", 2153201),
    # Additional NEOs for coverage
    (4581, "Asclepius", "1989 FC", "J89F00C", 2004581),
    (5604, "1992 FE", "1992 FE", "J92F00E", 2005604),
    (7341, "1991 VK", "1991 VK", "J91V00K", 2007341),
    (138175, None, "2000 EE104", "K00EA4E", 2138175),
    (85953, None, "1999 FK21", "J99F21K", 2085953),
    (66391, "Moshup", "1999 KW4", "J99K04W", 2066391),
    (136617, None, "1994 CC", "J94C00C", 2136617),
    (185851, None, "2000 DP107", "K00DA7P", 2185851),
    (276049, None, "2002 CE26", "K02C26E", 2276049),
    (341843, None, "2008 EV5", "K08E05V", 2341843),
    (153591, None, "2001 SN263", "K01SQ3N", 2153591),
    (3554, "Amun", "1986 EB", "J86E00B", 2003554),
    (1943, "Anteros", "1973 EC", "J73E00C", 2001943),
    (887, "Alinda", "1918 DB", "J18D00B", 2000887),
    (1580, "Betulia", "1950 KA", "J50K00A", 2001580),
    (2100, "Ra-Shalom", "1978 RA", "J78R00A", 2002100),
    (4486, "Mithra", "1987 SB", "J87S00B", 2004486),
    (5381, "Sekhmet", "1991 JY", "J91J00Y", 2005381),
]


# ---------------------------------------------------------------------------
# Cycle encoding/decoding
# ---------------------------------------------------------------------------
class TestCycleEncoding:
    """Tests for _encode_cycle / _decode_cycle."""

    @pytest.mark.parametrize(
        "n, expected",
        [
            (0, "00"),
            (1, "01"),
            (9, "09"),
            (10, "10"),
            (36, "36"),
            (99, "99"),
            (100, "A0"),
            (109, "A9"),
            (110, "B0"),
            (259, "P9"),
            (350, "Z0"),
            (359, "Z9"),
            (360, "a0"),
            (369, "a9"),
            (619, "z9"),
        ],
    )
    def test_encode(self, n, expected):
        assert _encode_cycle(n) == expected

    @pytest.mark.parametrize(
        "s, expected",
        [
            ("00", 0),
            ("01", 1),
            ("36", 36),
            ("99", 99),
            ("A0", 100),
            ("A9", 109),
            ("Z9", 359),
            ("a0", 360),
            ("z9", 619),
        ],
    )
    def test_decode(self, s, expected):
        assert _decode_cycle(s) == expected

    def test_roundtrip(self):
        for n in range(620):
            assert _decode_cycle(_encode_cycle(n)) == n

    def test_encode_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _encode_cycle(-1)

    def test_encode_too_large_raises(self):
        with pytest.raises(ValueError, match="too large"):
            _encode_cycle(620)


# ---------------------------------------------------------------------------
# Pack/unpack designation
# ---------------------------------------------------------------------------
class TestPackDesignation:
    """Tests for pack_designation / unpack_designation."""

    @pytest.mark.parametrize(
        "unpacked, packed",
        [
            (a[2], a[3])
            for a in KNOWN_ASTEROIDS
            if a[2] is not None and a[3] is not None
        ],
    )
    def test_pack_known_asteroids(self, unpacked, packed):
        assert pack_designation(unpacked) == packed

    @pytest.mark.parametrize(
        "unpacked, packed",
        [
            (a[2], a[3])
            for a in KNOWN_ASTEROIDS
            if a[2] is not None and a[3] is not None
        ],
    )
    def test_unpack_known_asteroids(self, unpacked, packed):
        assert unpack_designation(packed) == unpacked

    def test_roundtrip_pack_unpack(self):
        for a in KNOWN_ASTEROIDS:
            if a[2] is None:
                continue
            packed = pack_designation(a[2])
            assert unpack_designation(packed) == a[2]

    def test_pack_strips_whitespace(self):
        assert pack_designation("  1998 SF36  ") == "J98S36F"

    def test_pack_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            pack_designation("not a designation")

    def test_unpack_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            unpack_designation("XXXXXXX")

    def test_unpack_wrong_length_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            unpack_designation("J98S3")

    def test_pack_century_18(self):
        assert pack_designation("1898 DQ") == "I98D00Q"

    def test_pack_century_20(self):
        assert pack_designation("2004 MN4") == "K04M04N"


# ---------------------------------------------------------------------------
# Normalize designation
# ---------------------------------------------------------------------------
class TestNormalizeDesignation:
    def test_unpacked_passthrough(self):
        assert normalize_designation("1998 SF36") == "1998 SF36"

    def test_packed_to_unpacked(self):
        assert normalize_designation("J98S36F") == "1998 SF36"

    def test_strips_whitespace(self):
        assert normalize_designation("  1998 SF36  ") == "1998 SF36"

    def test_normalizes_extra_spaces(self):
        assert normalize_designation("1998  SF36") == "1998 SF36"

    def test_non_designation_passthrough(self):
        assert normalize_designation("Itokawa") == "Itokawa"


# ---------------------------------------------------------------------------
# SPK-ID conversion
# ---------------------------------------------------------------------------
class TestSpkidConversion:
    @pytest.mark.parametrize(
        "iau, spkid",
        [(a[0], a[4]) for a in KNOWN_ASTEROIDS],
    )
    def test_spkid_to_asteroid_id(self, iau, spkid):
        assert spkid_to_asteroid_id(spkid) == iau

    def test_unnumbered_passthrough(self):
        # SPK-IDs outside the numbered range pass through
        assert spkid_to_asteroid_id(3500000) == 3500000

    def test_boundary_low(self):
        assert spkid_to_asteroid_id(2000001) == 1

    def test_boundary_high(self):
        assert spkid_to_asteroid_id(2999999) == 999999


# ---------------------------------------------------------------------------
# MITHNEOS filename parsing
# ---------------------------------------------------------------------------
class TestMithneosFilename:
    def test_numbered(self):
        assert parse_mithneos_filename("a004179.sp05.txt") == 4179

    def test_numbered_with_path(self):
        assert parse_mithneos_filename("/data/spectra/a025143.sp01.txt") == 25143

    def test_numbered_short(self):
        assert parse_mithneos_filename("a000433.sp01.txt") == 433

    def test_invalid(self):
        assert parse_mithneos_filename("readme.txt") is None

    def test_unnumbered(self):
        result = parse_mithneos_filename("au2004mn4.sp01.txt")
        assert result == "2004 MN4"


# ---------------------------------------------------------------------------
# DB-backed resolution
# ---------------------------------------------------------------------------
@pytest.fixture
def db_with_asteroids():
    """In-memory DB populated with known asteroids for resolution tests."""
    conn = get_connection(":memory:")
    for iau, name, desig, packed, spkid in KNOWN_ASTEROIDS:
        # Build full_name similar to SBDB format
        parts = []
        if iau:
            parts.append(str(iau))
        if name:
            parts.append(name)
        if desig:
            parts.append(f"({desig})")
        full_name = " ".join(parts)

        conn.execute(
            "INSERT OR REPLACE INTO asteroids "
            "(asteroid_id, designation, name, full_name, neo, pha) "
            "VALUES (?, ?, ?, ?, 1, 0)",
            (iau, desig, name, full_name),
        )
    conn.commit()
    return conn


class TestResolveWithDB:
    """Test DB-backed identifier resolution."""

    def test_resolve_by_number(self, db_with_asteroids):
        result = resolve("25143", db_with_asteroids)
        assert result is not None
        assert result.asteroid_id == 25143
        assert result.designation == "1998 SF36"
        assert result.source == "number"

    def test_resolve_by_spkid(self, db_with_asteroids):
        result = resolve("2025143", db_with_asteroids)
        assert result is not None
        assert result.asteroid_id == 25143
        assert result.source == "spkid"

    def test_resolve_by_unpacked_designation(self, db_with_asteroids):
        result = resolve("1998 SF36", db_with_asteroids)
        assert result is not None
        assert result.asteroid_id == 25143
        assert result.packed == "J98S36F"
        assert result.source == "designation"

    def test_resolve_by_packed_designation(self, db_with_asteroids):
        result = resolve("J98S36F", db_with_asteroids)
        assert result is not None
        assert result.asteroid_id == 25143
        assert result.designation == "1998 SF36"
        assert result.source == "designation"

    def test_resolve_by_name(self, db_with_asteroids):
        result = resolve("Itokawa", db_with_asteroids)
        assert result is not None
        assert result.asteroid_id == 25143
        assert result.source == "name"

    def test_resolve_by_name_case_insensitive(self, db_with_asteroids):
        result = resolve("itokawa", db_with_asteroids)
        assert result is not None
        assert result.asteroid_id == 25143

    def test_resolve_bennu(self, db_with_asteroids):
        """Bennu: 101955 = 1999 RQ36 = J99R36Q"""
        for ident in ["101955", "2101955", "1999 RQ36", "J99R36Q", "Bennu"]:
            result = resolve(ident, db_with_asteroids)
            assert result is not None, f"Failed to resolve: {ident}"
            assert result.asteroid_id == 101955, f"Wrong ID for {ident}: {result}"

    def test_resolve_apophis(self, db_with_asteroids):
        """Apophis: 99942 = 2004 MN4 = K04M04N"""
        for ident in ["99942", "2004 MN4", "K04M04N", "Apophis"]:
            result = resolve(ident, db_with_asteroids)
            assert result is not None, f"Failed to resolve: {ident}"
            assert result.asteroid_id == 99942

    def test_resolve_ryugu(self, db_with_asteroids):
        """Ryugu: 162173 = 1999 JU3 = J99J03U"""
        for ident in ["162173", "1999 JU3", "J99J03U", "Ryugu"]:
            result = resolve(ident, db_with_asteroids)
            assert result is not None, f"Failed to resolve: {ident}"
            assert result.asteroid_id == 162173

    def test_resolve_eros(self, db_with_asteroids):
        """Eros: 433 = 1898 DQ = I98D00Q"""
        for ident in ["433", "1898 DQ", "I98D00Q", "Eros"]:
            result = resolve(ident, db_with_asteroids)
            assert result is not None, f"Failed to resolve: {ident}"
            assert result.asteroid_id == 433

    def test_resolve_psyche(self, db_with_asteroids):
        """Psyche: 16 = 1852 FA = I52F00A"""
        for ident in ["16", "1852 FA", "I52F00A", "Psyche"]:
            result = resolve(ident, db_with_asteroids)
            assert result is not None, f"Failed to resolve: {ident}"
            assert result.asteroid_id == 16

    def test_resolve_alias_in_full_name(self, db_with_asteroids):
        """Alias search via full_name field."""
        result = resolve("1950 DA", db_with_asteroids)
        assert result is not None
        assert result.asteroid_id == 29075

    def test_resolve_empty_string(self, db_with_asteroids):
        assert resolve("", db_with_asteroids) is None

    def test_resolve_unknown(self, db_with_asteroids):
        assert resolve("UnknownAsteroid99999", db_with_asteroids) is None

    def test_resolve_all_known_by_number(self, db_with_asteroids):
        """Verify all 50+ known asteroids resolve by IAU number."""
        for iau, name, desig, packed, spkid in KNOWN_ASTEROIDS:
            result = resolve(str(iau), db_with_asteroids)
            assert result is not None, f"Failed to resolve number {iau}"
            assert result.asteroid_id == iau

    def test_resolve_all_known_by_spkid(self, db_with_asteroids):
        """Verify all 50+ known asteroids resolve by SPK-ID."""
        for iau, name, desig, packed, spkid in KNOWN_ASTEROIDS:
            result = resolve(str(spkid), db_with_asteroids)
            assert result is not None, f"Failed to resolve SPK-ID {spkid}"
            assert result.asteroid_id == iau

    def test_resolve_all_known_by_designation(self, db_with_asteroids):
        """Verify all asteroids with designations resolve by unpacked designation."""
        for iau, name, desig, packed, spkid in KNOWN_ASTEROIDS:
            if desig is None:
                continue
            result = resolve(desig, db_with_asteroids)
            assert result is not None, f"Failed to resolve designation {desig}"
            assert result.asteroid_id == iau


# ---------------------------------------------------------------------------
# Resolution without DB (pure conversion only)
# ---------------------------------------------------------------------------
class TestResolveWithoutDB:
    """Test resolution in pure-conversion mode (no database)."""

    def test_number_no_db(self):
        result = resolve("25143")
        assert result is not None
        assert result.asteroid_id == 25143
        assert result.source == "number"

    def test_spkid_no_db(self):
        result = resolve("2025143")
        assert result is not None
        assert result.asteroid_id == 25143
        assert result.source == "spkid"

    def test_packed_designation_no_db(self):
        result = resolve("J98S36F")
        assert result is not None
        assert result.designation == "1998 SF36"
        assert result.packed == "J98S36F"
        assert result.asteroid_id is None  # no DB to look up

    def test_unpacked_designation_no_db(self):
        result = resolve("1998 SF36")
        assert result is not None
        assert result.designation == "1998 SF36"
        assert result.packed == "J98S36F"
        assert result.asteroid_id is None

    def test_name_no_db(self):
        # Names can't resolve without DB
        assert resolve("Itokawa") is None


# ---------------------------------------------------------------------------
# Coverage count verification
# ---------------------------------------------------------------------------
def test_known_asteroids_count():
    """Verify we have 50+ known asteroids for testing."""
    unique_ids = {a[0] for a in KNOWN_ASTEROIDS}
    assert len(unique_ids) >= 50, f"Only {len(unique_ids)} unique asteroids, need 50+"
