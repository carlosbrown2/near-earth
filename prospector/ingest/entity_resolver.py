"""Cross-survey entity resolution: normalize identifiers to canonical asteroid_id.

Canonical key: IAU asteroid number (integer).
Fallback: packed MPC designation (7-character string).

Supports: IAU number, SPK-ID, packed MPC designation, unpacked MPC designation,
MITHNEOS filename, asteroid name. See PRD §5E.

Usage:
    from prospector.ingest.entity_resolver import resolve, pack_designation

    # Pure conversion (no DB needed)
    packed = pack_designation("1998 SF36")   # "J98S36F"
    unpacked = unpack_designation("J98S36F") # "1998 SF36"

    # DB-backed resolution
    asteroid_id = resolve("1998 SF36", conn)  # returns 25143 (Itokawa)
    asteroid_id = resolve("25143", conn)      # same
"""

import logging
import re
import sqlite3
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# MPC century codes for packed designations
_CENTURY_PACK = {18: "I", 19: "J", 20: "K"}
_CENTURY_UNPACK = {"I": 18, "J": 19, "K": 20}

# SPK-ID range for numbered asteroids
_NUMBERED_OFFSET = 2_000_000
_NUMBERED_MAX = 2_999_999

# Provisional designation: "YYYY LL" or "YYYY LLN" where N is cycle number
_PROV_RE = re.compile(
    r"^(\d{4})\s+([A-HJ-Y])([A-HJ-Z])(\d*)$"
)

# Packed provisional designation: 7 chars, century+yy+half_month+cycle(2)+letter
_PACKED_PROV_RE = re.compile(
    r"^([IJK])(\d{2})([A-HJ-Y])([0-9A-Za-z]\d)([A-HJ-Z])$"
)

# MITHNEOS filename: aXXXXXX.spNN.txt (numbered) or au... (unnumbered)
_MITHNEOS_NUM_RE = re.compile(r"^a(\d+)\.sp\d+")
_MITHNEOS_DESIG_RE = re.compile(r"^au(\d{4})(\w+)\.sp\d+")

# Survey designation (comet-style not handled — only minor planets)
_SURVEY_RE = re.compile(r"^(\d{4})\s+([A-Z]{2}\d*)$")


@dataclass(frozen=True)
class ResolvedID:
    """Result of entity resolution."""

    asteroid_id: int | None  # IAU number if numbered, SPK-ID otherwise
    designation: str | None  # unpacked MPC designation
    packed: str | None  # packed MPC designation
    source: str  # how it was resolved: 'number', 'designation', 'spkid', 'name', 'alias'


def spkid_to_asteroid_id(spkid: int) -> int:
    """Convert SPK-ID to asteroid_id.

    Numbered asteroids: spkid in [2000001, 2999999] -> subtract 2000000.
    Unnumbered/other: use spkid directly.
    """
    if _NUMBERED_OFFSET < spkid <= _NUMBERED_MAX:
        return spkid - _NUMBERED_OFFSET
    return spkid


def _encode_cycle(n: int) -> str:
    """Encode cycle number as 2-char MPC packed format.

    0-99: two-digit number ("00"-"99").
    100-619: letter + digit (A0=100, B0=110, ..., Z0=350, a0=360, ..., z9=619).
    """
    if n < 0:
        raise ValueError(f"Cycle number must be non-negative: {n}")
    if n < 100:
        return f"{n:02d}"
    tens = n // 10
    ones = n % 10
    if 10 <= tens <= 35:
        return chr(ord("A") + tens - 10) + str(ones)
    if 36 <= tens <= 61:
        return chr(ord("a") + tens - 36) + str(ones)
    raise ValueError(f"Cycle number too large to encode: {n}")


def _decode_cycle(s: str) -> int:
    """Decode 2-char MPC packed cycle encoding."""
    c = s[0]
    d = int(s[1])
    if c.isdigit():
        return int(s)
    if c.isupper():
        return (ord(c) - ord("A") + 10) * 10 + d
    if c.islower():
        return (ord(c) - ord("a") + 36) * 10 + d
    raise ValueError(f"Cannot decode cycle: {s!r}")


def pack_designation(unpacked: str) -> str:
    """Convert unpacked provisional designation to packed MPC format.

    Examples:
        "1998 SF36" -> "J98S36F"
        "1989 FB"   -> "J89F00B"
        "2024 YR4"  -> "K24Y04R"
    """
    m = _PROV_RE.match(unpacked.strip())
    if not m:
        raise ValueError(f"Cannot parse designation: {unpacked!r}")

    year = int(m.group(1))
    half_month = m.group(2)
    second_letter = m.group(3)
    cycle = int(m.group(4)) if m.group(4) else 0

    century = year // 100
    yy = year % 100

    if century not in _CENTURY_PACK:
        raise ValueError(f"Unsupported century: {century}")

    cycle_str = _encode_cycle(cycle)
    return f"{_CENTURY_PACK[century]}{yy:02d}{half_month}{cycle_str}{second_letter}"


def unpack_designation(packed: str) -> str:
    """Convert packed MPC designation to unpacked form.

    Examples:
        "J98S36F" -> "1998 SF36"
        "J89F00B" -> "1989 FB"
        "K24Y04R" -> "2024 YR4"
    """
    m = _PACKED_PROV_RE.match(packed.strip())
    if not m:
        raise ValueError(f"Cannot parse packed designation: {packed!r}")

    century_code = m.group(1)
    yy = int(m.group(2))
    half_month = m.group(3)
    cycle_str = m.group(4)
    second_letter = m.group(5)

    century = _CENTURY_UNPACK[century_code]
    year = century * 100 + yy
    cycle = _decode_cycle(cycle_str)

    cycle_suffix = str(cycle) if cycle > 0 else ""
    return f"{year} {half_month}{second_letter}{cycle_suffix}"


def normalize_designation(desig: str) -> str:
    """Normalize a designation to unpacked form.

    Accepts packed or unpacked designations. Returns unpacked form.
    """
    desig = desig.strip()
    if _PACKED_PROV_RE.match(desig):
        return unpack_designation(desig)
    if _PROV_RE.match(desig):
        # Already unpacked — normalize whitespace
        m = _PROV_RE.match(desig)
        year = m.group(1)
        half = m.group(2)
        letter = m.group(3)
        cycle = m.group(4)
        return f"{year} {half}{letter}{cycle}"
    return desig


def parse_mithneos_filename(filename: str) -> int | str | None:
    """Extract asteroid ID or designation from MITHNEOS filename.

    Returns:
        int: asteroid number for numbered objects (e.g., a004179.sp05.txt -> 4179)
        str: unpacked designation for unnumbered objects
        None: if parsing fails
    """
    name = filename.split("/")[-1]  # strip path
    m = _MITHNEOS_NUM_RE.match(name)
    if m:
        return int(m.group(1))
    m = _MITHNEOS_DESIG_RE.match(name)
    if m:
        year = m.group(1)
        rest = m.group(2)
        if len(rest) >= 2:
            half_month = rest[0].upper()
            second_letter = rest[1].upper()
            cycle = rest[2:] if len(rest) > 2 else ""
            return f"{year} {half_month}{second_letter}{cycle}"
    return None


def resolve(
    identifier: str,
    conn: sqlite3.Connection | None = None,
) -> ResolvedID | None:
    """Resolve any asteroid identifier to canonical form.

    Resolution order:
    1. Pure integer → treat as IAU number
    2. SPK-ID range (2000001-2999999) → convert to IAU number
    3. Packed designation → unpack, look up in DB
    4. Unpacked designation → look up in DB
    5. Name → look up in DB
    6. Alias → look up via designation/full_name search

    Parameters
    ----------
    identifier : any string representation (number, designation, name, SPK-ID)
    conn : optional database connection for DB-backed lookup

    Returns
    -------
    ResolvedID or None if resolution fails
    """
    identifier = identifier.strip()
    if not identifier:
        return None

    # 1. Pure integer — IAU number or SPK-ID
    try:
        num = int(identifier)
        # Check if it's in SPK-ID range for numbered asteroids
        if _NUMBERED_OFFSET < num <= _NUMBERED_MAX:
            aid = num - _NUMBERED_OFFSET
            desig = _lookup_designation(aid, conn) if conn else None
            packed = pack_designation(desig) if desig and _PROV_RE.match(desig) else None
            return ResolvedID(
                asteroid_id=aid,
                designation=desig,
                packed=packed,
                source="spkid",
            )
        # Otherwise treat as IAU number
        if num > 0:
            desig = _lookup_designation(num, conn) if conn else None
            packed = pack_designation(desig) if desig and _PROV_RE.match(desig) else None
            return ResolvedID(
                asteroid_id=num,
                designation=desig,
                packed=packed,
                source="number",
            )
    except ValueError:
        pass

    # 2. Packed designation
    if _PACKED_PROV_RE.match(identifier):
        unpacked = unpack_designation(identifier)
        aid = _lookup_by_designation(unpacked, conn) if conn else None
        return ResolvedID(
            asteroid_id=aid,
            designation=unpacked,
            packed=identifier,
            source="designation",
        )

    # 3. Unpacked designation
    if _PROV_RE.match(identifier):
        normalized = normalize_designation(identifier)
        packed = pack_designation(normalized)
        aid = _lookup_by_designation(normalized, conn) if conn else None
        return ResolvedID(
            asteroid_id=aid,
            designation=normalized,
            packed=packed,
            source="designation",
        )

    # 4. Name lookup (DB required)
    if conn:
        aid = _lookup_by_name(identifier, conn)
        if aid is not None:
            desig = _lookup_designation(aid, conn)
            packed = pack_designation(desig) if desig and _PROV_RE.match(desig) else None
            return ResolvedID(
                asteroid_id=aid,
                designation=desig,
                packed=packed,
                source="name",
            )

        # 5. Alias search — check full_name field
        aid = _lookup_by_alias(identifier, conn)
        if aid is not None:
            desig = _lookup_designation(aid, conn)
            packed = pack_designation(desig) if desig and _PROV_RE.match(desig) else None
            return ResolvedID(
                asteroid_id=aid,
                designation=desig,
                packed=packed,
                source="alias",
            )

    logger.warning("Could not resolve identifier: %s", identifier)
    return None


def _lookup_designation(asteroid_id: int, conn: sqlite3.Connection | None) -> str | None:
    """Look up designation for an asteroid_id."""
    if conn is None:
        return None
    row = conn.execute(
        "SELECT designation FROM asteroids WHERE asteroid_id = ?",
        (asteroid_id,),
    ).fetchone()
    return row[0] if row else None


def _lookup_by_designation(
    designation: str, conn: sqlite3.Connection | None
) -> int | None:
    """Look up asteroid_id by unpacked designation."""
    if conn is None:
        return None
    row = conn.execute(
        "SELECT asteroid_id FROM asteroids WHERE designation = ?",
        (designation,),
    ).fetchone()
    if row:
        return row[0]
    # Try packed form as well
    try:
        packed = pack_designation(designation)
        row = conn.execute(
            "SELECT asteroid_id FROM asteroids WHERE designation = ?",
            (packed,),
        ).fetchone()
        if row:
            return row[0]
    except ValueError:
        pass
    return None


def _lookup_by_name(name: str, conn: sqlite3.Connection) -> int | None:
    """Look up asteroid_id by name (case-insensitive)."""
    row = conn.execute(
        "SELECT asteroid_id FROM asteroids WHERE LOWER(name) = LOWER(?)",
        (name,),
    ).fetchone()
    return row[0] if row else None


def _lookup_by_alias(identifier: str, conn: sqlite3.Connection) -> int | None:
    """Search full_name field for alias matches."""
    # Search full_name for the identifier substring
    row = conn.execute(
        "SELECT asteroid_id FROM asteroids WHERE full_name LIKE ?",
        (f"%{identifier}%",),
    ).fetchone()
    return row[0] if row else None
