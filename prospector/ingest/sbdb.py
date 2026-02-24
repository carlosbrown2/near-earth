"""SBDB bulk CSV ingestion into asteroids and orbits tables.

Reads JPL SBDB bulk CSV export and populates the asteroids and orbits
tables. SPK-ID is mapped to asteroid_id (IAU number for numbered objects).
Missing fields are stored as NULL.

Usage:
    from prospector.ingest.sbdb import ingest_sbdb_csv
    from prospector.db import get_connection

    conn = get_connection()
    count = ingest_sbdb_csv("data/sbdb/sbdb_query_results.csv", conn)
"""

import logging
import re
import sqlite3
from pathlib import Path

import pandas as pd

from prospector.db import get_connection, init_schema

logger = logging.getLogger(__name__)

# Numbered asteroid SPK-IDs: 2000000 + IAU_number
_NUMBERED_OFFSET = 2_000_000
_NUMBERED_MAX = 2_999_999

# Parse SBDB full_name field.
# Examples:
#   "  4179 Toutatis (1989 FB)"  → number=4179, name=Toutatis, desig=1989 FB
#   "     1 Ceres"               → number=1,    name=Ceres,    desig=None
#   "       (2024 YR4)"          → number=None,  name=None,     desig=2024 YR4
_FULLNAME_RE = re.compile(
    r"^\s*(?:(\d+)\s+)?"  # optional leading IAU number
    r"([^(]*?)\s*"  # optional name (before any parenthesized part)
    r"(?:\(([^)]+)\))?\s*$"  # optional designation in parentheses
)


def spkid_to_asteroid_id(spkid: int) -> int:
    """Convert SPK-ID to asteroid_id.

    Numbered asteroids: spkid in [2000001, 2999999] → subtract 2000000.
    Unnumbered/other: use spkid directly.
    """
    if _NUMBERED_OFFSET < spkid <= _NUMBERED_MAX:
        return spkid - _NUMBERED_OFFSET
    return spkid


def parse_full_name(full_name: str) -> dict[str, str | None]:
    """Parse SBDB full_name into number, name, and designation components.

    Returns dict with keys: number (str or None), name, designation.
    """
    if not full_name or not isinstance(full_name, str):
        return {"number": None, "name": None, "designation": None}

    m = _FULLNAME_RE.match(full_name.strip())
    if not m:
        return {"number": None, "name": None, "designation": full_name.strip()}

    number = m.group(1)  # IAU number as string, or None
    name = m.group(2).strip() if m.group(2) and m.group(2).strip() else None
    designation = m.group(3)  # provisional designation, or None

    return {"number": number, "name": name, "designation": designation}


def _flag_to_int(val) -> int | None:
    """Convert Y/N/NaN flag to 1/0/None for SQLite boolean storage."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().upper()
    if s == "Y":
        return 1
    if s == "N":
        return 0
    return None


def _safe_float(val) -> float | None:
    """Convert to float, returning None for missing/invalid values."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def ingest_sbdb_csv(
    csv_path: str | Path,
    conn: sqlite3.Connection,
    *,
    batch_size: int = 5000,
    neo_only: bool = False,
) -> int:
    """Ingest SBDB bulk CSV into asteroids + orbits tables.

    Parameters
    ----------
    csv_path : path to the SBDB CSV file
    conn : database connection (use get_connection() or get_connection(":memory:"))
    batch_size : rows per INSERT batch
    neo_only : if True, only ingest rows where neo='Y'

    Returns
    -------
    int : number of rows ingested
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"SBDB CSV not found: {csv_path}")

    init_schema(conn)
    count = 0

    for chunk in pd.read_csv(csv_path, chunksize=batch_size, low_memory=False):
        # Normalize column names to lowercase
        chunk.columns = [c.strip().lower() for c in chunk.columns]

        if neo_only:
            neo_col = chunk.get("neo", pd.Series(dtype=str))
            chunk = chunk[neo_col.astype(str).str.strip().str.upper() == "Y"]

        asteroid_rows = []
        orbit_rows = []

        for _, row in chunk.iterrows():
            spkid_val = row.get("spkid")
            if spkid_val is None or pd.isna(spkid_val):
                continue
            spkid = int(spkid_val)
            if spkid == 0:
                continue

            asteroid_id = spkid_to_asteroid_id(spkid)
            full_name_raw = row.get("full_name")
            full_name = (
                str(full_name_raw).strip()
                if full_name_raw is not None and not (isinstance(full_name_raw, float) and pd.isna(full_name_raw))
                else ""
            )
            parsed = parse_full_name(full_name)

            neo = _flag_to_int(row.get("neo"))
            pha = _flag_to_int(row.get("pha"))

            asteroid_rows.append((
                asteroid_id,
                parsed["designation"],
                parsed["name"],
                full_name if full_name else None,
                neo,
                pha,
            ))

            orbit_rows.append((
                asteroid_id,
                _safe_float(row.get("epoch")),
                _safe_float(row.get("e")),
                _safe_float(row.get("a")),
                _safe_float(row.get("i")),
                _safe_float(row.get("om")),
                _safe_float(row.get("w")),
                _safe_float(row.get("ma")),
                _safe_float(row.get("q")),
                _safe_float(row.get("h")),
                _safe_float(row.get("moid")),
                _safe_float(row.get("diameter")),
                _safe_float(row.get("diameter_sigma")),
            ))

        if asteroid_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO asteroids "
                "(asteroid_id, designation, name, full_name, neo, pha) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                asteroid_rows,
            )
            conn.executemany(
                "INSERT OR REPLACE INTO orbits "
                "(asteroid_id, epoch, e, a, i, om, w, ma, q, H, moid, diameter, diameter_sigma) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                orbit_rows,
            )
            conn.commit()
            count += len(asteroid_rows)

    logger.info("Ingested %d rows from %s", count, csv_path)
    return count
