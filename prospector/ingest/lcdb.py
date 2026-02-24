"""LCDB (Lightcurve Database) rotation period ingestion.

Reads the LCDB summary table (Warner, Harris & Pravec) and populates the
rotation_properties table.  Fast rotators (P < 2.2 hr) are flagged as
monolithic (above spin barrier) — easier to mine.  Slow rotators with
large amplitude are flagged as binary suspects.

Cross-matching to existing asteroids is by IAU number (preferred) or
provisional designation (fallback lookup in the asteroids table).

Usage:
    from prospector.ingest.lcdb import ingest_lcdb
    from prospector.db import get_connection

    conn = get_connection()
    count = ingest_lcdb("data/lcdb/LC_DAT_PUB.csv", conn)
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from prospector.db import init_schema

logger = logging.getLogger(__name__)

# Spin barrier threshold (hours) — objects rotating faster are likely monolithic
SPIN_BARRIER_PERIOD = 2.2

# Amplitude threshold (mag) for binary suspect classification
BINARY_AMPLITUDE_THRESHOLD = 0.7

# Minimum period (hours) for binary suspect — must be slow rotator
BINARY_PERIOD_THRESHOLD = 6.0

# Default column name mapping from LCDB summary file to our schema
DEFAULT_COLUMN_MAP = {
    "number": "Num",
    "name": "Name",
    "designation": "Desig",
    "period": "Per",
    "period_unc": "PerErr",
    "amplitude": "AmpMax",
    "amplitude_unc": "AmpErr",
    "quality_code": "U",
}

# Alternative column names found in various LCDB releases
_COLUMN_ALIASES = {
    "number": ["num", "number", "iau_number", "ast_number"],
    "name": ["name", "ast_name"],
    "designation": ["desig", "designation", "prov_desig"],
    "period": ["per", "period", "rot_per", "rotation_period"],
    "period_unc": ["pererr", "per_err", "period_err", "period_unc"],
    "amplitude": ["ampmax", "amp", "amplitude", "amp_max"],
    "amplitude_unc": ["amperr", "amp_err", "amplitude_err"],
    "quality_code": ["u", "qual", "quality", "u_code", "quality_code"],
}


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


def _safe_int(val) -> int | None:
    """Convert to int, returning None for missing/invalid values."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _resolve_columns(df_columns: list[str], column_map: dict[str, str]) -> dict[str, str | None]:
    """Resolve logical column names to actual DataFrame column names.

    Tries the explicit column_map first, then falls back to known aliases.
    Returns a dict mapping logical names to actual column names (or None).
    """
    df_cols_lower = {c.lower().strip(): c for c in df_columns}
    resolved = {}

    for logical_name, mapped_name in column_map.items():
        # Try explicitly mapped name first
        if mapped_name.lower().strip() in df_cols_lower:
            resolved[logical_name] = df_cols_lower[mapped_name.lower().strip()]
            continue

        # Try known aliases
        found = False
        for alias in _COLUMN_ALIASES.get(logical_name, []):
            if alias.lower() in df_cols_lower:
                resolved[logical_name] = df_cols_lower[alias.lower()]
                found = True
                break

        if not found:
            resolved[logical_name] = None

    return resolved


def _lookup_designation(conn: sqlite3.Connection, designation: str) -> int | None:
    """Look up asteroid_id by designation in the asteroids table."""
    if not designation or not isinstance(designation, str):
        return None
    designation = designation.strip()
    if not designation:
        return None
    row = conn.execute(
        "SELECT asteroid_id FROM asteroids WHERE designation = ?",
        (designation,),
    ).fetchone()
    return row[0] if row else None


def classify_spin(period: float | None, amplitude: float | None) -> tuple[bool | None, bool | None]:
    """Classify an asteroid's physical structure from rotation properties.

    Parameters
    ----------
    period : float or None
        Rotation period in hours.
    amplitude : float or None
        Lightcurve amplitude in magnitudes.

    Returns
    -------
    (is_monolithic, is_binary_suspect) : tuple of bool or None
        is_monolithic: True if period < 2.2 hr (above spin barrier).
        is_binary_suspect: True if slow rotator with large amplitude.
    """
    is_monolithic = None
    is_binary_suspect = None

    if period is not None and period > 0:
        is_monolithic = period < SPIN_BARRIER_PERIOD
        if amplitude is not None:
            is_binary_suspect = (
                period > BINARY_PERIOD_THRESHOLD
                and amplitude > BINARY_AMPLITUDE_THRESHOLD
            )

    return is_monolithic, is_binary_suspect


def ingest_lcdb(
    filepath: str | Path,
    conn: sqlite3.Connection,
    *,
    column_map: dict[str, str] | None = None,
    min_quality: str | None = None,
    batch_size: int = 5000,
) -> int:
    """Ingest LCDB rotation period data into rotation_properties table.

    Parameters
    ----------
    filepath : path to the LCDB data file (.csv)
    conn : database connection
    column_map : optional override for column name mapping
    min_quality : minimum quality code to accept ('1', '2', '2+', '3').
        If None, all quality codes are ingested.
    batch_size : rows per INSERT batch

    Returns
    -------
    int : number of rows ingested
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"LCDB data file not found: {filepath}")

    init_schema(conn)

    # Read data file
    df = pd.read_csv(filepath, low_memory=False)

    # Resolve column names
    cmap = column_map if column_map is not None else DEFAULT_COLUMN_MAP
    cols = _resolve_columns(list(df.columns), cmap)

    # Need at least number or designation for matching
    has_number = cols["number"] is not None
    has_desig = cols["designation"] is not None
    if not has_number and not has_desig:
        raise ValueError(
            f"Cannot find number or designation column in data. "
            f"Available columns: {list(df.columns)}. "
            f"Provide a column_map to specify the mapping."
        )

    # Need period column
    if cols["period"] is None:
        raise ValueError(
            f"Cannot find rotation period column in data. "
            f"Available columns: {list(df.columns)}."
        )

    # Quality code filtering
    quality_order = {"0": 0, "1": 1, "2": 2, "2+": 3, "3": 4}

    # Preload known asteroid_ids for FK validation
    known_ids = set(
        row[0]
        for row in conn.execute("SELECT asteroid_id FROM asteroids").fetchall()
    )

    count = 0
    skipped = 0

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start: start + batch_size]
        rows = []

        for _, record in batch.iterrows():
            # Resolve asteroid_id
            asteroid_id = None

            # Try IAU number first
            if has_number:
                candidate = _safe_int(record.get(cols["number"]))
                if candidate is not None and candidate in known_ids:
                    asteroid_id = candidate

            # Fallback to designation lookup
            if asteroid_id is None and has_desig:
                desig_val = record.get(cols["designation"])
                if desig_val is not None and not (isinstance(desig_val, float) and pd.isna(desig_val)):
                    asteroid_id = _lookup_designation(conn, str(desig_val).strip())

            if asteroid_id is None:
                skipped += 1
                continue

            # Extract rotation properties
            period = _safe_float(record.get(cols["period"])) if cols["period"] else None
            period_unc = _safe_float(record.get(cols["period_unc"])) if cols["period_unc"] else None
            amplitude = _safe_float(record.get(cols["amplitude"])) if cols["amplitude"] else None
            amplitude_unc = _safe_float(record.get(cols["amplitude_unc"])) if cols["amplitude_unc"] else None

            quality = None
            if cols["quality_code"] is not None:
                qval = record.get(cols["quality_code"])
                if qval is not None:
                    try:
                        if not pd.isna(qval):
                            quality = str(qval).strip()
                    except (TypeError, ValueError):
                        quality = str(qval).strip()

            # Apply quality filter
            if min_quality is not None and quality is not None:
                min_rank = quality_order.get(min_quality, 0)
                cur_rank = quality_order.get(quality, 0)
                if cur_rank < min_rank:
                    skipped += 1
                    continue

            # Skip rows with no period data
            if period is None:
                skipped += 1
                continue

            # Classify spin state
            is_monolithic, is_binary_suspect = classify_spin(period, amplitude)

            rows.append((
                asteroid_id,
                period,
                period_unc,
                amplitude,
                amplitude_unc,
                quality,
                is_monolithic,
                is_binary_suspect,
                "LCDB",
            ))

        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO rotation_properties "
                "(asteroid_id, rotation_period, period_unc, amplitude, "
                "amplitude_unc, quality_code, is_monolithic, is_binary_suspect, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            count += len(rows)

    if skipped:
        logger.info("Skipped %d rows (no match, no period, or below quality threshold)", skipped)
    logger.info("Ingested %d LCDB rows from %s", count, filepath)
    return count
