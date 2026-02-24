"""NEOWISE PDS SBN diameter/albedo ingestion into physical_properties table.

Reads the pre-computed NEOWISE diameter and albedo table (Mainzer et al.)
from PDS SBN and populates the physical_properties table.  Supports both
IPAC pipe-delimited table format (.tab) and CSV (.csv).

Cross-matching to existing asteroids is by IAU number (preferred) or
provisional designation (fallback lookup in the asteroids table).

Usage:
    from prospector.ingest.neowise import ingest_neowise
    from prospector.db import get_connection

    conn = get_connection()
    count = ingest_neowise("data/neowise/neowise_diameters_albedos.tab", conn)
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from prospector.db import init_schema

logger = logging.getLogger(__name__)

# Default column name mapping from PDS SBN NEOWISE table to our schema.
# Users can override this if their file uses different column names.
DEFAULT_COLUMN_MAP = {
    "number": "number",          # IAU asteroid number (int)
    "designation": "desig",      # provisional MPC designation
    "diameter_km": "diameter",   # diameter in km
    "diameter_err": "diameter_err",
    "albedo_pv": "pv",          # visible geometric albedo
    "albedo_pv_err": "pv_err",
    "beaming_eta": "beaming",   # NEATM beaming parameter
}

# Alternative column names commonly found in NEOWISE data products
_COLUMN_ALIASES = {
    "number": ["number", "num", "ast_number", "iau_number"],
    "designation": ["desig", "prov_desig", "designation", "name"],
    "diameter_km": ["diameter", "diam_km", "d", "diam", "wmean_d"],
    "diameter_err": ["diameter_err", "diam_err", "d_err", "d_sig", "diameter_sigma"],
    "albedo_pv": ["pv", "albedo", "albedo_pv", "p_v", "wmean_pv"],
    "albedo_pv_err": ["pv_err", "albedo_err", "albedo_pv_err", "pv_sig", "p_v_err"],
    "beaming_eta": ["beaming", "eta", "beaming_eta", "wmean_eta"],
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
    Returns a dict mapping logical names to actual column names (or None if not found).
    """
    df_cols_lower = {c.lower().strip(): c for c in df_columns}
    resolved = {}

    for logical_name, mapped_name in column_map.items():
        # Try the explicitly mapped name first
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


def _read_ipac_table(filepath: Path) -> pd.DataFrame:
    """Read an IPAC pipe-delimited table into a pandas DataFrame.

    Uses astropy.io.ascii for robust IPAC format parsing.
    """
    from astropy.io import ascii as astropy_ascii

    table = astropy_ascii.read(str(filepath), format="ipac")
    return table.to_pandas()


def _read_data_file(filepath: Path) -> pd.DataFrame:
    """Read a NEOWISE data file, auto-detecting format from extension.

    Supports .tab (IPAC format) and .csv (comma-separated).
    """
    suffix = filepath.suffix.lower()

    if suffix in (".tab", ".tbl"):
        return _read_ipac_table(filepath)
    elif suffix == ".csv":
        return pd.read_csv(filepath, low_memory=False)
    else:
        # Try CSV first, fall back to IPAC
        try:
            return pd.read_csv(filepath, low_memory=False)
        except Exception:
            return _read_ipac_table(filepath)


def _lookup_designation(conn: sqlite3.Connection, designation: str) -> int | None:
    """Look up asteroid_id by designation in the asteroids table.

    Returns the asteroid_id if found, None otherwise.
    """
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


def ingest_neowise(
    filepath: str | Path,
    conn: sqlite3.Connection,
    *,
    column_map: dict[str, str] | None = None,
    batch_size: int = 5000,
) -> int:
    """Ingest NEOWISE diameter/albedo data into physical_properties table.

    Parameters
    ----------
    filepath : path to the NEOWISE data file (.tab IPAC format or .csv)
    conn : database connection (use get_connection() or get_connection(":memory:"))
    column_map : optional override for column name mapping
    batch_size : rows per INSERT batch

    Returns
    -------
    int : number of rows ingested
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NEOWISE data file not found: {filepath}")

    init_schema(conn)

    # Read the data file
    df = _read_data_file(filepath)

    # Resolve column names
    cmap = column_map if column_map is not None else DEFAULT_COLUMN_MAP
    cols = _resolve_columns(list(df.columns), cmap)

    # We need at least one of: number or designation for matching
    has_number = cols["number"] is not None
    has_desig = cols["designation"] is not None
    if not has_number and not has_desig:
        raise ValueError(
            f"Cannot find number or designation column in data. "
            f"Available columns: {list(df.columns)}. "
            f"Provide a column_map to specify the mapping."
        )

    # We need at least one physical property column
    property_cols = ["diameter_km", "albedo_pv", "beaming_eta"]
    if not any(cols.get(c) for c in property_cols):
        raise ValueError(
            f"Cannot find any physical property columns (diameter, albedo, beaming) "
            f"in data. Available columns: {list(df.columns)}."
        )

    # Preload known asteroid_ids for FK validation
    known_ids = set(
        row[0]
        for row in conn.execute("SELECT asteroid_id FROM asteroids").fetchall()
    )

    count = 0
    skipped = 0

    # Process in batches
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        rows = []

        for _, record in batch.iterrows():
            # Resolve asteroid_id
            asteroid_id = None

            # Try IAU number first
            if has_number:
                candidate = _safe_int(record.get(cols["number"]))
                if candidate is not None and candidate in known_ids:
                    asteroid_id = candidate

            # Fallback to designation lookup (already validates existence)
            if asteroid_id is None and has_desig:
                desig_val = record.get(cols["designation"])
                if desig_val is not None and not (isinstance(desig_val, float) and pd.isna(desig_val)):
                    asteroid_id = _lookup_designation(conn, str(desig_val).strip())

            if asteroid_id is None:
                skipped += 1
                continue

            # Extract physical properties
            diameter_km = _safe_float(record.get(cols["diameter_km"])) if cols["diameter_km"] else None
            diameter_err = _safe_float(record.get(cols["diameter_err"])) if cols["diameter_err"] else None
            albedo_pv = _safe_float(record.get(cols["albedo_pv"])) if cols["albedo_pv"] else None
            albedo_pv_err = _safe_float(record.get(cols["albedo_pv_err"])) if cols["albedo_pv_err"] else None
            beaming_eta = _safe_float(record.get(cols["beaming_eta"])) if cols["beaming_eta"] else None

            # Skip rows with no useful data
            if diameter_km is None and albedo_pv is None and beaming_eta is None:
                skipped += 1
                continue

            rows.append((
                asteroid_id,
                albedo_pv,
                albedo_pv_err,
                diameter_km,
                diameter_err,
                beaming_eta,
                "NEOWISE",
            ))

        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO physical_properties "
                "(asteroid_id, albedo_pv, albedo_pv_err, diameter_km, diameter_err, beaming_eta, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            count += len(rows)

    if skipped:
        logger.info("Skipped %d rows (no matching asteroid or no useful data)", skipped)
    logger.info("Ingested %d NEOWISE rows from %s", count, filepath)
    return count
