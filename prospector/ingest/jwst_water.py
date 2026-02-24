"""JWST 6μm molecular water detection ingestion.

Ingests curated data on unambiguous molecular water detections from the
6μm H-O-H bending mode emission feature observed by JWST/MIRI and
SOFIA/FORCAST. Unlike the 3μm band (which can be OH, organics, or
ammonium), the 6μm feature is exclusively molecular water.

Reference: Arredondo et al. (2024) — first detection on S-type asteroids
(Iris, Massalia) at ~450 μg/g.

The data is sparse (currently ~4 asteroids, expanding to 30+ in JWST
Cycle 2+) and comes from individual publications, so ingestion is from
curated CSV or JSON files.

Usage:
    from prospector.ingest.jwst_water import ingest_jwst_water
    from prospector.db import get_connection

    conn = get_connection()
    count = ingest_jwst_water("data/jwst/water_detections.csv", conn)
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from prospector.db import init_schema

logger = logging.getLogger(__name__)

# Expected CSV columns and their aliases
DEFAULT_COLUMN_MAP = {
    "number": "number",
    "designation": "designation",
    "water_abundance": "water_abundance",
    "water_abundance_unc": "water_abundance_unc",
    "band_depth": "band_depth",
    "band_depth_unc": "band_depth_unc",
    "detection": "detection",
    "instrument": "instrument",
    "reference": "reference",
}

_COLUMN_ALIASES = {
    "number": ["number", "num", "asteroid_number", "iau_number"],
    "designation": ["designation", "desig", "name"],
    "water_abundance": ["water_abundance", "abundance", "h2o_ug_g", "water_ppm"],
    "water_abundance_unc": ["water_abundance_unc", "abundance_unc", "h2o_unc"],
    "band_depth": ["band_depth", "depth", "feature_depth", "band_6um_depth"],
    "band_depth_unc": ["band_depth_unc", "depth_unc"],
    "detection": ["detection", "detected", "confirmed", "water_detected"],
    "instrument": ["instrument", "observatory", "facility"],
    "reference": ["reference", "ref", "bibref", "citation"],
}


def _safe_float(val: Any) -> float | None:
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


def _safe_int(val: Any) -> int | None:
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


def _parse_bool(val: Any) -> bool | None:
    """Parse a boolean-like value. Returns True/False or None."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        lower = val.strip().lower()
        if lower in ("true", "yes", "1", "y", "confirmed"):
            return True
        if lower in ("false", "no", "0", "n", "unconfirmed"):
            return False
    return None


def _resolve_columns(df_columns: list[str], column_map: dict[str, str]) -> dict[str, str | None]:
    """Resolve logical column names to actual DataFrame column names."""
    df_cols_lower = {c.lower().strip(): c for c in df_columns}
    resolved: dict[str, str | None] = {}

    for logical_name, mapped_name in column_map.items():
        if mapped_name.lower().strip() in df_cols_lower:
            resolved[logical_name] = df_cols_lower[mapped_name.lower().strip()]
            continue

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


def ingest_jwst_water(
    filepath: str | Path,
    conn: sqlite3.Connection,
    *,
    column_map: dict[str, str] | None = None,
) -> int:
    """Ingest JWST 6μm water detection data from a curated CSV file.

    Expected CSV columns (flexible naming via aliases):
    - number or designation: asteroid identifier
    - detection: boolean (True/False) — water confirmed at 6μm
    - water_abundance: optional, μg/g (ppm by mass)
    - band_depth: optional, fractional 6μm feature depth
    - instrument: optional, e.g. 'JWST/MIRI'
    - reference: optional, e.g. 'Arredondo+2024'

    Parameters
    ----------
    filepath : path to CSV file
    conn : database connection
    column_map : optional column name mapping override

    Returns
    -------
    int : number of rows ingested
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JWST water data file not found: {filepath}")

    init_schema(conn)

    df = pd.read_csv(filepath, low_memory=False)
    cmap = column_map if column_map is not None else DEFAULT_COLUMN_MAP
    cols = _resolve_columns(list(df.columns), cmap)

    has_number = cols.get("number") is not None
    has_desig = cols.get("designation") is not None
    has_detection = cols.get("detection") is not None

    if not has_number and not has_desig:
        raise ValueError(
            f"Cannot find number or designation column. "
            f"Available columns: {list(df.columns)}"
        )

    if not has_detection:
        raise ValueError(
            f"Cannot find detection column (required). "
            f"Available columns: {list(df.columns)}"
        )

    known_ids = set(
        row[0] for row in conn.execute("SELECT asteroid_id FROM asteroids").fetchall()
    )

    count = 0
    skipped = 0

    for _, record in df.iterrows():
        asteroid_id = None

        if has_number:
            candidate = _safe_int(record.get(cols["number"]))
            if candidate is not None and candidate in known_ids:
                asteroid_id = candidate

        if asteroid_id is None and has_desig:
            desig_val = record.get(cols["designation"])
            if desig_val is not None and not (isinstance(desig_val, float) and pd.isna(desig_val)):
                asteroid_id = _lookup_designation(conn, str(desig_val).strip())

        if asteroid_id is None:
            skipped += 1
            continue

        detection = _parse_bool(record.get(cols["detection"]))
        if detection is None:
            skipped += 1
            continue

        water_abundance = _safe_float(
            record.get(cols["water_abundance"]) if cols.get("water_abundance") else None
        )
        water_abundance_unc = _safe_float(
            record.get(cols["water_abundance_unc"]) if cols.get("water_abundance_unc") else None
        )
        band_depth = _safe_float(
            record.get(cols["band_depth"]) if cols.get("band_depth") else None
        )
        band_depth_unc = _safe_float(
            record.get(cols["band_depth_unc"]) if cols.get("band_depth_unc") else None
        )
        instrument = None
        if cols.get("instrument"):
            val = record.get(cols["instrument"])
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                instrument = str(val).strip()
        reference = None
        if cols.get("reference"):
            val = record.get(cols["reference"])
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                reference = str(val).strip()

        conn.execute(
            "INSERT OR REPLACE INTO jwst_water "
            "(asteroid_id, water_abundance, water_abundance_unc, band_depth, "
            "band_depth_unc, detection, instrument, reference, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'JWST')",
            (
                asteroid_id,
                water_abundance,
                water_abundance_unc,
                band_depth,
                band_depth_unc,
                detection,
                instrument,
                reference,
            ),
        )
        count += 1

    conn.commit()

    if skipped:
        logger.info("Skipped %d rows (no match or missing detection)", skipped)
    logger.info("Ingested %d JWST water detection rows from %s", count, filepath)
    return count


def ingest_jwst_water_json(
    filepath: str | Path,
    conn: sqlite3.Connection,
) -> int:
    """Ingest JWST 6μm water detections from a JSON file.

    JSON format: list of objects with keys matching CSV columns.

    Parameters
    ----------
    filepath : path to JSON file
    conn : database connection

    Returns
    -------
    int : number of rows ingested
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JWST water JSON file not found: {filepath}")

    init_schema(conn)

    with open(filepath) as f:
        records = json.load(f)

    if not isinstance(records, list):
        records = [records]

    known_ids = set(
        row[0] for row in conn.execute("SELECT asteroid_id FROM asteroids").fetchall()
    )

    count = 0
    for record in records:
        asteroid_id = record.get("asteroid_id") or record.get("number")
        if asteroid_id is not None:
            asteroid_id = int(asteroid_id)
        else:
            desig = record.get("designation")
            if desig:
                asteroid_id = _lookup_designation(conn, str(desig))

        if asteroid_id is None or asteroid_id not in known_ids:
            continue

        detection = _parse_bool(record.get("detection"))
        if detection is None:
            continue

        conn.execute(
            "INSERT OR REPLACE INTO jwst_water "
            "(asteroid_id, water_abundance, water_abundance_unc, band_depth, "
            "band_depth_unc, detection, instrument, reference, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'JWST')",
            (
                asteroid_id,
                _safe_float(record.get("water_abundance")),
                _safe_float(record.get("water_abundance_unc")),
                _safe_float(record.get("band_depth")),
                _safe_float(record.get("band_depth_unc")),
                detection,
                record.get("instrument"),
                record.get("reference"),
            ),
        )
        count += 1

    conn.commit()
    logger.info("Ingested %d JWST water detection rows from %s", count, filepath)
    return count
