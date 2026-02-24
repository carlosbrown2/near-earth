"""SMASS II visible spectra ingestion into spectra table.

Reads three-column ASCII text files (wavelength in μm, reflectance, uncertainty)
from the SMASS II survey (Bus & Binzel 2002) — 1,341 asteroids at 0.44–0.92 μm.
Supplements Gaia DR3 for objects not in that release.

File format: whitespace-delimited ASCII, same structure as MITHNEOS.
Naming convention: similar to MITHNEOS (e.g., a004179.sp01.txt for numbered).

Usage:
    from prospector.ingest.smass import ingest_smass_dir, ingest_smass_file
    from prospector.db import get_connection

    conn = get_connection()
    count = ingest_smass_dir("data/spectra/smass/", conn)
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np

from prospector.db import init_schema
from prospector.ingest.entity_resolver import parse_mithneos_filename, resolve
from prospector.ingest.mithneos import _read_spectrum_file

logger = logging.getLogger(__name__)


def ingest_smass_file(
    filepath: str | Path,
    conn: sqlite3.Connection,
    *,
    known_ids: set[int] | None = None,
) -> bool:
    """Ingest a single SMASS II spectrum file into the spectra table.

    Parameters
    ----------
    filepath : path to the ASCII spectrum file
    conn : database connection
    known_ids : optional pre-loaded set of asteroid_ids for FK validation.
        If None, a DB lookup is performed for each file.

    Returns
    -------
    bool : True if the spectrum was ingested, False if skipped.
    """
    filepath = Path(filepath)

    # Parse asteroid identity from filename (same convention as MITHNEOS)
    identity = parse_mithneos_filename(filepath.name)
    if identity is None:
        logger.warning("Cannot parse SMASS filename: %s", filepath.name)
        return False

    # Resolve to asteroid_id
    if isinstance(identity, int):
        asteroid_id = identity
    else:
        result = resolve(str(identity), conn)
        if result is None or result.asteroid_id is None:
            logger.debug("Could not resolve designation %s from %s", identity, filepath.name)
            return False
        asteroid_id = result.asteroid_id

    # FK validation: check asteroid exists in DB
    if known_ids is not None:
        if asteroid_id not in known_ids:
            logger.debug("Asteroid %d not in DB, skipping %s", asteroid_id, filepath.name)
            return False
    else:
        row = conn.execute(
            "SELECT 1 FROM asteroids WHERE asteroid_id = ?", (asteroid_id,)
        ).fetchone()
        if row is None:
            logger.debug("Asteroid %d not in DB, skipping %s", asteroid_id, filepath.name)
            return False

    # Read the spectrum (reuses MITHNEOS reader — same ASCII format)
    try:
        wavelengths, reflectance, uncertainty = _read_spectrum_file(filepath)
    except ValueError as e:
        logger.warning("Skipping %s: %s", filepath.name, e)
        return False

    wl_min = float(wavelengths.min())
    wl_max = float(wavelengths.max())

    # Store as BLOBs
    wl_blob = wavelengths.tobytes()
    refl_blob = reflectance.tobytes()
    unc_blob = uncertainty.tobytes() if uncertainty is not None else None

    conn.execute(
        "INSERT INTO spectra "
        "(asteroid_id, survey, wavelengths, reflectance, uncertainty, wl_min, wl_max) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (asteroid_id, "SMASS", wl_blob, refl_blob, unc_blob, wl_min, wl_max),
    )

    return True


def ingest_smass_dir(
    dirpath: str | Path,
    conn: sqlite3.Connection,
    *,
    pattern: str = "*.txt",
) -> int:
    """Ingest all SMASS II spectra from a directory into the spectra table.

    Parameters
    ----------
    dirpath : path to directory containing SMASS II ASCII spectrum files
    conn : database connection
    pattern : glob pattern for spectrum files (default: ``*.txt``)

    Returns
    -------
    int : number of spectra successfully ingested
    """
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"SMASS directory not found: {dirpath}")

    init_schema(conn)

    # Preload known asteroid_ids for FK validation
    known_ids = set(
        row[0] for row in conn.execute("SELECT asteroid_id FROM asteroids").fetchall()
    )

    files = sorted(dirpath.glob(pattern))
    if not files:
        logger.warning("No files matching '%s' in %s", pattern, dirpath)
        return 0

    count = 0
    skipped = 0

    for filepath in files:
        if ingest_smass_file(filepath, conn, known_ids=known_ids):
            count += 1
        else:
            skipped += 1

    conn.commit()

    if skipped:
        logger.info("Skipped %d files (no match or parse error)", skipped)
    logger.info("Ingested %d SMASS II spectra from %s", count, dirpath)
    return count
