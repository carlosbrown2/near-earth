"""MITHNEOS ASCII spectra ingestion into spectra table.

Reads two-column (or three-column) ASCII text files from the MITHNEOS survey
(wavelength in μm, relative reflectance, optional uncertainty) and inserts
them into the spectra table.  Filenames encode the asteroid identity
(e.g. a004179.sp05.txt → number 4179), which is resolved via entity_resolver.

Usage:
    from prospector.ingest.mithneos import ingest_mithneos_dir, ingest_mithneos_file
    from prospector.db import get_connection

    conn = get_connection()
    count = ingest_mithneos_dir("data/spectra/mithneos/", conn)
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np

from prospector.db import init_schema
from prospector.ingest.entity_resolver import parse_mithneos_filename, resolve

logger = logging.getLogger(__name__)


def _read_spectrum_file(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read a MITHNEOS ASCII spectrum file.

    Expected format: whitespace-delimited columns.
      - Column 1: wavelength (μm)
      - Column 2: relative reflectance
      - Column 3 (optional): uncertainty

    Blank lines and lines starting with '#' are skipped.

    Returns
    -------
    wavelengths, reflectance, uncertainty (or None if only 2 columns)

    Raises
    ------
    ValueError
        If the file has fewer than 2 columns or no valid data rows.
    """
    wavelengths = []
    reflectances = []
    uncertainties = []
    has_uncertainty = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                wl = float(parts[0])
                refl = float(parts[1])
            except (ValueError, IndexError):
                continue

            wavelengths.append(wl)
            reflectances.append(refl)

            if has_uncertainty is None:
                has_uncertainty = len(parts) >= 3
            if has_uncertainty and len(parts) >= 3:
                try:
                    uncertainties.append(float(parts[2]))
                except ValueError:
                    uncertainties.append(np.nan)

    if not wavelengths:
        raise ValueError(f"No valid data rows in {filepath}")

    wl_arr = np.array(wavelengths, dtype=np.float64)
    refl_arr = np.array(reflectances, dtype=np.float64)
    unc_arr = np.array(uncertainties, dtype=np.float64) if has_uncertainty and uncertainties else None

    return wl_arr, refl_arr, unc_arr


def ingest_mithneos_file(
    filepath: str | Path,
    conn: sqlite3.Connection,
    *,
    known_ids: set[int] | None = None,
) -> bool:
    """Ingest a single MITHNEOS spectrum file into the spectra table.

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

    # Parse asteroid identity from filename
    identity = parse_mithneos_filename(filepath.name)
    if identity is None:
        logger.warning("Cannot parse MITHNEOS filename: %s", filepath.name)
        return False

    # Resolve to asteroid_id
    if isinstance(identity, int):
        asteroid_id = identity
    else:
        # Unnumbered object — resolve via entity_resolver
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

    # Read the spectrum
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
        (asteroid_id, "MITHNEOS", wl_blob, refl_blob, unc_blob, wl_min, wl_max),
    )

    return True


def ingest_mithneos_dir(
    dirpath: str | Path,
    conn: sqlite3.Connection,
    *,
    pattern: str = "*.txt",
) -> int:
    """Ingest all MITHNEOS spectra from a directory into the spectra table.

    Parameters
    ----------
    dirpath : path to directory containing MITHNEOS ASCII spectrum files
    conn : database connection
    pattern : glob pattern for spectrum files (default: ``*.txt``)

    Returns
    -------
    int : number of spectra successfully ingested
    """
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"MITHNEOS directory not found: {dirpath}")

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
        if ingest_mithneos_file(filepath, conn, known_ids=known_ids):
            count += 1
        else:
            skipped += 1

    conn.commit()

    if skipped:
        logger.info("Skipped %d files (no match or parse error)", skipped)
    logger.info("Ingested %d MITHNEOS spectra from %s", count, dirpath)
    return count
