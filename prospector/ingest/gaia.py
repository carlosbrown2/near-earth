"""Gaia DR3 asteroid reflectance spectra ingestion into spectra table.

Ingests the 16-band BP/RP reflectance spectra (0.374–1.034 μm) from Gaia DR3
(Tanga et al. 2022).  The Gaia archive table ``gaiadr3.sso_reflectance_spectrum``
stores one row per (asteroid, wavelength_bin) pair — 16 rows per asteroid.

Two ingest paths:
  1. TAP query via astroquery (``ingest_gaia_tap``) — downloads directly
  2. Local file (``ingest_gaia_file``) — loads a pre-downloaded CSV/VOTable

Cross-matching:
  - ``number_mp`` → IAU asteroid number (preferred)
  - ``denomination`` → entity_resolver for unnumbered objects

Usage:
    from prospector.ingest.gaia import ingest_gaia_file, ingest_gaia_tap
    from prospector.db import get_connection

    conn = get_connection()
    count = ingest_gaia_file("data/spectra/gaia/sso_reflectance.csv", conn)
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from prospector.db import init_schema
from prospector.ingest.entity_resolver import resolve

logger = logging.getLogger(__name__)

# ADQL query to download all SSO reflectance spectra from Gaia DR3.
# One row per (source_id, wavelength) — ~968K rows for ~60K asteroids × 16 bands.
GAIA_ADQL = """\
SELECT source_id, number_mp, denomination, nb_samples, num_of_spectra,
       wavelength, reflectance_spectrum, reflectance_spectrum_err,
       reflectance_spectrum_flag
FROM gaiadr3.sso_reflectance_spectrum
ORDER BY source_id, wavelength
"""

# Quality flag values (from Gaia DR3 data model)
FLAG_GOOD = 0
FLAG_POOR = 1
FLAG_COMPROMISED = 2


def _query_gaia_tap(*, row_limit: int = -1) -> pd.DataFrame:
    """Run the ADQL query against the Gaia TAP service.

    Parameters
    ----------
    row_limit : int
        Maximum rows to return.  -1 (default) means no limit.

    Returns
    -------
    pd.DataFrame with columns from GAIA_ADQL.
    """
    from astroquery.gaia import Gaia

    query = GAIA_ADQL
    if row_limit > 0:
        query = query.rstrip().rstrip(";") + f"\nTOP {row_limit}"

    # Large dataset — use async job
    job = Gaia.launch_job_async(query)
    table = job.get_results()
    return table.to_pandas()


def _load_gaia_file(filepath: Path) -> pd.DataFrame:
    """Load Gaia reflectance data from a local file.

    Supports CSV (.csv) and VOTable (.vot, .xml).

    Parameters
    ----------
    filepath : Path to local file.

    Returns
    -------
    pd.DataFrame with at least: source_id, number_mp, denomination,
        wavelength, reflectance_spectrum, reflectance_spectrum_err.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(filepath)
    elif suffix in (".vot", ".xml"):
        from astropy.io.votable import parse_single_table
        table = parse_single_table(str(filepath))
        df = table.to_table().to_pandas()
    else:
        # Try CSV as fallback
        df = pd.read_csv(filepath)

    # Normalize column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def _group_spectra(df: pd.DataFrame) -> dict[int | str, dict]:
    """Group per-wavelength rows into per-asteroid spectra.

    Groups by ``source_id`` and builds sorted wavelength/reflectance/error
    arrays for each asteroid.

    Returns
    -------
    dict mapping source_id → {
        'number_mp': int or NaN,
        'denomination': str or NaN,
        'wavelengths': np.ndarray (μm),
        'reflectance': np.ndarray,
        'uncertainty': np.ndarray or None,
        'num_of_spectra': int,
        'flags': np.ndarray (int8),
    }
    """
    spectra = {}

    for source_id, group in df.groupby("source_id"):
        group = group.sort_values("wavelength")

        # Wavelengths: Gaia stores in nm → convert to μm
        wl_nm = group["wavelength"].values.astype(np.float64)
        wl_um = wl_nm / 1000.0

        refl = group["reflectance_spectrum"].values.astype(np.float64)

        unc = None
        if "reflectance_spectrum_err" in group.columns:
            unc_vals = group["reflectance_spectrum_err"].values
            if not pd.isna(unc_vals).all():
                unc = unc_vals.astype(np.float64)

        flags = None
        if "reflectance_spectrum_flag" in group.columns:
            flags = group["reflectance_spectrum_flag"].values.astype(np.int8)

        # Extract identifiers from first row of this group
        first = group.iloc[0]
        number_mp = first.get("number_mp")
        denomination = first.get("denomination")
        num_of_spectra = first.get("num_of_spectra", 0)

        spectra[source_id] = {
            "number_mp": number_mp,
            "denomination": denomination,
            "wavelengths": wl_um,
            "reflectance": refl,
            "uncertainty": unc,
            "num_of_spectra": int(num_of_spectra) if pd.notna(num_of_spectra) else 0,
            "flags": flags,
        }

    return spectra


def _resolve_asteroid_id(
    number_mp,
    denomination,
    conn: sqlite3.Connection,
    known_ids: set[int],
) -> int | None:
    """Resolve a Gaia object to an asteroid_id.

    Strategy (per PRD §5E):
      1. ``number_mp`` → direct IAU number match
      2. ``denomination`` → entity_resolver lookup

    Returns asteroid_id or None if not in our DB.
    """
    # Strategy 1: numbered asteroid via number_mp
    if pd.notna(number_mp):
        aid = int(number_mp)
        if aid in known_ids:
            return aid
        # number_mp might be valid but asteroid not in our DB
        return None

    # Strategy 2: denomination → entity_resolver
    if pd.notna(denomination):
        denom = str(denomination).strip()
        # Skip Gaia-internal IDs (unidentified bundles)
        if denom.startswith("Gaia-DR3SSO"):
            return None
        result = resolve(denom, conn)
        if result is not None and result.asteroid_id is not None:
            if result.asteroid_id in known_ids:
                return result.asteroid_id
        return None

    return None


def _has_good_bands(flags: np.ndarray | None) -> bool:
    """Check if a spectrum has any non-compromised bands."""
    if flags is None:
        return True  # No flag data → assume usable
    return bool(np.any(flags < FLAG_COMPROMISED))


def ingest_gaia_file(
    filepath: str | Path,
    conn: sqlite3.Connection,
    *,
    skip_compromised: bool = True,
) -> int:
    """Ingest Gaia DR3 reflectance spectra from a local file.

    Parameters
    ----------
    filepath : path to CSV or VOTable with Gaia SSO reflectance data.
    conn : database connection (schema must already exist).
    skip_compromised : if True, skip spectra where ALL bands are flagged
        as compromised (flag == 2).

    Returns
    -------
    int : number of spectra ingested.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Gaia data file not found: {filepath}")

    df = _load_gaia_file(filepath)
    return _ingest_dataframe(df, conn, skip_compromised=skip_compromised)


def ingest_gaia_tap(
    conn: sqlite3.Connection,
    *,
    row_limit: int = -1,
    skip_compromised: bool = True,
) -> int:
    """Ingest Gaia DR3 reflectance spectra via TAP query.

    Parameters
    ----------
    conn : database connection (schema must already exist).
    row_limit : max rows from TAP query (-1 = all).
    skip_compromised : skip all-compromised spectra.

    Returns
    -------
    int : number of spectra ingested.
    """
    df = _query_gaia_tap(row_limit=row_limit)
    return _ingest_dataframe(df, conn, skip_compromised=skip_compromised)


def _ingest_dataframe(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    *,
    skip_compromised: bool = True,
) -> int:
    """Core ingestion: group spectra, resolve IDs, insert into spectra table.

    Parameters
    ----------
    df : DataFrame with per-wavelength rows from Gaia.
    conn : database connection.
    skip_compromised : skip spectra with all bands compromised.

    Returns
    -------
    int : number of spectra ingested.
    """
    init_schema(conn)

    # Preload known asteroid_ids for FK validation
    known_ids = set(
        row[0] for row in conn.execute("SELECT asteroid_id FROM asteroids").fetchall()
    )

    if not known_ids:
        logger.warning("No asteroids in DB — Gaia spectra cannot be cross-matched")
        return 0

    # Group per-wavelength rows into per-asteroid spectra
    spectra = _group_spectra(df)
    logger.info("Grouped %d Gaia sources from %d rows", len(spectra), len(df))

    count = 0
    skipped_fk = 0
    skipped_quality = 0

    for source_id, spec in spectra.items():
        # Quality gate: skip all-compromised spectra
        if skip_compromised and not _has_good_bands(spec["flags"]):
            skipped_quality += 1
            continue

        # Resolve to asteroid_id
        asteroid_id = _resolve_asteroid_id(
            spec["number_mp"], spec["denomination"], conn, known_ids
        )
        if asteroid_id is None:
            skipped_fk += 1
            continue

        # Build BLOBs
        wl_blob = spec["wavelengths"].tobytes()
        refl_blob = spec["reflectance"].tobytes()
        unc_blob = spec["uncertainty"].tobytes() if spec["uncertainty"] is not None else None

        wl_min = float(spec["wavelengths"].min())
        wl_max = float(spec["wavelengths"].max())

        conn.execute(
            "INSERT INTO spectra "
            "(asteroid_id, survey, wavelengths, reflectance, uncertainty, wl_min, wl_max) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (asteroid_id, "Gaia", wl_blob, refl_blob, unc_blob, wl_min, wl_max),
        )
        count += 1

    conn.commit()

    if skipped_fk:
        logger.info("Skipped %d Gaia sources (not in asteroids table)", skipped_fk)
    if skipped_quality:
        logger.info("Skipped %d Gaia sources (all bands compromised)", skipped_quality)
    logger.info("Ingested %d Gaia DR3 spectra", count)

    return count
