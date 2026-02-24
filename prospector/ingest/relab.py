"""RELAB PDS4 meteorite spectral library ingestion into lab_spectra table.

Reads paired .tab (spectral data) and .xml (PDS4 label) files from the RELAB
PDS archive (data_reflectance/ directory).  Extracts specimen metadata from XML
labels and spectral data from fixed-width ASCII tables.

RELAB spectra are meteorite/mineral lab samples, NOT asteroid observations.
They join to asteroids conceptually via meteorite type → taxonomy class mapping.

PDS archive structure:
    data_reflectance/
    ├── bdr2/          # Bidirectional reflectance (VNIR)
    ├── bdr3/          # Bidirectional reflectance (updated)
    ├── ftir1/         # FTIR spectra
    └── ftir2/         # FTIR spectra (updated)
    Each subdir contains paired <key>.tab + <key>.xml files.

Usage:
    from prospector.ingest.relab import ingest_relab_dir, ingest_relab_file
    from prospector.db import get_connection

    conn = get_connection()
    count = ingest_relab_dir("data/relab/data_reflectance/bdr2/", conn)
"""

import logging
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from prospector.db import init_schema

logger = logging.getLogger(__name__)

# PDS4 namespaces used in RELAB XML labels
_NS = {
    "pds": "http://pds.nasa.gov/pds4/pds/v1",
    "speclib": "http://pds.nasa.gov/pds4/speclib/v1",
}

# Meteorite specimen_type values that identify meteorite samples
_METEORITE_TYPES = {"Other Meteorite", "Ordinary Chondrite", "Meteorite"}

# Pattern to extract meteorite group from type string (e.g., "CM2" → "CM")
_GROUP_RE = re.compile(r"^([A-Z]{1,3})")


def _parse_tab_file(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a RELAB .tab spectrum file.

    Expected format: fixed-width or whitespace-delimited, 2 columns.
      - Column 1: wavelength (nm)
      - Column 2: reflectance (unitless)

    Returns wavelengths in μm (converted from nm) and reflectance.
    """
    wavelengths = []
    reflectances = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                wl_nm = float(parts[0])
                refl = float(parts[1])
            except (ValueError, IndexError):
                continue
            wavelengths.append(wl_nm / 1000.0)  # nm → μm
            reflectances.append(refl)

    if not wavelengths:
        raise ValueError(f"No valid data rows in {filepath}")

    return (
        np.array(wavelengths, dtype=np.float64),
        np.array(reflectances, dtype=np.float64),
    )


def _parse_xml_label(filepath: Path) -> dict | None:
    """Parse a RELAB PDS4 XML label for specimen metadata.

    Returns a dict with keys:
        sample_id, specimen_name, specimen_type, meteorite_type,
        meteorite_group, sample_desc, grain_size_min, grain_size_max

    Returns None if the XML cannot be parsed.
    """
    try:
        tree = ET.parse(filepath)
    except ET.ParseError:
        logger.warning("Cannot parse XML: %s", filepath)
        return None

    root = tree.getroot()

    def _find(tag: str) -> str | None:
        """Find first matching element text across speclib namespace."""
        el = root.find(f".//{{{_NS['speclib']}}}{tag}")
        return el.text.strip() if el is not None and el.text else None

    def _findall(tag: str) -> list[str]:
        """Find all matching element texts."""
        els = root.findall(f".//{{{_NS['speclib']}}}{tag}")
        return [el.text.strip() for el in els if el is not None and el.text]

    sample_id = _find("specimen_id")
    specimen_name = _find("specimen_name")
    specimen_type = _find("specimen_type")
    sample_desc = _find("specimen_description")

    # rock_subtype can have multiple values: e.g., ["Carbonaceous Chondrite", "CM2"]
    rock_subtypes = _findall("rock_subtype")
    rock_type = _find("rock_type")

    # Extract meteorite type and group from rock_subtype values
    meteorite_type = None
    meteorite_group = None
    if len(rock_subtypes) >= 2:
        # Second value is the specific classification (e.g., "CM2", "H5", "IVA")
        meteorite_type = rock_subtypes[1]
        meteorite_group = rock_subtypes[0]
    elif len(rock_subtypes) == 1:
        meteorite_type = rock_subtypes[0]
        meteorite_group = rock_subtypes[0]
    elif rock_type:
        # Fall back to rock_type (e.g., "Iron" for iron meteorites)
        meteorite_type = rock_type
        meteorite_group = rock_type

    # Grain size from specimen_min_size / specimen_max_size
    grain_min = None
    grain_max = None
    el_min = root.find(f".//{{{_NS['speclib']}}}specimen_min_size")
    el_max = root.find(f".//{{{_NS['speclib']}}}specimen_max_size")
    if el_min is not None and el_min.text:
        try:
            grain_min = float(el_min.text)
        except ValueError:
            pass
    if el_max is not None and el_max.text:
        try:
            grain_max = float(el_max.text)
        except ValueError:
            pass

    return {
        "sample_id": sample_id,
        "specimen_name": specimen_name,
        "specimen_type": specimen_type,
        "meteorite_type": meteorite_type,
        "meteorite_group": meteorite_group,
        "sample_desc": sample_desc,
        "grain_size_min": grain_min,
        "grain_size_max": grain_max,
    }


def is_meteorite(metadata: dict) -> bool:
    """Check if a RELAB sample is a meteorite based on specimen_type."""
    specimen_type = metadata.get("specimen_type") or ""
    return any(mt.lower() in specimen_type.lower() for mt in _METEORITE_TYPES)


def ingest_relab_file(
    tab_path: str | Path,
    conn: sqlite3.Connection,
    *,
    meteorites_only: bool = True,
) -> bool:
    """Ingest a single RELAB spectrum (.tab + .xml pair) into lab_spectra.

    Parameters
    ----------
    tab_path : path to the .tab spectral data file
    conn : database connection
    meteorites_only : if True (default), skip non-meteorite samples

    Returns
    -------
    bool : True if ingested, False if skipped.
    """
    tab_path = Path(tab_path)
    xml_path = tab_path.with_suffix(".xml")

    if not tab_path.exists():
        logger.warning("Tab file not found: %s", tab_path)
        return False

    # Parse XML metadata
    metadata = None
    if xml_path.exists():
        metadata = _parse_xml_label(xml_path)

    if metadata is None:
        metadata = {
            "sample_id": None,
            "specimen_name": None,
            "specimen_type": None,
            "meteorite_type": None,
            "meteorite_group": None,
            "sample_desc": None,
            "grain_size_min": None,
            "grain_size_max": None,
        }

    # Filter for meteorites only
    if meteorites_only and not is_meteorite(metadata):
        logger.debug("Skipping non-meteorite: %s", tab_path.name)
        return False

    # Parse spectral data
    try:
        wavelengths, reflectance = _parse_tab_file(tab_path)
    except ValueError as e:
        logger.warning("Skipping %s: %s", tab_path.name, e)
        return False

    wl_min = float(wavelengths.min())
    wl_max = float(wavelengths.max())

    # spectrum_key = filename stem (unique in the archive)
    spectrum_key = tab_path.stem
    sample_id = metadata["sample_id"] or spectrum_key

    conn.execute(
        "INSERT OR REPLACE INTO lab_spectra "
        "(sample_id, spectrum_key, meteorite_name, meteorite_type, "
        "meteorite_group, sample_desc, grain_size_min, grain_size_max, "
        "wavelengths, reflectance, wl_min, wl_max, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            sample_id,
            spectrum_key,
            metadata["specimen_name"],
            metadata["meteorite_type"],
            metadata["meteorite_group"],
            metadata["sample_desc"],
            metadata["grain_size_min"],
            metadata["grain_size_max"],
            wavelengths.tobytes(),
            reflectance.tobytes(),
            wl_min,
            wl_max,
            "RELAB",
        ),
    )

    return True


def ingest_relab_dir(
    dirpath: str | Path,
    conn: sqlite3.Connection,
    *,
    meteorites_only: bool = True,
    pattern: str = "*.tab",
) -> int:
    """Ingest all RELAB spectra from a directory into lab_spectra.

    Parameters
    ----------
    dirpath : path to a RELAB data directory (e.g., data_reflectance/bdr2/)
    conn : database connection
    meteorites_only : if True (default), only ingest meteorite samples
    pattern : glob pattern for tab files

    Returns
    -------
    int : number of spectra successfully ingested
    """
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"RELAB directory not found: {dirpath}")

    init_schema(conn)

    files = sorted(dirpath.glob(pattern))
    if not files:
        logger.warning("No files matching '%s' in %s", pattern, dirpath)
        return 0

    count = 0
    skipped = 0

    for tab_path in files:
        if ingest_relab_file(tab_path, conn, meteorites_only=meteorites_only):
            count += 1
        else:
            skipped += 1

    conn.commit()

    if skipped:
        logger.info("Skipped %d files (non-meteorite or parse error)", skipped)
    logger.info("Ingested %d RELAB spectra from %s", count, dirpath)
    return count
