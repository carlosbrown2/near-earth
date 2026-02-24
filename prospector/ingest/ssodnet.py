"""SsODNet (Solar System Open Database Network) supplementary data ingestion.

Queries the IMCCE SsODNet ssoCard REST API to retrieve best-estimate physical
and dynamical properties: mass, density, thermal inertia, taxonomy, and delta-v.
These supplement existing SBDB/NEOWISE data with aggregated literature values.

API reference: https://ssp.imcce.fr/webservices/ssodnet/api/ssocard/
Data paper: Berthier et al. (2023), A&A 671, A151

Usage:
    from prospector.ingest.ssodnet import ingest_ssodnet, ingest_ssodnet_json
    from prospector.db import get_connection

    conn = get_connection()
    # From live API (rate-limited):
    count = ingest_ssodnet([433, 4179, 101955], conn)
    # From pre-downloaded JSON files:
    count = ingest_ssodnet_json("data/ssodnet/433.json", conn)
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from prospector.db import init_schema

logger = logging.getLogger(__name__)

SSOCARD_URL = "https://ssp.imcce.fr/webservices/ssodnet/api/ssocard.php"
REQUEST_DELAY_S = 0.5  # polite rate limiting between API calls
REQUEST_TIMEOUT_S = 30


def _safe_float(val) -> float | None:
    """Extract float from a value or dict with 'value' key."""
    if val is None:
        return None
    if isinstance(val, dict):
        val = val.get("value")
    if val is None:
        return None
    try:
        f = float(val)
        return None if f != f else f  # NaN check
    except (ValueError, TypeError):
        return None


def _extract_error(obj: dict) -> float | None:
    """Extract symmetric uncertainty from SsODNet error dict.

    SsODNet uses {min: -X, max: X} format. We take the average absolute value.
    """
    if not isinstance(obj, dict):
        return None
    err = obj.get("error")
    if not isinstance(err, dict):
        return None
    lo = _safe_float(err.get("min"))
    hi = _safe_float(err.get("max"))
    if lo is not None and hi is not None:
        return (abs(lo) + abs(hi)) / 2.0
    if hi is not None:
        return abs(hi)
    if lo is not None:
        return abs(lo)
    return None


def fetch_ssocard(identifier: int | str, *, timeout: int = REQUEST_TIMEOUT_S) -> dict | None:
    """Fetch a single SsODNet ssoCard via REST API.

    Parameters
    ----------
    identifier : asteroid number (int) or name/designation (str)
    timeout : request timeout in seconds

    Returns
    -------
    dict or None : parsed JSON response, or None on failure
    """
    url = f"{SSOCARD_URL}?q={identifier}"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, json.JSONDecodeError, OSError) as exc:
        logger.warning("SsODNet fetch failed for %s: %s", identifier, exc)
        return None


def parse_ssocard(data: dict) -> dict:
    """Parse an SsODNet ssoCard JSON into a flat property dict.

    Returns a dict with keys matching ssodnet_properties table columns.
    Values are None when the property is absent from the card.
    """
    result = {
        "mass_kg": None,
        "mass_unc": None,
        "density_kgm3": None,
        "density_unc": None,
        "thermal_inertia": None,
        "thermal_inertia_unc": None,
        "taxonomy_class": None,
        "taxonomy_scheme": None,
        "taxonomy_complex": None,
        "taxonomy_waverange": None,
        "delta_v_km_s": None,
    }

    params = data.get("parameters", {})
    phys = params.get("physical", {})
    dyn = params.get("dynamical", {})

    # Mass (kg)
    mass = phys.get("mass", {})
    if isinstance(mass, dict):
        result["mass_kg"] = _safe_float(mass)
        result["mass_unc"] = _extract_error(mass)

    # Density (kg/m³)
    density = phys.get("density", {})
    if isinstance(density, dict):
        result["density_kgm3"] = _safe_float(density)
        result["density_unc"] = _extract_error(density)

    # Thermal inertia (J/m²/s^0.5/K)
    ti = phys.get("thermal_inertia", {})
    if isinstance(ti, dict):
        result["thermal_inertia"] = _safe_float(ti)
        result["thermal_inertia_unc"] = _extract_error(ti)

    # Taxonomy
    tax = phys.get("taxonomy", {})
    if isinstance(tax, dict):
        result["taxonomy_class"] = tax.get("class")
        result["taxonomy_scheme"] = tax.get("scheme")
        result["taxonomy_complex"] = tax.get("complex")
        result["taxonomy_waverange"] = tax.get("waverange")

    # Delta-v (km/s)
    dv = dyn.get("delta_v", {})
    if isinstance(dv, dict):
        result["delta_v_km_s"] = _safe_float(dv.get("delta_v"))

    return result


def _has_any_data(props: dict) -> bool:
    """Check if parsed properties contain any non-None values."""
    return any(v is not None for v in props.values())


def _insert_row(conn: sqlite3.Connection, asteroid_id: int, props: dict) -> None:
    """Insert or replace a single ssodnet_properties row."""
    conn.execute(
        "INSERT OR REPLACE INTO ssodnet_properties "
        "(asteroid_id, mass_kg, mass_unc, density_kgm3, density_unc, "
        "thermal_inertia, thermal_inertia_unc, taxonomy_class, taxonomy_scheme, "
        "taxonomy_complex, taxonomy_waverange, delta_v_km_s, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'SsODNet')",
        (
            asteroid_id,
            props["mass_kg"],
            props["mass_unc"],
            props["density_kgm3"],
            props["density_unc"],
            props["thermal_inertia"],
            props["thermal_inertia_unc"],
            props["taxonomy_class"],
            props["taxonomy_scheme"],
            props["taxonomy_complex"],
            props["taxonomy_waverange"],
            props["delta_v_km_s"],
        ),
    )


def ingest_ssodnet(
    asteroid_ids: list[int],
    conn: sqlite3.Connection,
    *,
    delay: float = REQUEST_DELAY_S,
) -> int:
    """Fetch and ingest SsODNet data for a list of asteroid IDs.

    Queries the SsODNet ssoCard API for each asteroid and stores
    supplementary physical properties. Only inserts rows for asteroids
    that exist in the asteroids table (FK safety).

    Parameters
    ----------
    asteroid_ids : list of IAU asteroid numbers to query
    conn : database connection
    delay : seconds between API requests (rate limiting)

    Returns
    -------
    int : number of rows ingested
    """
    init_schema(conn)

    known_ids = set(
        row[0] for row in conn.execute("SELECT asteroid_id FROM asteroids").fetchall()
    )

    count = 0
    for i, aid in enumerate(asteroid_ids):
        if aid not in known_ids:
            logger.debug("Skipping %d — not in asteroids table", aid)
            continue

        data = fetch_ssocard(aid)
        if data is None:
            continue

        props = parse_ssocard(data)
        if not _has_any_data(props):
            logger.debug("No useful data for asteroid %d", aid)
            continue

        _insert_row(conn, aid, props)
        count += 1

        # Rate limit (skip delay after last request)
        if delay > 0 and i < len(asteroid_ids) - 1:
            time.sleep(delay)

    conn.commit()
    logger.info("Ingested %d SsODNet rows for %d requested asteroids", count, len(asteroid_ids))
    return count


def ingest_ssodnet_json(
    filepath: str | Path,
    conn: sqlite3.Connection,
    *,
    asteroid_id: int | None = None,
) -> int:
    """Ingest SsODNet data from a pre-downloaded JSON file.

    Supports both single ssoCard JSON and a list of ssoCards.
    The asteroid_id is resolved from the JSON 'number' field,
    or can be explicitly provided.

    Parameters
    ----------
    filepath : path to JSON file
    conn : database connection
    asteroid_id : explicit asteroid ID (overrides JSON 'number' field)

    Returns
    -------
    int : number of rows ingested
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"SsODNet JSON file not found: {filepath}")

    init_schema(conn)

    known_ids = set(
        row[0] for row in conn.execute("SELECT asteroid_id FROM asteroids").fetchall()
    )

    with open(filepath) as f:
        raw = json.load(f)

    # Handle both single card and list of cards
    cards = raw if isinstance(raw, list) else [raw]

    count = 0
    for card in cards:
        aid = asteroid_id if asteroid_id is not None else card.get("number")
        if aid is None:
            logger.warning("No asteroid number in JSON card, skipping")
            continue

        aid = int(aid)
        if aid not in known_ids:
            logger.debug("Skipping %d — not in asteroids table", aid)
            continue

        props = parse_ssocard(card)
        if not _has_any_data(props):
            continue

        _insert_row(conn, aid, props)
        count += 1

    conn.commit()
    logger.info("Ingested %d SsODNet rows from %s", count, filepath)
    return count
