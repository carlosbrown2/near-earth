"""Unified database schema for the Asteroid Mining Prospector.

Defines the shared SQLite schema (asteroids, orbits, physical_properties,
spectra, taxonomy, band_analysis, scores) and provides connection management.
All ingest modules write to this schema.

BLOB columns store numpy arrays via array.tobytes() / np.frombuffer().
"""

import sqlite3
from pathlib import Path

# Default database path (data/ directory, gitignored)
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "prospector.db"

SCHEMA_SQL = """\
-- Core identity table. All other tables join here.
CREATE TABLE IF NOT EXISTS asteroids (
    asteroid_id    INTEGER PRIMARY KEY,  -- IAU number (canonical key)
    designation    TEXT,                  -- packed MPC designation (for unnumbered objects)
    name           TEXT,                  -- common name (Itokawa, Bennu, etc.)
    full_name      TEXT,                  -- SBDB full_name field
    neo            BOOLEAN,
    pha            BOOLEAN
);

-- Orbital elements from SBDB
CREATE TABLE IF NOT EXISTS orbits (
    asteroid_id    INTEGER PRIMARY KEY REFERENCES asteroids(asteroid_id),
    epoch          REAL,                  -- Julian date
    e              REAL,                  -- eccentricity
    a              REAL,                  -- semi-major axis (AU)
    i              REAL,                  -- inclination (deg)
    om             REAL,                  -- longitude of ascending node (deg)
    w              REAL,                  -- argument of perihelion (deg)
    ma             REAL,                  -- mean anomaly (deg)
    q              REAL,                  -- perihelion distance (AU)
    H              REAL,                  -- absolute magnitude
    moid           REAL,                  -- Earth MOID (AU)
    diameter       REAL,                  -- diameter (km), NULL if unknown
    diameter_sigma REAL
);

-- Physical properties from NEOWISE
CREATE TABLE IF NOT EXISTS physical_properties (
    asteroid_id    INTEGER PRIMARY KEY REFERENCES asteroids(asteroid_id),
    albedo_pv      REAL,                  -- visible geometric albedo
    albedo_pv_err  REAL,
    diameter_km    REAL,                  -- NEOWISE-derived diameter
    diameter_err   REAL,
    beaming_eta    REAL,                  -- NEATM beaming parameter
    source         TEXT                   -- 'NEOWISE', 'IRAS', etc.
);

-- Spectral observations (one asteroid can have many)
CREATE TABLE IF NOT EXISTS spectra (
    spectrum_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    asteroid_id    INTEGER REFERENCES asteroids(asteroid_id),
    survey         TEXT NOT NULL,          -- 'MITHNEOS', 'SMASS', 'Gaia', etc.
    wavelengths    BLOB,                  -- numpy array serialized
    reflectance    BLOB,                  -- numpy array serialized
    uncertainty    BLOB,                  -- numpy array serialized, NULL if unavailable
    wl_min         REAL,                  -- min wavelength (um) for quick filtering
    wl_max         REAL,                  -- max wavelength (um)
    normalized     BOOLEAN DEFAULT FALSE, -- has Stage 0 been applied?
    snr_estimate   REAL,
    quality_flag   TEXT                   -- 'good', 'low_snr', 'partial'
);

-- Taxonomy results from classy
CREATE TABLE IF NOT EXISTS taxonomy (
    asteroid_id    INTEGER PRIMARY KEY REFERENCES asteroids(asteroid_id),
    primary_class  TEXT,                  -- e.g., 'S', 'M', 'C'
    primary_prob   REAL,                  -- probability of primary class
    prob_vector    BLOB,                  -- full 17-class probability vector (numpy)
    classifier     TEXT,                  -- 'classy_mahlke2022', 'bus_demeo_fallback'
    input_coverage TEXT                   -- 'vnir', 'vis_only', 'nir_only', 'albedo_only'
);

-- Band analysis results (S-complex only)
CREATE TABLE IF NOT EXISTS band_analysis (
    asteroid_id    INTEGER PRIMARY KEY REFERENCES asteroids(asteroid_id),
    band1_center   REAL,                  -- Band I center (um)
    band2_center   REAL,                  -- Band II center (um)
    bar            REAL,                  -- Band Area Ratio
    ol_opx_ratio   REAL,                  -- ol/(ol+px) from Dunn calibration
    fa_mol_pct     REAL,                  -- fayalite mol%
    fs_mol_pct     REAL,                  -- ferrosilite mol%
    gaffey_subtype TEXT,                  -- S(I) through S(VII)
    calibration    TEXT                   -- 'dunn2010', 'gaffey1993_zone'
);

-- Scoring output
CREATE TABLE IF NOT EXISTS scores (
    asteroid_id       INTEGER PRIMARY KEY REFERENCES asteroids(asteroid_id),
    estimated_mass_kg REAL,
    grade_estimate    REAL,               -- concentration of target material
    target_material   TEXT,               -- 'PGM', 'water', 'iron', etc.
    unit_value        REAL,
    accessibility     REAL,               -- 0-1 score from MOID/Tisserand proxy
    composite_score   REAL,
    score_mode        TEXT                -- 'earth_return' or 'in_space'
);
"""

# Index definitions for common query patterns
INDEX_SQL = """\
CREATE INDEX IF NOT EXISTS idx_spectra_asteroid ON spectra(asteroid_id);
CREATE INDEX IF NOT EXISTS idx_spectra_survey ON spectra(survey);
CREATE INDEX IF NOT EXISTS idx_orbits_moid ON orbits(moid);
CREATE INDEX IF NOT EXISTS idx_orbits_a ON orbits(a);
CREATE INDEX IF NOT EXISTS idx_asteroids_neo ON asteroids(neo);
CREATE INDEX IF NOT EXISTS idx_scores_composite ON scores(composite_score);
"""


def get_connection(db_path: Path | str | None = None, *, create: bool = True) -> sqlite3.Connection:
    """Open (and optionally initialize) a SQLite connection.

    Parameters
    ----------
    db_path : Path or str, optional
        Path to the database file. Defaults to ``data/prospector.db``.
        Use ``:memory:`` for an in-memory database (useful for tests).
    create : bool
        If True (default), run the schema DDL on first connection.
        Set False to open an existing database without migration.

    Returns
    -------
    sqlite3.Connection
        A connection with WAL journal mode and foreign keys enabled.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    db_path = Path(db_path) if db_path != ":memory:" else db_path

    # Ensure parent directory exists for file-based databases
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    if create:
        init_schema(conn)

    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they don't already exist."""
    conn.executescript(SCHEMA_SQL)
    conn.executescript(INDEX_SQL)


def table_names(conn: sqlite3.Connection) -> list[str]:
    """Return sorted list of user table names in the database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]
