"""Ranked candidate list output (FR-8).

Generates a CSV with all scoring components for each ranked asteroid,
including taxonomy, analog match, composition, PGM tier, and accessibility.

Usage:
    from prospector.output import generate_ranked_csv
    generate_ranked_csv(conn, "output.csv", mode="earth_return")
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from prospector.scoring.granvik_prior import MAHLKE_CLASSES


# CSV column order
COLUMNS = [
    "rank",
    "asteroid_id",
    "name",
    "designation",
    "neo",
    "pha",
    # Orbital
    "a_au",
    "e",
    "i_deg",
    "moid_au",
    "diameter_km",
    # Taxonomy
    "taxonomy_class",
    "taxonomy_prob",
    "taxonomy_coverage",
    # Analog match
    "top_analog_name",
    "top_analog_type",
    "top_analog_wmse",
    # Composition (classical)
    "ol_opx_ratio",
    "fa_mol_pct",
    "fs_mol_pct",
    "gaffey_subtype",
    # Composition (CNN)
    "cnn_ol_pct",
    "cnn_opx_pct",
    "cnn_cpx_pct",
    "cnn_agreement",
    # PGM
    "pgm_tier",
    "pgm_confidence",
    "pgm_signal_count",
    # Scoring
    "composite_score",
    "estimated_mass_kg",
    "grade_estimate",
    "target_material",
    "unit_value",
    "accessibility",
    "score_mode",
    # Transfer windows
    "best_dv_total_km_s",
    "best_launch_jd",
    "best_tof_days",
]


def _fetch_candidates(conn, mode="earth_return", min_score=None, limit=None):
    """Query scored asteroids with all supporting data.

    Parameters
    ----------
    conn : sqlite3.Connection
    mode : str
        Score mode to filter by.
    min_score : float, optional
        Minimum composite_score to include.
    limit : int, optional
        Maximum number of results.

    Returns
    -------
    list[dict]
        One dict per asteroid with all output fields.
    """
    query = """
        SELECT
            a.asteroid_id, a.name, a.designation, a.neo, a.pha,
            o.a, o.e, o.i, o.moid,
            COALESCE(o.diameter, pp.diameter_km) AS diameter_km,
            t.primary_class, t.primary_prob, t.input_coverage,
            ba.ol_opx_ratio, ba.fa_mol_pct, ba.fs_mol_pct, ba.gaffey_subtype,
            cm.ol_pct, cm.opx_pct, cm.cpx_pct, cm.classical_agreement,
            pg.pgm_tier, pg.pgm_confidence, pg.signal_count,
            s.composite_score, s.estimated_mass_kg, s.grade_estimate,
            s.target_material, s.unit_value, s.accessibility, s.score_mode
        FROM scores s
        JOIN asteroids a ON s.asteroid_id = a.asteroid_id
        LEFT JOIN orbits o ON a.asteroid_id = o.asteroid_id
        LEFT JOIN physical_properties pp ON a.asteroid_id = pp.asteroid_id
        LEFT JOIN taxonomy t ON a.asteroid_id = t.asteroid_id
        LEFT JOIN band_analysis ba ON a.asteroid_id = ba.asteroid_id
        LEFT JOIN cnn_mineral cm ON a.asteroid_id = cm.asteroid_id
        LEFT JOIN pgm_convergence pg ON a.asteroid_id = pg.asteroid_id
        WHERE s.score_mode = ?
    """
    params = [mode]

    if min_score is not None:
        query += " AND s.composite_score >= ?"
        params.append(min_score)

    query += " ORDER BY s.composite_score DESC"

    if limit is not None:
        query += f" LIMIT {int(limit)}"

    rows = conn.execute(query, params).fetchall()

    # Column names from query
    col_names = [
        "asteroid_id", "name", "designation", "neo", "pha",
        "a", "e", "i", "moid", "diameter_km",
        "primary_class", "primary_prob", "input_coverage",
        "ol_opx_ratio", "fa_mol_pct", "fs_mol_pct", "gaffey_subtype",
        "cnn_ol_pct", "cnn_opx_pct", "cnn_cpx_pct", "cnn_agreement",
        "pgm_tier", "pgm_confidence", "pgm_signal_count",
        "composite_score", "estimated_mass_kg", "grade_estimate",
        "target_material", "unit_value", "accessibility", "score_mode",
    ]

    return [dict(zip(col_names, row)) for row in rows]


def _fetch_top_analog(conn, asteroid_id):
    """Get top meteorite analog match for an asteroid.

    Uses curve_match.find_analogs if lab_spectra data is available.
    Returns (name, type, wmse) or (None, None, None).
    """
    try:
        from prospector.spectral.curve_match import find_analogs
        analogs = find_analogs(asteroid_id, conn, top_n=1)
        if analogs:
            best = analogs[0]
            return (
                best.get("meteorite_name"),
                best.get("meteorite_type"),
                best.get("wmse"),
            )
    except Exception:
        pass
    return None, None, None


def _fetch_best_window(conn, asteroid_id):
    """Get best transfer window for an asteroid.

    Returns (dv_total, launch_jd, tof_days) or (None, None, None).
    """
    try:
        from prospector.ephemeris.lambert import search_windows
        windows = search_windows(asteroid_id, conn, top_n=1)
        if windows:
            w = windows[0]
            return w.dv_total_km_s, w.launch_jd, w.tof_days
    except Exception:
        pass
    return None, None, None


def generate_ranked_csv(
    conn,
    output_path,
    *,
    mode="earth_return",
    min_score=None,
    limit=None,
    include_analogs=False,
    include_windows=False,
):
    """Generate ranked candidate CSV from scored database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database with scored asteroids.
    output_path : str or Path
        Path for the output CSV file.
    mode : str
        Score mode ('earth_return' or 'in_space').
    min_score : float, optional
        Minimum composite_score threshold.
    limit : int, optional
        Maximum number of candidates to output.
    include_analogs : bool
        If True, fetch top meteorite analog for each asteroid (slow).
    include_windows : bool
        If True, compute best transfer window for each asteroid (slow).

    Returns
    -------
    int
        Number of candidates written.
    """
    candidates = _fetch_candidates(conn, mode=mode, min_score=min_score, limit=limit)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        for rank, cand in enumerate(candidates, 1):
            # Analog match
            analog_name, analog_type, analog_wmse = None, None, None
            if include_analogs:
                analog_name, analog_type, analog_wmse = _fetch_top_analog(
                    conn, cand["asteroid_id"]
                )

            # Transfer window
            best_dv, best_launch, best_tof = None, None, None
            if include_windows:
                best_dv, best_launch, best_tof = _fetch_best_window(
                    conn, cand["asteroid_id"]
                )

            row = {
                "rank": rank,
                "asteroid_id": cand["asteroid_id"],
                "name": cand["name"] or "",
                "designation": cand["designation"] or "",
                "neo": bool(cand["neo"]) if cand["neo"] is not None else "",
                "pha": bool(cand["pha"]) if cand["pha"] is not None else "",
                "a_au": _fmt(cand["a"], 6),
                "e": _fmt(cand["e"], 6),
                "i_deg": _fmt(cand["i"], 4),
                "moid_au": _fmt(cand["moid"], 6),
                "diameter_km": _fmt(cand["diameter_km"], 3),
                "taxonomy_class": cand["primary_class"] or "",
                "taxonomy_prob": _fmt(cand["primary_prob"], 3),
                "taxonomy_coverage": cand["input_coverage"] or "",
                "top_analog_name": analog_name or "",
                "top_analog_type": analog_type or "",
                "top_analog_wmse": _fmt(analog_wmse, 6),
                "ol_opx_ratio": _fmt(cand["ol_opx_ratio"], 3),
                "fa_mol_pct": _fmt(cand["fa_mol_pct"], 1),
                "fs_mol_pct": _fmt(cand["fs_mol_pct"], 1),
                "gaffey_subtype": cand["gaffey_subtype"] or "",
                "cnn_ol_pct": _fmt(cand["cnn_ol_pct"], 1),
                "cnn_opx_pct": _fmt(cand["cnn_opx_pct"], 1),
                "cnn_cpx_pct": _fmt(cand["cnn_cpx_pct"], 1),
                "cnn_agreement": cand["cnn_agreement"] or "",
                "pgm_tier": cand["pgm_tier"] or "",
                "pgm_confidence": _fmt(cand["pgm_confidence"], 3),
                "pgm_signal_count": cand["pgm_signal_count"] if cand["pgm_signal_count"] is not None else "",
                "composite_score": _fmt(cand["composite_score"], 2),
                "estimated_mass_kg": _fmt_sci(cand["estimated_mass_kg"]),
                "grade_estimate": _fmt_sci(cand["grade_estimate"]),
                "target_material": cand["target_material"] or "",
                "unit_value": _fmt(cand["unit_value"], 2),
                "accessibility": _fmt(cand["accessibility"], 4),
                "score_mode": cand["score_mode"] or "",
                "best_dv_total_km_s": _fmt(best_dv, 3),
                "best_launch_jd": _fmt(best_launch, 1),
                "best_tof_days": _fmt(best_tof, 1),
            }
            writer.writerow(row)

    return len(candidates)


def generate_ranked_dicts(
    conn,
    *,
    mode="earth_return",
    min_score=None,
    limit=None,
) -> list[dict]:
    """Return ranked candidates as a list of dicts (no file I/O).

    Useful for programmatic access and testing.

    Parameters
    ----------
    conn : sqlite3.Connection
    mode : str
    min_score : float, optional
    limit : int, optional

    Returns
    -------
    list[dict]
        Ranked candidate dicts with all fields.
    """
    candidates = _fetch_candidates(conn, mode=mode, min_score=min_score, limit=limit)
    results = []
    for rank, cand in enumerate(candidates, 1):
        cand["rank"] = rank
        results.append(cand)
    return results


def _fmt(value, decimals=2):
    """Format a numeric value or return empty string for None."""
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def _fmt_sci(value):
    """Format a value in scientific notation or return empty string."""
    if value is None:
        return ""
    return f"{value:.4e}"
