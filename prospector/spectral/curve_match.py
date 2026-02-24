"""Stage 3: Meteorite analog curve matching.

Match asteroid spectra against RELAB meteorite lab spectra to infer mineralogy.
Uses weighted MSE (or unweighted MSE when uncertainties unavailable) with free
scaling and optional slope adjustment.  Ranks by WMSE with correlation
coefficient ρ as tiebreaker.  Rejects matches with ρ < 0.90.

Taxonomy-consistency filter: meteorite types weighted by the asteroid's
taxonomy probability vector, so surprising-but-plausible matches survive.

Usage:
    from prospector.spectral.curve_match import find_analogs, match_all
    from prospector.db import get_connection

    conn = get_connection()
    results = find_analogs(4179, conn, top_n=5)
    # or batch:
    all_results = match_all(conn, top_n=5)
"""

import logging
import sqlite3

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# Minimum correlation to accept a match
RHO_THRESHOLD = 0.90

# Minimum number of overlapping wavelength points for a valid match
MIN_OVERLAP_POINTS = 20

# Taxonomy class → compatible meteorite groups (from PRD §10)
# Keys are Mahlke 2022 taxonomy classes; values are sets of RELAB meteorite_group
# keywords that partially match the group field.
TAXONOMY_METEORITE_MAP = {
    "S": {"Ordinary Chondrite", "Iron", "Stony-Iron", "Pallasite"},
    "Q": {"Ordinary Chondrite"},
    "K": {"Ordinary Chondrite", "Carbonaceous Chondrite"},
    "L": {"Ordinary Chondrite", "Stony-Iron"},
    "A": {"Pallasite", "Stony-Iron", "Brachinite"},
    "O": {"Ordinary Chondrite"},
    "R": {"Ordinary Chondrite"},
    "V": {"HED", "Achondrite", "Eucrite", "Diogenite", "Howardite"},
    "M": {"Iron", "Stony-Iron", "Enstatite Chondrite"},
    "X": {"Iron", "Enstatite Chondrite", "Stony-Iron"},
    "E": {"Enstatite Chondrite", "Aubrite"},
    "C": {"Carbonaceous Chondrite"},
    "B": {"Carbonaceous Chondrite"},
    "Ch": {"Carbonaceous Chondrite"},
    "D": {"Carbonaceous Chondrite"},
    "P": {"Carbonaceous Chondrite"},
    "Z": set(),  # Unclassifiable — accept anything
}

# Mahlke class order (same as taxonomy module)
MAHLKE_CLASSES = [
    "A", "B", "C", "Ch", "D", "E", "K", "L", "M",
    "O", "P", "Q", "R", "S", "V", "X", "Z",
]


def _resample_to_overlap(
    wl_ast: np.ndarray,
    refl_ast: np.ndarray,
    wl_lab: np.ndarray,
    refl_lab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Resample lab spectrum to asteroid wavelength grid in the overlap region.

    Returns (common_wl, refl_ast_overlap, refl_lab_resampled) or None
    if insufficient overlap.
    """
    # Find overlap range
    lo = max(wl_ast[0], wl_lab[0])
    hi = min(wl_ast[-1], wl_lab[-1])

    if hi - lo < 0.05:  # Less than 50 nm overlap
        return None

    # Select asteroid wavelengths in overlap
    mask = (wl_ast >= lo) & (wl_ast <= hi)
    common_wl = wl_ast[mask]
    refl_ast_ov = refl_ast[mask]

    # Remove NaN from asteroid spectrum
    valid = ~np.isnan(refl_ast_ov)
    if valid.sum() < MIN_OVERLAP_POINTS:
        return None

    common_wl = common_wl[valid]
    refl_ast_ov = refl_ast_ov[valid]

    # Interpolate lab spectrum onto asteroid grid
    try:
        f_lab = interp1d(
            wl_lab, refl_lab, kind="linear", bounds_error=False, fill_value=np.nan
        )
        refl_lab_interp = f_lab(common_wl)
    except ValueError:
        return None

    # Remove any NaN from interpolated lab spectrum
    valid2 = ~np.isnan(refl_lab_interp)
    if valid2.sum() < MIN_OVERLAP_POINTS:
        return None

    return common_wl[valid2], refl_ast_ov[valid2], refl_lab_interp[valid2]


def _best_scale_factor(ast_refl: np.ndarray, lab_refl: np.ndarray) -> float:
    """Find the scale factor α that minimizes MSE: Σ(ast - α*lab)².

    Solution: α = (ast · lab) / (lab · lab)
    """
    denom = np.dot(lab_refl, lab_refl)
    if denom == 0:
        return 1.0
    return float(np.dot(ast_refl, lab_refl) / denom)


def compute_match_score(
    ast_refl: np.ndarray,
    lab_refl: np.ndarray,
    uncertainty: np.ndarray | None = None,
) -> dict | None:
    """Compute WMSE and correlation between asteroid and lab spectra.

    Applies free scaling to the lab spectrum before computing metrics.

    Parameters
    ----------
    ast_refl : asteroid reflectance (on common grid)
    lab_refl : lab reflectance (on common grid, same length)
    uncertainty : per-channel uncertainty (same length), or None

    Returns
    -------
    dict with 'wmse', 'rho', 'scale_factor', 'n_points', or None if invalid.
    """
    if len(ast_refl) < MIN_OVERLAP_POINTS:
        return None

    # Free scaling
    alpha = _best_scale_factor(ast_refl, lab_refl)
    scaled_lab = alpha * lab_refl

    # Residuals
    residuals = ast_refl - scaled_lab

    # WMSE
    if uncertainty is not None and len(uncertainty) == len(ast_refl):
        # Weight by inverse variance
        weights = np.where(uncertainty > 0, 1.0 / (uncertainty**2), 1.0)
        wmse = float(np.mean(weights * residuals**2))
    else:
        wmse = float(np.mean(residuals**2))

    # Correlation coefficient
    if np.std(ast_refl) == 0 or np.std(scaled_lab) == 0:
        # Constant spectra: if residual is near-zero, treat as perfect match
        if wmse < 1e-10:
            rho = 1.0
        else:
            return None
    else:
        rho = float(np.corrcoef(ast_refl, scaled_lab)[0, 1])

    return {
        "wmse": wmse,
        "rho": rho,
        "scale_factor": alpha,
        "n_points": len(ast_refl),
    }


def _taxonomy_weight(
    prob_vector: np.ndarray | None,
    meteorite_group: str | None,
) -> float:
    """Compute taxonomy-consistency weight for a meteorite match.

    Returns a weight in [0, 1] based on how compatible the meteorite group
    is with the asteroid's taxonomy probability vector.

    If prob_vector is None (no taxonomy), returns 1.0 (no penalty).
    """
    if prob_vector is None or meteorite_group is None:
        return 1.0

    total_weight = 0.0
    for i, cls in enumerate(MAHLKE_CLASSES):
        if i >= len(prob_vector):
            break
        compatible_groups = TAXONOMY_METEORITE_MAP.get(cls, set())
        if any(cg.lower() in meteorite_group.lower() for cg in compatible_groups):
            total_weight += prob_vector[i]
        # Z (unclassifiable) → accept anything
        if cls == "Z":
            total_weight += prob_vector[i]

    return min(1.0, total_weight)


def match_spectrum(
    wl_ast: np.ndarray,
    refl_ast: np.ndarray,
    wl_lab: np.ndarray,
    refl_lab: np.ndarray,
    uncertainty: np.ndarray | None = None,
) -> dict | None:
    """Match an asteroid spectrum against a single lab spectrum.

    Resamples to common grid, applies free scaling, computes WMSE and ρ.

    Returns dict with match metrics, or None if no valid overlap.
    """
    overlap = _resample_to_overlap(wl_ast, refl_ast, wl_lab, refl_lab)
    if overlap is None:
        return None

    common_wl, ast_ov, lab_ov = overlap

    # Resample uncertainty if available
    unc_ov = None
    if uncertainty is not None:
        try:
            f_unc = interp1d(
                wl_ast, uncertainty, kind="linear",
                bounds_error=False, fill_value=np.nan,
            )
            unc_ov = f_unc(common_wl)
            unc_ov = np.where(np.isnan(unc_ov), 0.0, unc_ov)
        except ValueError:
            unc_ov = None

    return compute_match_score(ast_ov, lab_ov, unc_ov)


def find_analogs(
    asteroid_id: int,
    conn: sqlite3.Connection,
    *,
    top_n: int = 5,
    rho_threshold: float = RHO_THRESHOLD,
    taxonomy_filter: bool = True,
) -> list[dict]:
    """Find top-N meteorite analogs for an asteroid from RELAB lab_spectra.

    Parameters
    ----------
    asteroid_id : asteroid to match
    conn : database connection
    top_n : number of top matches to return
    rho_threshold : minimum correlation coefficient
    taxonomy_filter : whether to weight by taxonomy consistency

    Returns
    -------
    List of dicts with: spectrum_key, sample_id, meteorite_name,
    meteorite_type, meteorite_group, wmse, rho, scale_factor,
    taxonomy_weight, n_points.  Sorted by WMSE ascending.
    """
    # Get asteroid's best spectrum (prefer deweathered if available)
    spec_row = conn.execute(
        "SELECT wavelengths, reflectance, uncertainty, deweathered "
        "FROM spectra "
        "WHERE asteroid_id = ? AND normalized = TRUE "
        "ORDER BY "
        "  CASE quality_flag WHEN 'good' THEN 0 WHEN 'partial' THEN 1 ELSE 2 END, "
        "  (wl_max - wl_min) DESC "
        "LIMIT 1",
        (asteroid_id,),
    ).fetchone()

    if spec_row is None:
        return []

    wl_blob, refl_blob, unc_blob, dw_blob = spec_row
    wl_ast = np.frombuffer(wl_blob, dtype=np.float64).copy()

    # Prefer deweathered spectrum if available
    if dw_blob is not None:
        refl_ast = np.frombuffer(dw_blob, dtype=np.float64).copy()
    else:
        refl_ast = np.frombuffer(refl_blob, dtype=np.float64).copy()

    unc_ast = None
    if unc_blob is not None:
        unc_ast = np.frombuffer(unc_blob, dtype=np.float64).copy()

    # Get taxonomy prob_vector for consistency filtering
    prob_vector = None
    if taxonomy_filter:
        tax_row = conn.execute(
            "SELECT prob_vector FROM taxonomy WHERE asteroid_id = ?",
            (asteroid_id,),
        ).fetchone()
        if tax_row and tax_row[0]:
            prob_vector = np.frombuffer(tax_row[0], dtype=np.float64)

    # Fetch all lab spectra
    lab_rows = conn.execute(
        "SELECT spectrum_key, sample_id, meteorite_name, meteorite_type, "
        "meteorite_group, wavelengths, reflectance "
        "FROM lab_spectra "
        "WHERE wavelengths IS NOT NULL AND reflectance IS NOT NULL"
    ).fetchall()

    if not lab_rows:
        return []

    matches = []
    for row in lab_rows:
        (spectrum_key, sample_id, met_name, met_type, met_group,
         lab_wl_blob, lab_refl_blob) = row

        lab_wl = np.frombuffer(lab_wl_blob, dtype=np.float64)
        lab_refl = np.frombuffer(lab_refl_blob, dtype=np.float64)

        score = match_spectrum(wl_ast, refl_ast, lab_wl, lab_refl, unc_ast)
        if score is None:
            continue

        if score["rho"] < rho_threshold:
            continue

        tax_wt = _taxonomy_weight(prob_vector, met_group) if taxonomy_filter else 1.0

        matches.append({
            "spectrum_key": spectrum_key,
            "sample_id": sample_id,
            "meteorite_name": met_name,
            "meteorite_type": met_type,
            "meteorite_group": met_group,
            "wmse": score["wmse"],
            "rho": score["rho"],
            "scale_factor": score["scale_factor"],
            "taxonomy_weight": tax_wt,
            "n_points": score["n_points"],
        })

    # Sort: primary by WMSE ascending, secondary by rho descending
    matches.sort(key=lambda m: (m["wmse"], -m["rho"]))

    return matches[:top_n]


def match_all(
    conn: sqlite3.Connection,
    *,
    top_n: int = 5,
    rho_threshold: float = RHO_THRESHOLD,
) -> dict[int, list[dict]]:
    """Find meteorite analogs for all asteroids with normalized spectra.

    Returns a dict mapping asteroid_id → list of top-N analogs.
    """
    rows = conn.execute(
        "SELECT DISTINCT asteroid_id FROM spectra WHERE normalized = TRUE"
    ).fetchall()

    if not rows:
        logger.info("No normalized spectra for curve matching")
        return {}

    results = {}
    for (asteroid_id,) in rows:
        try:
            analogs = find_analogs(
                asteroid_id, conn, top_n=top_n, rho_threshold=rho_threshold
            )
            if analogs:
                results[asteroid_id] = analogs
        except Exception:
            logger.exception(
                "Failed curve matching for asteroid %d", asteroid_id
            )

    logger.info("Found analogs for %d asteroids", len(results))
    return results
