"""Stage 2: Classical band parameter extraction.

Continuum removal, band I/II center determination via polynomial fitting,
temperature correction (Sanchez 2012), BAR computation, Dunn et al. (2010)
mineral chemistry calibrations for S(IV), Lindsay (2015/2016) red-edge BAR
correction, and Gaffey S-subtype classification.

Requires NIR spectra with coverage >= 2.0 um (e.g., MITHNEOS).

Usage:
    from prospector.spectral.band_analysis import analyze_all
    from prospector.db import get_connection

    conn = get_connection()
    count = analyze_all(conn)
"""

import logging
import sqlite3

import numpy as np
from scipy.optimize import minimize_scalar

from prospector.schemas import BandAnalysisResult

logger = logging.getLogger(__name__)

# Silicate taxonomy classes eligible for band analysis
SILICATE_CLASSES = {"S", "Q", "K", "A", "L", "O", "R", "V"}

# Minimum NIR coverage required (um)
MIN_WL_MAX = 2.0

# Wavelength regions for finding continuum anchor peaks (um)
BAND1_LEFT_REGION = (0.65, 0.80)    # Peak before Band I
BAND1_RIGHT_REGION = (1.20, 1.80)   # Peak between bands (NaN-aware)
BAND2_RIGHT_REGION = (2.30, 2.50)   # Peak after Band II

# Band search regions for absorption minima
BAND1_SEARCH = (0.82, 1.15)
BAND2_SEARCH = (1.65, 2.30)

# Valid ranges for band centers (sanity check)
BAND1_VALID = (0.85, 1.10)
BAND2_VALID = (1.70, 2.30)

# Temperature correction (Sanchez et al. 2012, Icarus 220, 36-50)
DBIC_DT = 5.7e-5    # Band I center shift (um/K)
DBIIC_DT = 1.0e-4   # Band II center shift (um/K)
LAB_TEMPERATURE = 300.0

# Standard red-edge for BAR (Lindsay 2015/2016)
STANDARD_RED_EDGE = 2.44  # um, as used by Dunn (2010)


def find_peak(wl, refl, region):
    """Find reflectance peak wavelength within a region.

    Parameters
    ----------
    wl : wavelengths (um)
    refl : reflectance (may contain NaN)
    region : (min_wl, max_wl) search bounds

    Returns
    -------
    Peak wavelength (um) or None if insufficient data.
    """
    mask = (wl >= region[0]) & (wl <= region[1]) & ~np.isnan(refl)
    if mask.sum() < 3:
        return None
    return float(wl[mask][np.argmax(refl[mask])])


def linear_continuum(wl, refl, left_wl, right_wl):
    """Compute linear continuum between two anchor wavelengths.

    Returns continuum values at each point in wl.
    """
    valid = ~np.isnan(refl)
    wl_v, refl_v = wl[valid], refl[valid]

    left_idx = int(np.argmin(np.abs(wl_v - left_wl)))
    right_idx = int(np.argmin(np.abs(wl_v - right_wl)))

    left_r, left_w = refl_v[left_idx], wl_v[left_idx]
    right_r, right_w = refl_v[right_idx], wl_v[right_idx]

    if abs(right_w - left_w) < 1e-6:
        return np.full_like(wl, left_r, dtype=np.float64)

    slope = (right_r - left_r) / (right_w - left_w)
    return left_r + slope * (wl - left_w)


def remove_continuum(wl, refl, left_wl, right_wl):
    """Divide spectrum by linear continuum to isolate absorption band.

    Returns continuum-removed reflectance (< 1.0 indicates absorption).
    """
    continuum = linear_continuum(wl, refl, left_wl, right_wl)
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = np.where(continuum > 0, refl / continuum, np.nan)
    return cr


def find_band_center(wl, cr_refl, search_region, poly_degree=4):
    """Find absorption band center via polynomial fitting.

    Fits a polynomial to the absorption region and finds its minimum.

    Parameters
    ----------
    wl : wavelengths (um)
    cr_refl : continuum-removed reflectance
    search_region : (min_wl, max_wl) for the fit
    poly_degree : polynomial degree (default 4)

    Returns
    -------
    Band center wavelength (um) or None.
    """
    mask = (wl >= search_region[0]) & (wl <= search_region[1]) & ~np.isnan(cr_refl)
    n_valid = mask.sum()
    if n_valid < poly_degree + 2:
        return None

    wl_band = wl[mask]
    cr_band = cr_refl[mask]

    # Check there is actually an absorption (min < 1.0)
    if cr_band.min() >= 0.99:
        return None

    coeffs = np.polyfit(wl_band, cr_band, poly_degree)
    poly = np.poly1d(coeffs)

    result = minimize_scalar(
        lambda x: poly(x),
        bounds=(float(wl_band[0]), float(wl_band[-1])),
        method="bounded",
    )

    if result.success:
        center = float(result.x)
        if wl_band[0] <= center <= wl_band[-1]:
            return center

    # Fallback: data minimum
    return float(wl_band[np.argmin(cr_band)])


def compute_band_area(wl, cr_refl, left_wl, right_wl):
    """Compute absorption band area via trapezoidal integration.

    Area = integral of (1 - continuum_removed_reflectance) over the band.

    Returns
    -------
    Band area (positive for absorption) or None.
    """
    mask = (wl >= left_wl) & (wl <= right_wl) & ~np.isnan(cr_refl)
    if mask.sum() < 3:
        return None

    # np.trapezoid added in NumPy 2.0; np.trapz deprecated but works in 1.x
    _trapz = getattr(np, "trapezoid", np.trapz)
    area = float(_trapz(1.0 - cr_refl[mask], wl[mask]))
    return area if area > 0 else None


def estimate_temperature(semi_major_axis_au):
    """Estimate mean surface temperature from semi-major axis.

    Uses T ~ 280/sqrt(a) K (rapid-rotator approximation).
    """
    if semi_major_axis_au is None or semi_major_axis_au <= 0:
        return LAB_TEMPERATURE
    return 280.0 / np.sqrt(semi_major_axis_au)


def temperature_correct_band_centers(bic, biic, temperature):
    """Correct observed band centers to lab temperature (Sanchez 2012).

    Returns (corrected_bic, corrected_biic).
    """
    delta_t = temperature - LAB_TEMPERATURE
    bic_corr = bic - delta_t * DBIC_DT
    biic_corr = biic - delta_t * DBIIC_DT if biic is not None else None
    return bic_corr, biic_corr


def lindsay_bar_correction(bar, red_edge_wl):
    """Apply Lindsay (2015/2016) red-edge correction to BAR.

    Adjusts BAR for differences in the Band II continuum red-edge
    endpoint relative to the Dunn (2010) standard of ~2.44 um.
    """
    if red_edge_wl is None or abs(red_edge_wl - STANDARD_RED_EDGE) < 0.02:
        return bar
    # Approximate: BAR changes ~30% per um shift in red edge
    delta = red_edge_wl - STANDARD_RED_EDGE
    correction = 1.0 + 0.3 * delta
    if correction <= 0:
        return bar
    return bar / correction


def classify_gaffey_subtype(bic, bar):
    """Classify Gaffey S-subtype from Band I center and BAR.

    Simplified zone boundaries from Gaffey et al. (1993, Icarus 106, 573-602).

    Parameters
    ----------
    bic : Band I center (um), temperature-corrected
    bar : Band Area Ratio

    Returns
    -------
    Gaffey subtype: 'S(I)' through 'S(VII)'.
    """
    if bar < 0.10:
        return "S(I)"
    if bar > 1.80:
        return "S(VII)"
    if bic < 0.92 and bar > 1.20:
        return "S(VI)"
    if bic < 0.92 and 0.40 <= bar <= 1.20:
        return "S(V)"
    if 0.92 <= bic <= 1.04 and 0.80 <= bar <= 1.80:
        return "S(IV)"
    if 0.92 <= bic <= 1.02 and 0.10 <= bar < 0.80:
        return "S(III)"
    if bic > 1.02 and 0.10 <= bar < 0.80:
        return "S(II)"
    # Boundary cases
    if bar >= 0.80:
        return "S(IV)"
    if bic > 1.00:
        return "S(II)"
    return "S(III)"


def dunn_calibration(bar, bic, biic):
    """Apply Dunn et al. (2010) mineral chemistry calibrations.

    Valid only for S(IV)-subtype asteroids.

    Parameters
    ----------
    bar : Band Area Ratio
    bic : Band I center (um), temperature-corrected
    biic : Band II center (um), temperature-corrected, or None

    Returns
    -------
    dict with 'ol_opx_ratio', 'fa_mol_pct', 'fs_mol_pct'.
    """
    # ol/(ol+px) = -0.242 * BAR + 0.728 (Eq. 1; RMSE ~0.03)
    ol_ratio = max(0.0, min(1.0, -0.242 * bar + 0.728))

    # Fa (mol%) = -14.63 * BIC + 15.33 (RMSE ~1.3 mol%)
    fa = max(0.0, -14.63 * bic + 15.33)

    # Fs (mol%) = -53.46 * BIIC + 109.4 (RMSE ~1.4 mol%)
    fs = None
    if biic is not None:
        fs = max(0.0, -53.46 * biic + 109.4)

    return {"ol_opx_ratio": ol_ratio, "fa_mol_pct": fa, "fs_mol_pct": fs}


def analyze_spectrum(wl, refl, semi_major_axis=None):
    """Perform full band parameter analysis on a single spectrum.

    Parameters
    ----------
    wl : wavelength array (um), should cover ~0.7-2.5
    refl : reflectance array (may contain NaN in telluric regions)
    semi_major_axis : orbital semi-major axis (AU) for temperature correction

    Returns
    -------
    dict with band analysis results, or None if analysis fails.
    """
    # Find continuum anchor peaks
    peak_left = find_peak(wl, refl, BAND1_LEFT_REGION)
    peak_mid = find_peak(wl, refl, BAND1_RIGHT_REGION)
    peak_right = find_peak(wl, refl, BAND2_RIGHT_REGION)

    if peak_left is None or peak_mid is None:
        logger.debug("Cannot find Band I continuum anchors")
        return None

    # --- Band I ---
    cr1 = remove_continuum(wl, refl, peak_left, peak_mid)
    bic = find_band_center(wl, cr1, BAND1_SEARCH)
    if bic is None:
        logger.debug("Cannot determine Band I center")
        return None
    if not (BAND1_VALID[0] <= bic <= BAND1_VALID[1]):
        logger.debug("Band I center %.3f um outside valid range", bic)
        return None

    band1_area = compute_band_area(wl, cr1, peak_left, peak_mid)

    # --- Band II ---
    biic = None
    band2_area = None
    bar = None

    if peak_right is not None:
        cr2 = remove_continuum(wl, refl, peak_mid, peak_right)
        biic_candidate = find_band_center(wl, cr2, BAND2_SEARCH)
        if biic_candidate is not None and BAND2_VALID[0] <= biic_candidate <= BAND2_VALID[1]:
            biic = biic_candidate
            band2_area = compute_band_area(wl, cr2, peak_mid, peak_right)

    # --- BAR ---
    if band1_area is not None and band2_area is not None and band1_area > 0:
        bar = band2_area / band1_area
        bar = lindsay_bar_correction(bar, peak_right)

    # --- Temperature correction ---
    temperature = estimate_temperature(semi_major_axis)
    bic_corr, biic_corr = temperature_correct_band_centers(bic, biic, temperature)

    # --- Gaffey classification ---
    gaffey = None
    if bar is not None:
        gaffey = classify_gaffey_subtype(bic_corr, bar)
    elif bic_corr > 1.04:
        gaffey = "S(I)"  # No Band II → likely pure olivine

    # --- Mineral chemistry (S(IV) only) ---
    ol_opx = None
    fa = None
    fs = None
    calibration = None

    if gaffey == "S(IV)" and bar is not None:
        minerals = dunn_calibration(bar, bic_corr, biic_corr)
        ol_opx = minerals["ol_opx_ratio"]
        fa = minerals["fa_mol_pct"]
        fs = minerals["fs_mol_pct"]
        calibration = "dunn2010"
    elif gaffey is not None:
        calibration = "gaffey1993_zone"

    result = {
        "band1_center": bic_corr,
        "band2_center": biic_corr,
        "bar": bar,
        "ol_opx_ratio": ol_opx,
        "fa_mol_pct": fa,
        "fs_mol_pct": fs,
        "gaffey_subtype": gaffey,
        "calibration": calibration,
    }
    BandAnalysisResult.model_validate(result)
    return result


def analyze_asteroid(asteroid_id, conn):
    """Analyze an asteroid's best spectrum and store results in band_analysis.

    Returns True if analyzed, False if skipped.
    """
    # Check taxonomy — silicate classes only
    tax_row = conn.execute(
        "SELECT primary_class FROM taxonomy WHERE asteroid_id = ?",
        (asteroid_id,),
    ).fetchone()

    if tax_row is None or tax_row[0] not in SILICATE_CLASSES:
        return False

    # Best normalized spectrum with NIR coverage
    row = conn.execute(
        "SELECT spectrum_id, wavelengths, reflectance "
        "FROM spectra "
        "WHERE asteroid_id = ? AND normalized = TRUE AND wl_max >= ? "
        "ORDER BY "
        "  CASE quality_flag WHEN 'good' THEN 0 WHEN 'partial' THEN 1 ELSE 2 END, "
        "  (wl_max - wl_min) DESC "
        "LIMIT 1",
        (asteroid_id, MIN_WL_MAX),
    ).fetchone()

    if row is None:
        return False

    _spectrum_id, wl_blob, refl_blob = row
    wavelengths = np.frombuffer(wl_blob, dtype=np.float64).copy()
    reflectance = np.frombuffer(refl_blob, dtype=np.float64).copy()

    # Get semi-major axis for temperature correction
    orbit_row = conn.execute(
        "SELECT a FROM orbits WHERE asteroid_id = ?", (asteroid_id,),
    ).fetchone()
    semi_major = orbit_row[0] if orbit_row else None

    result = analyze_spectrum(wavelengths, reflectance, semi_major_axis=semi_major)
    if result is None:
        return False

    conn.execute(
        "INSERT OR REPLACE INTO band_analysis "
        "(asteroid_id, band1_center, band2_center, bar, ol_opx_ratio, "
        "fa_mol_pct, fs_mol_pct, gaffey_subtype, calibration) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            asteroid_id,
            result["band1_center"],
            result["band2_center"],
            result["bar"],
            result["ol_opx_ratio"],
            result["fa_mol_pct"],
            result["fs_mol_pct"],
            result["gaffey_subtype"],
            result["calibration"],
        ),
    )
    return True


def analyze_all(conn):
    """Analyze all eligible asteroids not yet in band_analysis table.

    Eligible: silicate taxonomy + normalized NIR spectrum + no existing entry.

    Returns the number of asteroids analyzed.
    """
    placeholders = ",".join("?" for _ in SILICATE_CLASSES)
    rows = conn.execute(
        "SELECT DISTINCT s.asteroid_id "
        "FROM spectra s "
        "JOIN taxonomy t ON s.asteroid_id = t.asteroid_id "
        "LEFT JOIN band_analysis b ON s.asteroid_id = b.asteroid_id "
        f"WHERE s.normalized = TRUE AND s.wl_max >= ? "
        f"AND t.primary_class IN ({placeholders}) "
        "AND b.asteroid_id IS NULL",
        (MIN_WL_MAX, *SILICATE_CLASSES),
    ).fetchall()

    if not rows:
        logger.info("No eligible asteroids for band analysis")
        return 0

    count = 0
    for (asteroid_id,) in rows:
        try:
            if analyze_asteroid(asteroid_id, conn):
                count += 1
        except Exception:
            logger.exception("Failed band analysis for asteroid %d", asteroid_id)

    conn.commit()
    logger.info("Analyzed %d asteroids", count)
    return count
