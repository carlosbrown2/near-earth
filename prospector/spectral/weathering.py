"""Stage 1B: Space weathering correction (Brunetto et al. 2006).

Apply ONLY to silicate-bearing spectra (S/Q/V/A/K/L/O/R-complex).
Do NOT apply to C-complex, M/X-complex, or featureless spectra — Brunetto-style
corrections can introduce spurious features on intrinsically featureless surfaces.

The model: R_observed(λ) = R_fresh(λ) × exp(Cs/λ)
  where Cs < 0 indicates reddening/darkening from solar wind irradiation.

De-weathered: R_fresh(λ) = R_observed(λ) × exp(-Cs/λ)

Cs is estimated via least-squares fit of ln(R_norm) vs. 1/λ in continuum regions
(avoiding the 1 μm and 2 μm absorption bands).

Usage:
    from prospector.spectral.weathering import deweather_all
    from prospector.db import get_connection

    conn = get_connection()
    count = deweather_all(conn)
"""

import logging
import sqlite3

import numpy as np

logger = logging.getLogger(__name__)

# Silicate taxonomy classes eligible for weathering correction
# Same set as band_analysis SILICATE_CLASSES
SILICATE_CLASSES = {"S", "Q", "K", "A", "L", "O", "R", "V"}

# Continuum regions for Cs fitting (avoid 1 μm and 2 μm absorption bands)
# Each tuple is (wl_min, wl_max) in μm
CONTINUUM_REGIONS = [
    (0.50, 0.70),   # Visible continuum (before Band I onset)
    (1.25, 1.35),   # Between bands (before telluric mask)
    (2.20, 2.50),   # After Band II
]

# Minimum number of valid continuum points for fitting
MIN_CONTINUUM_POINTS = 5

# Cs bounds for sanity check (μm); typical range is -0.5 to 0.0
CS_MIN = -2.0
CS_MAX = 0.1


def _select_continuum_points(wl, refl):
    """Select continuum points from defined wavelength regions.

    Returns (wavelengths, reflectances) of selected continuum points,
    excluding NaN values.
    """
    mask = np.zeros(len(wl), dtype=bool)
    for lo, hi in CONTINUUM_REGIONS:
        mask |= (wl >= lo) & (wl <= hi)
    # Exclude NaN reflectance
    mask &= ~np.isnan(refl) & (refl > 0)
    return wl[mask], refl[mask]


def fit_cs(wl, refl):
    """Fit the Brunetto Cs parameter from a normalized spectrum.

    Uses least-squares fit of ln(R_norm) = a + Cs/λ in continuum regions.

    Parameters
    ----------
    wl : wavelength array (μm)
    refl : normalized reflectance array (may contain NaN)

    Returns
    -------
    float : Cs parameter (μm), or None if fit fails.
    """
    wl_cont, refl_cont = _select_continuum_points(wl, refl)

    if len(wl_cont) < MIN_CONTINUUM_POINTS:
        logger.debug(
            "Insufficient continuum points (%d) for Cs fit", len(wl_cont)
        )
        return None

    # ln(R) = a + Cs * (1/λ)
    x = 1.0 / wl_cont
    y = np.log(refl_cont)

    # Reject non-finite values
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < MIN_CONTINUUM_POINTS:
        return None

    x, y = x[valid], y[valid]

    # Least-squares: y = a + Cs * x
    # Using numpy polyfit (degree 1): coeffs[0] = Cs, coeffs[1] = a
    try:
        coeffs = np.polyfit(x, y, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None

    cs = float(coeffs[0])

    # Sanity check
    if cs < CS_MIN or cs > CS_MAX:
        logger.debug("Cs = %.4f out of plausible range [%.1f, %.1f]", cs, CS_MIN, CS_MAX)
        return None

    return cs


def deweather_spectrum(wl, refl, cs):
    """Apply Brunetto correction to remove space weathering.

    R_fresh(λ) = R_observed(λ) × exp(-Cs/λ)

    Parameters
    ----------
    wl : wavelength array (μm)
    refl : reflectance array
    cs : Brunetto Cs parameter (μm)

    Returns
    -------
    De-weathered reflectance array (same shape as input).
    """
    correction = np.exp(-cs / wl)
    deweathered = refl * correction
    return deweathered


def deweather_asteroid(asteroid_id, conn):
    """Apply weathering correction to an asteroid's best spectrum.

    Only applies to silicate-class asteroids with normalized spectra.
    Stores Cs and deweathered reflectance in the spectra table.

    Returns True if corrected, False if skipped.
    """
    # Check taxonomy — silicate classes only
    tax_row = conn.execute(
        "SELECT primary_class FROM taxonomy WHERE asteroid_id = ?",
        (asteroid_id,),
    ).fetchone()

    if tax_row is None or tax_row[0] not in SILICATE_CLASSES:
        return False

    # Best normalized spectrum (preferring good quality, widest coverage)
    row = conn.execute(
        "SELECT spectrum_id, wavelengths, reflectance "
        "FROM spectra "
        "WHERE asteroid_id = ? AND normalized = TRUE AND deweathered IS NULL "
        "ORDER BY "
        "  CASE quality_flag WHEN 'good' THEN 0 WHEN 'partial' THEN 1 ELSE 2 END, "
        "  (wl_max - wl_min) DESC "
        "LIMIT 1",
        (asteroid_id,),
    ).fetchone()

    if row is None:
        return False

    spectrum_id, wl_blob, refl_blob = row
    wavelengths = np.frombuffer(wl_blob, dtype=np.float64).copy()
    reflectance = np.frombuffer(refl_blob, dtype=np.float64).copy()

    # Fit Cs
    cs = fit_cs(wavelengths, reflectance)
    if cs is None:
        logger.debug("Could not fit Cs for asteroid %d", asteroid_id)
        return False

    # Apply correction
    deweathered = deweather_spectrum(wavelengths, reflectance, cs)

    # Store results
    conn.execute(
        "UPDATE spectra SET weathering_cs = ?, deweathered = ? "
        "WHERE spectrum_id = ?",
        (cs, deweathered.tobytes(), spectrum_id),
    )

    return True


def deweather_all(conn):
    """Apply weathering correction to all eligible silicate-class asteroids.

    Eligible: silicate taxonomy + normalized spectrum + no existing correction.

    Returns the number of asteroids corrected.
    """
    placeholders = ",".join("?" for _ in SILICATE_CLASSES)
    rows = conn.execute(
        "SELECT DISTINCT s.asteroid_id "
        "FROM spectra s "
        "JOIN taxonomy t ON s.asteroid_id = t.asteroid_id "
        f"WHERE s.normalized = TRUE AND s.deweathered IS NULL "
        f"AND t.primary_class IN ({placeholders})",
        (*SILICATE_CLASSES,),
    ).fetchall()

    if not rows:
        logger.info("No eligible asteroids for weathering correction")
        return 0

    count = 0
    for (asteroid_id,) in rows:
        try:
            if deweather_asteroid(asteroid_id, conn):
                count += 1
        except Exception:
            logger.exception(
                "Failed weathering correction for asteroid %d", asteroid_id
            )

    conn.commit()
    logger.info("Applied weathering correction to %d asteroids", count)
    return count
