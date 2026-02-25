"""Stage 0: Spectral normalization and quality control.

Resample spectra to a common wavelength grid, normalize to reflectance=1.0
at 550 nm, mask telluric absorption regions, estimate SNR, and flag
low-quality spectra.  Updates the spectra table in-place (setting
normalized=TRUE, snr_estimate, and quality_flag).

Usage:
    from prospector.spectral.preprocessing import preprocess_all
    from prospector.db import get_connection

    conn = get_connection()
    count = preprocess_all(conn)
"""

import logging
import sqlite3

import deal
import numpy as np

logger = logging.getLogger(__name__)

# Default common wavelength grid: 0.35 – 2.55 μm at 5 nm spacing
DEFAULT_GRID = np.arange(0.35, 2.55, 0.005)

# Telluric absorption bands (μm) — mask for ground-based observations
TELLURIC_REGIONS = [(1.35, 1.45), (1.80, 2.00)]

# Surveys observed from ground-based telescopes
GROUND_BASED_SURVEYS = {"MITHNEOS", "SMASS"}

# Normalization reference wavelength (μm)
NORM_WAVELENGTH = 0.55

# SNR threshold for quality flagging
SNR_GOOD_THRESHOLD = 25.0

# Minimum wavelength span (μm) for "good" quality
MIN_COVERAGE_GOOD = 1.5


@deal.pre(lambda wavelengths, reflectance, *_, **__: len(wavelengths) == len(reflectance),
          message="wavelengths and reflectance must have same length")
@deal.pre(lambda wavelengths, *_, **__: len(wavelengths) >= 2,
          message="need at least 2 wavelength points")
def resample(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    uncertainty: np.ndarray | None,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Resample spectrum onto a common wavelength grid.

    Interpolates within the original wavelength range; NaN outside.

    Parameters
    ----------
    wavelengths : original wavelength array (μm), must be sorted ascending
    reflectance : corresponding reflectance values
    uncertainty : per-channel uncertainty (or None)
    grid : target wavelength grid (μm). Defaults to DEFAULT_GRID.

    Returns
    -------
    (grid, resampled_reflectance, resampled_uncertainty)
    """
    if grid is None:
        grid = DEFAULT_GRID

    wl_min, wl_max = float(wavelengths.min()), float(wavelengths.max())
    in_range = (grid >= wl_min) & (grid <= wl_max)

    refl_out = np.full(len(grid), np.nan, dtype=np.float64)
    refl_out[in_range] = np.interp(grid[in_range], wavelengths, reflectance)

    unc_out = None
    if uncertainty is not None:
        unc_out = np.full(len(grid), np.nan, dtype=np.float64)
        unc_out[in_range] = np.interp(grid[in_range], wavelengths, uncertainty)

    return grid.copy(), refl_out, unc_out


@deal.pre(lambda wavelengths, reflectance, *_, **__: len(wavelengths) == len(reflectance),
          message="wavelengths and reflectance must have same length")
def normalize(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    uncertainty: np.ndarray | None,
    ref_wl: float = NORM_WAVELENGTH,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalize reflectance to 1.0 at the reference wavelength.

    If the reference wavelength is outside the spectrum's valid range,
    falls back to the nearest valid channel.

    Returns
    -------
    (normalized_reflectance, normalized_uncertainty)
    """
    valid = ~np.isnan(reflectance)
    if not valid.any():
        return reflectance.copy(), uncertainty.copy() if uncertainty is not None else None

    valid_wl = wavelengths[valid]
    valid_refl = reflectance[valid]

    # Interpolate to find reflectance at ref_wl, or use nearest if out of range
    if ref_wl < valid_wl.min() or ref_wl > valid_wl.max():
        nearest_idx = int(np.argmin(np.abs(valid_wl - ref_wl)))
        norm_value = valid_refl[nearest_idx]
    else:
        norm_value = float(np.interp(ref_wl, valid_wl, valid_refl))

    if norm_value == 0 or np.isnan(norm_value):
        logger.warning("Cannot normalize: reflectance at %.3f μm is zero or NaN", ref_wl)
        return reflectance.copy(), uncertainty.copy() if uncertainty is not None else None

    norm_refl = reflectance / norm_value
    norm_unc = uncertainty / norm_value if uncertainty is not None else None

    return norm_refl, norm_unc


def mask_telluric(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    uncertainty: np.ndarray | None,
    regions: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Set reflectance to NaN in telluric absorption regions.

    Parameters
    ----------
    regions : list of (min_wl, max_wl) tuples defining telluric bands (μm).
        Defaults to TELLURIC_REGIONS.

    Returns
    -------
    (masked_reflectance, masked_uncertainty) — copies, originals unchanged.
    """
    if regions is None:
        regions = TELLURIC_REGIONS

    refl_out = reflectance.copy()
    unc_out = uncertainty.copy() if uncertainty is not None else None

    for lo, hi in regions:
        mask = (wavelengths >= lo) & (wavelengths <= hi)
        refl_out[mask] = np.nan
        if unc_out is not None:
            unc_out[mask] = np.nan

    return refl_out, unc_out


def estimate_snr(
    reflectance: np.ndarray,
    uncertainty: np.ndarray | None,
) -> float:
    """Estimate aggregate signal-to-noise ratio.

    If per-channel uncertainty is available: SNR = median(|refl| / unc).
    Otherwise: spectral smoothness heuristic via running-median residuals.

    Returns
    -------
    Estimated SNR (NaN if estimation fails).
    """
    valid = ~np.isnan(reflectance)
    if valid.sum() < 5:
        return float("nan")

    refl_valid = reflectance[valid]

    if uncertainty is not None:
        unc_valid = uncertainty[valid]
        good = ~np.isnan(unc_valid) & (unc_valid > 0)
        if good.sum() >= 3:
            per_channel = np.abs(refl_valid[good]) / unc_valid[good]
            return float(np.median(per_channel))

    return _snr_from_smoothness(refl_valid)


def _snr_from_smoothness(reflectance: np.ndarray, window: int = 7) -> float:
    """Estimate SNR from spectral smoothness using running-median residuals."""
    n = len(reflectance)
    if n < window:
        return float("nan")

    half = window // 2
    smoothed = np.array([
        np.median(reflectance[max(0, i - half):min(n, i + half + 1)])
        for i in range(n)
    ])
    residuals = reflectance - smoothed
    sigma = float(np.std(residuals))

    if sigma == 0:
        return float("inf")

    return float(np.median(np.abs(reflectance)) / sigma)


def assess_quality(
    snr: float,
    wl_min: float,
    wl_max: float,
    snr_threshold: float = SNR_GOOD_THRESHOLD,
    coverage_threshold: float = MIN_COVERAGE_GOOD,
) -> str:
    """Determine quality flag for a spectrum.

    Returns
    -------
    'good', 'low_snr', or 'partial'.
    """
    if np.isnan(snr) or snr < snr_threshold:
        return "low_snr"
    if (wl_max - wl_min) < coverage_threshold:
        return "partial"
    return "good"


def preprocess_spectrum(
    spectrum_id: int,
    conn: sqlite3.Connection,
    *,
    grid: np.ndarray | None = None,
) -> bool:
    """Apply Stage 0 preprocessing to a single spectrum in the database.

    Reads the raw spectrum, resamples, normalizes, masks tellurics
    (if ground-based), estimates SNR, assesses quality, and updates
    the spectra row in-place.

    Returns
    -------
    True if processed, False if skipped.
    """
    if grid is None:
        grid = DEFAULT_GRID

    row = conn.execute(
        "SELECT wavelengths, reflectance, uncertainty, survey, wl_min, wl_max "
        "FROM spectra WHERE spectrum_id = ?",
        (spectrum_id,),
    ).fetchone()

    if row is None:
        logger.warning("Spectrum %d not found", spectrum_id)
        return False

    wl_blob, refl_blob, unc_blob, survey, orig_wl_min, orig_wl_max = row

    # Deserialize BLOBs
    wavelengths = np.frombuffer(wl_blob, dtype=np.float64).copy()
    reflectance = np.frombuffer(refl_blob, dtype=np.float64).copy()
    uncertainty = np.frombuffer(unc_blob, dtype=np.float64).copy() if unc_blob else None

    # 1. Resample to common grid
    wl_grid, refl_rs, unc_rs = resample(wavelengths, reflectance, uncertainty, grid)

    # 2. Normalize at 550 nm
    refl_norm, unc_norm = normalize(wl_grid, refl_rs, unc_rs)

    # 3. Mask telluric regions (ground-based only)
    if survey in GROUND_BASED_SURVEYS:
        refl_norm, unc_norm = mask_telluric(wl_grid, refl_norm, unc_norm)

    # 4. Estimate SNR
    snr = estimate_snr(refl_norm, unc_norm)

    # 5. Assess quality (use original coverage for assessment)
    quality = assess_quality(snr, orig_wl_min, orig_wl_max)

    # 6. Compute valid wavelength bounds after masking
    valid_mask = ~np.isnan(refl_norm)
    if valid_mask.any():
        new_wl_min = float(wl_grid[valid_mask].min())
        new_wl_max = float(wl_grid[valid_mask].max())
    else:
        new_wl_min, new_wl_max = orig_wl_min, orig_wl_max

    # 7. Update DB row
    conn.execute(
        "UPDATE spectra SET "
        "wavelengths = ?, reflectance = ?, uncertainty = ?, "
        "wl_min = ?, wl_max = ?, "
        "normalized = TRUE, snr_estimate = ?, quality_flag = ? "
        "WHERE spectrum_id = ?",
        (
            wl_grid.tobytes(),
            refl_norm.tobytes(),
            unc_norm.tobytes() if unc_norm is not None else None,
            new_wl_min,
            new_wl_max,
            snr if not np.isnan(snr) else None,
            quality,
            spectrum_id,
        ),
    )

    return True


def preprocess_all(
    conn: sqlite3.Connection,
    *,
    grid: np.ndarray | None = None,
) -> int:
    """Apply Stage 0 preprocessing to all un-normalized spectra.

    Returns
    -------
    Number of spectra processed.
    """
    if grid is None:
        grid = DEFAULT_GRID

    rows = conn.execute(
        "SELECT spectrum_id FROM spectra WHERE normalized = FALSE OR normalized IS NULL"
    ).fetchall()

    if not rows:
        logger.info("No un-normalized spectra found")
        return 0

    count = 0
    for (spectrum_id,) in rows:
        if preprocess_spectrum(spectrum_id, conn, grid=grid):
            count += 1

    conn.commit()
    logger.info("Preprocessed %d spectra", count)
    return count
