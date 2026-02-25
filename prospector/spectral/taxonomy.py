"""Stage 1: Unified probabilistic taxonomic classification (Mahlke et al. 2022).

Wraps the ``classy`` package (space-classy on PyPI) to classify asteroid spectra
using the Mahlke 2022 MCFA-based taxonomy.  Accepts reflectance spectra of any
wavelength coverage plus optional NEOWISE albedo.  Returns a 17-class probability
vector and stores results in the taxonomy table.

Usage:
    from prospector.spectral.taxonomy import classify_all
    from prospector.db import get_connection

    conn = get_connection()
    count = classify_all(conn)
"""

import json
import logging
import sqlite3

import deal
import numpy as np

logger = logging.getLogger(__name__)

# Mahlke 2022 taxonomy classes (alphabetical order — canonical)
MAHLKE_CLASSES = [
    "A", "B", "C", "Ch", "D", "E", "K", "L", "M",
    "O", "P", "Q", "R", "S", "V", "X", "Z",
]

# Coverage thresholds (μm)
VIS_UPPER = 0.8   # visible extends below this
NIR_LOWER = 1.0   # NIR extends above this


def determine_coverage(wl_min: float, wl_max: float) -> str:
    """Determine spectral coverage type from wavelength bounds.

    Returns one of: 'vnir', 'vis_only', 'nir_only'.
    """
    has_vis = wl_min < VIS_UPPER
    has_nir = wl_max > NIR_LOWER

    if has_vis and has_nir:
        return "vnir"
    elif has_vis:
        return "vis_only"
    else:
        return "nir_only"


@deal.pre(lambda wave, refl, *_, **__: len(wave) == len(refl),
          message="wave and refl must have same length")
@deal.pre(lambda wave, *_, **__: len(wave) >= 3,
          message="need at least 3 wavelength points")
def classify_spectrum(
    wave: np.ndarray,
    refl: np.ndarray,
    refl_err: np.ndarray | None = None,
    pV: float | None = None,
) -> dict | None:
    """Classify a spectrum using Mahlke 2022 taxonomy via classy.

    Parameters
    ----------
    wave : wavelengths in μm
    refl : reflectance values
    refl_err : reflectance uncertainties (optional)
    pV : visual geometric albedo (optional, MUST use pV= kwarg)

    Returns
    -------
    dict with keys: 'class', 'prob', 'probabilities', 'prob_vector'
    or None if the spectrum is not classifiable.
    """
    try:
        import classy
    except ImportError:
        raise ImportError(
            "classy is required for taxonomy classification. "
            "Install with: pip install space-classy tf-keras 'tensorflow-probability==0.24.*'"
        )

    # Filter out NaN channels before passing to classy
    valid = ~np.isnan(refl)
    if valid.sum() < 3:
        logger.warning("Too few valid channels (%d) for classification", valid.sum())
        return None

    wave_clean = wave[valid].copy()
    refl_clean = refl[valid].copy()
    err_clean = refl_err[valid].copy() if refl_err is not None and len(refl_err) == len(refl) else None

    kwargs = {}
    if pV is not None and np.isfinite(pV):
        kwargs["pV"] = float(pV)

    spec = classy.Spectrum(wave=wave_clean, refl=refl_clean, refl_err=err_clean, **kwargs)

    if not spec.is_classifiable("mahlke"):
        logger.info("Spectrum not classifiable by Mahlke taxonomy")
        return None

    spec.classify(taxonomy="mahlke")

    # Extract 17-class probabilities
    probabilities = {}
    for c in MAHLKE_CLASSES:
        probabilities[c] = float(getattr(spec, f"class_{c}", 0.0))

    # Build ordered numpy probability vector (same order as MAHLKE_CLASSES)
    prob_vector = np.array([probabilities[c] for c in MAHLKE_CLASSES], dtype=np.float64)

    return {
        "class": str(spec.class_mahlke),
        "prob": float(spec.prob),
        "probabilities": probabilities,
        "prob_vector": prob_vector,
    }


def classify_asteroid(
    asteroid_id: int,
    conn: sqlite3.Connection,
) -> bool:
    """Classify an asteroid from its best available spectrum + albedo.

    Reads the best preprocessed spectrum (highest quality, widest coverage)
    and optional albedo from physical_properties.  Stores result in taxonomy table.

    Returns True if classified, False if skipped.
    """
    # Find best spectrum: normalized, prefer 'good' quality, widest coverage
    row = conn.execute(
        "SELECT spectrum_id, wavelengths, reflectance, uncertainty, wl_min, wl_max "
        "FROM spectra "
        "WHERE asteroid_id = ? AND normalized = TRUE "
        "ORDER BY "
        "  CASE quality_flag WHEN 'good' THEN 0 WHEN 'partial' THEN 1 ELSE 2 END, "
        "  (wl_max - wl_min) DESC "
        "LIMIT 1",
        (asteroid_id,),
    ).fetchone()

    if row is None:
        logger.debug("No normalized spectrum for asteroid %d", asteroid_id)
        return False

    spectrum_id, wl_blob, refl_blob, unc_blob, wl_min, wl_max = row

    wavelengths = np.frombuffer(wl_blob, dtype=np.float64).copy()
    reflectance = np.frombuffer(refl_blob, dtype=np.float64).copy()
    uncertainty = np.frombuffer(unc_blob, dtype=np.float64).copy() if unc_blob else None

    # Get albedo if available
    pV = None
    albedo_row = conn.execute(
        "SELECT albedo_pv FROM physical_properties WHERE asteroid_id = ?",
        (asteroid_id,),
    ).fetchone()
    if albedo_row and albedo_row[0] is not None:
        pV = albedo_row[0]

    # Classify
    result = classify_spectrum(wavelengths, reflectance, uncertainty, pV=pV)

    if result is None:
        logger.info("Could not classify asteroid %d", asteroid_id)
        return False

    # Determine coverage
    coverage = determine_coverage(wl_min, wl_max)
    if pV is not None and coverage == "nir_only":
        coverage = "nir_only"  # albedo supplements but coverage label stays spectral
    elif pV is not None and result is not None:
        # If we only had albedo and no usable spectrum, coverage would be 'albedo_only'
        # But here we always have a spectrum, so coverage is spectral-based
        pass

    # Store in taxonomy table (INSERT OR REPLACE for idempotent re-runs)
    conn.execute(
        "INSERT OR REPLACE INTO taxonomy "
        "(asteroid_id, primary_class, primary_prob, prob_vector, classifier, input_coverage) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            asteroid_id,
            result["class"],
            result["prob"],
            result["prob_vector"].tobytes(),
            "classy_mahlke2022",
            coverage,
        ),
    )

    return True


def classify_all(conn: sqlite3.Connection) -> int:
    """Classify all asteroids with normalized spectra not yet in taxonomy table.

    Returns the number of asteroids classified.
    """
    # Find asteroids with normalized spectra but no taxonomy entry
    rows = conn.execute(
        "SELECT DISTINCT s.asteroid_id "
        "FROM spectra s "
        "LEFT JOIN taxonomy t ON s.asteroid_id = t.asteroid_id "
        "WHERE s.normalized = TRUE AND t.asteroid_id IS NULL"
    ).fetchall()

    if not rows:
        logger.info("No unclassified asteroids with normalized spectra")
        return 0

    count = 0
    for (asteroid_id,) in rows:
        try:
            if classify_asteroid(asteroid_id, conn):
                count += 1
        except Exception:
            logger.exception("Failed to classify asteroid %d", asteroid_id)

    conn.commit()
    logger.info("Classified %d asteroids", count)
    return count
