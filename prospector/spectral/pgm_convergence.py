"""Stage 5: PGM candidate identification via multi-signal convergence.

PGMs are never directly detectable via reflectance spectroscopy. Identification
requires convergence of multiple independent signals:

    1. M-complex taxonomy probability (Mahlke 2022)
    2. Featureless NIR with reddish slope (MITHNEOS)
    3. Absence of 1/2 um silicate absorption bands
    4. Iron meteorite as best analog match (RELAB curve matching)
    5. High radar albedo (>0.25) -- supplementary, often NULL
    6. NEATM beaming parameter (eta) -- weak indicator
    7. S-type metal fraction (OC mineralogy) -- Tier S pathway

Confidence tiers (from PRD Section 4 Stage 5):
    Tier 1 (High):    M-type (p>0.7) + featureless NIR + iron match + radar
    Tier 2 (Medium):  M-type (p>0.5) + featureless NIR + iron match (no radar)
    Tier 3 (Low):     M-type (p>0.3) + albedo only (no NIR spectra)
    Tier 4 (Speculative): X-complex, no albedo, Mahlke ambiguous
    Tier S (S-type metal): S/Q with confirmed OC mineralogy (metal ~10-20%)

Usage:
    from prospector.spectral.pgm_convergence import assess_asteroid, assess_all
    from prospector.db import get_connection

    conn = get_connection()
    result = assess_asteroid(4179, conn)
    # or batch:
    count = assess_all(conn)
"""

import logging
import sqlite3

import numpy as np

from prospector.spectral.curve_match import find_analogs

logger = logging.getLogger(__name__)

# Mahlke class order (same as taxonomy/curve_match modules)
MAHLKE_CLASSES = [
    "A", "B", "C", "Ch", "D", "E", "K", "L", "M",
    "O", "P", "Q", "R", "S", "V", "X", "Z",
]

M_INDEX = MAHLKE_CLASSES.index("M")
X_INDEX = MAHLKE_CLASSES.index("X")
E_INDEX = MAHLKE_CLASSES.index("E")
S_INDEX = MAHLKE_CLASSES.index("S")
Q_INDEX = MAHLKE_CLASSES.index("Q")

# Classes in each PGM pathway
METALLIC_CLASSES = {"M", "X", "E"}
S_TYPE_METAL_CLASSES = {"S", "Q"}

# M-type probability thresholds for tier assignment (PRD Stage 5)
METAL_PROB_TIER1 = 0.7
METAL_PROB_TIER2 = 0.5
METAL_PROB_TIER3 = 0.3

# Iron meteorite groups in RELAB
IRON_METEORITE_GROUPS = {"Iron", "Stony-Iron", "Pallasite"}

# Spectral slope: any positive slope counts as reddish
REDDISH_SLOPE_THRESHOLD = 0.0

# Featureless criterion: relative std of residuals around linear fit
# M-type spectra are very smooth (<3%); S-types with bands are >10%
FEATURELESS_RESIDUAL_THRESHOLD = 0.03

# Radar albedo threshold for metallic surface (PRD)
RADAR_ALBEDO_THRESHOLD = 0.25


def _get_m_type_prob(prob_vector):
    """Get M-type probability from taxonomy prob_vector."""
    if prob_vector is None or len(prob_vector) < len(MAHLKE_CLASSES):
        return 0.0
    return float(prob_vector[M_INDEX])


def _get_metallic_prob(prob_vector):
    """Sum M + X + E probabilities from taxonomy prob_vector."""
    if prob_vector is None or len(prob_vector) < len(MAHLKE_CLASSES):
        return 0.0
    return float(prob_vector[M_INDEX] + prob_vector[X_INDEX] + prob_vector[E_INDEX])


def _get_s_type_prob(prob_vector):
    """Sum S + Q probabilities from taxonomy prob_vector."""
    if prob_vector is None or len(prob_vector) < len(MAHLKE_CLASSES):
        return 0.0
    return float(prob_vector[S_INDEX] + prob_vector[Q_INDEX])


def assess_nir_slope(wl, refl):
    """Assess NIR spectral slope and featurelessness.

    Parameters
    ----------
    wl : numpy array of wavelengths (um)
    refl : numpy array of reflectance values

    Returns
    -------
    dict with has_nir, slope, is_reddish, is_featureless.
    """
    empty = {
        "has_nir": False, "slope": 0.0,
        "is_reddish": False, "is_featureless": False,
    }

    if wl is None or refl is None or len(wl) == 0:
        return empty

    valid = ~np.isnan(refl)
    if valid.sum() < 5:
        return empty

    wl_valid = wl[valid]
    refl_valid = refl[valid]
    has_nir = float(wl_valid.max()) >= 1.5

    # Compute slope in NIR region (0.8-2.5 um)
    nir_mask = (wl_valid >= 0.8) & (wl_valid <= 2.5)
    if nir_mask.sum() < 5:
        return {**empty, "has_nir": has_nir}

    wl_nir = wl_valid[nir_mask]
    refl_nir = refl_valid[nir_mask]

    # Linear fit
    coeffs = np.polyfit(wl_nir, refl_nir, 1)
    slope = float(coeffs[0])

    # Featurelessness: residual std relative to mean reflectance
    linear_fit = np.polyval(coeffs, wl_nir)
    residuals = refl_nir - linear_fit
    mean_refl = float(np.mean(np.abs(refl_nir)))
    rel_residual_std = (
        float(np.std(residuals) / mean_refl) if mean_refl > 0 else 1.0
    )

    return {
        "has_nir": has_nir,
        "slope": slope,
        "is_reddish": slope > REDDISH_SLOPE_THRESHOLD,
        "is_featureless": rel_residual_std < FEATURELESS_RESIDUAL_THRESHOLD,
    }


def check_silicate_bands(asteroid_id, conn):
    """Check whether silicate absorption bands are absent.

    Returns True if bands are ABSENT (positive PGM indicator),
    False if bands are present, None if unknown.
    """
    row = conn.execute(
        "SELECT band1_center, band2_center FROM band_analysis "
        "WHERE asteroid_id = ?",
        (asteroid_id,),
    ).fetchone()

    if row is not None:
        # Has band analysis results — bands ARE present
        return False

    # Not in band_analysis — check if it's a silicate class that just
    # hasn't been processed vs a non-silicate class where no bands expected
    tax_row = conn.execute(
        "SELECT primary_class FROM taxonomy WHERE asteroid_id = ?",
        (asteroid_id,),
    ).fetchone()

    if tax_row and tax_row[0] in {"S", "Q", "K", "A", "L", "O", "R", "V"}:
        return None  # Silicate class but not yet processed — unknown

    return True  # Non-silicate class, no bands expected


def check_iron_analog(asteroid_id, conn):
    """Check if best meteorite analog is an iron meteorite.

    Returns dict with has_iron_match, best_iron_wmse, best_iron_rho,
    best_iron_name.
    """
    no_match = {
        "has_iron_match": False, "best_iron_wmse": None,
        "best_iron_rho": None, "best_iron_name": None,
    }

    try:
        analogs = find_analogs(asteroid_id, conn, top_n=10)
    except Exception:
        logger.debug("Could not find analogs for asteroid %d", asteroid_id)
        return no_match

    return _check_iron_in_analogs(analogs)


def _check_iron_in_analogs(analogs):
    """Check pre-computed analog list for iron meteorite matches."""
    no_match = {
        "has_iron_match": False, "best_iron_wmse": None,
        "best_iron_rho": None, "best_iron_name": None,
    }

    if not analogs:
        return no_match

    for analog in analogs:
        group = (analog.get("meteorite_group") or "").lower()
        if any(ig.lower() in group for ig in IRON_METEORITE_GROUPS):
            return {
                "has_iron_match": True,
                "best_iron_wmse": analog.get("wmse"),
                "best_iron_rho": analog.get("rho"),
                "best_iron_name": analog.get("meteorite_name"),
            }

    return no_match


def _assign_tier(signals):
    """Assign PGM confidence tier based on converging signals.

    Returns (tier, confidence) or (None, 0.0) if not a PGM candidate.
    """
    m_prob = signals.get("m_type_prob", 0.0)
    metallic_prob = signals.get("metallic_prob", 0.0)
    featureless = signals.get("featureless_nir", False)
    no_bands = signals.get("no_silicate_bands")
    iron_match = signals.get("iron_analog_match", False)
    radar = signals.get("radar_albedo")
    s_type_metal = signals.get("s_type_metal", False)

    # Tier S: S-type metal pathway (separate from metallic tiers)
    if s_type_metal:
        s_prob = signals.get("s_type_prob", 0.0)
        confidence = min(0.30, s_prob * 0.30)
        return "Tier S", confidence

    # Count supporting spectral signals
    positive_signals = 0
    if featureless:
        positive_signals += 1
    if no_bands is True:
        positive_signals += 1
    if iron_match:
        positive_signals += 1
    if radar is not None and radar > RADAR_ALBEDO_THRESHOLD:
        positive_signals += 1

    # Tier 1: M-type (p>0.7) + featureless NIR + iron match + radar
    if (m_prob > METAL_PROB_TIER1 and featureless and iron_match
            and radar is not None and radar > RADAR_ALBEDO_THRESHOLD):
        confidence = 0.70 + 0.20 * min(1.0, m_prob)
        return "Tier 1", min(0.90, confidence)

    # Tier 2: M-type (p>0.5) + featureless NIR + iron match (no radar)
    if m_prob > METAL_PROB_TIER2 and featureless and iron_match:
        confidence = 0.40 + 0.20 * min(1.0, m_prob)
        return "Tier 2", min(0.60, confidence)

    # Tier 3: M-type (p>0.3) + some supporting evidence
    if m_prob > METAL_PROB_TIER3:
        confidence = 0.20 + 0.10 * min(1.0, positive_signals / 2)
        return "Tier 3", min(0.30, confidence)

    # Tier 4: Any significant metallic probability
    if metallic_prob > 0.1:
        confidence = metallic_prob * 0.10
        return "Tier 4", min(0.10, confidence)

    return None, 0.0


def assess_asteroid(asteroid_id, conn, *, analogs=None):
    """Assess PGM candidacy for a single asteroid.

    Parameters
    ----------
    asteroid_id : int
    conn : sqlite3.Connection
    analogs : list of dicts, optional
        Pre-computed analog results from find_analogs().

    Returns
    -------
    dict with signal assessments, tier, and confidence, or None
    if the asteroid is not a PGM candidate.
    """
    # Get taxonomy
    tax_row = conn.execute(
        "SELECT primary_class, primary_prob, prob_vector "
        "FROM taxonomy WHERE asteroid_id = ?",
        (asteroid_id,),
    ).fetchone()

    prob_vector = None
    primary_class = None
    if tax_row:
        primary_class = tax_row[0]
        if tax_row[2]:
            prob_vector = np.frombuffer(tax_row[2], dtype=np.float64).copy()

    m_prob = _get_m_type_prob(prob_vector)
    metallic_prob = _get_metallic_prob(prob_vector)
    s_prob = _get_s_type_prob(prob_vector)

    # Quick exit: not worth assessing if no metallic or S-type probability
    if metallic_prob < 0.05 and s_prob < 0.5:
        return None

    signals = {
        "asteroid_id": asteroid_id,
        "primary_class": primary_class,
        "m_type_prob": m_prob,
        "metallic_prob": metallic_prob,
        "s_type_prob": s_prob,
    }

    # --- S-type metal pathway ---
    s_type_metal = False
    metal_fraction = None
    if s_prob > 0.5 and primary_class in S_TYPE_METAL_CLASSES:
        # Check for OC mineralogy via band_analysis S(IV) subtype
        ba_row = conn.execute(
            "SELECT gaffey_subtype, ol_opx_ratio "
            "FROM band_analysis WHERE asteroid_id = ?",
            (asteroid_id,),
        ).fetchone()
        if ba_row and ba_row[0] == "S(IV)":
            s_type_metal = True
            metal_fraction = 15.0  # OC metal ~10-20% (Cannon 2023)

        # CNN mineral: high non-silicate fraction also indicates metal
        cnn_row = conn.execute(
            "SELECT ol_pct, opx_pct, cpx_pct "
            "FROM cnn_mineral WHERE asteroid_id = ?",
            (asteroid_id,),
        ).fetchone()
        if cnn_row and all(v is not None for v in cnn_row):
            silicate_total = sum(cnn_row)
            if silicate_total < 95.0:
                s_type_metal = True
                metal_fraction = max(metal_fraction or 0, 100.0 - silicate_total)

    signals["s_type_metal"] = s_type_metal
    signals["metal_fraction_pct"] = metal_fraction

    # --- Metallic-class pathway ---
    if metallic_prob >= 0.05:
        # Assess NIR spectral slope
        spec_row = conn.execute(
            "SELECT wavelengths, reflectance, deweathered "
            "FROM spectra "
            "WHERE asteroid_id = ? AND normalized = TRUE "
            "ORDER BY "
            "  CASE quality_flag WHEN 'good' THEN 0 "
            "  WHEN 'partial' THEN 1 ELSE 2 END, "
            "  (wl_max - wl_min) DESC "
            "LIMIT 1",
            (asteroid_id,),
        ).fetchone()

        if spec_row and spec_row[0]:
            wl = np.frombuffer(spec_row[0], dtype=np.float64).copy()
            refl_blob = spec_row[2] if spec_row[2] else spec_row[1]
            refl = np.frombuffer(refl_blob, dtype=np.float64).copy()

            slope_info = assess_nir_slope(wl, refl)
            signals["has_nir"] = slope_info["has_nir"]
            signals["nir_slope"] = slope_info["slope"]
            signals["featureless_nir"] = (
                slope_info["is_reddish"] and slope_info["is_featureless"]
            )
        else:
            signals["has_nir"] = False
            signals["nir_slope"] = None
            signals["featureless_nir"] = False

        # Check silicate band absence
        signals["no_silicate_bands"] = check_silicate_bands(asteroid_id, conn)

        # Iron analog check
        if analogs is not None:
            iron_info = _check_iron_in_analogs(analogs)
        else:
            iron_info = check_iron_analog(asteroid_id, conn)

        signals["iron_analog_match"] = iron_info["has_iron_match"]
        signals["iron_analog_wmse"] = iron_info["best_iron_wmse"]
    else:
        signals["has_nir"] = False
        signals["nir_slope"] = None
        signals["featureless_nir"] = False
        signals["no_silicate_bands"] = None
        signals["iron_analog_match"] = False
        signals["iron_analog_wmse"] = None

    # Supplementary data
    pp_row = conn.execute(
        "SELECT beaming_eta FROM physical_properties WHERE asteroid_id = ?",
        (asteroid_id,),
    ).fetchone()
    signals["beaming_eta"] = pp_row[0] if pp_row else None
    signals["radar_albedo"] = None  # Not yet ingested

    # Assign tier
    tier, confidence = _assign_tier(signals)
    if tier is None:
        return None

    signals["pgm_tier"] = tier
    signals["pgm_confidence"] = confidence

    # Count positive signals
    signals["signal_count"] = sum([
        bool(signals.get("featureless_nir")),
        signals.get("no_silicate_bands") is True,
        bool(signals.get("iron_analog_match")),
        (signals.get("radar_albedo") or 0) > RADAR_ALBEDO_THRESHOLD,
        bool(signals.get("s_type_metal")),
        m_prob > METAL_PROB_TIER3,
    ])

    # Notes summary
    notes_parts = []
    if tier == "Tier S":
        notes_parts.append(f"S-type metal candidate (metal ~{metal_fraction:.0f}%)")
    if signals.get("featureless_nir"):
        notes_parts.append("featureless NIR")
    if signals.get("no_silicate_bands") is True:
        notes_parts.append("no silicate bands")
    if signals.get("iron_analog_match"):
        notes_parts.append("iron meteorite analog match")
    if m_prob > METAL_PROB_TIER3:
        notes_parts.append(f"M-type p={m_prob:.2f}")
    signals["notes"] = "; ".join(notes_parts) if notes_parts else None

    return signals


def _store_result(result, conn):
    """Store PGM convergence result in database."""
    conn.execute(
        "INSERT OR REPLACE INTO pgm_convergence "
        "(asteroid_id, pgm_tier, pgm_confidence, m_type_prob, "
        " featureless_nir, no_silicate_bands, iron_analog_match, "
        " iron_analog_wmse, radar_albedo, beaming_eta, "
        " s_type_metal, metal_fraction_pct, signal_count, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            result["asteroid_id"],
            result["pgm_tier"],
            result["pgm_confidence"],
            result["m_type_prob"],
            result.get("featureless_nir", False),
            result.get("no_silicate_bands"),
            result.get("iron_analog_match", False),
            result.get("iron_analog_wmse"),
            result.get("radar_albedo"),
            result.get("beaming_eta"),
            result.get("s_type_metal", False),
            result.get("metal_fraction_pct"),
            result["signal_count"],
            result.get("notes"),
        ),
    )


def assess_all(conn):
    """Assess PGM candidacy for all asteroids with taxonomy data.

    Writes results to pgm_convergence table. Returns count of
    PGM candidates found.
    """
    rows = conn.execute("SELECT asteroid_id FROM taxonomy").fetchall()

    if not rows:
        logger.info("No taxonomized asteroids for PGM assessment")
        return 0

    count = 0
    for (asteroid_id,) in rows:
        try:
            result = assess_asteroid(asteroid_id, conn)
            if result is not None:
                _store_result(result, conn)
                count += 1
        except Exception:
            logger.exception(
                "PGM assessment failed for asteroid %d", asteroid_id
            )

    conn.commit()
    logger.info("Found %d PGM candidates out of %d assessed", count, len(rows))
    return count
