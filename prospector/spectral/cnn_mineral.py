"""Stage 2B: CNN-based mineral quantification (Korda et al. 2023).

Convolutional neural network for modal mineral abundances (olivine/orthopyroxene/
clinopyroxene) and chemistry (Fa, Fs, Wo) from VNIR reflectance spectra.

Only valid for S*-complex (silicate-bearing) asteroids. The CNN is trained on
RELAB + C-Tape lab spectra of OL/OPX/CPX mixtures and meteorites.

Requires the ``Asteroid-spectra`` package (https://github.com/Sirrah91/Asteroid-spectra)
and PyTorch.  These are optional dependencies — the module degrades gracefully
when unavailable.

Usage:
    from prospector.spectral.cnn_mineral import predict_all
    from prospector.db import get_connection

    conn = get_connection()
    count = predict_all(conn)
"""

import logging
import sqlite3

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# Silicate taxonomy classes eligible for CNN mineral quantification
# Same domain restriction as band_analysis — Korda 2023 only validated for S*-complex
SILICATE_CLASSES = {"S", "Q", "K", "A", "L", "O", "R", "V"}

# CNN input wavelength grid (Korda et al. 2023): 0.45 to 2.45 μm at 5nm spacing
CNN_WL_MIN = 0.45
CNN_WL_MAX = 2.45
CNN_WL_STEP = 0.005
CNN_GRID = np.arange(CNN_WL_MIN, CNN_WL_MAX + CNN_WL_STEP / 2, CNN_WL_STEP)
CNN_N_CHANNELS = len(CNN_GRID)  # 401 channels

# Minimum fraction of non-NaN channels for valid prediction
MIN_VALID_FRACTION = 0.80

# Agreement thresholds (OL% difference between classical and CNN)
GOOD_AGREEMENT = 10.0   # within 10 percentage points
FAIR_AGREEMENT = 20.0   # within 20 percentage points

# Number of MC dropout forward passes for uncertainty estimation
DEFAULT_N_MC = 50


def _check_dependencies():
    """Check that PyTorch and Asteroid-spectra are importable.

    Returns True if available, False otherwise.
    """
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def resample_to_cnn_grid(wl, refl):
    """Resample a spectrum to the CNN's expected wavelength grid.

    Parameters
    ----------
    wl : wavelength array (μm)
    refl : reflectance array (may contain NaN)

    Returns
    -------
    Resampled reflectance on CNN_GRID, NaN outside original range.
    Returns None if insufficient coverage.
    """
    valid = ~np.isnan(refl)
    if valid.sum() < 10:
        return None

    wl_valid = wl[valid]
    refl_valid = refl[valid]

    # Check coverage — need at least the core of the VNIR range
    wl_min, wl_max = float(wl_valid.min()), float(wl_valid.max())
    if wl_max - wl_min < 1.0:
        return None

    # Interpolate onto CNN grid (linear, NaN outside range)
    interp_func = interp1d(
        wl_valid, refl_valid,
        kind="linear", bounds_error=False, fill_value=np.nan,
    )
    resampled = interp_func(CNN_GRID)

    # Check enough valid channels remain
    n_valid = np.sum(~np.isnan(resampled))
    if n_valid / CNN_N_CHANNELS < MIN_VALID_FRACTION:
        return None

    return resampled


def _fill_nan_channels(refl):
    """Fill NaN channels (e.g. telluric gaps) via linear interpolation.

    The CNN expects a continuous input — NaN gaps from telluric masking
    must be interpolated through.  Only fills interior gaps; leading/trailing
    NaN stay as NaN.

    Returns filled reflectance array (copy).
    """
    result = refl.copy()
    valid = ~np.isnan(result)
    if valid.all() or valid.sum() < 2:
        return result

    indices = np.arange(len(result))
    result[~valid] = np.interp(indices[~valid], indices[valid], result[valid])
    return result


def predict_spectrum(wl, refl, n_mc=DEFAULT_N_MC, _model=None):
    """Run CNN mineral prediction on a single spectrum.

    Parameters
    ----------
    wl : wavelength array (μm)
    refl : reflectance array (may contain NaN)
    n_mc : number of MC dropout passes for uncertainty
    _model : optional pre-loaded model (for batch efficiency)

    Returns
    -------
    dict with prediction results, or None if prediction fails.
    Keys: ol_pct, opx_pct, cpx_pct, fa_mol_pct, fs_mol_pct, wo_mol_pct,
          ol_unc, opx_unc, cpx_unc, fa_unc, fs_unc, wo_unc, method
    """
    # Resample to CNN grid
    resampled = resample_to_cnn_grid(wl, refl)
    if resampled is None:
        logger.debug("Insufficient spectral coverage for CNN prediction")
        return None

    # Fill NaN gaps (telluric) for continuous CNN input
    filled = _fill_nan_channels(resampled)

    # Trim to valid range (no leading/trailing NaN)
    valid_mask = ~np.isnan(filled)
    if not valid_mask.any():
        return None

    first_valid = int(np.argmax(valid_mask))
    last_valid = len(valid_mask) - 1 - int(np.argmax(valid_mask[::-1]))
    spectrum_input = filled[first_valid:last_valid + 1]

    if len(spectrum_input) < int(CNN_N_CHANNELS * MIN_VALID_FRACTION):
        return None

    if not _check_dependencies():
        raise ImportError(
            "PyTorch is required for CNN mineral quantification. "
            "Install with: pip install torch\n"
            "Also install Asteroid-spectra: "
            "pip install git+https://github.com/Sirrah91/Asteroid-spectra.git"
        )

    # Import and run model
    import torch

    if _model is None:
        _model = _load_model()

    # Prepare input tensor
    input_tensor = torch.tensor(
        spectrum_input.reshape(1, 1, -1), dtype=torch.float32
    )

    # MC dropout predictions for uncertainty estimation
    _model.train()  # enable dropout
    predictions = []
    with torch.no_grad():
        for _ in range(n_mc):
            pred = _model(input_tensor)
            predictions.append(pred.cpu().numpy().flatten())

    predictions = np.array(predictions)
    means = predictions.mean(axis=0)
    stds = predictions.std(axis=0)

    # Map outputs: [OL%, OPX%, CPX%, Fa, Fs, Wo]
    ol_pct = float(np.clip(means[0], 0, 100))
    opx_pct = float(np.clip(means[1], 0, 100))
    cpx_pct = float(np.clip(means[2], 0, 100))

    # Normalize modal abundances to sum to 100%
    total = ol_pct + opx_pct + cpx_pct
    if total > 0:
        ol_pct = ol_pct / total * 100
        opx_pct = opx_pct / total * 100
        cpx_pct = cpx_pct / total * 100

    return {
        "ol_pct": ol_pct,
        "opx_pct": opx_pct,
        "cpx_pct": cpx_pct,
        "fa_mol_pct": float(np.clip(means[3], 0, 100)),
        "fs_mol_pct": float(np.clip(means[4], 0, 100)),
        "wo_mol_pct": float(np.clip(means[5], 0, 100)),
        "ol_unc": float(stds[0]),
        "opx_unc": float(stds[1]),
        "cpx_unc": float(stds[2]),
        "fa_unc": float(stds[3]),
        "fs_unc": float(stds[4]),
        "wo_unc": float(stds[5]),
        "method": "korda2023",
    }


def _load_model():
    """Load the pre-trained CNN model from Asteroid-spectra package.

    Returns the model in eval mode.
    Raises ImportError if the package is not installed.
    """
    try:
        from asteroid_spectra.modules.NN_models_v2 import (
            MyNet as KordaCNN,
        )
    except ImportError:
        raise ImportError(
            "Asteroid-spectra package not found. "
            "Install with: pip install git+https://github.com/Sirrah91/Asteroid-spectra.git"
        )

    import torch

    model = KordaCNN()
    # Load pre-trained weights (packaged with Asteroid-spectra)
    try:
        from importlib.resources import files as pkg_files
        weights_path = str(
            pkg_files("asteroid_spectra").joinpath("weights", "model_weights.pth")
        )
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        # Fallback: look for weights in standard locations
        from pathlib import Path
        candidates = [
            Path.home() / ".asteroid_spectra" / "model_weights.pth",
            Path(__file__).parent.parent.parent / "data" / "models" / "korda2023_weights.pth",
        ]
        weights_path = None
        for p in candidates:
            if p.exists():
                weights_path = str(p)
                break
        if weights_path is None:
            raise FileNotFoundError(
                "CNN model weights not found. Download from "
                "https://github.com/Sirrah91/Asteroid-spectra and place in "
                "~/.asteroid_spectra/model_weights.pth"
            )

    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model


def assess_agreement(cnn_result, classical_result):
    """Compare CNN prediction with classical band analysis result.

    Parameters
    ----------
    cnn_result : dict from predict_spectrum()
    classical_result : dict or row from band_analysis table
        Must have 'ol_opx_ratio' key (classical ol/(ol+px) from Dunn).

    Returns
    -------
    'good', 'fair', 'poor', or 'no_classical'.
    """
    if classical_result is None:
        return "no_classical"

    classical_ol_opx = classical_result.get("ol_opx_ratio")
    if classical_ol_opx is None:
        return "no_classical"

    # Convert CNN OL% / (OL% + OPX%) to ol/(ol+px) ratio for comparison
    cnn_ol = cnn_result["ol_pct"]
    cnn_opx = cnn_result["opx_pct"]
    denom = cnn_ol + cnn_opx
    if denom <= 0:
        return "poor"

    cnn_ratio = cnn_ol / denom

    # Compare ratios (both are 0-1 scale)
    diff_pct = abs(cnn_ratio - classical_ol_opx) * 100

    if diff_pct <= GOOD_AGREEMENT:
        return "good"
    elif diff_pct <= FAIR_AGREEMENT:
        return "fair"
    else:
        return "poor"


def predict_asteroid(asteroid_id, conn, n_mc=DEFAULT_N_MC, _model=None):
    """Run CNN prediction for an asteroid and store results.

    Returns True if predicted, False if skipped.
    """
    # Domain check: silicate class only
    tax_row = conn.execute(
        "SELECT primary_class FROM taxonomy WHERE asteroid_id = ?",
        (asteroid_id,),
    ).fetchone()

    if tax_row is None or tax_row[0] not in SILICATE_CLASSES:
        return False

    # Best normalized spectrum with VNIR coverage
    row = conn.execute(
        "SELECT spectrum_id, wavelengths, reflectance "
        "FROM spectra "
        "WHERE asteroid_id = ? AND normalized = TRUE AND wl_max >= 2.0 "
        "ORDER BY "
        "  CASE quality_flag WHEN 'good' THEN 0 WHEN 'partial' THEN 1 ELSE 2 END, "
        "  (wl_max - wl_min) DESC "
        "LIMIT 1",
        (asteroid_id,),
    ).fetchone()

    if row is None:
        return False

    _spectrum_id, wl_blob, refl_blob = row
    wavelengths = np.frombuffer(wl_blob, dtype=np.float64).copy()
    reflectance = np.frombuffer(refl_blob, dtype=np.float64).copy()

    # Prefer deweathered spectrum when available
    try:
        dw_row = conn.execute(
            "SELECT deweathered FROM spectra WHERE spectrum_id = ?",
            (_spectrum_id,),
        ).fetchone()
        if dw_row and dw_row[0] is not None:
            reflectance = np.frombuffer(dw_row[0], dtype=np.float64).copy()
    except Exception:
        pass  # deweathered column may not exist in older schemas

    result = predict_spectrum(wavelengths, reflectance, n_mc=n_mc, _model=_model)
    if result is None:
        return False

    # Get classical band analysis for agreement check
    ba_row = conn.execute(
        "SELECT ol_opx_ratio FROM band_analysis WHERE asteroid_id = ?",
        (asteroid_id,),
    ).fetchone()
    classical = {"ol_opx_ratio": ba_row[0]} if ba_row else None
    agreement = assess_agreement(result, classical)

    conn.execute(
        "INSERT OR REPLACE INTO cnn_mineral "
        "(asteroid_id, ol_pct, opx_pct, cpx_pct, fa_mol_pct, fs_mol_pct, wo_mol_pct, "
        "ol_unc, opx_unc, cpx_unc, fa_unc, fs_unc, wo_unc, "
        "classical_agreement, method) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            asteroid_id,
            result["ol_pct"], result["opx_pct"], result["cpx_pct"],
            result["fa_mol_pct"], result["fs_mol_pct"], result["wo_mol_pct"],
            result["ol_unc"], result["opx_unc"], result["cpx_unc"],
            result["fa_unc"], result["fs_unc"], result["wo_unc"],
            agreement, result["method"],
        ),
    )
    return True


def predict_all(conn, n_mc=DEFAULT_N_MC):
    """Run CNN prediction for all eligible asteroids not yet in cnn_mineral.

    Eligible: silicate taxonomy + normalized VNIR spectrum + no existing entry.

    Returns the number of asteroids predicted.
    """
    placeholders = ",".join("?" for _ in SILICATE_CLASSES)
    rows = conn.execute(
        "SELECT DISTINCT s.asteroid_id "
        "FROM spectra s "
        "JOIN taxonomy t ON s.asteroid_id = t.asteroid_id "
        "LEFT JOIN cnn_mineral c ON s.asteroid_id = c.asteroid_id "
        f"WHERE s.normalized = TRUE AND s.wl_max >= 2.0 "
        f"AND t.primary_class IN ({placeholders}) "
        "AND c.asteroid_id IS NULL",
        (*SILICATE_CLASSES,),
    ).fetchall()

    if not rows:
        logger.info("No eligible asteroids for CNN mineral quantification")
        return 0

    # Load model once for batch
    model = None
    if _check_dependencies():
        try:
            model = _load_model()
        except (ImportError, FileNotFoundError):
            logger.warning("CNN model not available — skipping batch prediction")
            return 0

    count = 0
    for (asteroid_id,) in rows:
        try:
            if predict_asteroid(asteroid_id, conn, n_mc=n_mc, _model=model):
                count += 1
        except ImportError:
            logger.warning("CNN dependencies not available — stopping batch")
            break
        except Exception:
            logger.exception("Failed CNN prediction for asteroid %d", asteroid_id)

    conn.commit()
    logger.info("CNN predicted %d asteroids", count)
    return count
