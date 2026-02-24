"""Phase curve analysis using the H, G1, G2 photometric system.

Fits the three-parameter H, G1, G2 phase function (Muinonen et al. 2010)
to reduced magnitude vs phase angle observations.  Phase curve parameters
constrain albedo, surface roughness, and taxonomy without spectroscopy.

Basis functions Phi_1, Phi_2, Phi_3 are cubic-spline interpolations of the
tabulated values from Muinonen et al. (2010), Table 3.

Key outputs per asteroid:
- H (absolute magnitude), G1, G2 (slope parameters)
- Phase integral q = 0.009082 + 0.4061*G1 + 0.8768*G2
- Taxonomy hint from G1-G2 clustering (Penttila et al. 2016)
- Geometric albedo (when diameter is known): p_V = (1329/D_km)^2 * 10^(-0.4*H)

Usage:
    from prospector.spectral.phase_curve import fit_phase_curve, fit_all
    result = fit_phase_curve(alpha_deg, reduced_mag)
    fit_all(conn)
"""

import logging
import sqlite3

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from prospector.db import init_schema

logger = logging.getLogger(__name__)

# --- Basis function tabulation (Muinonen et al. 2010, Table 3) ---

_ALPHA_TABLE = np.array([
    0.0, 0.3, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0,
    30.0, 60.0, 90.0, 120.0, 150.0,
])

_PHI1_TABLE = np.array([
    1.0, 0.8340, 0.6300, 0.4684, 0.2833, 0.1341, 0.0756,
    0.0330, 0.0135, 0.0013, 2.4e-4, 0.0, 0.0,
])

_PHI2_TABLE = np.array([
    1.0, 0.9194, 0.7823, 0.6327, 0.4092, 0.2085, 0.1220,
    0.0556, 0.0226, 0.0020, 3.2e-4, 0.0, 0.0,
])

_PHI3_TABLE = np.array([
    1.0, 0.9983, 0.9899, 0.9740, 0.9325, 0.8330, 0.7437,
    0.5683, 0.3918, 0.1150, 0.0282, 0.0048, 0.0002,
])

# Build cubic splines once at import time
_phi1_spline = CubicSpline(_ALPHA_TABLE, _PHI1_TABLE, bc_type='natural')
_phi2_spline = CubicSpline(_ALPHA_TABLE, _PHI2_TABLE, bc_type='natural')
_phi3_spline = CubicSpline(_ALPHA_TABLE, _PHI3_TABLE, bc_type='natural')


def _phi1(alpha_deg: np.ndarray) -> np.ndarray:
    """Basis function Phi_1 evaluated at phase angles (degrees)."""
    return np.clip(_phi1_spline(alpha_deg), 0.0, 1.0)


def _phi2(alpha_deg: np.ndarray) -> np.ndarray:
    """Basis function Phi_2 evaluated at phase angles (degrees)."""
    return np.clip(_phi2_spline(alpha_deg), 0.0, 1.0)


def _phi3(alpha_deg: np.ndarray) -> np.ndarray:
    """Basis function Phi_3 evaluated at phase angles (degrees)."""
    return np.clip(_phi3_spline(alpha_deg), 0.0, 1.0)


def h_g1g2_model(alpha_deg: np.ndarray, H: float, G1: float, G2: float) -> np.ndarray:
    """Compute reduced magnitude using the H, G1, G2 phase function.

    Parameters
    ----------
    alpha_deg : array-like
        Phase angles in degrees.
    H : float
        Absolute magnitude.
    G1, G2 : float
        Slope parameters. Constrained: G1 >= 0, G2 >= 0, G1+G2 <= 1.

    Returns
    -------
    np.ndarray
        Predicted reduced magnitudes V(alpha).
    """
    alpha_deg = np.asarray(alpha_deg, dtype=np.float64)
    G3 = 1.0 - G1 - G2
    flux = G1 * _phi1(alpha_deg) + G2 * _phi2(alpha_deg) + G3 * _phi3(alpha_deg)
    # Clamp to avoid log10(0)
    flux = np.maximum(flux, 1e-30)
    return H - 2.5 * np.log10(flux)


def phase_integral(G1: float, G2: float) -> float:
    """Compute the phase integral q from G1, G2.

    q = 0.009082 + 0.4061*G1 + 0.8768*G2  (Muinonen et al. 2010)
    """
    return 0.009082 + 0.4061 * G1 + 0.8768 * G2


def geometric_albedo(H: float, diameter_km: float) -> float:
    """Compute geometric albedo from absolute magnitude and diameter.

    p_V = (1329 / D_km)^2 * 10^(-0.4 * H)
    """
    if diameter_km <= 0:
        return float('nan')
    return (1329.0 / diameter_km) ** 2 * 10 ** (-0.4 * H)


# --- Taxonomy clustering in G1-G2 space (Penttila et al. 2016) ---
# Centroids in (G1, G2) space for major taxonomic classes
_TAX_CENTROIDS = {
    'S': (0.25, 0.21),
    'C': (0.82, 0.02),
    'X': (0.58, 0.10),
    'B': (0.09, 0.55),
    'D': (0.90, 0.01),
    'V': (0.19, 0.28),
    'L': (0.45, 0.12),
    'Q': (0.28, 0.19),
}

# Maximum distance from centroid to count as a match
_TAX_MAX_DIST = 0.25


def taxonomy_hint(G1: float, G2: float) -> str | None:
    """Suggest a taxonomic class from G1, G2 position.

    Uses Euclidean distance to class centroids in G1-G2 space
    (Penttila et al. 2016).  Returns None if no centroid is close enough.
    """
    best_cls = None
    best_dist = _TAX_MAX_DIST
    for cls, (cg1, cg2) in _TAX_CENTROIDS.items():
        d = ((G1 - cg1) ** 2 + (G2 - cg2) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_cls = cls
    return best_cls


# --- Phase curve fitting ---

# Minimum observations required for a meaningful fit
MIN_OBS = 5


def fit_phase_curve(
    alpha_deg: np.ndarray,
    reduced_mag: np.ndarray,
    *,
    mag_unc: np.ndarray | None = None,
) -> dict | None:
    """Fit H, G1, G2 to observed phase curve data.

    Uses a two-stage approach: coarse grid search over (G1, G2) with
    analytical H solution in flux space, then local refinement.  This is
    more robust than pure gradient descent for the H,G1,G2 system.

    Parameters
    ----------
    alpha_deg : array-like
        Phase angles in degrees.
    reduced_mag : array-like
        Reduced magnitudes (corrected for helio/geocentric distances).
    mag_unc : array-like, optional
        Per-observation magnitude uncertainties for weighted fitting.

    Returns
    -------
    dict or None
        Keys: H, G1, G2, h_unc, g1_unc, g2_unc, phase_integral,
              rms_residual, n_obs, alpha_min, alpha_max, taxonomy_hint.
        None if fewer than MIN_OBS valid observations.
    """
    alpha_deg = np.asarray(alpha_deg, dtype=np.float64)
    reduced_mag = np.asarray(reduced_mag, dtype=np.float64)

    # Remove NaN/inf observations
    valid = np.isfinite(alpha_deg) & np.isfinite(reduced_mag)
    alpha_deg = alpha_deg[valid]
    reduced_mag = reduced_mag[valid]

    if len(alpha_deg) < MIN_OBS:
        return None

    if mag_unc is not None:
        mag_unc = np.asarray(mag_unc, dtype=np.float64)[valid]
        mag_unc = np.maximum(mag_unc, 0.01)
        weights = 1.0 / mag_unc ** 2
    else:
        weights = np.ones_like(alpha_deg)

    phi1 = _phi1(alpha_deg)
    phi2 = _phi2(alpha_deg)
    phi3 = _phi3(alpha_deg)

    # Stage 1: Grid search over (G1, G2) in magnitude space.
    # For each (G1, G2), the optimal H is the weighted mean offset
    # between observed magnitudes and the phase function shape.
    n_grid = 21
    best_wss = np.inf
    best_g1, best_g2, best_H = 0.15, 0.15, float(np.median(reduced_mag))

    for ig1 in range(n_grid):
        g1 = ig1 / (n_grid - 1)
        n_g2 = max(2, int((1.0 - g1) * (n_grid - 1)) + 1)
        for ig2 in range(n_g2):
            g2 = ig2 * (1.0 - g1) / (n_g2 - 1) if n_g2 > 1 else 0.0
            phase_fn = g1 * phi1 + g2 * phi2 + (1 - g1 - g2) * phi3
            phase_fn = np.maximum(phase_fn, 1e-30)
            shape = -2.5 * np.log10(phase_fn)  # magnitude offset from H
            # Optimal H: minimize sum(w * (mag - (H + shape))^2)
            # → H = weighted_mean(mag - shape)
            H_opt = np.sum(weights * (reduced_mag - shape)) / np.sum(weights)
            model = H_opt + shape
            wss = np.sum(weights * (reduced_mag - model) ** 2)
            if wss < best_wss:
                best_wss = wss
                best_g1, best_g2, best_H = g1, g2, H_opt

    H0 = best_H

    # Stage 2: Local refinement in magnitude space from the grid-search start
    x0 = np.array([H0, best_g1, best_g2])

    def objective(params):
        H, G1, G2 = params
        model = h_g1g2_model(alpha_deg, H, G1, G2)
        return np.sum(weights * (reduced_mag - model) ** 2)

    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=[(None, None), (0.0, 1.0), (0.0, 1.0)],
        options={'maxiter': 500},
    )

    H_fit, G1_fit, G2_fit = result.x

    # Enforce G1 + G2 <= 1
    if G1_fit + G2_fit > 1.0:
        scale = 1.0 / (G1_fit + G2_fit)
        G1_fit *= scale
        G2_fit *= scale

    # Compute residuals and RMS in magnitude space
    model_mag = h_g1g2_model(alpha_deg, H_fit, G1_fit, G2_fit)
    residuals = reduced_mag - model_mag
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    # Estimate uncertainties from Hessian
    h_unc, g1_unc, g2_unc = _estimate_uncertainties(
        alpha_deg, reduced_mag, weights, H_fit, G1_fit, G2_fit,
    )

    return {
        'H': float(H_fit),
        'G1': float(G1_fit),
        'G2': float(G2_fit),
        'h_unc': h_unc,
        'g1_unc': g1_unc,
        'g2_unc': g2_unc,
        'phase_integral': phase_integral(G1_fit, G2_fit),
        'rms_residual': rms,
        'n_obs': len(alpha_deg),
        'alpha_min': float(alpha_deg.min()),
        'alpha_max': float(alpha_deg.max()),
        'taxonomy_hint': taxonomy_hint(G1_fit, G2_fit),
    }


def _estimate_uncertainties(
    alpha_deg, reduced_mag, weights, H, G1, G2,
) -> tuple[float, float, float]:
    """Estimate parameter uncertainties via finite-difference Hessian."""
    n = len(alpha_deg)
    if n <= 3:
        return float('nan'), float('nan'), float('nan')

    params = np.array([H, G1, G2])
    eps = np.array([1e-4, 1e-4, 1e-4])

    def chi2(p):
        model = h_g1g2_model(alpha_deg, p[0], max(0, p[1]), max(0, p[2]))
        return np.sum(weights * (reduced_mag - model) ** 2)

    c0 = chi2(params)
    uncertainties = []
    for i in range(3):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += eps[i]
        p_minus[i] -= eps[i]
        d2 = (chi2(p_plus) + chi2(p_minus) - 2 * c0) / eps[i] ** 2
        if d2 > 0:
            uncertainties.append(float(np.sqrt(2.0 / d2)))
        else:
            uncertainties.append(float('nan'))

    return tuple(uncertainties)


# --- Database integration ---

def reduce_magnitude(
    apparent_mag: float, r_au: float, delta_au: float
) -> float:
    """Convert apparent magnitude to reduced magnitude.

    V_reduced = V_apparent - 5 * log10(r * delta)

    Parameters
    ----------
    apparent_mag : float
        Observed apparent visual magnitude.
    r_au : float
        Heliocentric distance in AU.
    delta_au : float
        Geocentric distance in AU.
    """
    return apparent_mag - 5.0 * np.log10(r_au * delta_au)


def fit_all(
    conn: sqlite3.Connection,
    *,
    table: str = 'mpc_photometry',
    min_obs: int = MIN_OBS,
) -> int:
    """Fit H, G1, G2 for all asteroids with photometry in the database.

    Reads phase angle and reduced magnitude from the photometry table,
    fits the H, G1, G2 model, and writes results to the phase_curve table.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection with schema initialized.
    table : str
        Name of the table containing photometry observations.
        Expected columns: asteroid_id, phase_angle_deg, reduced_mag.
    min_obs : int
        Minimum number of observations required for fitting.

    Returns
    -------
    int
        Number of asteroids successfully fitted.
    """
    init_schema(conn)

    # Check if the photometry table exists
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if table not in tables:
        logger.warning("Photometry table '%s' not found — skipping fit_all", table)
        return 0

    # Get distinct asteroids with enough observations
    rows = conn.execute(
        f"SELECT asteroid_id FROM {table} "  # noqa: S608
        f"GROUP BY asteroid_id HAVING COUNT(*) >= ?",
        (min_obs,),
    ).fetchall()

    count = 0
    for (asteroid_id,) in rows:
        obs = conn.execute(
            f"SELECT phase_angle_deg, reduced_mag, mag_unc FROM {table} "  # noqa: S608
            "WHERE asteroid_id = ?",
            (asteroid_id,),
        ).fetchall()

        alpha = np.array([r[0] for r in obs])
        mag = np.array([r[1] for r in obs])
        unc_vals = [r[2] for r in obs]
        unc = np.array(unc_vals) if all(v is not None for v in unc_vals) else None

        result = fit_phase_curve(alpha, mag, mag_unc=unc)
        if result is None:
            continue

        conn.execute(
            "INSERT OR REPLACE INTO phase_curve "
            "(asteroid_id, h_fit, g1, g2, h_unc, g1_unc, g2_unc, "
            " phase_integral, n_obs, rms_residual, alpha_min, alpha_max, "
            " taxonomy_hint, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'MPC')",
            (
                asteroid_id,
                result['H'],
                result['G1'],
                result['G2'],
                result['h_unc'],
                result['g1_unc'],
                result['g2_unc'],
                result['phase_integral'],
                result['n_obs'],
                result['rms_residual'],
                result['alpha_min'],
                result['alpha_max'],
                result['taxonomy_hint'],
            ),
        )
        count += 1

    conn.commit()
    logger.info("Fitted phase curves for %d asteroids", count)
    return count
