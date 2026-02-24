"""Stage 3B: Hapke radiative transfer forward modeling.

Physics-based spectral modeling for top-N mining candidates.  Computes
synthetic VNIR spectra from mineral endmember assemblages via Hapke
(1981, 2012) intimate-mixture theory, then inverts to solve for modal
mineralogy, grain sizes, and space weathering state.

Endmembers: olivine, orthopyroxene, plagioclase, troilite, Fe-Ni metal.
Mixing: cross-sectional area fractions (intimate mixture).
Weathering: submicroscopic metallic iron (SMFe) per Hapke (2001).
Fitting: L-BFGS-B with free scaling + bootstrap uncertainty.

The inversion is ill-posed -- multiple parameter combinations can produce
similar spectra.  Results are presented with uncertainty ranges from
bootstrap resampling, not as point estimates.

References:
    Hapke (1981, 2012): Bidirectional reflectance theory
    Hapke (2001): Space weathering by nanophase iron
    Lawrence & Lucey (2007, JGR): Asteroid Hapke modeling
    Clark (1995): Mineral mixing model validation
    Cahill & Castellano (1971): Optical constants of metallic iron

Usage:
    from prospector.spectral.hapke_model import model_asteroid, model_all
    from prospector.db import get_connection

    conn = get_connection()
    result = model_asteroid(4179, conn)
    # or batch:
    all_results = model_all(conn, top_n=20)
"""

import logging
import sqlite3
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default viewing geometry (standard lab-like: i=30 deg, e=0 deg)
DEFAULT_INCIDENCE_DEG = 30.0
DEFAULT_EMISSION_DEG = 0.0

# Grain size bounds (um)
MIN_GRAIN_UM = 5.0
MAX_GRAIN_UM = 500.0

# SMFe volume fraction bounds
MIN_SMFE = 0.0
MAX_SMFE = 0.05

# Bootstrap iterations for uncertainty
DEFAULT_N_BOOTSTRAP = 50

# Wavelength range for fitting (um)
FIT_WL_MIN = 0.45
FIT_WL_MAX = 2.50

# Number of endmembers in the default set
N_ENDMEMBERS = 5

# Parameter vector length: (N-1) fractions + N grain sizes + 1 SMFe
N_PARAMS = (N_ENDMEMBERS - 1) + N_ENDMEMBERS + 1  # = 10

# Characteristic SMFe rind thickness (um) -- Hapke (2001)
SMFE_RIND_THICKNESS = 0.1

# ---------------------------------------------------------------------------
# Metallic iron optical constants (Cahill & Castellano 1971)
# ---------------------------------------------------------------------------

_FE_WL = np.array([0.40, 0.55, 0.70, 1.00, 1.50, 2.00, 2.50])
_FE_N = np.array([1.95, 2.87, 2.95, 3.05, 3.40, 3.80, 4.20])
_FE_K = np.array([3.14, 3.36, 3.50, 3.67, 4.00, 4.50, 5.00])


def _iron_nk(wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate metallic iron optical constants at given wavelengths (um)."""
    n = np.interp(wavelengths, _FE_WL, _FE_N)
    k = np.interp(wavelengths, _FE_WL, _FE_K)
    return n, k


# ---------------------------------------------------------------------------
# Endmember definitions
# ---------------------------------------------------------------------------


@dataclass
class Endmember:
    """A mineral endmember for Hapke modeling.

    Optical constants are parameterized as:
    - n: real refractive index (approximately constant over VNIR)
    - k(lam): imaginary index = baseline + sum of Gaussian absorption bands

    Each band tuple is (center_um, amplitude, sigma_um).
    """

    name: str
    n: float
    density: float  # g/cm3
    k_baseline: float
    bands: list[tuple[float, float, float]] = field(default_factory=list)

    def k(self, wavelengths: np.ndarray) -> np.ndarray:
        """Compute imaginary refractive index at given wavelengths (um)."""
        result = np.full_like(wavelengths, self.k_baseline, dtype=np.float64)
        for center, amplitude, sigma in self.bands:
            result += amplitude * np.exp(
                -((wavelengths - center) ** 2) / (2 * sigma**2)
            )
        return result


# Representative optical constants (Lucey 1998, Lawrence & Lucey 2007).
OLIVINE = Endmember(
    name="olivine",
    n=1.65,
    density=3.32,
    k_baseline=2e-5,
    bands=[(1.05, 3e-3, 0.15), (0.85, 5e-4, 0.05)],
)

PYROXENE = Endmember(
    name="pyroxene",
    n=1.68,
    density=3.35,
    k_baseline=1e-5,
    bands=[(0.90, 2e-3, 0.10), (1.90, 3e-3, 0.12)],
)

PLAGIOCLASE = Endmember(
    name="plagioclase",
    n=1.55,
    density=2.68,
    k_baseline=5e-6,
    bands=[(1.25, 5e-4, 0.08)],
)

TROILITE = Endmember(
    name="troilite",
    n=2.5,
    density=4.61,
    k_baseline=0.5,
    bands=[],
)

FE_NI_METAL = Endmember(
    name="metal",
    n=2.8,
    density=7.87,
    k_baseline=3.0,
    bands=[],
)

DEFAULT_ENDMEMBERS = [OLIVINE, PYROXENE, PLAGIOCLASE, TROILITE, FE_NI_METAL]

# Endmember names in canonical order (for DB columns and parameter mapping)
ENDMEMBER_NAMES = [em.name for em in DEFAULT_ENDMEMBERS]


# ---------------------------------------------------------------------------
# Hapke SSA computation
# ---------------------------------------------------------------------------


def compute_ssa(
    endmember: Endmember,
    wavelengths: np.ndarray,
    grain_size_um: float,
) -> np.ndarray:
    """Compute single scattering albedo (SSA) for an endmember.

    Uses Hapke (1981, 2012) theory:
        w = Se + (1 - Se)(1 - Si) * Theta / (1 - Si * Theta)

    where Se, Si are external/internal Fresnel reflection coefficients
    and Theta = exp(-alpha * D) is the internal transmission.

    Parameters
    ----------
    endmember : mineral with optical constants
    wavelengths : array in um
    grain_size_um : particle diameter in um

    Returns
    -------
    SSA array, values in [0, 1].
    """
    n = endmember.n
    k_arr = endmember.k(wavelengths)

    # Absorption coefficient (1/um)
    alpha = 4.0 * np.pi * k_arr / wavelengths

    # Internal transmission
    theta = np.exp(-alpha * grain_size_um)

    # External Fresnel reflection (normal-incidence average)
    Se = ((n - 1) ** 2 + k_arr**2) / ((n + 1) ** 2 + k_arr**2)

    # Internal Fresnel reflection (approximate, Hapke 2012 eq. 5.25)
    Si = float(np.clip(1.0 - 4.0 / (n * (n + 1) ** 2), 0.0, 1.0))

    # SSA
    denom = 1.0 - Si * theta
    denom = np.where(denom == 0, 1e-30, denom)
    w = Se + (1.0 - Se) * (1.0 - Si) * theta / denom

    return np.clip(w, 0.0, 1.0)


def mix_ssa(
    endmembers: list[Endmember],
    area_fractions: np.ndarray,
    wavelengths: np.ndarray,
    grain_sizes_um: np.ndarray,
) -> np.ndarray:
    """Compute mixed SSA for an intimate mixture.

    The mixed SSA is the cross-sectional-area-fraction-weighted average
    of the individual SSAs:  w_mix = sum(X_i * w_i).

    Parameters
    ----------
    endmembers : list of Endmember objects
    area_fractions : array of area fractions (should sum to 1)
    wavelengths : array in um
    grain_sizes_um : grain size per endmember in um

    Returns
    -------
    Mixed SSA array.
    """
    w_mix = np.zeros_like(wavelengths, dtype=np.float64)
    for em, frac, gs in zip(endmembers, area_fractions, grain_sizes_um):
        if frac > 0:
            w_mix += frac * compute_ssa(em, wavelengths, gs)
    return np.clip(w_mix, 0.0, 1.0)


def apply_smfe(
    ssa: np.ndarray,
    wavelengths: np.ndarray,
    smfe_vol_frac: float,
    n_host: float = 1.65,
) -> np.ndarray:
    """Apply submicroscopic metallic iron (SMFe) space weathering.

    SMFe particles in the regolith rind darken and redden the spectrum.
    Uses Hapke (2001): the SMFe contribution to the absorption coefficient is

        delta_alpha = 36 pi n_host f n_Fe k_Fe / (lam |m_Fe^2 - n_host^2|^2)

    Applied over an effective rind thickness to modify SSA.

    Parameters
    ----------
    ssa : SSA array before weathering
    wavelengths : array in um
    smfe_vol_frac : SMFe volume fraction (typically 0 to 0.01)
    n_host : real refractive index of host silicate

    Returns
    -------
    Modified SSA array (darkened and reddened).
    """
    if smfe_vol_frac <= 0:
        return ssa.copy()

    n_fe, k_fe = _iron_nk(wavelengths)

    # |m_Fe^2 - n_host^2|^2
    real_part = n_fe**2 - k_fe**2 - n_host**2
    imag_part = 2.0 * n_fe * k_fe
    denom = real_part**2 + imag_part**2
    denom = np.where(denom == 0, 1e-30, denom)

    delta_alpha = (
        36.0 * np.pi * n_host * smfe_vol_frac * n_fe * k_fe
    ) / (wavelengths * denom)

    # Damping through rind (enter + exit = factor of 2)
    damping = np.exp(-2.0 * delta_alpha * SMFE_RIND_THICKNESS)

    return np.clip(ssa * damping, 0.0, 1.0)


def ssa_to_reflectance(
    ssa: np.ndarray,
    incidence_deg: float = DEFAULT_INCIDENCE_DEG,
    emission_deg: float = DEFAULT_EMISSION_DEG,
) -> np.ndarray:
    """Convert SSA to radiance factor (I/F) via Hapke BRDF.

    Uses isotropic phase function (P=1) and no opposition effect (B=0):
        RADF = (w/4) * mu0/(mu0+mu) * H(mu0) * H(mu)

    with the Chandrasekhar H-function approximation
        H(x) = (1 + 2x) / (1 + 2 gamma x),  gamma = sqrt(1 - w).

    Parameters
    ----------
    ssa : single scattering albedo array
    incidence_deg : solar incidence angle (degrees)
    emission_deg : observer emission angle (degrees)

    Returns
    -------
    Radiance factor (RADF) array.
    """
    mu0 = np.cos(np.radians(incidence_deg))
    mu = np.cos(np.radians(emission_deg))

    gamma = np.sqrt(np.maximum(1.0 - ssa, 0.0))

    def _H(x, g):
        denom = 1.0 + 2.0 * g * x
        denom = np.where(denom == 0, 1e-30, denom)
        return (1.0 + 2.0 * x) / denom

    h_mu0 = _H(mu0, gamma)
    h_mu = _H(mu, gamma)

    radf = (ssa / 4.0) * (mu0 / (mu0 + mu)) * h_mu0 * h_mu
    return np.maximum(radf, 0.0)


# ---------------------------------------------------------------------------
# Forward model and fitting
# ---------------------------------------------------------------------------

# Parameter layout for 5 endmembers:
#   [0:4]  area fractions for first 4 (metal = 1 - sum)
#   [4:9]  grain sizes in um for all 5
#   [9]    SMFe volume fraction

_N_FRAC = N_ENDMEMBERS - 1  # 4 independent fractions


def _unpack_params(
    params: np.ndarray,
    endmembers: list[Endmember] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Unpack parameter vector into (area_fractions, grain_sizes, smfe).

    Returns
    -------
    fracs : array of length N_ENDMEMBERS, sums to 1
    grain_sizes : array of length N_ENDMEMBERS (um)
    smfe : SMFe volume fraction
    """
    n_em = len(endmembers) if endmembers else N_ENDMEMBERS
    n_frac = n_em - 1

    raw_fracs = np.clip(params[:n_frac], 0.0, 1.0)
    metal_frac = max(0.0, 1.0 - raw_fracs.sum())
    fracs = np.append(raw_fracs, metal_frac)

    # Renormalise if sum exceeded 1
    total = fracs.sum()
    if total > 0:
        fracs /= total
    else:
        fracs = np.ones(n_em) / n_em

    grain_sizes = np.clip(params[n_frac : n_frac + n_em], MIN_GRAIN_UM, MAX_GRAIN_UM)
    smfe = float(np.clip(params[n_frac + n_em], MIN_SMFE, MAX_SMFE))

    return fracs, grain_sizes, smfe


def forward_model(
    params: np.ndarray,
    wavelengths: np.ndarray,
    endmembers: list[Endmember] | None = None,
) -> np.ndarray:
    """Compute synthetic reflectance from model parameters.

    Parameters
    ----------
    params : parameter vector (see _unpack_params)
    wavelengths : array in um
    endmembers : list of Endmember objects (default: DEFAULT_ENDMEMBERS)

    Returns
    -------
    Synthetic reflectance (radiance factor) array.
    """
    if endmembers is None:
        endmembers = DEFAULT_ENDMEMBERS

    fracs, grain_sizes, smfe = _unpack_params(params, endmembers)

    ssa = mix_ssa(endmembers, fracs, wavelengths, grain_sizes)

    # Weighted-average host n for SMFe calculation
    active = [(em, f) for em, f in zip(endmembers, fracs) if f > 0.01]
    n_host = (
        sum(em.n * f for em, f in active) / sum(f for _, f in active)
        if active
        else 1.65
    )

    ssa_w = apply_smfe(ssa, wavelengths, smfe, n_host=n_host)
    return ssa_to_reflectance(ssa_w)


def _objective(
    params: np.ndarray,
    wavelengths: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    endmembers: list[Endmember] | None,
) -> float:
    """Weighted MSE objective with free scaling.

    Finds the optimal scale factor alpha = (obs . mod) / (mod . mod)
    and returns the weighted mean squared residual.
    """
    model = forward_model(params, wavelengths, endmembers)

    # Optimal free scale factor
    w_mod = weights * model
    denom = np.dot(w_mod, model)
    if denom == 0:
        return 1e10
    alpha = np.dot(weights * observed, model) / denom

    residual = observed - alpha * model
    return float(np.mean(weights * residual**2))


def _default_initial_guess() -> np.ndarray:
    """Default initial parameters: generic silicate mixture."""
    # fracs: ol=0.35, opx=0.30, plag=0.10, tro=0.05  =>  metal=0.20
    fracs = [0.35, 0.30, 0.10, 0.05]
    # grain sizes: 50 um for all
    grain_sizes = [50.0] * N_ENDMEMBERS
    # SMFe: small initial value
    smfe = [0.001]
    return np.array(fracs + grain_sizes + smfe)


def _param_bounds() -> list[tuple[float, float]]:
    """Parameter bounds for L-BFGS-B."""
    frac_bounds = [(0.0, 1.0)] * _N_FRAC
    gs_bounds = [(MIN_GRAIN_UM, MAX_GRAIN_UM)] * N_ENDMEMBERS
    smfe_bounds = [(MIN_SMFE, MAX_SMFE)]
    return frac_bounds + gs_bounds + smfe_bounds


def fit_spectrum(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    uncertainty: np.ndarray | None = None,
    endmembers: list[Endmember] | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> dict:
    """Fit Hapke model to an observed reflectance spectrum.

    Optimises mineral abundances, grain sizes, and SMFe via L-BFGS-B
    with free scaling.  Runs bootstrap with noise injection for
    uncertainty estimation.

    Parameters
    ----------
    wavelengths : observed wavelengths (um)
    reflectance : observed reflectance (normalised)
    uncertainty : per-channel uncertainty, or None (assumes 2%)
    endmembers : endmember set (default: DEFAULT_ENDMEMBERS)
    n_bootstrap : bootstrap iterations for uncertainty

    Returns
    -------
    dict with keys:
        fractions   - dict name -> area fraction
        grain_sizes - dict name -> grain size (um)
        smfe        - SMFe volume fraction
        fit_rmse    - RMS of best-fit residuals
        fit_rho     - Pearson correlation coefficient
        uncertainties - dict name -> 1-sigma uncertainty
        n_bootstrap - number of bootstrap iterations
        synthetic   - best-fit synthetic spectrum on input wavelengths
    """
    if endmembers is None:
        endmembers = DEFAULT_ENDMEMBERS

    # Mask to valid wavelength range and non-NaN values
    mask = (
        (wavelengths >= FIT_WL_MIN)
        & (wavelengths <= FIT_WL_MAX)
        & np.isfinite(reflectance)
    )
    wl_fit = wavelengths[mask]
    obs_fit = reflectance[mask]

    if len(wl_fit) < 20:
        raise ValueError(
            f"Insufficient valid wavelengths for fitting: {len(wl_fit)} < 20"
        )

    # Weights (inverse variance)
    if uncertainty is not None and len(uncertainty) == len(wavelengths):
        unc_fit = uncertainty[mask]
        weights = np.where(unc_fit > 0, 1.0 / unc_fit**2, 1.0)
    else:
        unc_fit = obs_fit * 0.02  # 2% assumed uncertainty
        weights = np.ones_like(obs_fit)

    # ----- Best-fit optimisation -----
    x0 = _default_initial_guess()
    bounds = _param_bounds()

    result = minimize(
        _objective,
        x0,
        args=(wl_fit, obs_fit, weights, endmembers),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    best_params = result.x
    fracs, grain_sizes, smfe = _unpack_params(best_params, endmembers)

    # Compute best-fit synthetic on full wavelength grid
    synthetic_full = forward_model(best_params, wavelengths, endmembers)

    # Best-fit metrics on the fitting region
    synth_fit = forward_model(best_params, wl_fit, endmembers)
    w_mod = weights * synth_fit
    denom = np.dot(w_mod, synth_fit)
    alpha_best = np.dot(weights * obs_fit, synth_fit) / denom if denom > 0 else 1.0
    residual = obs_fit - alpha_best * synth_fit
    fit_rmse = float(np.sqrt(np.mean(residual**2)))
    if np.std(obs_fit) > 0 and np.std(synth_fit) > 0:
        fit_rho = float(np.corrcoef(obs_fit, synth_fit)[0, 1])
    else:
        fit_rho = 0.0

    # ----- Bootstrap uncertainty -----
    rng = np.random.default_rng(42)
    boot_params = []
    for _ in range(n_bootstrap):
        noise = rng.normal(0, unc_fit)
        obs_noisy = obs_fit + noise

        res_b = minimize(
            _objective,
            best_params,  # warm-start from best fit
            args=(wl_fit, obs_noisy, weights, endmembers),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        boot_params.append(res_b.x)

    boot_arr = np.array(boot_params)

    # Extract uncertainties per parameter
    names = [em.name for em in endmembers]
    uncertainties = {}
    for i, name in enumerate(names[:-1]):
        uncertainties[f"{name}_frac"] = float(np.std(boot_arr[:, i]))
    # Metal fraction uncertainty from sum constraint
    metal_fracs = np.clip(1.0 - boot_arr[:, :_N_FRAC].sum(axis=1), 0.0, 1.0)
    uncertainties["metal_frac"] = float(np.std(metal_fracs))

    for i, name in enumerate(names):
        col = _N_FRAC + i
        uncertainties[f"{name}_grain"] = float(np.std(boot_arr[:, col]))

    uncertainties["smfe"] = float(np.std(boot_arr[:, -1]))

    return {
        "fractions": {name: float(f) for name, f in zip(names, fracs)},
        "grain_sizes": {name: float(gs) for name, gs in zip(names, grain_sizes)},
        "smfe": float(smfe),
        "fit_rmse": fit_rmse,
        "fit_rho": fit_rho,
        "uncertainties": uncertainties,
        "n_bootstrap": n_bootstrap,
        "synthetic": synthetic_full,
    }


# ---------------------------------------------------------------------------
# Database integration
# ---------------------------------------------------------------------------


def model_asteroid(
    asteroid_id: int,
    conn: sqlite3.Connection,
    *,
    endmembers: list[Endmember] | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> dict | None:
    """Fit Hapke model to an asteroid's spectrum and store results.

    Retrieves the best normalised spectrum (prefers deweathered),
    runs the forward-model optimisation, and writes to hapke_modeling.

    Parameters
    ----------
    asteroid_id : target asteroid
    conn : database connection
    endmembers : endmember set (default: DEFAULT_ENDMEMBERS)
    n_bootstrap : bootstrap iterations

    Returns
    -------
    dict with fit results, or None if no suitable spectrum.
    """
    if endmembers is None:
        endmembers = DEFAULT_ENDMEMBERS

    # Get best normalised spectrum (prefer deweathered)
    row = conn.execute(
        "SELECT wavelengths, reflectance, uncertainty, deweathered "
        "FROM spectra "
        "WHERE asteroid_id = ? AND normalized = TRUE "
        "ORDER BY "
        "  CASE quality_flag WHEN 'good' THEN 0 WHEN 'partial' THEN 1 ELSE 2 END, "
        "  (wl_max - wl_min) DESC "
        "LIMIT 1",
        (asteroid_id,),
    ).fetchone()

    if row is None:
        logger.info("No normalised spectrum for asteroid %d", asteroid_id)
        return None

    wl_blob, refl_blob, unc_blob, dw_blob = row
    wl = np.frombuffer(wl_blob, dtype=np.float64).copy()

    if dw_blob is not None:
        refl = np.frombuffer(dw_blob, dtype=np.float64).copy()
    else:
        refl = np.frombuffer(refl_blob, dtype=np.float64).copy()

    unc = None
    if unc_blob is not None:
        unc = np.frombuffer(unc_blob, dtype=np.float64).copy()

    try:
        result = fit_spectrum(
            wl, refl, uncertainty=unc, endmembers=endmembers,
            n_bootstrap=n_bootstrap,
        )
    except ValueError as exc:
        logger.warning("Cannot fit asteroid %d: %s", asteroid_id, exc)
        return None

    # Store in DB
    fracs = result["fractions"]
    grain_sizes = result["grain_sizes"]
    unc_dict = result["uncertainties"]
    names = ENDMEMBER_NAMES

    conn.execute(
        "INSERT OR REPLACE INTO hapke_modeling "
        "(asteroid_id, olivine_frac, pyroxene_frac, plagioclase_frac, "
        " troilite_frac, metal_frac, "
        " olivine_grain_um, pyroxene_grain_um, plagioclase_grain_um, "
        " troilite_grain_um, metal_grain_um, "
        " smfe_fraction, fit_rmse, fit_rho, "
        " olivine_unc, pyroxene_unc, metal_unc, smfe_unc, "
        " n_bootstrap) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            asteroid_id,
            fracs.get("olivine", 0),
            fracs.get("pyroxene", 0),
            fracs.get("plagioclase", 0),
            fracs.get("troilite", 0),
            fracs.get("metal", 0),
            grain_sizes.get("olivine", 50),
            grain_sizes.get("pyroxene", 50),
            grain_sizes.get("plagioclase", 50),
            grain_sizes.get("troilite", 50),
            grain_sizes.get("metal", 50),
            result["smfe"],
            result["fit_rmse"],
            result["fit_rho"],
            unc_dict.get("olivine_frac", 0),
            unc_dict.get("pyroxene_frac", 0),
            unc_dict.get("metal_frac", 0),
            unc_dict.get("smfe", 0),
            result["n_bootstrap"],
        ),
    )
    conn.commit()

    logger.info(
        "Hapke model for asteroid %d: RMSE=%.4f rho=%.3f "
        "ol=%.1f%% opx=%.1f%% met=%.1f%% SMFe=%.4f",
        asteroid_id,
        result["fit_rmse"],
        result["fit_rho"],
        fracs.get("olivine", 0) * 100,
        fracs.get("pyroxene", 0) * 100,
        fracs.get("metal", 0) * 100,
        result["smfe"],
    )

    return result


def model_all(
    conn: sqlite3.Connection,
    *,
    top_n: int = 20,
    endmembers: list[Endmember] | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> dict[int, dict]:
    """Run Hapke modeling for top-N scoring candidates.

    Selects asteroids by composite_score (descending) from the scores
    table, limited to those with normalised spectra.

    Parameters
    ----------
    conn : database connection
    top_n : number of top candidates to model
    endmembers : endmember set
    n_bootstrap : bootstrap iterations

    Returns
    -------
    dict mapping asteroid_id -> fit result dict.
    """
    # Get top-N candidates with normalised spectra
    rows = conn.execute(
        "SELECT s.asteroid_id "
        "FROM scores s "
        "JOIN spectra sp ON s.asteroid_id = sp.asteroid_id "
        "WHERE sp.normalized = TRUE "
        "GROUP BY s.asteroid_id "
        "ORDER BY s.composite_score DESC "
        "LIMIT ?",
        (top_n,),
    ).fetchall()

    if not rows:
        # Fallback: model all asteroids with normalised spectra
        rows = conn.execute(
            "SELECT DISTINCT asteroid_id FROM spectra "
            "WHERE normalized = TRUE "
            "LIMIT ?",
            (top_n,),
        ).fetchall()

    if not rows:
        logger.info("No candidates for Hapke modeling")
        return {}

    results = {}
    for (asteroid_id,) in rows:
        try:
            result = model_asteroid(
                asteroid_id, conn, endmembers=endmembers,
                n_bootstrap=n_bootstrap,
            )
            if result is not None:
                results[asteroid_id] = result
        except Exception:
            logger.exception("Hapke modeling failed for asteroid %d", asteroid_id)

    logger.info("Completed Hapke modeling for %d asteroids", len(results))
    return results
