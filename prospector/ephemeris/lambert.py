"""Lambert transfer solver and approach window search.

Solves Lambert's problem to find transfer orbits between Earth and NEO
candidates.  Uses two-body Keplerian propagation for grid search (fast)
with the universal variable Lambert algorithm (Bate, Mueller & White 1971;
Curtis 2014).

Usage:
    from prospector.ephemeris.lambert import search_windows, prefilter
    from prospector.db import get_connection

    conn = get_connection()
    windows = search_windows(25143, conn, horizon_years=10)
    # returns list of TransferWindow sorted by total delta-v

PRD reference: US-006 (Approach Window Calculator), FR-6.
"""

import logging
import sqlite3
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants (SI: metres, seconds, kilograms)
# ---------------------------------------------------------------------------
MU_SUN = 1.32712440018e20       # Sun standard gravitational parameter (m^3/s^2)
MU_EARTH = 3.986004418e14       # Earth gravitational parameter (m^3/s^2)
AU = 1.495978707e11             # Astronomical unit (m)
R_EARTH = 6378136.0             # Earth mean equatorial radius (m)
LEO_ALT = 200_000.0             # Default parking orbit altitude (m)
DAY = 86400.0                   # Seconds per day
J2000 = 2451545.0               # Julian date of J2000.0 epoch

# Earth J2000 osculating elements (Standish 1992 / Meeus)
_EARTH = {
    "a": 1.00000261,        # AU
    "e": 0.01671123,
    "i_deg": 0.00005,       # ecliptic inclination (essentially zero)
    "om_deg": -11.26064,    # longitude of ascending node
    "wbar_deg": 102.93768,  # longitude of perihelion
    "L0_deg": 100.46457,    # mean longitude at J2000
    "n_deg_day": 0.9856091, # mean motion (deg/day)
}


# ---------------------------------------------------------------------------
# Stumpff functions
# ---------------------------------------------------------------------------

def stumpff_c(z):
    """Stumpff function C(z) = (1 - cos(sqrt(z))) / z."""
    if z > 1e-6:
        sz = np.sqrt(z)
        return (1.0 - np.cos(sz)) / z
    elif z < -1e-6:
        sz = np.sqrt(-z)
        if sz > 500:
            return np.inf  # prevent cosh overflow
        return (np.cosh(sz) - 1.0) / (-z)
    else:
        return 0.5 - z / 24.0 + z * z / 720.0


def stumpff_s(z):
    """Stumpff function S(z) = (sqrt(z) - sin(sqrt(z))) / sqrt(z)^3."""
    if z > 1e-6:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / (sz * sz * sz)
    elif z < -1e-6:
        sz = np.sqrt(-z)
        if sz > 500:
            return np.inf  # prevent sinh overflow
        return (np.sinh(sz) - sz) / (sz * sz * sz)
    else:
        return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


# ---------------------------------------------------------------------------
# Kepler's equation
# ---------------------------------------------------------------------------

def solve_kepler(M, e, tol=1e-12, maxiter=50):
    """Solve Kepler's equation  M = E - e sin(E)  for eccentric anomaly E.

    Uses Newton-Raphson iteration.  *M* in radians, returns *E* in radians.
    """
    E = M + 0.85 * e * np.sign(np.sin(M))  # Markley starting value
    for _ in range(maxiter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = f / fp
        E -= dE
        if abs(dE) < tol:
            break
    return E


# ---------------------------------------------------------------------------
# Keplerian → Cartesian
# ---------------------------------------------------------------------------

def _rotation_matrix(i, om, w):
    """Rotation from perifocal to heliocentric ecliptic frame.

    All angles in **radians**.
    """
    ci, si = np.cos(i), np.sin(i)
    co, so = np.cos(om), np.sin(om)
    cw, sw = np.cos(w), np.sin(w)

    return np.array([
        [co * cw - so * sw * ci, -co * sw - so * cw * ci,  so * si],
        [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
        [sw * si,                 cw * si,                  ci],
    ])


def keplerian_to_cartesian(a_m, e, i_rad, om_rad, w_rad, M_rad, mu=MU_SUN):
    """Convert Keplerian elements to position & velocity vectors.

    Parameters
    ----------
    a_m : semi-major axis (m)
    e : eccentricity
    i_rad, om_rad, w_rad, M_rad : angles in radians
    mu : gravitational parameter (m^3/s^2)

    Returns
    -------
    (r_vec, v_vec) : 3-element numpy arrays in heliocentric ecliptic frame.
    """
    E = solve_kepler(M_rad, e)

    # True anomaly
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0),
    )

    # Distance
    r = a_m * (1.0 - e * np.cos(E))

    # Semi-latus rectum
    p = a_m * (1.0 - e * e)

    # Perifocal coordinates
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # Rotate to ecliptic
    R = _rotation_matrix(i_rad, om_rad, w_rad)
    return R @ r_pf, R @ v_pf


def earth_state(jd, mu=MU_SUN):
    """Heliocentric state vector of Earth at Julian date *jd*.

    Uses J2000 osculating elements with linear mean-longitude propagation.
    Accurate to ~0.01 AU over decades (sufficient for transfer grid search).
    """
    a_m = _EARTH["a"] * AU
    e = _EARTH["e"]
    i_rad = np.radians(_EARTH["i_deg"])
    om_rad = np.radians(_EARTH["om_deg"])
    wbar_rad = np.radians(_EARTH["wbar_deg"])
    w_rad = wbar_rad - om_rad  # argument of perihelion

    dt_days = jd - J2000
    L_deg = _EARTH["L0_deg"] + _EARTH["n_deg_day"] * dt_days
    M_rad = np.radians(L_deg - _EARTH["wbar_deg"])  # mean anomaly

    return keplerian_to_cartesian(a_m, e, i_rad, om_rad, w_rad, M_rad, mu)


def asteroid_state(a_m, e, i_rad, om_rad, w_rad, ma_rad, epoch_jd, jd, mu=MU_SUN):
    """Propagate asteroid Keplerian elements to Julian date *jd*.

    Parameters are in SI (a_m in metres, angles in radians).
    """
    n = np.sqrt(mu / a_m**3)  # mean motion (rad/s)
    dt_sec = (jd - epoch_jd) * DAY
    M_rad = ma_rad + n * dt_sec
    return keplerian_to_cartesian(a_m, e, i_rad, om_rad, w_rad, M_rad, mu)


# ---------------------------------------------------------------------------
# Lambert solver (universal variable method)
# ---------------------------------------------------------------------------

def solve_lambert(r1_vec, r2_vec, tof, mu=MU_SUN, prograde=True):
    """Solve Lambert's problem for a single-revolution transfer.

    Parameters
    ----------
    r1_vec, r2_vec : departure and arrival position vectors (m)
    tof : time of flight (s), must be > 0
    mu : gravitational parameter (m^3/s^2)
    prograde : True for prograde (short-way) transfers

    Returns
    -------
    (v1_vec, v2_vec) : departure and arrival velocity vectors (m/s),
    or None if the solver does not converge.

    Algorithm
    ---------
    Universal variable method (Bate, Mueller & White 1971; Curtis 2014
    Algorithm 5.2).  Root found by bisection for robustness.
    """
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    cos_dnu = np.clip(np.dot(r1_vec, r2_vec) / (r1 * r2), -1.0, 1.0)

    # Transfer angle — depends on direction
    cross = np.cross(r1_vec, r2_vec)
    if prograde:
        dnu = np.arccos(cos_dnu) if cross[2] >= 0 else 2.0 * np.pi - np.arccos(cos_dnu)
    else:
        dnu = np.arccos(cos_dnu) if cross[2] < 0 else 2.0 * np.pi - np.arccos(cos_dnu)

    # Auxiliary variable A
    sin_dnu = np.sin(dnu)
    if abs(sin_dnu) < 1e-14:
        return None  # degenerate (collinear or 180° transfer)

    A = sin_dnu * np.sqrt(r1 * r2 / (1.0 - cos_dnu))

    def _y(z):
        c = stumpff_c(z)
        s = stumpff_s(z)
        sqc = np.sqrt(max(c, 1e-30))
        return r1 + r2 + A * (z * s - 1.0) / sqc

    def _F(z):
        y = _y(z)
        if y < 0 or not np.isfinite(y):
            return 1e30
        c = stumpff_c(z)
        s = stumpff_s(z)
        if c < 1e-30:
            return 1e30
        val = (y / c) ** 1.5 * s + A * np.sqrt(y) - np.sqrt(mu) * tof
        return val if np.isfinite(val) else 1e30

    # --- Bracket the root via bisection ---
    # F(z_low) should be < 0 (hyperbolic, short TOF)
    # F(z_high) should be > 0 (tight elliptical, long TOF)
    # z < 0 → hyperbolic;  0 < z < (2π)² → elliptical single-rev

    # Find lower bound where F < 0
    z_low = 0.0
    f_low = _F(z_low)
    if f_low > 0:
        # TOF is shorter than parabolic — need hyperbolic (z < 0)
        z_low = -2.0
        for _ in range(20):
            f_low = _F(z_low)
            if f_low < 0 and np.isfinite(f_low):
                break
            z_low *= 2.0
            if z_low < -1e4:
                return None
    elif not np.isfinite(f_low):
        return None

    # Find upper bound where F > 0
    z_high = 4.0 * np.pi * np.pi * 0.99  # just below one revolution
    f_high = _F(z_high)
    if f_high <= 0 or not np.isfinite(f_high):
        # Try smaller upper bounds
        for z_try in [30.0, 20.0, 10.0, 5.0, 2.0, 1.0]:
            f_try = _F(z_try)
            if f_try > 0 and np.isfinite(f_try):
                z_high = z_try
                f_high = f_try
                break
        else:
            return None

    # Ensure we have a valid bracket
    if _F(z_low) * _F(z_high) > 0:
        return None

    # Bisection
    for _ in range(80):
        z_mid = 0.5 * (z_low + z_high)
        f_mid = _F(z_mid)
        if abs(f_mid) < 1e-6 or (z_high - z_low) < 1e-12:
            break
        if f_mid < 0:
            z_low = z_mid
        else:
            z_high = z_mid
    z = 0.5 * (z_low + z_high)

    # Lagrange coefficients
    y = _y(z)
    if y < 0 or not np.isfinite(y):
        return None

    f = 1.0 - y / r1
    g = A * np.sqrt(y / mu)
    g_dot = 1.0 - y / r2

    if abs(g) < 1e-20:
        return None

    v1 = (r2_vec - f * r1_vec) / g
    v2 = (g_dot * r2_vec - r1_vec) / g

    return v1, v2


# ---------------------------------------------------------------------------
# Delta-v computations
# ---------------------------------------------------------------------------

def departure_dv(v_inf_vec, r_park=R_EARTH + LEO_ALT, mu_body=MU_EARTH):
    """Δv for departure from circular parking orbit (Oberth manoeuvre).

    Parameters
    ----------
    v_inf_vec : hyperbolic excess velocity vector (m/s)
    r_park : parking orbit radius (m), default 200 km LEO
    mu_body : central body gravitational parameter

    Returns
    -------
    Δv in m/s.
    """
    v_inf = np.linalg.norm(v_inf_vec)
    v_circ = np.sqrt(mu_body / r_park)
    v_peri = np.sqrt(v_inf * v_inf + 2.0 * mu_body / r_park)
    return v_peri - v_circ


def arrival_dv(v_inf_vec):
    """Δv for rendezvous with a small body (negligible gravity well).

    Full braking from approach velocity is required.
    """
    return float(np.linalg.norm(v_inf_vec))


# ---------------------------------------------------------------------------
# Prefiltering
# ---------------------------------------------------------------------------

def prefilter(moid_au, incl_deg, a_au,
              max_moid_au=0.3, max_incl_deg=35.0, max_a_au=4.5):
    """Fast prefilter: reject asteroids unlikely to have low-Δv transfers.

    Parameters
    ----------
    moid_au : Minimum Orbit Intersection Distance (AU), may be None
    incl_deg : orbital inclination (degrees)
    a_au : semi-major axis (AU)
    max_moid_au : MOID cutoff (default 0.3 AU)
    max_incl_deg : inclination cutoff (default 35°)
    max_a_au : semi-major axis cutoff (default 4.5 AU)

    Returns True if the asteroid passes the filter.
    """
    if a_au is None or a_au <= 0:
        return False
    if a_au > max_a_au:
        return False
    if incl_deg is not None and incl_deg > max_incl_deg:
        return False
    if moid_au is not None and moid_au > max_moid_au:
        return False
    return True


def synodic_period_years(a_au):
    """Synodic period relative to Earth in years.

    T_syn = 1 / |1 - 1/P_ast|  where P_ast in years ≈ a^(3/2).
    """
    if a_au is None or a_au <= 0:
        return float("inf")
    p_ast = a_au ** 1.5  # years (Kepler's third law)
    denom = abs(1.0 - 1.0 / p_ast)
    if denom < 1e-10:
        return float("inf")  # co-orbital
    return 1.0 / denom


# ---------------------------------------------------------------------------
# TransferWindow result
# ---------------------------------------------------------------------------

@dataclass
class TransferWindow:
    """A feasible transfer opportunity."""
    launch_jd: float
    arrival_jd: float
    tof_days: float
    dv_departure_km_s: float    # from 200 km LEO
    dv_arrival_km_s: float      # rendezvous braking
    dv_total_km_s: float        # departure + arrival


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def search_windows(
    asteroid_id,
    conn,
    *,
    start_jd=None,
    horizon_years=10,
    step_days=5,
    tof_min_days=100,
    tof_max_days=500,
    tof_step_days=20,
    max_dv_km_s=20.0,
    top_n=10,
):
    """Search for transfer windows to an asteroid.

    Parameters
    ----------
    asteroid_id : int
    conn : sqlite3.Connection
    start_jd : starting Julian date (default: ~2026-01-01)
    horizon_years : search horizon in years
    step_days : launch-date grid spacing
    tof_min_days, tof_max_days, tof_step_days : TOF grid
    max_dv_km_s : reject windows above this total Δv
    top_n : return only the N best windows

    Returns
    -------
    List of TransferWindow sorted by total Δv ascending.
    """
    # Get orbital elements
    row = conn.execute(
        "SELECT a, e, i, om, w, ma, epoch FROM orbits "
        "WHERE asteroid_id = ?",
        (asteroid_id,),
    ).fetchone()

    if row is None:
        return []

    a_au, e, i_deg, om_deg, w_deg, ma_deg, epoch_jd = row
    if a_au is None or e is None:
        return []

    # Convert to SI / radians
    a_m = a_au * AU
    i_rad = np.radians(i_deg or 0.0)
    om_rad = np.radians(om_deg or 0.0)
    w_rad = np.radians(w_deg or 0.0)
    ma_rad = np.radians(ma_deg or 0.0)
    epoch_jd = epoch_jd or J2000

    if start_jd is None:
        start_jd = 2461041.5  # ~2026-02-23

    max_dv_m_s = max_dv_km_s * 1000.0
    windows = []

    n_launch = int(horizon_years * 365 / step_days)
    tof_range = list(range(tof_min_days, tof_max_days + 1, tof_step_days))

    for i_launch in range(n_launch):
        launch_jd = start_jd + i_launch * step_days

        # Earth state at launch
        r_earth, v_earth = earth_state(launch_jd)

        for tof_days in tof_range:
            arrival_jd = launch_jd + tof_days
            tof_sec = tof_days * DAY

            # Asteroid state at arrival
            r_ast, v_ast = asteroid_state(
                a_m, e, i_rad, om_rad, w_rad, ma_rad, epoch_jd, arrival_jd,
            )

            # Solve Lambert
            result = solve_lambert(r_earth, r_ast, tof_sec)
            if result is None:
                continue

            v1, v2 = result

            # Delta-v
            v_inf_dep = v1 - v_earth
            v_inf_arr = v2 - v_ast
            dv_dep = departure_dv(v_inf_dep)
            dv_arr = arrival_dv(v_inf_arr)
            dv_total = dv_dep + dv_arr

            if dv_total > max_dv_m_s:
                continue

            windows.append(TransferWindow(
                launch_jd=launch_jd,
                arrival_jd=arrival_jd,
                tof_days=tof_days,
                dv_departure_km_s=dv_dep / 1000.0,
                dv_arrival_km_s=dv_arr / 1000.0,
                dv_total_km_s=dv_total / 1000.0,
            ))

    # Sort by total Δv
    windows.sort(key=lambda w: w.dv_total_km_s)
    return windows[:top_n]


def search_all(
    conn,
    *,
    max_moid_au=0.3,
    max_incl_deg=35.0,
    top_n_per_asteroid=3,
    **search_kwargs,
):
    """Search transfer windows for all prefiltered asteroids.

    Returns dict mapping asteroid_id → list of TransferWindow.
    """
    rows = conn.execute(
        "SELECT asteroid_id, a, i, moid FROM orbits"
    ).fetchall()

    results = {}
    for asteroid_id, a_au, i_deg, moid in rows:
        if not prefilter(moid, i_deg, a_au,
                         max_moid_au=max_moid_au, max_incl_deg=max_incl_deg):
            continue

        try:
            windows = search_windows(
                asteroid_id, conn,
                top_n=top_n_per_asteroid,
                **search_kwargs,
            )
            if windows:
                results[asteroid_id] = windows
        except Exception:
            logger.exception(
                "Lambert search failed for asteroid %d", asteroid_id
            )

    logger.info(
        "Found transfer windows for %d of %d asteroids",
        len(results), len(rows),
    )
    return results
