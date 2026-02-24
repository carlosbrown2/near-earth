"""Granvik et al. (2018) debiased population prior for NEO taxonomy.

Maps orbital elements (a, e, i) to source region probabilities and then to
a taxonomy class distribution.  Every NEO gets a 17-class Mahlke probability
vector even without spectral data — this is the prior that the Monte Carlo
scorer uses for uncharacterised objects.

Source regions (Granvik et al. 2018, Icarus 312, 181-207):
    nu6      — ν₆ secular resonance (inner main belt, ~2.1 AU)
    3:1      — 3:1 mean-motion resonance (~2.5 AU)
    5:2      — 5:2 mean-motion resonance (~2.82 AU)
    2:1      — 2:1 mean-motion resonance (~3.27 AU)
    hungaria — Hungaria region (~1.9 AU, high i)
    phocaea  — Phocaea region (~2.4 AU, high i)
    jfc      — Jupiter-family comets (Tisserand 2 < T_J < 3)

Source-to-taxonomy mapping derived from:
    DeMeo & Carry (2014, Nature 505, 629-631) — main belt compositional gradient
    Binzel et al. (2019, Icarus 324, 41-76) — NEO source-taxonomy correlations
    Carry (2012, P&SS 73, 98-118) — taxonomy class properties

Usage:
    from prospector.scoring.granvik_prior import taxonomy_prior
    prob_vector = taxonomy_prior(a=1.64, e=0.77, i_deg=26.1)
"""

import math

import numpy as np

# Mahlke 2022 taxonomy classes — canonical order (must match taxonomy.py)
MAHLKE_CLASSES = [
    "A", "B", "C", "Ch", "D", "E", "K", "L", "M",
    "O", "P", "Q", "R", "S", "V", "X", "Z",
]

# Index lookup for building probability vectors
_CLASS_IDX = {c: i for i, c in enumerate(MAHLKE_CLASSES)}

# 7 Granvik source regions
SOURCE_REGIONS = ["nu6", "3:1", "5:2", "2:1", "hungaria", "phocaea", "jfc"]

# Jupiter semi-major axis (AU) for Tisserand parameter
_A_JUPITER = 5.2044

# ---------------------------------------------------------------------------
# Source region characteristic orbital elements
# From Granvik et al. (2018) Table 1 + Bottke et al. (2002)
# Each entry: (a_center, e_center, i_center_deg, sigma_a, sigma_e, sigma_i)
# The sigmas define Gaussian kernel widths for proximity weighting.
# ---------------------------------------------------------------------------
_REGION_PARAMS = {
    "nu6": {"a": 2.12, "e": 0.30, "i": 6.0, "sa": 0.30, "se": 0.20, "si": 8.0},
    "3:1": {"a": 2.50, "e": 0.35, "i": 10.0, "sa": 0.20, "se": 0.20, "si": 10.0},
    "5:2": {"a": 2.82, "e": 0.40, "i": 10.0, "sa": 0.20, "se": 0.20, "si": 10.0},
    "2:1": {"a": 3.27, "e": 0.45, "i": 12.0, "sa": 0.25, "se": 0.20, "si": 12.0},
    "hungaria": {"a": 1.90, "e": 0.10, "i": 22.0, "sa": 0.15, "se": 0.10, "si": 6.0},
    "phocaea": {"a": 2.36, "e": 0.20, "i": 24.0, "sa": 0.20, "se": 0.15, "si": 6.0},
    "jfc": {"a": 3.50, "e": 0.55, "i": 12.0, "sa": 1.00, "se": 0.20, "si": 15.0},
}

# Relative steady-state delivery rates from Granvik et al. (2018) Table 3.
# These weight the orbital-element proximity scores.
_DELIVERY_RATES = {
    "nu6": 0.372,
    "3:1": 0.275,
    "5:2": 0.130,
    "2:1": 0.034,
    "hungaria": 0.063,
    "phocaea": 0.021,
    "jfc": 0.061,
}

# ---------------------------------------------------------------------------
# Source region → Mahlke taxonomy class probability vectors
# Derived from DeMeo & Carry (2014) main belt compositional gradient and
# Binzel et al. (2019) NEO source-taxonomy correlations.
#
# Each dict maps Mahlke class → probability.  Unlisted classes get 0.
# Probabilities must sum to 1.0.
# ---------------------------------------------------------------------------
_SOURCE_TAXONOMY = {
    # Inner belt: S-complex dominated with subordinate E, X, Q, V
    "nu6": {
        "S": 0.52, "Q": 0.10, "V": 0.05, "L": 0.04, "K": 0.03,
        "A": 0.02, "R": 0.01, "O": 0.01,
        "C": 0.06, "Ch": 0.02, "B": 0.01,
        "E": 0.04, "M": 0.03, "X": 0.03, "P": 0.01, "D": 0.01,
        "Z": 0.01,
    },
    # Central belt: S/C mixed
    "3:1": {
        "S": 0.35, "Q": 0.05, "L": 0.04, "K": 0.05,
        "A": 0.01, "V": 0.02, "R": 0.01, "O": 0.01,
        "C": 0.18, "Ch": 0.08, "B": 0.03,
        "M": 0.04, "X": 0.04, "E": 0.02, "P": 0.03, "D": 0.02,
        "Z": 0.02,
    },
    # Mid-outer belt: C-complex increasing
    "5:2": {
        "S": 0.15, "Q": 0.02, "K": 0.03, "L": 0.02,
        "A": 0.01, "V": 0.01,
        "C": 0.30, "Ch": 0.12, "B": 0.06,
        "P": 0.08, "D": 0.06, "X": 0.05, "M": 0.03, "E": 0.01,
        "R": 0.01, "O": 0.01, "Z": 0.03,
    },
    # Outer belt: C/P/D dominated
    "2:1": {
        "S": 0.05, "Q": 0.01, "K": 0.02, "L": 0.01,
        "C": 0.32, "Ch": 0.10, "B": 0.08,
        "P": 0.15, "D": 0.12, "X": 0.04, "M": 0.02, "E": 0.01,
        "A": 0.01, "V": 0.01, "R": 0.01, "O": 0.01, "Z": 0.03,
    },
    # Hungaria: E-type dominated (enstatite achondrites)
    "hungaria": {
        "E": 0.50, "X": 0.12, "S": 0.15, "L": 0.05,
        "C": 0.04, "Ch": 0.02, "M": 0.03, "A": 0.02,
        "Q": 0.02, "K": 0.01, "B": 0.01, "P": 0.01,
        "V": 0.01, "D": 0.01,
    },
    # Phocaea: S-complex with K/L component
    "phocaea": {
        "S": 0.50, "K": 0.08, "L": 0.07, "Q": 0.04,
        "C": 0.08, "Ch": 0.03, "X": 0.05, "M": 0.03,
        "E": 0.03, "B": 0.02, "A": 0.02, "V": 0.02,
        "P": 0.01, "D": 0.01, "R": 0.01,
    },
    # JFC: D/P-type dominated (cometary)
    "jfc": {
        "D": 0.40, "P": 0.20, "C": 0.10, "Ch": 0.03, "B": 0.05,
        "X": 0.08, "S": 0.04, "M": 0.02, "K": 0.02,
        "L": 0.01, "Q": 0.01, "E": 0.01, "A": 0.01,
        "V": 0.01, "Z": 0.01,
    },
}


def _build_source_vectors() -> dict[str, np.ndarray]:
    """Pre-compute 17-element probability vectors for each source region."""
    vectors = {}
    for region, class_probs in _SOURCE_TAXONOMY.items():
        vec = np.zeros(len(MAHLKE_CLASSES), dtype=np.float64)
        for cls, prob in class_probs.items():
            vec[_CLASS_IDX[cls]] = prob
        # Normalise to exactly 1.0 (accounts for rounding in the tables)
        total = vec.sum()
        if total > 0:
            vec /= total
        vectors[region] = vec
    return vectors


_SOURCE_VECTORS = _build_source_vectors()


def tisserand(a: float, e: float, i_deg: float) -> float:
    """Tisserand parameter relative to Jupiter.

    T_J = a_J/a + 2·cos(i)·sqrt(a/a_J · (1 - e²))

    Parameters
    ----------
    a : float
        Semi-major axis in AU.
    e : float
        Eccentricity.
    i_deg : float
        Inclination in degrees.

    Returns
    -------
    float
        Tisserand parameter.
    """
    if a <= 0:
        return float("inf")
    i_rad = math.radians(i_deg)
    return _A_JUPITER / a + 2.0 * math.cos(i_rad) * math.sqrt(a / _A_JUPITER * (1.0 - e * e))


def _gaussian_weight(val: float, center: float, sigma: float) -> float:
    """Unnormalised Gaussian kernel weight."""
    if sigma <= 0:
        return 0.0
    return math.exp(-0.5 * ((val - center) / sigma) ** 2)


def source_region_probabilities(
    a: float, e: float, i_deg: float
) -> dict[str, float]:
    """Compute probability of origin from each Granvik source region.

    Uses Gaussian proximity kernels in (a, e, i) space weighted by the
    published steady-state delivery rates from Granvik et al. (2018).
    JFC probability receives an additional Tisserand boost when 2 < T_J < 3.

    Parameters
    ----------
    a : float
        Semi-major axis in AU.
    e : float
        Eccentricity (0–1).
    i_deg : float
        Inclination in degrees.

    Returns
    -------
    dict
        {region_name: probability} summing to 1.0.
    """
    raw_weights = {}

    for region in SOURCE_REGIONS:
        p = _REGION_PARAMS[region]
        w_a = _gaussian_weight(a, p["a"], p["sa"])
        w_e = _gaussian_weight(e, p["e"], p["se"])
        w_i = _gaussian_weight(i_deg, p["i"], p["si"])
        raw_weights[region] = w_a * w_e * w_i * _DELIVERY_RATES[region]

    # Tisserand boost for JFC: objects with 2 < T_J < 3 are dynamically
    # consistent with Jupiter-family comets (Levison 1996).
    t_j = tisserand(a, e, i_deg)
    if 2.0 < t_j < 3.0:
        raw_weights["jfc"] *= 3.0  # boost cometary association
    elif t_j >= 3.0:
        raw_weights["jfc"] *= 0.1  # suppress for main-belt-like orbits

    # Normalise to probabilities
    total = sum(raw_weights.values())
    if total == 0:
        # Fallback: uniform prior if no region matches (extreme orbit)
        uniform = 1.0 / len(SOURCE_REGIONS)
        return {r: uniform for r in SOURCE_REGIONS}

    return {r: w / total for r, w in raw_weights.items()}


def taxonomy_prior(a: float, e: float, i_deg: float) -> np.ndarray:
    """Compute 17-class Mahlke taxonomy prior from orbital elements.

    Marginalises over source regions:
        P(class | a,e,i) = Σ_r  P(class | region_r) × P(region_r | a,e,i)

    Parameters
    ----------
    a : float
        Semi-major axis in AU.
    e : float
        Eccentricity (0–1).
    i_deg : float
        Inclination in degrees.

    Returns
    -------
    numpy.ndarray
        17-element probability vector in MAHLKE_CLASSES order, summing to 1.0.
    """
    region_probs = source_region_probabilities(a, e, i_deg)
    prior = np.zeros(len(MAHLKE_CLASSES), dtype=np.float64)

    for region, p_region in region_probs.items():
        prior += p_region * _SOURCE_VECTORS[region]

    # Safety normalisation (should already sum to ~1.0)
    total = prior.sum()
    if total > 0:
        prior /= total

    return prior


def taxonomy_prior_dict(a: float, e: float, i_deg: float) -> dict[str, float]:
    """Like :func:`taxonomy_prior` but returns {class_name: probability} dict."""
    vec = taxonomy_prior(a, e, i_deg)
    return {cls: float(vec[i]) for i, cls in enumerate(MAHLKE_CLASSES)}
