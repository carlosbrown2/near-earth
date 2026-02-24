"""Monte Carlo composite mining scorer.

Implements the scoring formula from PRD §6 US-005:

    Expected Mining Utility ≈ estimated_mass × grade × recoverability
                              × unit_value × confidence × accessibility

For each asteroid, the scorer:
1. Gets taxonomy probability distribution (spectral or Granvik prior)
2. Runs N Monte Carlo samples: draws taxonomy class, samples density/grade
3. Computes mass, material values, and composite score
4. Stores mean score and metadata in the scores table

Supports two economic modes:
- 'earth_return': PGMs and precious metals dominate
- 'in_space': Water and structural materials dominate

Usage:
    from prospector.scoring.scorer import score_all, score_asteroid
    score_all(conn, mode='earth_return', n_samples=1000)
"""

import math
from pathlib import Path

import numpy as np
import yaml

from prospector.scoring.granvik_prior import MAHLKE_CLASSES, taxonomy_prior

# Default config path
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scoring" / "grade_density_priors.yaml"
)

# Materials scored by the system
SCORED_MATERIALS = ["pgm", "water", "iron", "olivine", "pyroxene"]

# Default material unit values ($/kg) when not in config
DEFAULT_MATERIAL_VALUES = {
    "earth_return": {
        "pgm": 50_000,     # combined PGM (Pt+Pd+Rh+Ru+Os+Ir)
        "water": 0.001,    # nearly worthless on Earth
        "iron": 0.10,
        "olivine": 0.01,
        "pyroxene": 0.01,
    },
    "in_space": {
        "pgm": 50_000,
        "water": 500,      # propellant / life support
        "iron": 5,         # structural in orbit
        "olivine": 1,
        "pyroxene": 1,
    },
}

# Simplified Phase 1 recoverability factors: {material: {class: factor}}
# Full design deferred to recoverability-accessibility bead (Phase 2).
DEFAULT_RECOVERABILITY = {
    "pgm": {"M": 0.30, "X": 0.15, "S": 0.05, "Q": 0.05, "E": 0.05, "K": 0.02},
    "water": {"C": 0.70, "Ch": 0.70, "B": 0.50, "D": 0.30, "K": 0.30, "P": 0.20},
    "iron": {"M": 0.80, "X": 0.30, "S": 0.10, "Q": 0.10, "E": 0.15, "A": 0.10, "V": 0.02},
    "olivine": {
        "S": 0.50, "Q": 0.50, "A": 0.70, "O": 0.70,
        "K": 0.30, "L": 0.30, "R": 0.50, "V": 0.05,
    },
    "pyroxene": {
        "S": 0.30, "Q": 0.30, "V": 0.50, "E": 0.50,
        "K": 0.20, "L": 0.20, "R": 0.20,
    },
}

# MOID exponential decay scale (AU) for accessibility proxy
_MOID_SCALE = 0.1

# Fallback density for unknown taxonomy classes
_FALLBACK_DENSITY = {"mean": 2.0, "std": 1.0, "min": 0.5, "max": 5.0}


def load_config(config_path=None):
    """Load grade/density priors from YAML config.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to grade_density_priors.yaml.  Defaults to ``scoring/`` dir.

    Returns
    -------
    dict
        Parsed YAML config.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)


def sample_from_dist(dist, rng=None):
    """Sample from a ``{mean, std, min, max}`` normal distribution, clipped.

    Parameters
    ----------
    dist : dict
        Keys: mean, std, min, max.
    rng : numpy.random.Generator, optional

    Returns
    -------
    float
    """
    if rng is None:
        rng = np.random.default_rng()
    val = rng.normal(dist["mean"], dist["std"])
    return float(np.clip(val, dist["min"], dist["max"]))


def estimate_mass_kg(diameter_km, density_gcm3):
    """Compute asteroid mass assuming a sphere.

    Parameters
    ----------
    diameter_km : float
        Diameter in kilometres.
    density_gcm3 : float
        Bulk density in g/cm³.

    Returns
    -------
    float
        Mass in kg.
    """
    radius_m = diameter_km * 1000.0 / 2.0
    density_kgm3 = density_gcm3 * 1000.0
    return (4.0 / 3.0) * math.pi * radius_m ** 3 * density_kgm3


def compute_accessibility(moid_au=None, a=None, e=None):
    """MOID-based accessibility proxy (0–1 scale).

    Uses exponential decay on MOID: lower MOID → higher score.
    Falls back to perihelion-distance penalty when MOID unavailable.

    Parameters
    ----------
    moid_au : float, optional
        Earth Minimum Orbit Intersection Distance in AU.
    a : float, optional
        Semi-major axis in AU (fallback).
    e : float, optional
        Eccentricity (fallback).

    Returns
    -------
    float
        Accessibility in [0.01, 1.0].
    """
    if moid_au is not None and moid_au >= 0:
        score = math.exp(-moid_au / _MOID_SCALE)
    elif a is not None and e is not None:
        q = a * (1.0 - e)
        score = max(0.0, 1.0 - abs(q - 1.0) / 0.5)
    else:
        return 0.01
    return float(np.clip(score, 0.01, 1.0))


def compute_confidence(prob_vector):
    """Confidence from taxonomy probability vector entropy.

    confidence = 1 − H / H_max

    where H is Shannon entropy and H_max = log₂(17).
    A peaked distribution (low entropy) gives high confidence.

    Parameters
    ----------
    prob_vector : numpy.ndarray
        17-element probability vector.

    Returns
    -------
    float
        Confidence in [0.01, 1.0].
    """
    p = np.asarray(prob_vector, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.01
    entropy = -float(np.sum(p * np.log2(p)))
    max_entropy = math.log2(len(MAHLKE_CLASSES))
    return max(0.01, 1.0 - entropy / max_entropy)


def _grade_to_fraction(grade_value, unit):
    """Convert a grade value to a mass fraction."""
    if unit == "ppm":
        return grade_value / 1e6
    elif unit == "wt_pct":
        return grade_value / 100.0
    return grade_value


def _get_material_values(config, mode):
    """Resolve material unit values from config or defaults."""
    cfg_vals = config.get("material_values", {})
    if mode in cfg_vals:
        return cfg_vals[mode]
    return DEFAULT_MATERIAL_VALUES.get(mode, DEFAULT_MATERIAL_VALUES["earth_return"])


def _get_recoverability(config):
    """Resolve recoverability factors from config or defaults."""
    return config.get("recoverability_defaults", DEFAULT_RECOVERABILITY)


def score_asteroid(
    diameter_km,
    a,
    e,
    i_deg,
    moid=None,
    prob_vector=None,
    config=None,
    n_samples=1000,
    mode="earth_return",
    material_values=None,
    recoverability=None,
    rng=None,
):
    """Monte Carlo composite mining score for a single asteroid.

    Parameters
    ----------
    diameter_km : float
        Asteroid diameter in km.
    a, e, i_deg : float
        Orbital elements.
    moid : float, optional
        Earth MOID in AU.
    prob_vector : numpy.ndarray, optional
        17-element taxonomy probability vector.  Uses Granvik prior if None.
    config : dict, optional
        Loaded YAML config.  Loaded from default path if None.
    n_samples : int
        Number of Monte Carlo draws.
    mode : str
        ``'earth_return'`` or ``'in_space'``.
    material_values : dict, optional
        ``{material: $/kg}`` override.
    recoverability : dict, optional
        ``{material: {class: factor}}`` override.
    rng : numpy.random.Generator, optional

    Returns
    -------
    dict
        Keys: composite_score, estimated_mass_kg, grade_estimate,
        target_material, unit_value, accessibility, confidence,
        score_mode, material_contributions.
    """
    if config is None:
        config = load_config()
    if rng is None:
        rng = np.random.default_rng()
    if material_values is None:
        material_values = _get_material_values(config, mode)
    if recoverability is None:
        recoverability = _get_recoverability(config)

    # Taxonomy distribution
    if prob_vector is None:
        prob_vector = taxonomy_prior(a, e, i_deg)

    # Non-varying components
    accessibility = compute_accessibility(moid, a, e)
    confidence = compute_confidence(prob_vector)

    # Config sections
    density_priors = config["density_priors"]["classes"]
    grade_estimates = config["grade_estimates"]
    materials = [m for m in SCORED_MATERIALS if m in material_values]

    # Normalise prob vector for sampling
    pv = np.asarray(prob_vector, dtype=np.float64).copy()
    pv_sum = pv.sum()
    if pv_sum > 0:
        pv /= pv_sum
    else:
        pv = np.ones(len(MAHLKE_CLASSES)) / len(MAHLKE_CLASSES)

    sample_scores = np.zeros(n_samples)
    mass_samples = np.zeros(n_samples)
    material_totals = {m: 0.0 for m in materials}

    for i in range(n_samples):
        # 1. Sample taxonomy class
        class_idx = rng.choice(len(MAHLKE_CLASSES), p=pv)
        tax_class = MAHLKE_CLASSES[class_idx]

        # 2. Sample density
        if tax_class in density_priors:
            density = sample_from_dist(density_priors[tax_class], rng)
        else:
            density = sample_from_dist(_FALLBACK_DENSITY, rng)

        # 3. Compute mass
        mass_kg = estimate_mass_kg(diameter_km, density)
        mass_samples[i] = mass_kg

        # 4. Sum value across materials
        total_value = 0.0
        for material in materials:
            mat_cfg = grade_estimates.get(material, {})
            mat_classes = mat_cfg.get("classes", {})
            mat_unit = mat_cfg.get("unit", "wt_pct")

            if tax_class in mat_classes:
                grade_raw = sample_from_dist(mat_classes[tax_class], rng)
                grade_frac = _grade_to_fraction(grade_raw, mat_unit)
            else:
                grade_frac = 0.0

            recover = recoverability.get(material, {}).get(tax_class, 0.0)
            uv = material_values.get(material, 0.0)
            value = mass_kg * grade_frac * recover * uv
            total_value += value
            material_totals[material] += value

        # 5. Apply confidence and accessibility
        sample_scores[i] = total_value * confidence * accessibility

    # Aggregate
    composite_score = float(np.mean(sample_scores))
    estimated_mass_kg = float(np.mean(mass_samples))

    # Normalise material totals to per-sample means
    material_contributions = {m: v / n_samples for m, v in material_totals.items()}

    # Dominant target material
    best_material = max(material_contributions, key=material_contributions.get)
    best_unit_value = material_values.get(best_material, 0.0)

    # Expected grade for dominant material (weighted by taxonomy distribution)
    best_grade = 0.0
    mat_cfg = grade_estimates.get(best_material, {})
    mat_classes = mat_cfg.get("classes", {})
    mat_unit = mat_cfg.get("unit", "wt_pct")
    for ci, cls in enumerate(MAHLKE_CLASSES):
        if cls in mat_classes:
            best_grade += pv[ci] * _grade_to_fraction(mat_classes[cls]["mean"], mat_unit)

    return {
        "composite_score": composite_score,
        "estimated_mass_kg": estimated_mass_kg,
        "grade_estimate": best_grade,
        "target_material": best_material,
        "unit_value": best_unit_value,
        "accessibility": accessibility,
        "confidence": confidence,
        "score_mode": mode,
        "material_contributions": material_contributions,
    }


def score_all(conn, config_path=None, n_samples=1000, mode="earth_return"):
    """Score all asteroids with diameter data and write to the scores table.

    Queries asteroids that have orbital elements and a diameter estimate
    (from SBDB or NEOWISE).  Uses spectral taxonomy when available,
    otherwise falls back to the Granvik orbital prior.

    Parameters
    ----------
    conn : sqlite3.Connection
    config_path : str or Path, optional
    n_samples : int
    mode : str
        ``'earth_return'`` or ``'in_space'``.

    Returns
    -------
    int
        Number of asteroids scored.
    """
    config = load_config(config_path)
    rng = np.random.default_rng(42)

    rows = conn.execute(
        """
        SELECT
            a.asteroid_id,
            o.a, o.e, o.i, o.moid,
            COALESCE(o.diameter, pp.diameter_km) AS diameter_km,
            t.prob_vector
        FROM asteroids a
        JOIN orbits o ON a.asteroid_id = o.asteroid_id
        LEFT JOIN physical_properties pp ON a.asteroid_id = pp.asteroid_id
        LEFT JOIN taxonomy t ON a.asteroid_id = t.asteroid_id
        WHERE COALESCE(o.diameter, pp.diameter_km) IS NOT NULL
          AND o.a IS NOT NULL
          AND o.e IS NOT NULL
          AND o.i IS NOT NULL
        """
    ).fetchall()

    count = 0
    for row in rows:
        asteroid_id, a, e_val, i_deg, moid, diameter_km, prob_blob = row

        prob_vector = None
        if prob_blob is not None:
            prob_vector = np.frombuffer(prob_blob, dtype=np.float64)
            if len(prob_vector) != len(MAHLKE_CLASSES):
                prob_vector = None

        result = score_asteroid(
            diameter_km=diameter_km,
            a=a,
            e=e_val,
            i_deg=i_deg,
            moid=moid,
            prob_vector=prob_vector,
            config=config,
            n_samples=n_samples,
            mode=mode,
            rng=rng,
        )

        conn.execute(
            """INSERT OR REPLACE INTO scores
               (asteroid_id, estimated_mass_kg, grade_estimate, target_material,
                unit_value, accessibility, composite_score, score_mode)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                asteroid_id,
                result["estimated_mass_kg"],
                result["grade_estimate"],
                result["target_material"],
                result["unit_value"],
                result["accessibility"],
                result["composite_score"],
                result["score_mode"],
            ),
        )
        count += 1

    conn.commit()
    return count
