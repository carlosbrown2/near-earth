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
from dataclasses import dataclass, field
from pathlib import Path

import deal
import numpy as np
import yaml

from prospector.schemas import ScoringMode, ScoringResult
from prospector.scoring.config_schema import validate_config_checksum
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

# Spin-barrier scoring modifiers
_MONOLITHIC_BONUS = 1.15      # 15% bonus for fast rotators (easier to mine)
_BINARY_SUSPECT_PENALTY = 0.80  # 20% penalty for likely rubble piles/binaries

# Thermal depletion model (Toliou et al. 2021)
# Low-perihelion NEOs experience >600 K surface temps that dehydrate phyllosilicates.
_THERMAL_Q_CENTER = 0.6   # AU — 50% water retention point
_THERMAL_STEEPNESS = 8.0  # logistic steepness
_THERMAL_Q_SAFE = 1.0     # AU — no penalty above this

# JWST 6μm water confirmation boost (Arredondo et al. 2024)
# Unambiguous molecular water detection via H-O-H bending mode
# boosts water grade confidence relative to taxonomy-only estimate.
_JWST_WATER_BOOST = 3.0   # multiplicative boost to water grade

# Fallback density for unknown taxonomy classes
_FALLBACK_DENSITY = {"mean": 2.0, "std": 1.0, "min": 0.5, "max": 5.0}

# Number of taxonomy classes
_N_CLASSES = len(MAHLKE_CLASSES)

# Class name → index mapping for array construction
_CLASS_TO_IDX = {c: i for i, c in enumerate(MAHLKE_CLASSES)}


@dataclass(frozen=True)
class ConfigArrays:
    """Pre-built numpy arrays from YAML config for vectorized MC lookups.

    All arrays are shape ``(17,)`` indexed by Mahlke class index.
    Missing class entries are filled with zeros (or fallback for density).
    """

    # Density priors — shape (17,) each
    density_mean: np.ndarray
    density_std: np.ndarray
    density_min: np.ndarray
    density_max: np.ndarray

    # Grade arrays — {material: {stat: ndarray(17,)}}
    grade_mean: dict = field(default_factory=dict)
    grade_std: dict = field(default_factory=dict)
    grade_min: dict = field(default_factory=dict)
    grade_max: dict = field(default_factory=dict)

    # Unit conversion factors — {material: float}
    # Converts grade unit → mass fraction (ppm→1e-6, wt_pct→0.01, else 1.0)
    unit_factors: dict = field(default_factory=dict)

    # Recoverability — {material: ndarray(17,)}
    recoverability: dict = field(default_factory=dict)


def _unit_to_factor(unit: str) -> float:
    """Convert a unit string to its mass-fraction conversion factor."""
    if unit == "ppm":
        return 1e-6
    elif unit == "wt_pct":
        return 0.01
    return 1.0


def _dist_to_arrays(classes_dict, fallback=None):
    """Convert a ``{class_name: {mean, std, min, max}}`` dict to four (17,) arrays.

    Parameters
    ----------
    classes_dict : dict
        Mapping from taxonomy class names to distribution dicts.
    fallback : dict or None
        If provided, fill missing classes with this distribution.
        If None, fill with zeros.

    Returns
    -------
    tuple of (mean, std, min, max) np.ndarray, each shape (17,).
    """
    means = np.zeros(_N_CLASSES, dtype=np.float64)
    stds = np.zeros(_N_CLASSES, dtype=np.float64)
    mins = np.zeros(_N_CLASSES, dtype=np.float64)
    maxs = np.zeros(_N_CLASSES, dtype=np.float64)

    for cls_name, dist in classes_dict.items():
        if cls_name not in _CLASS_TO_IDX:
            continue
        idx = _CLASS_TO_IDX[cls_name]
        means[idx] = dist["mean"]
        stds[idx] = dist["std"]
        mins[idx] = dist["min"]
        maxs[idx] = dist["max"]

    if fallback is not None:
        for i in range(_N_CLASSES):
            cls = MAHLKE_CLASSES[i]
            if cls not in classes_dict:
                means[i] = fallback["mean"]
                stds[i] = fallback["std"]
                mins[i] = fallback["min"]
                maxs[i] = fallback["max"]

    return means, stds, mins, maxs


def build_config_arrays(config: dict, mode: ScoringMode = "earth_return") -> ConfigArrays:
    """Convert nested YAML config into flat numpy arrays for vectorized lookups.

    Parameters
    ----------
    config : dict
        Parsed YAML config from :func:`load_config`.
    mode : str
        ``'earth_return'`` or ``'in_space'`` — selects material unit values.

    Returns
    -------
    ConfigArrays
        Dataclass with pre-built ``(17,)`` arrays.
    """
    # --- Density ---
    density_classes = config["density_priors"]["classes"]
    d_mean, d_std, d_min, d_max = _dist_to_arrays(density_classes, fallback=_FALLBACK_DENSITY)

    # --- Grades per material ---
    grade_estimates = config.get("grade_estimates", {})
    g_mean: dict[str, np.ndarray] = {}
    g_std: dict[str, np.ndarray] = {}
    g_min: dict[str, np.ndarray] = {}
    g_max: dict[str, np.ndarray] = {}
    u_factors: dict[str, float] = {}

    for material in SCORED_MATERIALS:
        mat_cfg = grade_estimates.get(material, {})
        mat_classes = mat_cfg.get("classes", {})
        m, s, mn, mx = _dist_to_arrays(mat_classes)  # zeros for missing
        g_mean[material] = m
        g_std[material] = s
        g_min[material] = mn
        g_max[material] = mx
        u_factors[material] = _unit_to_factor(mat_cfg.get("unit", "wt_pct"))

    # --- Recoverability ---
    recover_cfg = config.get("recoverability_defaults", DEFAULT_RECOVERABILITY)
    recover_arrays: dict[str, np.ndarray] = {}
    for material in SCORED_MATERIALS:
        arr = np.zeros(_N_CLASSES, dtype=np.float64)
        mat_recover = recover_cfg.get(material, {})
        for cls_name, factor in mat_recover.items():
            if cls_name in _CLASS_TO_IDX:
                arr[_CLASS_TO_IDX[cls_name]] = factor
        recover_arrays[material] = arr

    return ConfigArrays(
        density_mean=d_mean,
        density_std=d_std,
        density_min=d_min,
        density_max=d_max,
        grade_mean=g_mean,
        grade_std=g_std,
        grade_min=g_min,
        grade_max=g_max,
        unit_factors=u_factors,
        recoverability=recover_arrays,
    )


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
    config_path = Path(config_path)
    validate_config_checksum(config_path)
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


@deal.pre(lambda diameter_km, density_gcm3: diameter_km >= 0, message="diameter must be non-negative")
@deal.pre(lambda diameter_km, density_gcm3: density_gcm3 > 0, message="density must be positive")
@deal.post(lambda result: math.isfinite(result) and result >= 0)
def estimate_mass_kg(diameter_km: float, density_gcm3: float) -> float:
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


@deal.post(lambda result: 0.01 <= result <= 1.0)
def compute_accessibility(moid_au: float | None = None, a: float | None = None, e: float | None = None) -> float:
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


@deal.post(lambda result: 0.01 <= result <= 1.0)
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


def compute_spin_modifier(is_monolithic=None, is_binary_suspect=None):
    """Scoring modifier from rotation properties (spin-barrier filter).

    Monolithic bodies (fast rotators, P < 2.2 hr) get a bonus — they lack
    the rubble-pile structure that complicates mining operations.
    Binary suspects (slow + high amplitude) get a penalty.

    Parameters
    ----------
    is_monolithic : bool or None
        True if rotation period is below the spin barrier (2.2 hr).
    is_binary_suspect : bool or None
        True if slow rotator with large lightcurve amplitude.

    Returns
    -------
    float
        Modifier in [0.80, 1.15].  Returns 1.0 if no rotation data.
    """
    if is_monolithic is True:
        return _MONOLITHIC_BONUS
    if is_binary_suspect is True:
        return _BINARY_SUSPECT_PENALTY
    return 1.0


@deal.post(lambda result: 0.0 <= result <= 1.0)
def compute_thermal_depletion_factor(q_au: float | None = None) -> float:
    """Water retention factor based on perihelion distance (Toliou et al. 2021).

    Near-Sun dwell time thermally depletes phyllosilicate-bound water.
    Surface temperatures exceed 600 K at q < 0.5 AU, dehydrating C/Ch/B-type
    asteroids.  Returns a multiplicative factor applied to water grades only.

    Parameters
    ----------
    q_au : float or None
        Perihelion distance in AU.  If None, returns 1.0 (no penalty).

    Returns
    -------
    float
        Water retention factor in [0, 1].  1.0 = fully retained, 0 = depleted.
    """
    if q_au is None:
        return 1.0
    if q_au >= _THERMAL_Q_SAFE:
        return 1.0
    # Logistic sigmoid: steep transition around q_center
    factor = 1.0 / (1.0 + math.exp(-_THERMAL_STEEPNESS * (q_au - _THERMAL_Q_CENTER)))
    return max(0.0, min(1.0, factor))


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


@deal.pre(lambda diameter_km, *_, **__: diameter_km > 0, message="diameter_km must be positive for scoring")
@deal.post(lambda result: result["composite_score"] >= 0, message="composite_score must be non-negative")
@deal.post(lambda result: math.isfinite(result["composite_score"]), message="composite_score must be finite")
def score_asteroid(
    diameter_km,
    a,
    e,
    i_deg,
    moid=None,
    prob_vector=None,
    config=None,
    n_samples=1000,
    mode: ScoringMode = "earth_return",
    material_values=None,
    recoverability=None,
    rng=None,
    is_monolithic=None,
    is_binary_suspect=None,
    q_au=None,
    jwst_water_confirmed=None,
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
    is_monolithic : bool, optional
        True if fast rotator (P < 2.2 hr).  Applies bonus to score.
    is_binary_suspect : bool, optional
        True if slow rotator with large amplitude.  Applies penalty.
    q_au : float, optional
        Perihelion distance in AU.  Used by the thermal depletion filter
        (Toliou et al. 2021) to penalise water grades for low-perihelion
        orbits.  Computed from ``a * (1 - e)`` if not provided explicitly.
    jwst_water_confirmed : bool, optional
        True if JWST 6μm H-O-H bending mode confirms molecular water
        (Arredondo et al. 2024).  Applies multiplicative boost to water
        grade since detection is unambiguous.

    Returns
    -------
    dict
        Keys: composite_score, estimated_mass_kg, grade_estimate,
        target_material, unit_value, accessibility, confidence,
        spin_modifier, thermal_depletion_factor, jwst_water_boost,
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
    spin_modifier = compute_spin_modifier(is_monolithic, is_binary_suspect)

    # Thermal depletion factor — derive q from orbital elements if not given
    if q_au is None and a is not None and e is not None:
        q_au = a * (1.0 - e)
    thermal_depl = compute_thermal_depletion_factor(q_au)

    # JWST 6μm water confirmation boost
    jwst_boost = _JWST_WATER_BOOST if jwst_water_confirmed else 1.0

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
                # Water-specific modifiers
                if material == "water":
                    grade_frac *= thermal_depl   # thermal depletion penalty
                    grade_frac *= jwst_boost     # JWST 6μm confirmation boost
            else:
                grade_frac = 0.0

            recover = recoverability.get(material, {}).get(tax_class, 0.0)
            uv = material_values.get(material, 0.0)
            value = mass_kg * grade_frac * recover * uv
            total_value += value
            material_totals[material] += value

        # 5. Apply confidence, accessibility, and spin modifier
        sample_scores[i] = total_value * confidence * accessibility * spin_modifier

    # Aggregate
    composite_score = float(np.mean(sample_scores))
    estimated_mass_kg = float(np.mean(mass_samples))

    # Normalise material totals to per-sample means
    material_contributions = {m: v / n_samples for m, v in material_totals.items()}

    # Dominant target material
    best_material = max(material_contributions, key=lambda k: material_contributions[k])
    best_unit_value = material_values.get(best_material, 0.0)

    # Expected grade for dominant material (weighted by taxonomy distribution)
    best_grade = 0.0
    mat_cfg = grade_estimates.get(best_material, {})
    mat_classes = mat_cfg.get("classes", {})
    mat_unit = mat_cfg.get("unit", "wt_pct")
    for ci, cls in enumerate(MAHLKE_CLASSES):
        if cls in mat_classes:
            best_grade += pv[ci] * _grade_to_fraction(mat_classes[cls]["mean"], mat_unit)

    result = {
        "composite_score": composite_score,
        "estimated_mass_kg": estimated_mass_kg,
        "grade_estimate": best_grade,
        "target_material": best_material,
        "unit_value": best_unit_value,
        "accessibility": accessibility,
        "confidence": confidence,
        "spin_modifier": spin_modifier,
        "thermal_depletion_factor": thermal_depl,
        "jwst_water_boost": jwst_boost,
        "score_mode": mode,
        "material_contributions": material_contributions,
    }
    ScoringResult.model_validate(result)
    return result


def score_all(conn, config_path=None, n_samples=1000, mode: ScoringMode = "earth_return"):
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
            o.a, o.e, o.i, o.moid, o.q,
            COALESCE(o.diameter, pp.diameter_km) AS diameter_km,
            t.prob_vector,
            rp.is_monolithic,
            rp.is_binary_suspect,
            jw.detection AS jwst_water_confirmed
        FROM asteroids a
        JOIN orbits o ON a.asteroid_id = o.asteroid_id
        LEFT JOIN physical_properties pp ON a.asteroid_id = pp.asteroid_id
        LEFT JOIN taxonomy t ON a.asteroid_id = t.asteroid_id
        LEFT JOIN rotation_properties rp ON a.asteroid_id = rp.asteroid_id
        LEFT JOIN jwst_water jw ON a.asteroid_id = jw.asteroid_id
        WHERE COALESCE(o.diameter, pp.diameter_km) IS NOT NULL
          AND o.a IS NOT NULL
          AND o.e IS NOT NULL
          AND o.i IS NOT NULL
        """
    ).fetchall()

    count = 0
    for row in rows:
        (asteroid_id, a, e_val, i_deg, moid, q_au, diameter_km, prob_blob,
         is_monolithic, is_binary_suspect, jwst_water_confirmed) = row

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
            is_monolithic=bool(is_monolithic) if is_monolithic is not None else None,
            is_binary_suspect=bool(is_binary_suspect) if is_binary_suspect is not None else None,
            q_au=q_au,
            jwst_water_confirmed=bool(jwst_water_confirmed) if jwst_water_confirmed is not None else None,
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
