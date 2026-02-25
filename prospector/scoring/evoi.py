"""Expected Value of Information (EVOI) ranking.

For each uncharacterised NEO, estimates how much a hypothetical observation
would reduce scoring uncertainty.  Ranks NEOs to answer:
  "Which asteroid should we observe next?"

Approach — preposterior analysis:
  1. Score asteroid under current prior → score distribution σ_prior
  2. For each possible observation outcome j (weighted by P(j) from prior):
     - Construct posterior taxonomy distribution
     - Score under posterior → σ_post_j
  3. EVOI(τ) = σ_prior − E[σ_post | observation τ]

Observation types modelled:
  - vnir_spectroscopy: narrows taxonomy to peaked single-class distribution
  - vis_spectroscopy:  narrows to complex level (S/C/X/end-member)
  - radar:             resolves metallic vs non-metallic surface
  - albedo:            resolves X-complex degeneracy (M vs E vs P)

Usage:
    from prospector.scoring.evoi import compute_evoi, rank_all
    ranking = rank_all(conn, mode='earth_return')
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from prospector.schemas import EVOIResultSchema, ObservationType, ScoringMode
from prospector.scoring.granvik_prior import MAHLKE_CLASSES, taxonomy_prior
from prospector.scoring.scorer import (
    _vectorized_mc_samples,
    compute_accessibility,
    compute_confidence,
    load_config,
    score_asteroid,
)

_CLASS_IDX = {c: i for i, c in enumerate(MAHLKE_CLASSES)}

# Number of MC samples for scoring during EVOI computation.
# Lower than production scoring — trades accuracy for speed over many asteroids.
_EVOI_SAMPLES = 200

# Observation models: define how each observation type narrows the prior.
# peak_weight: probability mass concentrated on the "true" class
# spread_weight: residual spread across related classes
# Taxonomy complexes (for vis_spectroscopy coarse narrowing)
_COMPLEXES = {
    "S": ["S", "Q", "A", "K", "L", "O", "R", "V"],
    "C": ["C", "Ch", "B", "D", "P"],
    "X": ["X", "M", "E"],
    "end": ["Z"],
}


def _class_to_complex(cls: str) -> str:
    """Map a Mahlke class to its Bus-DeMeo complex."""
    for cplx, members in _COMPLEXES.items():
        if cls in members:
            return cplx
    return "end"


def _make_peaked_posterior(true_class: str, peak_prob: float = 0.85) -> np.ndarray:
    """Build a posterior that concentrates probability on one class.

    Parameters
    ----------
    true_class : str
        The revealed taxonomy class.
    peak_prob : float
        Probability mass on the true class (remainder spread uniformly).

    Returns
    -------
    numpy.ndarray
        17-element probability vector.
    """
    n = len(MAHLKE_CLASSES)
    residual = (1.0 - peak_prob) / max(1, n - 1)
    post = np.full(n, residual)
    idx = _CLASS_IDX.get(true_class)
    if idx is not None:
        post[idx] = peak_prob
    post /= post.sum()
    return post


def _make_complex_posterior(true_class: str, peak_prob: float = 0.70) -> np.ndarray:
    """Build a posterior concentrated on a complex, not a single class.

    Simulates visible-only spectroscopy that identifies the complex but
    not the specific sub-class.

    Parameters
    ----------
    true_class : str
        The revealed class (used to determine complex).
    peak_prob : float
        Probability mass shared among the complex members.

    Returns
    -------
    numpy.ndarray
        17-element probability vector.
    """
    n = len(MAHLKE_CLASSES)
    cplx = _class_to_complex(true_class)
    members = _COMPLEXES.get(cplx, [true_class])

    residual_total = 1.0 - peak_prob
    residual_per = residual_total / max(1, n - len(members))
    per_member = peak_prob / len(members)

    post = np.full(n, residual_per)
    for m in members:
        idx = _CLASS_IDX.get(m)
        if idx is not None:
            post[idx] = per_member
    post /= post.sum()
    return post


def _make_radar_posterior(
    prior: np.ndarray, is_metallic: bool
) -> np.ndarray:
    """Update prior based on radar metallic/non-metallic result.

    Metallic boosts M, X, E; non-metallic suppresses them.

    Parameters
    ----------
    prior : numpy.ndarray
        Current 17-element prior.
    is_metallic : bool
        Whether radar indicates metallic surface.

    Returns
    -------
    numpy.ndarray
        Updated probability vector.
    """
    metallic_classes = {"M", "X", "E"}
    post = prior.copy()
    for i, cls in enumerate(MAHLKE_CLASSES):
        if cls in metallic_classes:
            post[i] *= 3.0 if is_metallic else 0.2
        else:
            post[i] *= 0.5 if is_metallic else 1.2
    total = post.sum()
    if total > 0:
        post /= total
    else:
        post = np.ones(len(MAHLKE_CLASSES)) / len(MAHLKE_CLASSES)
    return post


def _make_albedo_posterior(
    prior: np.ndarray, albedo_pv: float
) -> np.ndarray:
    """Update prior based on albedo measurement.

    Albedo resolves X-complex degeneracy:
      - High albedo (>0.30) → E-type (enstatite achondrite)
      - Moderate (0.10-0.30) → M-type (metallic)
      - Low (<0.10) → P/C/D-type (primitive, dark)

    Parameters
    ----------
    prior : numpy.ndarray
        Current 17-element prior.
    albedo_pv : float
        Hypothetical albedo measurement.

    Returns
    -------
    numpy.ndarray
        Updated probability vector.
    """
    post = prior.copy()
    for i, cls in enumerate(MAHLKE_CLASSES):
        if albedo_pv > 0.30:
            # High albedo: boost bright types
            if cls in ("E", "V"):
                post[i] *= 3.0
            elif cls in ("S", "Q", "A", "L", "K", "O", "R"):
                post[i] *= 1.5
            elif cls in ("C", "Ch", "B", "D", "P"):
                post[i] *= 0.2
        elif albedo_pv > 0.10:
            # Moderate: boost S-complex and M
            if cls in ("S", "Q", "M", "K", "L"):
                post[i] *= 2.0
            elif cls in ("C", "Ch", "B", "D", "P"):
                post[i] *= 0.5
        else:
            # Low albedo: boost dark types
            if cls in ("C", "Ch", "B", "D", "P"):
                post[i] *= 3.0
            elif cls in ("S", "Q", "A", "E", "V"):
                post[i] *= 0.2
            elif cls == "M":
                post[i] *= 0.5
    total = post.sum()
    if total > 0:
        post /= total
    else:
        post = np.ones(len(MAHLKE_CLASSES)) / len(MAHLKE_CLASSES)
    return post


# Canonical albedo bins for preposterior expectation
_ALBEDO_BINS = [0.05, 0.15, 0.35]
_ALBEDO_PROBS = [0.35, 0.45, 0.20]  # approximate NEO albedo distribution


@dataclass
class EVOIResult:
    """Result of EVOI computation for one asteroid."""

    asteroid_id: int
    current_score_mean: float
    current_score_std: float
    evoi_vnir: float
    evoi_vis: float
    evoi_radar: float
    evoi_albedo: float
    best_observation: ObservationType
    best_evoi: float

    @property
    def evoi_by_type(self) -> dict[str, float]:
        return {
            "vnir_spectroscopy": self.evoi_vnir,
            "vis_spectroscopy": self.evoi_vis,
            "radar": self.evoi_radar,
            "albedo": self.evoi_albedo,
        }


def _score_distribution(
    diameter_km: float,
    a: float,
    e: float,
    i_deg: float,
    moid: float | None,
    prob_vector: np.ndarray,
    config: dict,
    mode: ScoringMode,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run MC scoring and return the array of per-sample composite scores."""
    accessibility = compute_accessibility(moid, a, e)
    confidence = compute_confidence(prob_vector)

    sample_values, _, _ = _vectorized_mc_samples(
        diameter_km, prob_vector, config, mode, n_samples, rng,
    )
    return sample_values * confidence * accessibility


def compute_evoi(
    diameter_km: float,
    a: float,
    e: float,
    i_deg: float,
    moid: float | None = None,
    prior: np.ndarray | None = None,
    config: dict | None = None,
    mode: ScoringMode = "earth_return",
    n_samples: int = _EVOI_SAMPLES,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Compute EVOI for each observation type.

    Parameters
    ----------
    diameter_km : float
        Asteroid diameter in km.
    a, e, i_deg : float
        Orbital elements.
    moid : float, optional
        Earth MOID in AU.
    prior : numpy.ndarray, optional
        Current 17-element taxonomy prior.  Granvik prior if None.
    config : dict, optional
        Scorer config (loaded from default if None).
    mode : str
        Economic mode.
    n_samples : int
        MC samples per scoring run.
    rng : numpy.random.Generator, optional

    Returns
    -------
    dict
        Keys: score_mean, score_std, evoi_vnir, evoi_vis, evoi_radar,
        evoi_albedo, best_observation, best_evoi.
    """
    if config is None:
        config = load_config()
    if rng is None:
        rng = np.random.default_rng()
    if prior is None:
        prior = taxonomy_prior(a, e, i_deg)

    prior = np.asarray(prior, dtype=np.float64)

    # --- Current score distribution under prior ---
    scores_prior = _score_distribution(
        diameter_km, a, e, i_deg, moid, prior, config, mode, n_samples, rng
    )
    std_prior = float(np.std(scores_prior))
    mean_prior = float(np.mean(scores_prior))

    # --- EVOI for VNIR spectroscopy ---
    # Possible outcome: each class j revealed with probability prior[j]
    expected_std_vnir = 0.0
    for j, cls in enumerate(MAHLKE_CLASSES):
        p_j = prior[j]
        if p_j < 1e-6:
            continue
        post_j = _make_peaked_posterior(cls)
        rng_j = np.random.default_rng(rng.integers(0, 2**31))
        scores_j = _score_distribution(
            diameter_km, a, e, i_deg, moid, post_j, config, mode, n_samples, rng_j
        )
        expected_std_vnir += p_j * float(np.std(scores_j))
    evoi_vnir = max(0.0, std_prior - expected_std_vnir)

    # --- EVOI for visible-only spectroscopy ---
    expected_std_vis = 0.0
    for j, cls in enumerate(MAHLKE_CLASSES):
        p_j = prior[j]
        if p_j < 1e-6:
            continue
        post_j = _make_complex_posterior(cls)
        rng_j = np.random.default_rng(rng.integers(0, 2**31))
        scores_j = _score_distribution(
            diameter_km, a, e, i_deg, moid, post_j, config, mode, n_samples, rng_j
        )
        expected_std_vis += p_j * float(np.std(scores_j))
    evoi_vis = max(0.0, std_prior - expected_std_vis)

    # --- EVOI for radar ---
    # Two outcomes: metallic or non-metallic
    metallic_classes = {"M", "X", "E"}
    p_metallic = sum(prior[_CLASS_IDX[c]] for c in metallic_classes if c in _CLASS_IDX)
    p_nonmetallic = 1.0 - p_metallic

    expected_std_radar = 0.0
    for is_metallic, p_outcome in [(True, p_metallic), (False, p_nonmetallic)]:
        if p_outcome < 1e-6:
            continue
        post_r = _make_radar_posterior(prior, is_metallic)
        rng_r = np.random.default_rng(rng.integers(0, 2**31))
        scores_r = _score_distribution(
            diameter_km, a, e, i_deg, moid, post_r, config, mode, n_samples, rng_r
        )
        expected_std_radar += p_outcome * float(np.std(scores_r))
    evoi_radar = max(0.0, std_prior - expected_std_radar)

    # --- EVOI for albedo measurement ---
    expected_std_albedo = 0.0
    for albedo_val, p_alb in zip(_ALBEDO_BINS, _ALBEDO_PROBS):
        post_a = _make_albedo_posterior(prior, albedo_val)
        rng_a = np.random.default_rng(rng.integers(0, 2**31))
        scores_a = _score_distribution(
            diameter_km, a, e, i_deg, moid, post_a, config, mode, n_samples, rng_a
        )
        expected_std_albedo += p_alb * float(np.std(scores_a))
    evoi_albedo = max(0.0, std_prior - expected_std_albedo)

    # Best observation
    evoi_map: dict[ObservationType, float] = {
        "vnir_spectroscopy": evoi_vnir,
        "vis_spectroscopy": evoi_vis,
        "radar": evoi_radar,
        "albedo": evoi_albedo,
    }
    best_obs: ObservationType = max(evoi_map, key=lambda k: evoi_map[k])

    result = {
        "score_mean": mean_prior,
        "score_std": std_prior,
        "evoi_vnir": evoi_vnir,
        "evoi_vis": evoi_vis,
        "evoi_radar": evoi_radar,
        "evoi_albedo": evoi_albedo,
        "best_observation": best_obs,
        "best_evoi": evoi_map[best_obs],
    }
    EVOIResultSchema.model_validate(result)
    return result


def compute_evoi_for_asteroid(
    asteroid_id: int,
    conn,
    *,
    config: dict | None = None,
    mode: ScoringMode = "earth_return",
    n_samples: int = _EVOI_SAMPLES,
    rng: np.random.Generator | None = None,
) -> EVOIResult | None:
    """Compute EVOI for a single asteroid from the database.

    Only computes for asteroids that lack spectral taxonomy (i.e. using
    Granvik prior — the ones that would most benefit from new observations).

    Parameters
    ----------
    asteroid_id : int
    conn : sqlite3.Connection
    config, mode, n_samples, rng : optional

    Returns
    -------
    EVOIResult or None
        None if asteroid lacks orbital elements or diameter.
    """
    row = conn.execute(
        """
        SELECT o.a, o.e, o.i, o.moid,
               COALESCE(o.diameter, pp.diameter_km) AS diameter_km,
               t.prob_vector
        FROM orbits o
        LEFT JOIN physical_properties pp ON o.asteroid_id = pp.asteroid_id
        LEFT JOIN taxonomy t ON o.asteroid_id = t.asteroid_id
        WHERE o.asteroid_id = ?
          AND o.a IS NOT NULL AND o.e IS NOT NULL AND o.i IS NOT NULL
          AND COALESCE(o.diameter, pp.diameter_km) IS NOT NULL
        """,
        (asteroid_id,),
    ).fetchone()

    if row is None:
        return None

    a, e_val, i_deg, moid, diameter_km, prob_blob = row

    # Use spectral taxonomy if available, otherwise Granvik prior
    if prob_blob is not None:
        prior = np.frombuffer(prob_blob, dtype=np.float64)
        if len(prior) != len(MAHLKE_CLASSES):
            prior = taxonomy_prior(a, e_val, i_deg)
    else:
        prior = taxonomy_prior(a, e_val, i_deg)

    result = compute_evoi(
        diameter_km=diameter_km,
        a=a,
        e=e_val,
        i_deg=i_deg,
        moid=moid,
        prior=prior,
        config=config,
        mode=mode,
        n_samples=n_samples,
        rng=rng,
    )

    return EVOIResult(
        asteroid_id=asteroid_id,
        current_score_mean=result["score_mean"],
        current_score_std=result["score_std"],
        evoi_vnir=result["evoi_vnir"],
        evoi_vis=result["evoi_vis"],
        evoi_radar=result["evoi_radar"],
        evoi_albedo=result["evoi_albedo"],
        best_observation=result["best_observation"],
        best_evoi=result["best_evoi"],
    )


def rank_all(
    conn,
    *,
    config_path=None,
    mode: ScoringMode = "earth_return",
    n_samples: int = _EVOI_SAMPLES,
    neo_only: bool = True,
    max_asteroids: int | None = None,
) -> list[EVOIResult]:
    """Rank uncharacterised asteroids by EVOI.

    Targets asteroids without spectral taxonomy (using Granvik prior)
    since these are the ones where new observations add the most value.

    Parameters
    ----------
    conn : sqlite3.Connection
    config_path : str or Path, optional
    mode : str
    n_samples : int
    neo_only : bool
        If True, only rank NEOs (asteroids.neo=1).
    max_asteroids : int, optional
        Limit number of asteroids to evaluate (for performance).

    Returns
    -------
    list[EVOIResult]
        Sorted by best_evoi descending (highest information value first).
    """
    config = load_config(config_path)

    # Find asteroids without spectral taxonomy
    query = """
        SELECT a.asteroid_id
        FROM asteroids a
        JOIN orbits o ON a.asteroid_id = o.asteroid_id
        LEFT JOIN physical_properties pp ON a.asteroid_id = pp.asteroid_id
        LEFT JOIN taxonomy t ON a.asteroid_id = t.asteroid_id
        WHERE t.asteroid_id IS NULL
          AND o.a IS NOT NULL AND o.e IS NOT NULL AND o.i IS NOT NULL
          AND COALESCE(o.diameter, pp.diameter_km) IS NOT NULL
    """
    if neo_only:
        query += " AND a.neo = 1"
    query += " ORDER BY a.asteroid_id"
    if max_asteroids is not None:
        query += f" LIMIT {int(max_asteroids)}"

    rows = conn.execute(query).fetchall()

    results = []
    for (asteroid_id,) in rows:
        # Per-asteroid SeedSequence for DB-independence
        rng = np.random.default_rng(
            np.random.SeedSequence(42, spawn_key=(int(asteroid_id),))
        )
        evoi_result = compute_evoi_for_asteroid(
            asteroid_id, conn, config=config, mode=mode,
            n_samples=n_samples, rng=rng,
        )
        if evoi_result is not None:
            results.append(evoi_result)

    results.sort(key=lambda r: r.best_evoi, reverse=True)
    return results
