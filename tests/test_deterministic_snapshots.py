"""Deterministic snapshot tests for numerical pipeline reproducibility.

These tests lock down the exact numerical output of the scoring and EVOI
pipelines for fixed inputs and RNG seeds.  They detect accidental numerical
drift from code changes, dependency bumps, or loop reordering.

**Updating snapshots**: If an algorithm change is intentional, re-generate the
expected values by running each function with the documented seed and inputs,
then update the inline dicts below.  Do NOT update snapshots to make a failing
test pass without understanding WHY the output changed.
"""

import numpy as np
import pytest

from prospector.db import get_connection, init_schema
from prospector.scoring.evoi import compute_evoi, rank_all
from prospector.scoring.granvik_prior import MAHLKE_CLASSES, taxonomy_prior
from prospector.scoring.scorer import load_config, score_all, score_asteroid


# Tight tolerance for seeded MC outputs — should be bit-identical across runs.
_REL_TOL = 1e-12


@pytest.fixture
def config():
    return load_config()


# ---------------------------------------------------------------------------
# 1. taxonomy_prior: deterministic (no RNG), pure function of (a, e, i)
# ---------------------------------------------------------------------------

class TestTaxonomyPriorSnapshot:
    """taxonomy_prior(1.64, 0.77, 26.1) — orbit consistent with JFC/D-type."""

    EXPECTED = {
        "A": 0.014225285800405909,
        "B": 0.03307487568600677,
        "C": 0.08319686599264903,
        "Ch": 0.025835876160392835,
        "D": 0.23475167364977814,
        "E": 0.02268768252360125,
        "K": 0.024263451309525674,
        "L": 0.02271335008390685,
        "M": 0.024249608882774845,
        "O": 0.004237112241522652,
        "P": 0.1195128139574257,
        "Q": 0.0480741966845429,
        "R": 0.004237448699927388,
        "S": 0.24319004009113618,
        "V": 0.026912293900884024,
        "X": 0.058825595159734935,
        "Z": 0.010011829175784803,
    }

    def test_exact_probability_vector(self):
        prior = taxonomy_prior(a=1.64, e=0.77, i_deg=26.1)
        assert len(prior) == 17
        for i, cls in enumerate(MAHLKE_CLASSES):
            assert prior[i] == pytest.approx(
                self.EXPECTED[cls], rel=_REL_TOL
            ), f"taxonomy_prior mismatch for class {cls}"

    def test_sums_to_one(self):
        prior = taxonomy_prior(a=1.64, e=0.77, i_deg=26.1)
        assert float(prior.sum()) == pytest.approx(1.0, abs=1e-14)


# ---------------------------------------------------------------------------
# 2. score_asteroid: seeded MC with Granvik prior
# ---------------------------------------------------------------------------

class TestScoreAsteroidSnapshot:
    """score_asteroid(d=1.0, a=1.5, e=0.3, i=10.0, moid=0.05, seed=42, n=1000)."""

    EXPECTED = {
        "composite_score": 7988086806.079954,
        "estimated_mass_kg": 1355662696781.752,
        "grade_estimate": 2.267212947812078e-06,
        "target_material": "pgm",
        "unit_value": 50000,
        "accessibility": 0.6065306597126334,
        "confidence": 0.32396780811224635,
        "spin_modifier": 1.0,
        "thermal_depletion_factor": 1.0,
        "jwst_water_boost": 1.0,
        "score_mode": "earth_return",
    }

    EXPECTED_CONTRIBUTIONS = {
        "pgm": 31132284351.884666,
        "water": 5676227.5733847255,
        "iron": 6409869403.101304,
        "olivine": 2057805019.4406676,
        "pyroxene": 1046949296.8077171,
    }

    def test_exact_output(self, config):
        result = score_asteroid(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            moid=0.05, config=config, n_samples=1000,
            rng=np.random.default_rng(42), mode="earth_return",
        )
        for key, expected in self.EXPECTED.items():
            if isinstance(expected, str):
                assert result[key] == expected, f"Mismatch for {key}"
            else:
                assert result[key] == pytest.approx(
                    expected, rel=_REL_TOL
                ), f"Snapshot mismatch for {key}: {result[key]!r} != {expected!r}"

    def test_exact_material_contributions(self, config):
        result = score_asteroid(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            moid=0.05, config=config, n_samples=1000,
            rng=np.random.default_rng(42), mode="earth_return",
        )
        for material, expected in self.EXPECTED_CONTRIBUTIONS.items():
            assert result["material_contributions"][material] == pytest.approx(
                expected, rel=_REL_TOL
            ), f"Material contribution mismatch for {material}"

    def test_repeated_calls_identical(self, config):
        """Two calls with same seed must produce bit-identical output."""
        kwargs = dict(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            moid=0.05, config=config, n_samples=1000, mode="earth_return",
        )
        r1 = score_asteroid(**kwargs, rng=np.random.default_rng(42))
        r2 = score_asteroid(**kwargs, rng=np.random.default_rng(42))
        assert r1["composite_score"] == r2["composite_score"]
        for m in r1["material_contributions"]:
            assert r1["material_contributions"][m] == r2["material_contributions"][m]


# ---------------------------------------------------------------------------
# 3. score_asteroid with peaked M-type prob vector
# ---------------------------------------------------------------------------

class TestScoreAsteroidMTypeSnapshot:
    """score_asteroid with 100% M-type, seed=42, n=1000."""

    EXPECTED = {
        "composite_score": 34457737439.635704,
        "estimated_mass_kg": 254322773303.50085,
        "grade_estimate": 2e-05,
        "target_material": "pgm",
        "unit_value": 50000,
        "accessibility": 0.36787944117144233,
        "confidence": 1.0,
        "spin_modifier": 1.0,
        "thermal_depletion_factor": 1.0,
        "jwst_water_boost": 1.0,
        "score_mode": "earth_return",
    }

    EXPECTED_CONTRIBUTIONS = {
        "pgm": 76607169838.07536,
        "water": 0.0,
        "iron": 17058671693.899359,
        "olivine": 0.0,
        "pyroxene": 0.0,
    }

    def test_exact_output(self, config):
        pv = np.zeros(17)
        pv[8] = 1.0  # M-type
        result = score_asteroid(
            diameter_km=0.5, a=2.0, e=0.2, i_deg=5.0,
            moid=0.1, prob_vector=pv, config=config, n_samples=1000,
            rng=np.random.default_rng(42), mode="earth_return",
        )
        for key, expected in self.EXPECTED.items():
            if isinstance(expected, str):
                assert result[key] == expected
            else:
                assert result[key] == pytest.approx(
                    expected, rel=_REL_TOL
                ), f"M-type snapshot mismatch for {key}"

    def test_exact_material_contributions(self, config):
        pv = np.zeros(17)
        pv[8] = 1.0
        result = score_asteroid(
            diameter_km=0.5, a=2.0, e=0.2, i_deg=5.0,
            moid=0.1, prob_vector=pv, config=config, n_samples=1000,
            rng=np.random.default_rng(42), mode="earth_return",
        )
        for material, expected in self.EXPECTED_CONTRIBUTIONS.items():
            assert result["material_contributions"][material] == pytest.approx(
                expected, rel=_REL_TOL
            ), f"M-type material mismatch for {material}"


# ---------------------------------------------------------------------------
# 4. compute_evoi: seeded preposterior analysis
# ---------------------------------------------------------------------------

class TestComputeEVOISnapshot:
    """compute_evoi(d=1.0, a=1.5, e=0.3, i=10.0, moid=0.05, seed=42, n=200)."""

    EXPECTED = {
        "score_mean": 6354458197.133442,
        "score_std": 22983458307.414104,
        "evoi_vnir": 0.0,
        "evoi_vis": 12455857845.245924,
        "evoi_radar": 16085821141.78535,
        "evoi_albedo": 0.0,
        "best_observation": "radar",
        "best_evoi": 16085821141.78535,
    }

    def test_exact_output(self, config):
        result = compute_evoi(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            moid=0.05, config=config, n_samples=200,
            rng=np.random.default_rng(42), mode="earth_return",
        )
        for key, expected in self.EXPECTED.items():
            if isinstance(expected, str):
                assert result[key] == expected, f"EVOI mismatch for {key}"
            else:
                assert result[key] == pytest.approx(
                    expected, rel=_REL_TOL
                ), f"EVOI snapshot mismatch for {key}: {result[key]!r} != {expected!r}"

    def test_repeated_calls_identical(self, config):
        kwargs = dict(
            diameter_km=1.0, a=1.5, e=0.3, i_deg=10.0,
            moid=0.05, config=config, n_samples=200, mode="earth_return",
        )
        r1 = compute_evoi(**kwargs, rng=np.random.default_rng(42))
        r2 = compute_evoi(**kwargs, rng=np.random.default_rng(42))
        assert r1["score_mean"] == r2["score_mean"]
        assert r1["best_evoi"] == r2["best_evoi"]


# ---------------------------------------------------------------------------
# 5. Full pipeline replay: seed DB → score_all → rank_all
# ---------------------------------------------------------------------------

class TestFullPipelineReplay:
    """End-to-end deterministic replay with 5 synthetic NEOs."""

    # Synthetic NEOs: (id, name, neo, a, e, i, moid, diameter)
    NEOS = [
        (1, "Alpha", 1, 1.3, 0.2, 5.0, 0.01, 1.0),
        (2, "Beta", 1, 1.5, 0.3, 10.0, 0.05, 0.5),
        (3, "Gamma", 1, 2.0, 0.5, 15.0, 0.10, 2.0),
        (4, "Delta", 1, 1.1, 0.1, 2.0, 0.005, 0.3),
        (5, "Epsilon", 1, 2.5, 0.6, 20.0, 0.20, 3.0),
    ]

    # Expected score_all ranking (descending composite_score), seed=42, n=500
    EXPECTED_SCORE_RANKING = [3, 5, 1, 2, 4]

    # Expected EVOI ranking (descending best_evoi), seed=42, n=200
    EXPECTED_EVOI_RANKING = [3, 5, 1, 2, 4]

    # Expected composite scores from score_all (seed=42, n=500)
    EXPECTED_SCORES = {
        3: 31410619264.921436,
        5: 23221696523.554543,
        1: 10488795037.019817,
        2: 1096031965.3805215,
        4: 268587666.143565,
    }

    @pytest.fixture
    def pipeline_db(self):
        conn = get_connection(":memory:")
        for aid, name, neo, a, e, i, moid, diam in self.NEOS:
            conn.execute(
                "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, ?)",
                (aid, name, neo),
            )
            conn.execute(
                "INSERT INTO orbits (asteroid_id, a, e, i, moid, diameter) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (aid, a, e, i, moid, diam),
            )
        conn.commit()
        return conn

    def test_score_all_ranking(self, pipeline_db):
        count = score_all(pipeline_db, n_samples=500, mode="earth_return")
        assert count == 5
        rows = pipeline_db.execute(
            "SELECT asteroid_id FROM scores ORDER BY composite_score DESC"
        ).fetchall()
        ranking = [r[0] for r in rows]
        assert ranking == self.EXPECTED_SCORE_RANKING

    def test_score_all_values(self, pipeline_db):
        score_all(pipeline_db, n_samples=500, mode="earth_return")
        for aid, expected_score in self.EXPECTED_SCORES.items():
            row = pipeline_db.execute(
                "SELECT composite_score FROM scores WHERE asteroid_id = ?",
                (aid,),
            ).fetchone()
            assert row is not None, f"No score for asteroid {aid}"
            assert row[0] == pytest.approx(
                expected_score, rel=_REL_TOL
            ), f"Score mismatch for asteroid {aid}"

    def test_rank_all_evoi_ordering(self, pipeline_db):
        results = rank_all(
            pipeline_db, n_samples=200, mode="earth_return", neo_only=False
        )
        evoi_ranking = [r.asteroid_id for r in results]
        assert evoi_ranking == self.EXPECTED_EVOI_RANKING

    def test_rank_all_values_stable(self, pipeline_db):
        """Two rank_all calls on identical DBs produce identical EVOI values."""
        r1 = rank_all(pipeline_db, n_samples=200, mode="earth_return", neo_only=False)

        # Re-create DB for second run (rank_all uses internal seed=42)
        conn2 = get_connection(":memory:")
        for aid, name, neo, a, e, i, moid, diam in self.NEOS:
            conn2.execute(
                "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, ?)",
                (aid, name, neo),
            )
            conn2.execute(
                "INSERT INTO orbits (asteroid_id, a, e, i, moid, diameter) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (aid, a, e, i, moid, diam),
            )
        conn2.commit()
        r2 = rank_all(conn2, n_samples=200, mode="earth_return", neo_only=False)

        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.asteroid_id == b.asteroid_id
            assert a.best_evoi == b.best_evoi
            assert a.current_score_mean == b.current_score_mean
