"""Hypothesis stateful tests for the ingest→score pipeline.

Models the pipeline as a state machine using RuleBasedStateMachine.
Fuzzes transition sequences: interleaved ingests, partial data, repeated
scoring. Explores combinatorial state sequences that manual tests miss.

Invariants checked after every step:
- Scores non-negative
- Probability vectors normalized (sum ≈ 1.0)
- Accessibility bounded [0.01, 1.0]
- Foreign key integrity (scored asteroids exist in asteroids table)
- score_all count matches scoreable asteroids
"""

import math

import numpy as np
import pytest
from hypothesis import settings, note
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

from prospector.db import get_connection
from prospector.scoring.granvik_prior import MAHLKE_CLASSES
from prospector.scoring.scorer import load_config, score_all, score_asteroid


# ---------------------------------------------------------------------------
# Strategy helpers — physically reasonable asteroid parameters
# ---------------------------------------------------------------------------

asteroid_ids = st.integers(min_value=1, max_value=999_999)
orbital_a = st.floats(0.5, 6.0, allow_nan=False, allow_infinity=False)
orbital_e = st.floats(0.0, 0.99, allow_nan=False, allow_infinity=False)
orbital_i = st.floats(0.0, 90.0, allow_nan=False, allow_infinity=False)
moid_au = st.one_of(
    st.none(),
    st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False),
)
diameter_km = st.floats(0.001, 500.0, allow_nan=False, allow_infinity=False)
perihelion_q = st.one_of(
    st.none(),
    st.floats(0.05, 5.0, allow_nan=False, allow_infinity=False),
)
h_mag = st.one_of(
    st.none(),
    st.floats(10.0, 35.0, allow_nan=False, allow_infinity=False),
)
albedo = st.one_of(
    st.none(),
    st.floats(0.01, 0.60, allow_nan=False, allow_infinity=False),
)


def prob_vector_strategy():
    """Generate a valid 17-element probability vector (sums to 1)."""
    return (
        st.lists(
            st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False),
            min_size=17,
            max_size=17,
        )
        .map(lambda xs: np.array(xs))
        .map(lambda a: a / a.sum())
    )


prob_vectors = st.one_of(st.none(), prob_vector_strategy())


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class IngestScorePipeline(RuleBasedStateMachine):
    """Stateful test exercising the ingest→score pipeline.

    State tracks which asteroid IDs have been inserted (with various
    completeness of data), and which have been scored. Rules randomly
    interleave ingestion of asteroids (with full or partial data) and
    scoring operations. Invariants verify contracts after every step.
    """

    def __init__(self):
        super().__init__()
        self.conn = None
        self.config = None
        # Track inserted asteroid IDs and their data completeness
        self.inserted_ids: set[int] = set()
        self.has_diameter: set[int] = set()
        self.has_orbits: set[int] = set()
        self.has_taxonomy: set[int] = set()
        self.scored: bool = False

    @initialize()
    def setup(self):
        self.conn = get_connection(":memory:")
        self.config = load_config()
        self.inserted_ids = set()
        self.has_diameter = set()
        self.has_orbits = set()
        self.has_taxonomy = set()
        self.scored = False

    # --- Ingest rules ---

    @rule(
        aid=asteroid_ids,
        a=orbital_a,
        e=orbital_e,
        i=orbital_i,
        moid=moid_au,
        diam=diameter_km,
        q=perihelion_q,
        hmag=h_mag,
    )
    def ingest_full_asteroid(self, aid, a, e, i, moid, diam, q, hmag):
        """Insert an asteroid with full orbital and diameter data."""
        self.conn.execute(
            "INSERT OR REPLACE INTO asteroids (asteroid_id, neo, pha) VALUES (?, 1, 0)",
            (aid,),
        )
        # Compute q from orbit if not provided
        q_val = q if q is not None else a * (1.0 - e)
        self.conn.execute(
            """INSERT OR REPLACE INTO orbits
               (asteroid_id, a, e, i, moid, diameter, q, H)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (aid, a, e, i, moid, diam, q_val, hmag),
        )
        self.conn.commit()
        self.inserted_ids.add(aid)
        self.has_diameter.add(aid)
        self.has_orbits.add(aid)
        note(f"Ingested full asteroid {aid}: a={a:.2f}, e={e:.2f}, diam={diam:.3f}")

    @rule(aid=asteroid_ids, a=orbital_a, e=orbital_e, i=orbital_i)
    def ingest_partial_no_diameter(self, aid, a, e, i):
        """Insert asteroid with orbital elements but NO diameter — should be skipped by scorer."""
        self.conn.execute(
            "INSERT OR REPLACE INTO asteroids (asteroid_id, neo, pha) VALUES (?, 1, 0)",
            (aid,),
        )
        self.conn.execute(
            """INSERT OR REPLACE INTO orbits
               (asteroid_id, a, e, i, moid, diameter)
               VALUES (?, ?, ?, ?, NULL, NULL)""",
            (aid, a, e, i),
        )
        self.conn.commit()
        self.inserted_ids.add(aid)
        self.has_orbits.add(aid)
        self.has_diameter.discard(aid)
        note(f"Ingested partial asteroid {aid} (no diameter)")

    @rule(aid=asteroid_ids, pv=prob_vector_strategy())
    def add_taxonomy(self, aid, pv):
        """Add taxonomy probability vector to an existing asteroid."""
        if aid not in self.inserted_ids:
            return
        primary_idx = int(np.argmax(pv))
        primary_class = MAHLKE_CLASSES[primary_idx]
        self.conn.execute(
            """INSERT OR REPLACE INTO taxonomy
               (asteroid_id, primary_class, primary_prob, prob_vector, classifier)
               VALUES (?, ?, ?, ?, 'test')""",
            (aid, primary_class, float(pv[primary_idx]), pv.tobytes()),
        )
        self.conn.commit()
        self.has_taxonomy.add(aid)
        note(f"Added taxonomy to asteroid {aid}: primary={primary_class}")

    @rule(aid=asteroid_ids, alb=st.floats(0.01, 0.60, allow_nan=False, allow_infinity=False))
    def add_physical_properties(self, aid, alb):
        """Add NEOWISE-like physical properties to an existing asteroid."""
        if aid not in self.inserted_ids:
            return
        self.conn.execute(
            """INSERT OR REPLACE INTO physical_properties
               (asteroid_id, albedo_pv, source)
               VALUES (?, ?, 'test')""",
            (aid, alb),
        )
        self.conn.commit()
        note(f"Added albedo={alb:.3f} to asteroid {aid}")

    # --- Scoring rules ---

    @precondition(lambda self: len(self.inserted_ids) > 0)
    @rule(mode=st.sampled_from(["earth_return", "in_space"]))
    def score_all_asteroids(self, mode):
        """Run score_all on the database and verify counts."""
        count = score_all(self.conn, n_samples=50, mode=mode)

        # Count how many asteroids SHOULD be scoreable
        # (have diameter AND a, e, i)
        scoreable = self.conn.execute(
            """SELECT COUNT(*) FROM asteroids a
               JOIN orbits o ON a.asteroid_id = o.asteroid_id
               WHERE COALESCE(o.diameter, (
                   SELECT pp.diameter_km FROM physical_properties pp
                   WHERE pp.asteroid_id = a.asteroid_id
               )) IS NOT NULL
               AND o.a IS NOT NULL AND o.e IS NOT NULL AND o.i IS NOT NULL"""
        ).fetchone()[0]

        assert count == scoreable, (
            f"score_all returned {count} but {scoreable} asteroids are scoreable"
        )
        self.scored = True
        note(f"Scored {count} asteroids in '{mode}' mode")

    @precondition(lambda self: len(self.has_diameter & self.has_orbits) > 0)
    @rule(
        mode=st.sampled_from(["earth_return", "in_space"]),
        seed=st.integers(0, 2**31 - 1),
    )
    def score_single_asteroid(self, mode, seed):
        """Score a single asteroid directly and validate the result schema."""
        # Pick a scoreable asteroid from the DB
        row = self.conn.execute(
            """SELECT o.asteroid_id, o.a, o.e, o.i, o.moid, o.diameter, o.q,
                      t.prob_vector
               FROM orbits o
               LEFT JOIN taxonomy t ON o.asteroid_id = t.asteroid_id
               WHERE o.diameter IS NOT NULL
                 AND o.a IS NOT NULL AND o.e IS NOT NULL AND o.i IS NOT NULL
               LIMIT 1"""
        ).fetchone()
        if row is None:
            return

        aid, a, e, i, moid, diam, q, prob_blob = row
        prob_vector = None
        if prob_blob is not None:
            prob_vector = np.frombuffer(prob_blob, dtype=np.float64)
            if len(prob_vector) != 17:
                prob_vector = None

        result = score_asteroid(
            diameter_km=diam,
            a=a,
            e=e,
            i_deg=i,
            moid=moid,
            prob_vector=prob_vector,
            config=self.config,
            n_samples=50,
            mode=mode,
            rng=np.random.default_rng(seed),
            q_au=q,
        )

        # Validate through Pydantic schema
        from prospector.schemas import ScoringResult

        validated = ScoringResult.model_validate(result)
        assert validated.composite_score >= 0
        assert 0.01 <= validated.accessibility <= 1.0
        assert 0.01 <= validated.confidence <= 1.0
        assert validated.estimated_mass_kg >= 0
        assert validated.score_mode == mode
        note(f"Scored asteroid {aid}: composite={validated.composite_score:.4f}")

    # --- Re-ingest rules (update existing data) ---

    @rule(
        aid=asteroid_ids,
        diam=diameter_km,
    )
    def update_diameter(self, aid, diam):
        """Update an existing asteroid's diameter — re-scoring should reflect changes."""
        if aid not in self.has_orbits:
            return
        self.conn.execute(
            "UPDATE orbits SET diameter = ? WHERE asteroid_id = ?",
            (diam, aid),
        )
        self.conn.commit()
        self.has_diameter.add(aid)
        note(f"Updated diameter of asteroid {aid} to {diam:.3f}")

    # --- Invariants (checked after EVERY step) ---

    @invariant()
    def scores_non_negative(self):
        """All composite_score values in the scores table must be >= 0."""
        rows = self.conn.execute(
            "SELECT asteroid_id, composite_score FROM scores"
        ).fetchall()
        for aid, score in rows:
            assert score >= 0, f"Asteroid {aid} has negative score: {score}"

    @invariant()
    def accessibility_bounded(self):
        """All accessibility values must be in [0.01, 1.0]."""
        rows = self.conn.execute(
            "SELECT asteroid_id, accessibility FROM scores"
        ).fetchall()
        for aid, acc in rows:
            assert 0.01 <= acc <= 1.0, (
                f"Asteroid {aid} accessibility out of bounds: {acc}"
            )

    @invariant()
    def scored_asteroids_exist(self):
        """Every scored asteroid must exist in the asteroids table (FK integrity)."""
        orphans = self.conn.execute(
            """SELECT s.asteroid_id FROM scores s
               LEFT JOIN asteroids a ON s.asteroid_id = a.asteroid_id
               WHERE a.asteroid_id IS NULL"""
        ).fetchall()
        assert len(orphans) == 0, f"Orphan scores found: {[r[0] for r in orphans]}"

    @invariant()
    def taxonomy_vectors_normalized(self):
        """All taxonomy probability vectors must sum to ≈ 1.0."""
        rows = self.conn.execute(
            "SELECT asteroid_id, prob_vector FROM taxonomy WHERE prob_vector IS NOT NULL"
        ).fetchall()
        for aid, blob in rows:
            pv = np.frombuffer(blob, dtype=np.float64)
            assert len(pv) == 17, (
                f"Asteroid {aid} prob_vector has wrong length: {len(pv)}"
            )
            assert abs(pv.sum() - 1.0) < 1e-6, (
                f"Asteroid {aid} prob_vector sum = {pv.sum()}, expected ≈ 1.0"
            )
            assert (pv >= 0).all(), (
                f"Asteroid {aid} has negative probability in prob_vector"
            )

    @invariant()
    def mass_estimates_non_negative(self):
        """All estimated_mass_kg values must be >= 0."""
        rows = self.conn.execute(
            "SELECT asteroid_id, estimated_mass_kg FROM scores"
        ).fetchall()
        for aid, mass in rows:
            assert mass >= 0, f"Asteroid {aid} has negative mass: {mass}"

    @invariant()
    def scores_are_finite(self):
        """All numeric score fields must be finite (no NaN/inf)."""
        rows = self.conn.execute(
            """SELECT asteroid_id, composite_score, estimated_mass_kg,
                      accessibility, grade_estimate, unit_value
               FROM scores"""
        ).fetchall()
        for aid, cs, mass, acc, grade, uv in rows:
            for name, val in [
                ("composite_score", cs),
                ("estimated_mass_kg", mass),
                ("accessibility", acc),
                ("grade_estimate", grade),
                ("unit_value", uv),
            ]:
                assert math.isfinite(val), (
                    f"Asteroid {aid} has non-finite {name}: {val}"
                )

    def teardown(self):
        if self.conn is not None:
            self.conn.close()


# Hypothesis test runner — configure for reasonable runtime
TestIngestScorePipeline = IngestScorePipeline.TestCase
TestIngestScorePipeline.settings = settings(
    max_examples=50,
    stateful_step_count=15,
    deadline=None,
)
