"""Tests for prospector.output — ranked candidate CSV output."""

import csv
import math

import numpy as np
import pytest

from prospector.db import get_connection
from prospector.output import (
    COLUMNS,
    _fetch_candidates,
    _fmt,
    _fmt_sci,
    generate_ranked_csv,
    generate_ranked_dicts,
)
from prospector.scoring.granvik_prior import MAHLKE_CLASSES


# --- Helpers ---

def _make_prob_vector(**class_probs):
    """Build a 17-element prob vector from keyword class probabilities."""
    vec = np.zeros(len(MAHLKE_CLASSES))
    for cls, prob in class_probs.items():
        idx = MAHLKE_CLASSES.index(cls)
        vec[idx] = prob
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def _populate_db(conn, n=3):
    """Insert n asteroids with full pipeline data for testing."""
    asteroids = [
        (25143, "Itokawa", "1998 SF36", True, True),
        (101955, "Bennu", "1999 RQ36", True, True),
        (16, "Psyche", None, False, False),
        (433, "Eros", None, True, True),
        (99942, "Apophis", "2004 MN4", True, True),
    ]

    orbits = [
        (25143, 1.324, 0.280, 1.622, 0.014, 0.33),
        (101955, 1.126, 0.204, 6.035, 0.003, 0.49),
        (16, 2.920, 0.140, 3.095, 1.534, 226.0),
        (433, 1.458, 0.223, 10.83, 0.149, 16.84),
        (99942, 0.922, 0.191, 3.331, 0.000, 0.37),
    ]

    scores = [
        (25143, 1.5e10, 5e-7, "pgm", 50000.0, 0.87, 8.5e9, "earth_return"),
        (101955, 5.0e10, 0.07, "water", 0.001, 0.97, 3.5e6, "earth_return"),
        (16, 2.7e19, 1e-6, "pgm", 50000.0, 0.01, 1.4e13, "earth_return"),
        (433, 6.7e15, 3e-7, "pgm", 50000.0, 0.22, 2.2e8, "earth_return"),
        (99942, 2.7e10, 4e-7, "pgm", 50000.0, 1.00, 5.4e9, "earth_return"),
    ]

    for aid, name, desig, neo, pha in asteroids[:n]:
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, designation, neo, pha) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, name, desig, neo, pha),
        )

    for aid, a, e, i, moid, diam in orbits[:n]:
        conn.execute(
            "INSERT INTO orbits (asteroid_id, a, e, i, moid, diameter) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (aid, a, e, i, moid, diam),
        )

    for aid, mass, grade, mat, uv, acc, score, mode in scores[:n]:
        conn.execute(
            "INSERT INTO scores (asteroid_id, estimated_mass_kg, grade_estimate, "
            "target_material, unit_value, accessibility, composite_score, score_mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (aid, mass, grade, mat, uv, acc, score, mode),
        )

    # Add taxonomy for first asteroid
    prob = _make_prob_vector(S=0.85, Q=0.10, K=0.05)
    conn.execute(
        "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, prob_vector, "
        "classifier, input_coverage) VALUES (?, ?, ?, ?, ?, ?)",
        (asteroids[0][0], "S", 0.85, prob.tobytes(), "classy_mahlke2022", "vnir"),
    )

    conn.commit()
    return n


# --- Format Helper Tests ---

class TestFormatHelpers:
    """Tests for _fmt and _fmt_sci."""

    def test_fmt_none(self):
        assert _fmt(None) == ""

    def test_fmt_float(self):
        assert _fmt(3.14159, 2) == "3.14"

    def test_fmt_zero_decimals(self):
        assert _fmt(42.7, 0) == "43"

    def test_fmt_sci_none(self):
        assert _fmt_sci(None) == ""

    def test_fmt_sci_value(self):
        result = _fmt_sci(1.5e10)
        assert "e+10" in result

    def test_fmt_sci_small(self):
        result = _fmt_sci(5e-7)
        assert "e-07" in result


# --- Fetch Candidates Tests ---

class TestFetchCandidates:
    """Tests for _fetch_candidates."""

    def test_returns_scored_asteroids(self):
        conn = get_connection(":memory:")
        n = _populate_db(conn, 3)
        results = _fetch_candidates(conn, mode="earth_return")
        assert len(results) == n

    def test_ordered_by_composite_score_desc(self):
        conn = get_connection(":memory:")
        _populate_db(conn, 3)
        results = _fetch_candidates(conn, mode="earth_return")
        scores = [r["composite_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_limit_parameter(self):
        conn = get_connection(":memory:")
        _populate_db(conn, 5)
        results = _fetch_candidates(conn, mode="earth_return", limit=2)
        assert len(results) == 2

    def test_min_score_filter(self):
        conn = get_connection(":memory:")
        _populate_db(conn, 3)
        # Get the max score to set a threshold
        results_all = _fetch_candidates(conn, mode="earth_return")
        max_score = results_all[0]["composite_score"]
        results_filtered = _fetch_candidates(
            conn, mode="earth_return", min_score=max_score
        )
        assert len(results_filtered) == 1

    def test_includes_taxonomy_data(self):
        conn = get_connection(":memory:")
        _populate_db(conn, 1)  # Itokawa has taxonomy
        results = _fetch_candidates(conn, mode="earth_return")
        assert results[0]["primary_class"] == "S"
        assert results[0]["primary_prob"] == pytest.approx(0.85)

    def test_empty_database(self):
        conn = get_connection(":memory:")
        results = _fetch_candidates(conn, mode="earth_return")
        assert results == []

    def test_wrong_mode_returns_empty(self):
        conn = get_connection(":memory:")
        _populate_db(conn, 3)
        results = _fetch_candidates(conn, mode="in_space")
        assert results == []


# --- Generate Ranked CSV Tests ---

class TestGenerateRankedCSV:
    """Tests for generate_ranked_csv."""

    def test_creates_csv_file(self, tmp_path):
        conn = get_connection(":memory:")
        _populate_db(conn, 3)
        out = tmp_path / "ranked.csv"
        count = generate_ranked_csv(conn, out, mode="earth_return")
        assert out.exists()
        assert count == 3

    def test_csv_has_correct_headers(self, tmp_path):
        conn = get_connection(":memory:")
        _populate_db(conn, 1)
        out = tmp_path / "ranked.csv"
        generate_ranked_csv(conn, out, mode="earth_return")

        with open(out) as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == COLUMNS

    def test_csv_rows_are_ranked(self, tmp_path):
        conn = get_connection(":memory:")
        _populate_db(conn, 3)
        out = tmp_path / "ranked.csv"
        generate_ranked_csv(conn, out, mode="earth_return")

        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        ranks = [int(r["rank"]) for r in rows]
        assert ranks == [1, 2, 3]

    def test_csv_scores_descending(self, tmp_path):
        conn = get_connection(":memory:")
        _populate_db(conn, 3)
        out = tmp_path / "ranked.csv"
        generate_ranked_csv(conn, out, mode="earth_return")

        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        scores = [float(r["composite_score"]) for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_csv_limit(self, tmp_path):
        conn = get_connection(":memory:")
        _populate_db(conn, 5)
        out = tmp_path / "ranked.csv"
        count = generate_ranked_csv(conn, out, mode="earth_return", limit=2)
        assert count == 2

        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2

    def test_csv_contains_asteroid_names(self, tmp_path):
        conn = get_connection(":memory:")
        _populate_db(conn, 3)
        out = tmp_path / "ranked.csv"
        generate_ranked_csv(conn, out, mode="earth_return")

        with open(out) as f:
            reader = csv.DictReader(f)
            names = [r["name"] for r in reader]
        # First 3 asteroids should be in output
        assert any(n in names for n in ["Itokawa", "Bennu", "Psyche"])

    def test_creates_parent_directories(self, tmp_path):
        conn = get_connection(":memory:")
        _populate_db(conn, 1)
        out = tmp_path / "nested" / "dir" / "ranked.csv"
        generate_ranked_csv(conn, out, mode="earth_return")
        assert out.exists()

    def test_empty_database_creates_header_only(self, tmp_path):
        conn = get_connection(":memory:")
        out = tmp_path / "ranked.csv"
        count = generate_ranked_csv(conn, out, mode="earth_return")
        assert count == 0
        assert out.exists()

        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 0

    def test_taxonomy_fields_populated(self, tmp_path):
        conn = get_connection(":memory:")
        _populate_db(conn, 1)
        out = tmp_path / "ranked.csv"
        generate_ranked_csv(conn, out, mode="earth_return")

        with open(out) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["taxonomy_class"] == "S"
        assert float(row["taxonomy_prob"]) == pytest.approx(0.85, abs=0.01)

    def test_orbital_fields_populated(self, tmp_path):
        conn = get_connection(":memory:")
        _populate_db(conn, 1)
        out = tmp_path / "ranked.csv"
        generate_ranked_csv(conn, out, mode="earth_return")

        with open(out) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert float(row["a_au"]) > 0
        assert float(row["moid_au"]) >= 0

    def test_pgm_fields_when_available(self, tmp_path):
        conn = get_connection(":memory:")
        _populate_db(conn, 1)
        # Add PGM data
        conn.execute(
            "INSERT INTO pgm_convergence (asteroid_id, pgm_tier, pgm_confidence, signal_count) "
            "VALUES (?, ?, ?, ?)",
            (25143, "Tier S", 0.25, 2),
        )
        conn.commit()

        out = tmp_path / "ranked.csv"
        generate_ranked_csv(conn, out, mode="earth_return")

        with open(out) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["pgm_tier"] == "Tier S"
        assert float(row["pgm_confidence"]) == pytest.approx(0.25, abs=0.01)


# --- Generate Ranked Dicts Tests ---

class TestGenerateRankedDicts:
    """Tests for generate_ranked_dicts."""

    def test_returns_list_of_dicts(self):
        conn = get_connection(":memory:")
        _populate_db(conn, 3)
        results = generate_ranked_dicts(conn, mode="earth_return")
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_includes_rank_field(self):
        conn = get_connection(":memory:")
        _populate_db(conn, 3)
        results = generate_ranked_dicts(conn, mode="earth_return")
        ranks = [r["rank"] for r in results]
        assert ranks == [1, 2, 3]

    def test_ordered_by_score(self):
        conn = get_connection(":memory:")
        _populate_db(conn, 3)
        results = generate_ranked_dicts(conn, mode="earth_return")
        scores = [r["composite_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_limit_parameter(self):
        conn = get_connection(":memory:")
        _populate_db(conn, 5)
        results = generate_ranked_dicts(conn, mode="earth_return", limit=2)
        assert len(results) == 2


# --- Validation Tests ---

class TestValidation:
    """Validate sensible ranking for known asteroids."""

    def test_itokawa_and_bennu_both_ranked(self, tmp_path):
        """Both Itokawa and Bennu should appear in output."""
        conn = get_connection(":memory:")
        _populate_db(conn, 2)
        results = generate_ranked_dicts(conn, mode="earth_return")
        ids = {r["asteroid_id"] for r in results}
        assert 25143 in ids  # Itokawa
        assert 101955 in ids  # Bennu

    def test_all_scoring_fields_present(self, tmp_path):
        conn = get_connection(":memory:")
        _populate_db(conn, 1)
        results = generate_ranked_dicts(conn, mode="earth_return")
        r = results[0]
        assert r["composite_score"] is not None
        assert r["estimated_mass_kg"] is not None
        assert r["target_material"] is not None
        assert r["accessibility"] is not None

    def test_columns_list_complete(self):
        """COLUMNS should include all required FR-8 fields."""
        required = [
            "asteroid_id", "name", "taxonomy_class", "composite_score",
            "estimated_mass_kg", "pgm_tier", "accessibility", "moid_au",
            "target_material", "score_mode",
        ]
        for field in required:
            assert field in COLUMNS, f"Missing required column: {field}"

    def test_csv_roundtrip_preserves_data(self, tmp_path):
        """Data written to CSV should be readable and parseable."""
        conn = get_connection(":memory:")
        _populate_db(conn, 3)
        out = tmp_path / "ranked.csv"
        generate_ranked_csv(conn, out, mode="earth_return")

        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Verify all rows have all columns
        for row in rows:
            assert set(row.keys()) == set(COLUMNS)

        # Verify numeric fields are parseable
        for row in rows:
            assert float(row["composite_score"]) > 0
            assert float(row["estimated_mass_kg"]) > 0
