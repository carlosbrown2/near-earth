"""Tests for Stage 5 PGM candidate identification via multi-signal convergence."""

import numpy as np
import pytest

from prospector.db import get_connection, init_schema
from prospector.spectral.pgm_convergence import (
    FEATURELESS_RESIDUAL_THRESHOLD,
    IRON_METEORITE_GROUPS,
    MAHLKE_CLASSES,
    METAL_PROB_TIER1,
    METAL_PROB_TIER2,
    METAL_PROB_TIER3,
    M_INDEX,
    RADAR_ALBEDO_THRESHOLD,
    S_INDEX,
    X_INDEX,
    _assign_tier,
    _check_iron_in_analogs,
    _get_m_type_prob,
    _get_metallic_prob,
    _get_s_type_prob,
    assess_all,
    assess_asteroid,
    assess_nir_slope,
    check_iron_analog,
    check_silicate_bands,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prob_vector(**class_probs):
    """Build a 17-element Mahlke probability vector from keyword class probs.

    Unspecified classes get 0.0.  The vector is NOT normalised to 1.0
    unless the caller arranges that.
    """
    vec = np.zeros(len(MAHLKE_CLASSES), dtype=np.float64)
    for cls, prob in class_probs.items():
        idx = MAHLKE_CLASSES.index(cls)
        vec[idx] = prob
    return vec


def _setup_m_type_asteroid(conn, asteroid_id=16, m_prob=0.8):
    """Insert an M-type asteroid with full supporting data."""
    conn.execute(
        "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, 1)",
        (asteroid_id, "Psyche"),
    )
    conn.execute(
        "INSERT INTO orbits (asteroid_id, a, e, i, moid) VALUES (?, 2.92, 0.14, 3.1, 0.5)",
        (asteroid_id,),
    )
    pv = _make_prob_vector(M=m_prob, X=0.1, S=0.05)
    conn.execute(
        "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, prob_vector, classifier) "
        "VALUES (?, 'M', ?, ?, 'classy_mahlke2022')",
        (asteroid_id, m_prob, pv.tobytes()),
    )
    # Featureless reddish NIR spectrum
    wl = np.arange(0.4, 2.55, 0.005)
    refl = 0.8 + 0.1 * (wl - 0.4) / 2.0  # gentle positive slope, no features
    conn.execute(
        "INSERT INTO spectra (asteroid_id, survey, wavelengths, reflectance, "
        "wl_min, wl_max, normalized, quality_flag) "
        "VALUES (?, 'MITHNEOS', ?, ?, 0.40, 2.50, TRUE, 'good')",
        (asteroid_id, wl.tobytes(), refl.tobytes()),
    )
    conn.commit()


def _setup_s_type_asteroid(conn, asteroid_id=25143, subtype="S(IV)"):
    """Insert an S-type asteroid with band analysis."""
    conn.execute(
        "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, 1)",
        (asteroid_id, "Itokawa"),
    )
    conn.execute(
        "INSERT INTO orbits (asteroid_id, a, e, i, moid) VALUES (?, 1.32, 0.28, 1.6, 0.01)",
        (asteroid_id,),
    )
    pv = _make_prob_vector(S=0.75, Q=0.15, L=0.05)
    conn.execute(
        "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, prob_vector, classifier) "
        "VALUES (?, 'S', 0.75, ?, 'classy_mahlke2022')",
        (asteroid_id, pv.tobytes()),
    )
    if subtype:
        conn.execute(
            "INSERT INTO band_analysis (asteroid_id, band1_center, band2_center, "
            "bar, ol_opx_ratio, gaffey_subtype, calibration) "
            "VALUES (?, 0.99, 2.01, 1.2, 0.65, ?, 'dunn2010')",
            (asteroid_id, subtype),
        )
    conn.commit()


def _setup_c_type_asteroid(conn, asteroid_id=101955):
    """Insert a C-type asteroid (should not be a PGM candidate)."""
    conn.execute(
        "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, 1)",
        (asteroid_id, "Bennu"),
    )
    pv = _make_prob_vector(C=0.70, B=0.15, Ch=0.10)
    conn.execute(
        "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, prob_vector, classifier) "
        "VALUES (?, 'C', 0.70, ?, 'classy_mahlke2022')",
        (asteroid_id, pv.tobytes()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    """In-memory DB with full schema."""
    c = get_connection(":memory:")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Tests: probability helpers
# ---------------------------------------------------------------------------

class TestProbHelpers:

    def test_m_type_prob(self):
        pv = _make_prob_vector(M=0.6, X=0.2, S=0.1)
        assert _get_m_type_prob(pv) == pytest.approx(0.6)

    def test_metallic_prob(self):
        pv = _make_prob_vector(M=0.6, X=0.2, E=0.05, S=0.1)
        assert _get_metallic_prob(pv) == pytest.approx(0.85)

    def test_s_type_prob(self):
        pv = _make_prob_vector(S=0.6, Q=0.15)
        assert _get_s_type_prob(pv) == pytest.approx(0.75)

    def test_none_prob_vector(self):
        assert _get_m_type_prob(None) == 0.0
        assert _get_metallic_prob(None) == 0.0
        assert _get_s_type_prob(None) == 0.0

    def test_empty_prob_vector(self):
        assert _get_m_type_prob(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Tests: assess_nir_slope
# ---------------------------------------------------------------------------

class TestAssessNirSlope:

    def test_featureless_reddish(self):
        """Flat reddish spectrum with no features."""
        wl = np.arange(0.4, 2.55, 0.005)
        refl = 0.8 + 0.05 * (wl - 0.4)  # gentle positive slope
        result = assess_nir_slope(wl, refl)
        assert result["has_nir"] is True
        assert result["slope"] > 0
        assert result["is_reddish"] is True
        assert result["is_featureless"] is True

    def test_spectrum_with_absorption_bands(self):
        """Spectrum with strong 1-um band should not be featureless."""
        wl = np.arange(0.4, 2.55, 0.005)
        refl = np.ones_like(wl)
        # Add deep absorption band at 1.0 um
        refl -= 0.4 * np.exp(-((wl - 1.0) ** 2) / (2 * 0.05**2))
        result = assess_nir_slope(wl, refl)
        assert result["has_nir"] is True
        assert result["is_featureless"] is False

    def test_no_nir_coverage(self):
        """Visible-only spectrum (wl_max < 1.5)."""
        wl = np.arange(0.4, 0.92, 0.005)
        refl = np.ones_like(wl) * 0.8
        result = assess_nir_slope(wl, refl)
        assert result["has_nir"] is False

    def test_blue_slope(self):
        """Negative (blue) slope should not be reddish."""
        wl = np.arange(0.4, 2.55, 0.005)
        refl = 1.0 - 0.1 * (wl - 0.4)  # decreasing with wavelength
        result = assess_nir_slope(wl, refl)
        assert result["is_reddish"] is False

    def test_empty_input(self):
        result = assess_nir_slope(None, None)
        assert result["has_nir"] is False

    def test_all_nan(self):
        wl = np.arange(0.4, 2.55, 0.005)
        refl = np.full_like(wl, np.nan)
        result = assess_nir_slope(wl, refl)
        assert result["has_nir"] is False

    def test_few_valid_points(self):
        """Too few valid points."""
        wl = np.array([0.5, 1.0, 2.0])
        refl = np.array([1.0, 1.0, 1.0])
        result = assess_nir_slope(wl, refl)
        # Only 3 points < 5 minimum for NIR slope
        assert result["slope"] == 0.0


# ---------------------------------------------------------------------------
# Tests: check_silicate_bands
# ---------------------------------------------------------------------------

class TestCheckSilicateBands:

    def test_bands_present(self, conn):
        """Asteroid in band_analysis table has bands present."""
        conn.execute("INSERT INTO asteroids (asteroid_id) VALUES (1)")
        conn.execute(
            "INSERT INTO taxonomy (asteroid_id, primary_class) VALUES (1, 'S')"
        )
        conn.execute(
            "INSERT INTO band_analysis (asteroid_id, band1_center) VALUES (1, 0.99)"
        )
        conn.commit()
        assert check_silicate_bands(1, conn) is False  # bands present

    def test_nonsilicate_no_bands(self, conn):
        """M-type not in band_analysis => bands absent."""
        conn.execute("INSERT INTO asteroids (asteroid_id) VALUES (2)")
        conn.execute(
            "INSERT INTO taxonomy (asteroid_id, primary_class) VALUES (2, 'M')"
        )
        conn.commit()
        assert check_silicate_bands(2, conn) is True  # no bands

    def test_silicate_unknown(self, conn):
        """S-type not in band_analysis => unknown (not processed yet)."""
        conn.execute("INSERT INTO asteroids (asteroid_id) VALUES (3)")
        conn.execute(
            "INSERT INTO taxonomy (asteroid_id, primary_class) VALUES (3, 'S')"
        )
        conn.commit()
        assert check_silicate_bands(3, conn) is None  # unknown

    def test_no_taxonomy(self, conn):
        """No taxonomy entry, not in band_analysis => assume no bands."""
        conn.execute("INSERT INTO asteroids (asteroid_id) VALUES (4)")
        conn.commit()
        assert check_silicate_bands(4, conn) is True


# ---------------------------------------------------------------------------
# Tests: iron analog check
# ---------------------------------------------------------------------------

class TestCheckIronAnalog:

    def test_iron_match_found(self):
        analogs = [
            {"meteorite_group": "Iron", "wmse": 0.001, "rho": 0.97,
             "meteorite_name": "Canyon Diablo"},
        ]
        result = _check_iron_in_analogs(analogs)
        assert result["has_iron_match"] is True
        assert result["best_iron_wmse"] == 0.001
        assert result["best_iron_name"] == "Canyon Diablo"

    def test_stony_iron_match(self):
        analogs = [
            {"meteorite_group": "Stony-Iron", "wmse": 0.002, "rho": 0.95,
             "meteorite_name": "Vaca Muerta"},
        ]
        result = _check_iron_in_analogs(analogs)
        assert result["has_iron_match"] is True

    def test_no_iron_match(self):
        analogs = [
            {"meteorite_group": "Ordinary Chondrite", "wmse": 0.001,
             "rho": 0.98, "meteorite_name": "Allende"},
        ]
        result = _check_iron_in_analogs(analogs)
        assert result["has_iron_match"] is False

    def test_empty_analogs(self):
        result = _check_iron_in_analogs([])
        assert result["has_iron_match"] is False

    def test_iron_not_first_but_present(self):
        """Iron is second in list, not first — should still find it."""
        analogs = [
            {"meteorite_group": "Ordinary Chondrite", "wmse": 0.001,
             "rho": 0.99, "meteorite_name": "Allende"},
            {"meteorite_group": "Iron", "wmse": 0.002,
             "rho": 0.96, "meteorite_name": "Sikhote-Alin"},
        ]
        result = _check_iron_in_analogs(analogs)
        assert result["has_iron_match"] is True
        assert result["best_iron_name"] == "Sikhote-Alin"


# ---------------------------------------------------------------------------
# Tests: _assign_tier
# ---------------------------------------------------------------------------

class TestAssignTier:

    def test_tier_1(self):
        """Full convergence with radar."""
        signals = {
            "m_type_prob": 0.8,
            "metallic_prob": 0.9,
            "featureless_nir": True,
            "no_silicate_bands": True,
            "iron_analog_match": True,
            "radar_albedo": 0.35,
            "s_type_metal": False,
        }
        tier, conf = _assign_tier(signals)
        assert tier == "Tier 1"
        assert 0.70 <= conf <= 0.90

    def test_tier_2(self):
        """Good evidence but no radar."""
        signals = {
            "m_type_prob": 0.6,
            "metallic_prob": 0.7,
            "featureless_nir": True,
            "no_silicate_bands": True,
            "iron_analog_match": True,
            "radar_albedo": None,
            "s_type_metal": False,
        }
        tier, conf = _assign_tier(signals)
        assert tier == "Tier 2"
        assert 0.40 <= conf <= 0.60

    def test_tier_3(self):
        """Moderate M-prob, limited evidence."""
        signals = {
            "m_type_prob": 0.4,
            "metallic_prob": 0.5,
            "featureless_nir": False,
            "no_silicate_bands": None,
            "iron_analog_match": False,
            "radar_albedo": None,
            "s_type_metal": False,
        }
        tier, conf = _assign_tier(signals)
        assert tier == "Tier 3"
        assert 0.20 <= conf <= 0.30

    def test_tier_4(self):
        """Low metallic prob, speculative."""
        signals = {
            "m_type_prob": 0.1,
            "metallic_prob": 0.2,
            "featureless_nir": False,
            "no_silicate_bands": None,
            "iron_analog_match": False,
            "radar_albedo": None,
            "s_type_metal": False,
        }
        tier, conf = _assign_tier(signals)
        assert tier == "Tier 4"
        assert conf <= 0.10

    def test_tier_s(self):
        """S-type metal pathway."""
        signals = {
            "m_type_prob": 0.0,
            "metallic_prob": 0.0,
            "s_type_metal": True,
            "s_type_prob": 0.9,
        }
        tier, conf = _assign_tier(signals)
        assert tier == "Tier S"
        assert 0.0 < conf <= 0.30

    def test_not_candidate(self):
        """No metallic or S-type probability."""
        signals = {
            "m_type_prob": 0.0,
            "metallic_prob": 0.05,
            "featureless_nir": False,
            "no_silicate_bands": None,
            "iron_analog_match": False,
            "radar_albedo": None,
            "s_type_metal": False,
        }
        tier, conf = _assign_tier(signals)
        assert tier is None
        assert conf == 0.0

    def test_tier_s_takes_priority_over_metallic(self):
        """S-type metal should be assigned Tier S even if metallic_prob > 0."""
        signals = {
            "m_type_prob": 0.1,
            "metallic_prob": 0.15,
            "s_type_metal": True,
            "s_type_prob": 0.8,
        }
        tier, _ = _assign_tier(signals)
        assert tier == "Tier S"


# ---------------------------------------------------------------------------
# Tests: assess_asteroid (DB integration)
# ---------------------------------------------------------------------------

class TestAssessAsteroid:

    def test_m_type_asteroid(self, conn):
        """M-type asteroid gets Tier 3 (no lab spectra for iron match)."""
        _setup_m_type_asteroid(conn, asteroid_id=16, m_prob=0.8)
        result = assess_asteroid(16, conn)
        assert result is not None
        assert result["pgm_tier"] in {"Tier 2", "Tier 3"}
        assert result["m_type_prob"] == pytest.approx(0.8)
        assert result["featureless_nir"] is True  # smooth reddish spectrum
        assert "M-type" in result["notes"]

    def test_s_type_s4_asteroid(self, conn):
        """S(IV) S-type should be Tier S candidate."""
        _setup_s_type_asteroid(conn, asteroid_id=25143, subtype="S(IV)")
        result = assess_asteroid(25143, conn)
        assert result is not None
        assert result["pgm_tier"] == "Tier S"
        assert result["s_type_metal"] is True
        assert result["metal_fraction_pct"] == pytest.approx(15.0)

    def test_s_type_non_s4(self, conn):
        """S-type with non-S(IV) subtype is NOT Tier S unless CNN agrees."""
        _setup_s_type_asteroid(conn, asteroid_id=433, subtype="S(III)")
        result = assess_asteroid(433, conn)
        # S-type but not S(IV) and no CNN data — not a PGM candidate
        assert result is None

    def test_c_type_not_candidate(self, conn):
        """C-type asteroid should not be a PGM candidate."""
        _setup_c_type_asteroid(conn, asteroid_id=101955)
        result = assess_asteroid(101955, conn)
        assert result is None

    def test_no_taxonomy(self, conn):
        """Asteroid without taxonomy data returns None."""
        conn.execute("INSERT INTO asteroids (asteroid_id) VALUES (999)")
        conn.commit()
        result = assess_asteroid(999, conn)
        assert result is None

    def test_precomputed_analogs(self, conn):
        """Pass pre-computed analogs to avoid find_analogs call."""
        _setup_m_type_asteroid(conn, asteroid_id=16, m_prob=0.6)
        analogs = [
            {"meteorite_group": "Iron", "wmse": 0.001, "rho": 0.97,
             "meteorite_name": "Canyon Diablo"},
        ]
        result = assess_asteroid(16, conn, analogs=analogs)
        assert result is not None
        assert result["iron_analog_match"] is True
        # M-prob=0.6 > 0.5, featureless NIR, iron match → Tier 2
        assert result["pgm_tier"] == "Tier 2"

    def test_with_radar_albedo_tier1(self, conn):
        """M-type with all signals including radar → Tier 1 is possible.

        We can't set radar_albedo via the current schema (not ingested),
        but we can verify the tier assignment logic via _assign_tier directly.
        """
        signals = {
            "m_type_prob": 0.8,
            "metallic_prob": 0.9,
            "featureless_nir": True,
            "no_silicate_bands": True,
            "iron_analog_match": True,
            "radar_albedo": 0.35,
            "s_type_metal": False,
        }
        tier, conf = _assign_tier(signals)
        assert tier == "Tier 1"
        assert conf >= 0.70

    def test_x_type_speculative(self, conn):
        """X-type with no albedo → Tier 4 speculative."""
        conn.execute("INSERT INTO asteroids (asteroid_id, neo) VALUES (100, 1)")
        pv = _make_prob_vector(X=0.5, M=0.1, E=0.1)
        conn.execute(
            "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, prob_vector, classifier) "
            "VALUES (100, 'X', 0.5, ?, 'classy_mahlke2022')",
            (pv.tobytes(),),
        )
        conn.commit()
        result = assess_asteroid(100, conn)
        assert result is not None
        # Low M-prob but metallic_prob > 0.1 → Tier 3 or 4
        assert result["pgm_tier"] in {"Tier 3", "Tier 4"}

    def test_signal_count(self, conn):
        """Signal count reflects positive signals."""
        _setup_m_type_asteroid(conn, asteroid_id=16, m_prob=0.8)
        result = assess_asteroid(16, conn)
        assert result["signal_count"] >= 1  # at least m_prob > 0.3


# ---------------------------------------------------------------------------
# Tests: assess_all (batch)
# ---------------------------------------------------------------------------

class TestAssessAll:

    def test_batch_processing(self, conn):
        """Batch assess finds PGM candidates among mixed population."""
        _setup_m_type_asteroid(conn, asteroid_id=16, m_prob=0.8)
        _setup_s_type_asteroid(conn, asteroid_id=25143, subtype="S(IV)")
        _setup_c_type_asteroid(conn, asteroid_id=101955)

        count = assess_all(conn)
        assert count == 2  # M-type + S-type; C-type excluded

        # Verify DB storage
        rows = conn.execute(
            "SELECT asteroid_id, pgm_tier FROM pgm_convergence ORDER BY asteroid_id"
        ).fetchall()
        assert len(rows) == 2
        tiers = {r[0]: r[1] for r in rows}
        assert tiers[25143] == "Tier S"

    def test_empty_taxonomy(self, conn):
        """No taxonomy data → 0 candidates."""
        count = assess_all(conn)
        assert count == 0

    def test_idempotent(self, conn):
        """Running assess_all twice produces same results (INSERT OR REPLACE)."""
        _setup_m_type_asteroid(conn, asteroid_id=16, m_prob=0.8)
        count1 = assess_all(conn)
        count2 = assess_all(conn)
        assert count1 == count2

        rows = conn.execute("SELECT COUNT(*) FROM pgm_convergence").fetchone()
        assert rows[0] == 1  # not duplicated


# ---------------------------------------------------------------------------
# Tests: DB table
# ---------------------------------------------------------------------------

class TestPgmConvergenceTable:

    def test_table_exists(self, conn):
        """pgm_convergence table is created by schema init."""
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pgm_convergence'"
        ).fetchall()
        assert len(tables) == 1

    def test_insert_and_read(self, conn):
        """Basic round-trip insert and read."""
        conn.execute("INSERT INTO asteroids (asteroid_id) VALUES (1)")
        conn.execute(
            "INSERT INTO pgm_convergence "
            "(asteroid_id, pgm_tier, pgm_confidence, m_type_prob, "
            "featureless_nir, no_silicate_bands, iron_analog_match, "
            "s_type_metal, signal_count) "
            "VALUES (1, 'Tier 2', 0.55, 0.65, 1, 1, 1, 0, 3)"
        )
        conn.commit()

        row = conn.execute(
            "SELECT pgm_tier, pgm_confidence, m_type_prob, signal_count "
            "FROM pgm_convergence WHERE asteroid_id = 1"
        ).fetchone()
        assert row[0] == "Tier 2"
        assert row[1] == pytest.approx(0.55)
        assert row[2] == pytest.approx(0.65)
        assert row[3] == 3

    def test_fk_enforced(self, conn):
        """FK to asteroids table is enforced."""
        with pytest.raises(Exception):
            conn.execute(
                "INSERT INTO pgm_convergence (asteroid_id, pgm_tier) VALUES (9999, 'Tier 3')"
            )


# ---------------------------------------------------------------------------
# Tests: physical sanity
# ---------------------------------------------------------------------------

class TestPhysicalSanity:

    def test_confidence_bounds(self):
        """Confidence is always in [0, 1]."""
        test_cases = [
            {"m_type_prob": 0.9, "metallic_prob": 0.95, "featureless_nir": True,
             "no_silicate_bands": True, "iron_analog_match": True,
             "radar_albedo": 0.4, "s_type_metal": False},
            {"m_type_prob": 0.6, "metallic_prob": 0.7, "featureless_nir": True,
             "iron_analog_match": True, "radar_albedo": None, "s_type_metal": False},
            {"m_type_prob": 0.35, "metallic_prob": 0.4, "featureless_nir": False,
             "iron_analog_match": False, "radar_albedo": None, "s_type_metal": False},
            {"m_type_prob": 0.0, "metallic_prob": 0.15, "featureless_nir": False,
             "iron_analog_match": False, "radar_albedo": None, "s_type_metal": False},
            {"m_type_prob": 0.0, "metallic_prob": 0.0, "s_type_metal": True,
             "s_type_prob": 0.8},
        ]
        for signals in test_cases:
            tier, conf = _assign_tier(signals)
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of bounds for {signals}"

    def test_tier_ordering(self):
        """Higher tiers should generally have higher confidence."""
        tier1 = _assign_tier({
            "m_type_prob": 0.8, "metallic_prob": 0.9, "featureless_nir": True,
            "no_silicate_bands": True, "iron_analog_match": True,
            "radar_albedo": 0.35, "s_type_metal": False})[1]
        tier2 = _assign_tier({
            "m_type_prob": 0.6, "metallic_prob": 0.7, "featureless_nir": True,
            "iron_analog_match": True, "radar_albedo": None, "s_type_metal": False})[1]
        tier3 = _assign_tier({
            "m_type_prob": 0.4, "metallic_prob": 0.5, "featureless_nir": False,
            "iron_analog_match": False, "radar_albedo": None, "s_type_metal": False})[1]
        tier4 = _assign_tier({
            "m_type_prob": 0.05, "metallic_prob": 0.15, "featureless_nir": False,
            "iron_analog_match": False, "radar_albedo": None, "s_type_metal": False})[1]

        assert tier1 > tier2 > tier3 > tier4

    def test_psyche_like_m_type(self, conn):
        """16 Psyche-like asteroid should be a PGM candidate."""
        _setup_m_type_asteroid(conn, asteroid_id=16, m_prob=0.8)
        result = assess_asteroid(16, conn)
        assert result is not None
        assert result["pgm_confidence"] > 0.0
        assert result["pgm_tier"] in {"Tier 2", "Tier 3"}

    def test_itokawa_like_s_type(self, conn):
        """25143 Itokawa-like S(IV) should be Tier S."""
        _setup_s_type_asteroid(conn, asteroid_id=25143, subtype="S(IV)")
        result = assess_asteroid(25143, conn)
        assert result is not None
        assert result["pgm_tier"] == "Tier S"
        assert result["metal_fraction_pct"] > 0

    def test_cnn_metal_detection(self, conn):
        """S-type with CNN showing low silicate total → metal detected."""
        conn.execute("INSERT INTO asteroids (asteroid_id, neo) VALUES (500, 1)")
        pv = _make_prob_vector(S=0.8, Q=0.1)
        conn.execute(
            "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, "
            "prob_vector, classifier) VALUES (500, 'S', 0.8, ?, 'classy_mahlke2022')",
            (pv.tobytes(),),
        )
        # CNN shows only 80% silicates → 20% metal
        conn.execute(
            "INSERT INTO cnn_mineral (asteroid_id, ol_pct, opx_pct, cpx_pct) "
            "VALUES (500, 50.0, 25.0, 5.0)"
        )
        conn.commit()

        result = assess_asteroid(500, conn)
        assert result is not None
        assert result["s_type_metal"] is True
        assert result["metal_fraction_pct"] >= 20.0
        assert result["pgm_tier"] == "Tier S"

    def test_mahlke_classes_consistent(self):
        """MAHLKE_CLASSES constant matches expected 17 classes."""
        assert len(MAHLKE_CLASSES) == 17
        assert "M" in MAHLKE_CLASSES
        assert "S" in MAHLKE_CLASSES
        assert "X" in MAHLKE_CLASSES
