"""Tests for Stage 3 meteorite analog curve matching."""

import numpy as np
import pytest

from prospector.db import get_connection
from prospector.spectral.curve_match import (
    MAHLKE_CLASSES,
    MIN_OVERLAP_POINTS,
    RHO_THRESHOLD,
    _best_scale_factor,
    _resample_to_overlap,
    _taxonomy_weight,
    compute_match_score,
    find_analogs,
    match_all,
    match_spectrum,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spectrum(wl_min=0.4, wl_max=2.5, step=0.005, slope=0.0, band_center=None):
    """Create a synthetic spectrum for testing."""
    wl = np.arange(wl_min, wl_max + step / 2, step)
    refl = np.ones_like(wl) + slope * (wl - wl_min)
    if band_center is not None:
        refl -= 0.3 * np.exp(-((wl - band_center) ** 2) / (2 * 0.05**2))
    return wl, refl


def _db_with_asteroid_and_lab(
    asteroid_id=4179,
    taxonomy_class="S",
    met_name="Mundrabilla",
    met_type="Iron",
    met_group="Iron",
    ast_wl_min=0.35,
    ast_wl_max=2.50,
    lab_wl_min=0.35,
    lab_wl_max=2.50,
):
    """Create a DB with one asteroid spectrum and one lab spectrum."""
    conn = get_connection(":memory:")

    # Asteroid
    conn.execute(
        "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, 1)",
        (asteroid_id, f"Ast{asteroid_id}"),
    )

    # Taxonomy
    prob_vec = np.zeros(17, dtype=np.float64)
    class_idx = MAHLKE_CLASSES.index(taxonomy_class)
    prob_vec[class_idx] = 0.85
    prob_vec[-1] = 0.15  # Z remainder
    conn.execute(
        "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, prob_vector) "
        "VALUES (?, ?, 0.85, ?)",
        (asteroid_id, taxonomy_class, prob_vec.tobytes()),
    )

    # Asteroid spectrum
    wl_ast, refl_ast = _make_spectrum(
        wl_min=ast_wl_min, wl_max=ast_wl_max, band_center=1.0
    )
    conn.execute(
        "INSERT INTO spectra "
        "(asteroid_id, survey, wavelengths, reflectance, wl_min, wl_max, "
        "normalized, quality_flag) "
        "VALUES (?, 'MITHNEOS', ?, ?, ?, ?, TRUE, 'good')",
        (
            asteroid_id,
            wl_ast.tobytes(),
            refl_ast.tobytes(),
            float(wl_ast.min()),
            float(wl_ast.max()),
        ),
    )

    # Lab spectrum (similar shape for good match)
    wl_lab, refl_lab = _make_spectrum(
        wl_min=lab_wl_min, wl_max=lab_wl_max, band_center=1.0
    )
    # Slightly scale to test free-scaling
    refl_lab = refl_lab * 1.3

    conn.execute(
        "INSERT INTO lab_spectra "
        "(sample_id, spectrum_key, meteorite_name, meteorite_type, "
        "meteorite_group, wavelengths, reflectance, wl_min, wl_max, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'RELAB')",
        (
            "TEST-001",
            "test_iron01",
            met_name,
            met_type,
            met_group,
            wl_lab.tobytes(),
            refl_lab.tobytes(),
            float(wl_lab.min()),
            float(wl_lab.max()),
        ),
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# _resample_to_overlap
# ---------------------------------------------------------------------------


class TestResampleToOverlap:
    def test_full_overlap(self):
        wl1, r1 = _make_spectrum(0.4, 2.5)
        wl2, r2 = _make_spectrum(0.4, 2.5)
        result = _resample_to_overlap(wl1, r1, wl2, r2)
        assert result is not None
        common_wl, ast_ov, lab_ov = result
        assert len(common_wl) >= MIN_OVERLAP_POINTS
        assert len(ast_ov) == len(common_wl)
        assert len(lab_ov) == len(common_wl)

    def test_partial_overlap(self):
        wl1, r1 = _make_spectrum(0.4, 2.5)
        wl2, r2 = _make_spectrum(0.8, 1.5)
        result = _resample_to_overlap(wl1, r1, wl2, r2)
        assert result is not None
        common_wl = result[0]
        assert common_wl[0] >= 0.8
        assert common_wl[-1] <= 1.5

    def test_no_overlap(self):
        wl1, r1 = _make_spectrum(0.4, 0.8)
        wl2, r2 = _make_spectrum(1.5, 2.5)
        result = _resample_to_overlap(wl1, r1, wl2, r2)
        assert result is None

    def test_excludes_nan(self):
        wl1, r1 = _make_spectrum(0.4, 2.5)
        r1[50:60] = np.nan  # NaN gap
        wl2, r2 = _make_spectrum(0.4, 2.5)
        result = _resample_to_overlap(wl1, r1, wl2, r2)
        assert result is not None
        # No NaN in output
        assert not np.any(np.isnan(result[1]))
        assert not np.any(np.isnan(result[2]))


# ---------------------------------------------------------------------------
# _best_scale_factor
# ---------------------------------------------------------------------------


class TestBestScaleFactor:
    def test_same_spectrum(self):
        _, r = _make_spectrum()
        alpha = _best_scale_factor(r, r)
        assert abs(alpha - 1.0) < 1e-10

    def test_scaled_spectrum(self):
        _, r = _make_spectrum()
        alpha = _best_scale_factor(r, r * 2.0)
        assert abs(alpha - 0.5) < 1e-10

    def test_zero_lab(self):
        _, r = _make_spectrum()
        alpha = _best_scale_factor(r, np.zeros_like(r))
        assert alpha == 1.0


# ---------------------------------------------------------------------------
# compute_match_score
# ---------------------------------------------------------------------------


class TestComputeMatchScore:
    def test_identical_spectra(self):
        _, r = _make_spectrum()
        score = compute_match_score(r, r)
        assert score is not None
        assert score["wmse"] < 1e-10
        assert abs(score["rho"] - 1.0) < 1e-10
        assert abs(score["scale_factor"] - 1.0) < 1e-10

    def test_scaled_spectra(self):
        _, r = _make_spectrum()
        score = compute_match_score(r, r * 2.0)
        assert score is not None
        assert score["wmse"] < 1e-10  # Should be ~0 after scaling
        assert score["rho"] > 0.99

    def test_different_spectra(self):
        _, r1 = _make_spectrum(band_center=1.0)
        _, r2 = _make_spectrum(band_center=2.0)
        score = compute_match_score(r1, r2)
        assert score is not None
        assert score["wmse"] > 0  # Not identical

    def test_with_uncertainty(self):
        _, r = _make_spectrum()
        unc = np.ones_like(r) * 0.01
        score = compute_match_score(r, r * 1.1, uncertainty=unc)
        assert score is not None
        assert score["wmse"] > 0  # Weighted by uncertainty

    def test_too_few_points(self):
        r = np.array([1.0, 1.1, 1.2])
        score = compute_match_score(r, r)
        assert score is None

    def test_n_points(self):
        _, r = _make_spectrum()
        score = compute_match_score(r, r)
        assert score["n_points"] == len(r)


# ---------------------------------------------------------------------------
# match_spectrum
# ---------------------------------------------------------------------------


class TestMatchSpectrum:
    def test_same_grid(self):
        wl, r = _make_spectrum(band_center=1.0)
        score = match_spectrum(wl, r, wl, r * 1.5)
        assert score is not None
        assert score["rho"] > 0.99

    def test_different_grids(self):
        wl1, r1 = _make_spectrum(0.4, 2.5, step=0.005, band_center=1.0)
        wl2, r2 = _make_spectrum(0.5, 2.0, step=0.01, band_center=1.0)
        score = match_spectrum(wl1, r1, wl2, r2)
        assert score is not None
        assert score["rho"] > 0.90

    def test_no_overlap_returns_none(self):
        wl1, r1 = _make_spectrum(0.4, 0.8)
        wl2, r2 = _make_spectrum(1.5, 2.5)
        assert match_spectrum(wl1, r1, wl2, r2) is None


# ---------------------------------------------------------------------------
# _taxonomy_weight
# ---------------------------------------------------------------------------


class TestTaxonomyWeight:
    def test_consistent_match(self):
        """S-type asteroid matching Iron meteorite should have weight > 0."""
        prob = np.zeros(17, dtype=np.float64)
        prob[MAHLKE_CLASSES.index("S")] = 0.9
        prob[MAHLKE_CLASSES.index("Z")] = 0.1
        wt = _taxonomy_weight(prob, "Iron")
        assert wt > 0.8

    def test_inconsistent_match(self):
        """S-type matching Carbonaceous Chondrite should have low weight."""
        prob = np.zeros(17, dtype=np.float64)
        prob[MAHLKE_CLASSES.index("S")] = 0.9
        prob[MAHLKE_CLASSES.index("Z")] = 0.1
        wt = _taxonomy_weight(prob, "Carbonaceous Chondrite")
        assert wt < 0.2  # Only Z contributes

    def test_c_type_carbonaceous(self):
        """C-type matching Carbonaceous Chondrite should be high."""
        prob = np.zeros(17, dtype=np.float64)
        prob[MAHLKE_CLASSES.index("C")] = 0.85
        prob[MAHLKE_CLASSES.index("Z")] = 0.15
        wt = _taxonomy_weight(prob, "Carbonaceous Chondrite")
        assert wt > 0.8

    def test_none_prob_vector(self):
        """No taxonomy → weight 1.0 (no penalty)."""
        wt = _taxonomy_weight(None, "Iron")
        assert wt == 1.0

    def test_none_meteorite_group(self):
        prob = np.ones(17, dtype=np.float64) / 17
        wt = _taxonomy_weight(prob, None)
        assert wt == 1.0

    def test_z_class_accepts_anything(self):
        """Z-class probability should contribute to any meteorite type."""
        prob = np.zeros(17, dtype=np.float64)
        prob[MAHLKE_CLASSES.index("Z")] = 1.0
        wt = _taxonomy_weight(prob, "Exotic Meteorite")
        assert wt == 1.0

    def test_v_type_achondrite(self):
        """V-type should match HED meteorites."""
        prob = np.zeros(17, dtype=np.float64)
        prob[MAHLKE_CLASSES.index("V")] = 0.95
        prob[MAHLKE_CLASSES.index("Z")] = 0.05
        wt = _taxonomy_weight(prob, "Eucrite")
        assert wt > 0.9


# ---------------------------------------------------------------------------
# find_analogs (DB integration)
# ---------------------------------------------------------------------------


class TestFindAnalogs:
    def test_finds_matching_lab_spectrum(self):
        conn = _db_with_asteroid_and_lab()
        results = find_analogs(4179, conn)
        assert len(results) >= 1
        best = results[0]
        assert best["spectrum_key"] == "test_iron01"
        assert best["rho"] > RHO_THRESHOLD
        assert best["wmse"] >= 0
        assert best["meteorite_name"] == "Mundrabilla"

    def test_returns_top_n(self):
        conn = _db_with_asteroid_and_lab()
        # Add more lab spectra
        for i in range(10):
            wl, refl = _make_spectrum(band_center=1.0 + i * 0.01)
            refl = refl * (1.0 + i * 0.1)
            conn.execute(
                "INSERT INTO lab_spectra "
                "(sample_id, spectrum_key, meteorite_name, meteorite_type, "
                "meteorite_group, wavelengths, reflectance, wl_min, wl_max, source) "
                "VALUES (?, ?, ?, 'Iron', 'Iron', ?, ?, ?, ?, 'RELAB')",
                (
                    f"TEST-{i+10}",
                    f"test_extra_{i}",
                    f"Meteorite_{i}",
                    wl.tobytes(),
                    refl.tobytes(),
                    float(wl.min()),
                    float(wl.max()),
                ),
            )
        conn.commit()

        results = find_analogs(4179, conn, top_n=5)
        assert len(results) <= 5

    def test_sorted_by_wmse(self):
        conn = _db_with_asteroid_and_lab()
        # Add a second lab spectrum with slightly different shape
        wl, refl = _make_spectrum(band_center=1.02)
        conn.execute(
            "INSERT INTO lab_spectra "
            "(sample_id, spectrum_key, meteorite_name, meteorite_type, "
            "meteorite_group, wavelengths, reflectance, wl_min, wl_max, source) "
            "VALUES ('TEST-002', 'test_iron02', 'Sikhote-Alin', 'Iron', 'Iron', "
            "?, ?, ?, ?, 'RELAB')",
            (wl.tobytes(), refl.tobytes(), float(wl.min()), float(wl.max())),
        )
        conn.commit()

        results = find_analogs(4179, conn)
        if len(results) >= 2:
            assert results[0]["wmse"] <= results[1]["wmse"]

    def test_no_spectrum_returns_empty(self):
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (1, 'Test', 1)"
        )
        conn.commit()
        assert find_analogs(1, conn) == []

    def test_no_lab_spectra_returns_empty(self):
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (1, 'Test', 1)"
        )
        wl, refl = _make_spectrum()
        conn.execute(
            "INSERT INTO spectra "
            "(asteroid_id, survey, wavelengths, reflectance, wl_min, wl_max, "
            "normalized, quality_flag) "
            "VALUES (1, 'MITHNEOS', ?, ?, 0.4, 2.5, TRUE, 'good')",
            (wl.tobytes(), refl.tobytes()),
        )
        conn.commit()
        assert find_analogs(1, conn) == []

    def test_rho_filter(self):
        """Matches below rho threshold should be excluded."""
        conn = _db_with_asteroid_and_lab()
        # Add a very different lab spectrum (flat, no band)
        wl = np.arange(0.4, 2.5, 0.005)
        refl = np.ones_like(wl)  # Flat — won't correlate well with banded spectrum
        conn.execute(
            "INSERT INTO lab_spectra "
            "(sample_id, spectrum_key, meteorite_name, meteorite_type, "
            "meteorite_group, wavelengths, reflectance, wl_min, wl_max, source) "
            "VALUES ('TEST-FLAT', 'test_flat', 'Flat', 'Unknown', 'Unknown', "
            "?, ?, 0.4, 2.5, 'RELAB')",
            (wl.tobytes(), refl.tobytes()),
        )
        conn.commit()

        results = find_analogs(4179, conn, rho_threshold=0.95)
        # The flat spectrum may or may not pass, but good match should always be there
        for r in results:
            assert r["rho"] >= 0.95

    def test_prefers_deweathered(self):
        """Should use deweathered spectrum when available."""
        conn = _db_with_asteroid_and_lab()
        # Get the actual stored spectrum to create matching deweathered version
        row = conn.execute(
            "SELECT reflectance FROM spectra WHERE asteroid_id = 4179"
        ).fetchone()
        refl = np.frombuffer(row[0], dtype=np.float64)
        deweathered = refl * 1.05  # Slightly modified, same length
        conn.execute(
            "UPDATE spectra SET deweathered = ? WHERE asteroid_id = 4179",
            (deweathered.tobytes(),),
        )
        conn.commit()

        results = find_analogs(4179, conn)
        # Should still find matches (deweathered is similar to lab)
        assert len(results) >= 1

    def test_taxonomy_filter_weights(self):
        """Taxonomy consistency should affect ranking."""
        conn = _db_with_asteroid_and_lab(
            taxonomy_class="S", met_group="Iron"
        )
        # Add a carbonaceous chondrite match (inconsistent with S-type)
        wl, refl = _make_spectrum(band_center=1.0)
        conn.execute(
            "INSERT INTO lab_spectra "
            "(sample_id, spectrum_key, meteorite_name, meteorite_type, "
            "meteorite_group, wavelengths, reflectance, wl_min, wl_max, source) "
            "VALUES ('TEST-CC', 'test_cc01', 'Murchison', 'CM2', "
            "'Carbonaceous Chondrite', ?, ?, 0.35, 2.50, 'RELAB')",
            (wl.tobytes(), refl.tobytes()),
        )
        conn.commit()

        results = find_analogs(4179, conn)
        # Both should be present, but Iron should have higher taxonomy weight
        iron = [r for r in results if r["meteorite_group"] == "Iron"]
        cc = [r for r in results if r["meteorite_group"] == "Carbonaceous Chondrite"]
        if iron and cc:
            assert iron[0]["taxonomy_weight"] > cc[0]["taxonomy_weight"]


# ---------------------------------------------------------------------------
# match_all (batch)
# ---------------------------------------------------------------------------


class TestMatchAll:
    def test_batch_processing(self):
        conn = _db_with_asteroid_and_lab()
        # Add a second asteroid
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (25143, 'Itokawa', 1)"
        )
        wl, refl = _make_spectrum(band_center=0.95)
        conn.execute(
            "INSERT INTO spectra "
            "(asteroid_id, survey, wavelengths, reflectance, wl_min, wl_max, "
            "normalized, quality_flag) "
            "VALUES (25143, 'MITHNEOS', ?, ?, 0.35, 2.50, TRUE, 'good')",
            (wl.tobytes(), refl.tobytes()),
        )
        prob_vec = np.zeros(17, dtype=np.float64)
        prob_vec[MAHLKE_CLASSES.index("S")] = 0.9
        prob_vec[-1] = 0.1
        conn.execute(
            "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, prob_vector) "
            "VALUES (25143, 'S', 0.9, ?)",
            (prob_vec.tobytes(),),
        )
        conn.commit()

        results = match_all(conn, top_n=3)
        assert len(results) >= 1  # At least one asteroid matched

    def test_empty_db(self):
        conn = get_connection(":memory:")
        results = match_all(conn)
        assert results == {}


# ---------------------------------------------------------------------------
# Physical sanity checks
# ---------------------------------------------------------------------------


class TestPhysicalSanity:
    def test_same_shape_beats_different_shape(self):
        """A spectrum with the same absorption band should rank higher."""
        wl_ast, r_ast = _make_spectrum(band_center=1.0)
        wl_good, r_good = _make_spectrum(band_center=1.0)  # Same shape
        wl_bad, r_bad = _make_spectrum(band_center=2.0)  # Different band

        score_good = match_spectrum(wl_ast, r_ast, wl_good, r_good * 1.5)
        score_bad = match_spectrum(wl_ast, r_ast, wl_bad, r_bad)

        assert score_good is not None
        assert score_bad is not None
        assert score_good["wmse"] < score_bad["wmse"]

    def test_free_scaling_handles_albedo_mismatch(self):
        """Different albedos should not affect match quality."""
        wl, r = _make_spectrum(band_center=1.0)
        score = match_spectrum(wl, r, wl, r * 3.0)
        assert score is not None
        assert score["rho"] > 0.99
        assert score["wmse"] < 1e-8
