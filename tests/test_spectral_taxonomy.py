"""Tests for prospector.spectral.taxonomy (Stage 1 classification)."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from prospector.db import get_connection, init_schema
from prospector.spectral.taxonomy import (
    MAHLKE_CLASSES,
    classify_all,
    classify_asteroid,
    classify_spectrum,
    determine_coverage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn():
    """In-memory DB with schema initialized."""
    return get_connection(":memory:")


def _insert_asteroid(conn, asteroid_id=4179, name="Toutatis"):
    conn.execute(
        "INSERT OR IGNORE INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, 1)",
        (asteroid_id, name),
    )


def _insert_normalized_spectrum(
    conn,
    asteroid_id=4179,
    survey="MITHNEOS",
    wl_min=0.8,
    wl_max=2.5,
    n_channels=200,
    quality="good",
    uncertainty=None,
):
    """Insert a normalized spectrum and return its spectrum_id."""
    wavelengths = np.linspace(wl_min, wl_max, n_channels)
    reflectance = 0.8 + 0.3 * (wavelengths - wl_min)  # simple slope

    wl_blob = wavelengths.astype(np.float64).tobytes()
    refl_blob = reflectance.astype(np.float64).tobytes()
    unc_blob = uncertainty.astype(np.float64).tobytes() if uncertainty is not None else None

    cur = conn.execute(
        "INSERT INTO spectra "
        "(asteroid_id, survey, wavelengths, reflectance, uncertainty, "
        " wl_min, wl_max, normalized, quality_flag) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, TRUE, ?)",
        (asteroid_id, survey, wl_blob, refl_blob, unc_blob, wl_min, wl_max, quality),
    )
    conn.commit()
    return cur.lastrowid


def _insert_albedo(conn, asteroid_id=4179, albedo=0.15):
    conn.execute(
        "INSERT OR REPLACE INTO physical_properties (asteroid_id, albedo_pv, source) "
        "VALUES (?, ?, 'NEOWISE')",
        (asteroid_id, albedo),
    )
    conn.commit()


def _make_mock_classy():
    """Build a mock classy module that simulates Mahlke 2022 classification."""
    mock_classy = MagicMock()

    def make_spectrum(**kwargs):
        spec = MagicMock()
        spec.is_classifiable.return_value = True

        # After classify(), set Mahlke results
        def do_classify(taxonomy="mahlke"):
            spec.class_mahlke = "S"
            spec.prob = 0.72
            # Set per-class probabilities
            for c in MAHLKE_CLASSES:
                setattr(spec, f"class_{c}", 0.01)
            spec.class_S = 0.72
            spec.class_Q = 0.15
            spec.class_V = 0.05

        spec.classify.side_effect = do_classify
        return spec

    mock_classy.Spectrum.side_effect = make_spectrum
    return mock_classy


# ---------------------------------------------------------------------------
# determine_coverage
# ---------------------------------------------------------------------------


class TestDetermineCoverage:
    def test_vnir(self):
        assert determine_coverage(0.45, 2.5) == "vnir"

    def test_vis_only(self):
        assert determine_coverage(0.374, 0.8) == "vis_only"

    def test_nir_only(self):
        assert determine_coverage(0.8, 2.5) == "nir_only"

    def test_boundary_vis(self):
        # wl_min exactly at VIS_UPPER — not visible (0.8 < 0.8 is False)
        assert determine_coverage(0.8, 0.95) == "nir_only"

    def test_wide_nir(self):
        assert determine_coverage(0.85, 2.5) == "nir_only"

    def test_full_range(self):
        assert determine_coverage(0.35, 2.55) == "vnir"


# ---------------------------------------------------------------------------
# classify_spectrum (with mocked classy)
# ---------------------------------------------------------------------------


class TestClassifySpectrum:
    def test_basic_classification(self):
        mock_classy = _make_mock_classy()
        with patch.dict("sys.modules", {"classy": mock_classy}):
            wave = np.linspace(0.8, 2.5, 200)
            refl = np.ones_like(wave)

            result = classify_spectrum(wave, refl)

        assert result is not None
        assert result["class"] == "S"
        assert result["prob"] == pytest.approx(0.72)
        assert len(result["probabilities"]) == 17
        assert result["probabilities"]["S"] == pytest.approx(0.72)
        assert isinstance(result["prob_vector"], np.ndarray)
        assert len(result["prob_vector"]) == 17

    def test_with_albedo(self):
        mock_classy = _make_mock_classy()
        with patch.dict("sys.modules", {"classy": mock_classy}):
            wave = np.linspace(0.8, 2.5, 200)
            refl = np.ones_like(wave)

            result = classify_spectrum(wave, refl, pV=0.15)

        assert result is not None
        # Verify pV was passed to Spectrum constructor
        call_kwargs = mock_classy.Spectrum.call_args
        assert call_kwargs.kwargs.get("pV") == 0.15

    def test_with_uncertainty(self):
        mock_classy = _make_mock_classy()
        with patch.dict("sys.modules", {"classy": mock_classy}):
            wave = np.linspace(0.8, 2.5, 200)
            refl = np.ones_like(wave)
            err = np.full_like(wave, 0.01)

            result = classify_spectrum(wave, refl, refl_err=err)

        assert result is not None

    def test_not_classifiable_returns_none(self):
        mock_classy = MagicMock()

        def make_unclassifiable(**kwargs):
            spec = MagicMock()
            spec.is_classifiable.return_value = False
            return spec

        mock_classy.Spectrum.side_effect = make_unclassifiable
        with patch.dict("sys.modules", {"classy": mock_classy}):
            wave = np.linspace(0.8, 2.5, 5)
            refl = np.ones_like(wave)

            result = classify_spectrum(wave, refl)

        assert result is None

    def test_too_few_valid_channels(self):
        mock_classy = _make_mock_classy()
        with patch.dict("sys.modules", {"classy": mock_classy}):
            wave = np.array([1.0, 1.5, 2.0])
            refl = np.array([np.nan, np.nan, 1.0])  # only 1 valid

            result = classify_spectrum(wave, refl)

        assert result is None

    def test_nan_channels_filtered(self):
        mock_classy = _make_mock_classy()
        with patch.dict("sys.modules", {"classy": mock_classy}):
            wave = np.linspace(0.8, 2.5, 100)
            refl = np.ones(100)
            # Insert NaN gap (like telluric masking)
            refl[40:60] = np.nan

            result = classify_spectrum(wave, refl)

        assert result is not None
        # classy.Spectrum should receive only valid channels
        call_args = mock_classy.Spectrum.call_args
        assert len(call_args.kwargs["wave"]) == 80  # 100 - 20 NaN

    def test_prob_vector_order_matches_classes(self):
        mock_classy = _make_mock_classy()
        with patch.dict("sys.modules", {"classy": mock_classy}):
            wave = np.linspace(0.8, 2.5, 200)
            refl = np.ones_like(wave)

            result = classify_spectrum(wave, refl)

        # S is at index 13 in MAHLKE_CLASSES
        s_idx = MAHLKE_CLASSES.index("S")
        assert result["prob_vector"][s_idx] == pytest.approx(0.72)

    def test_nan_albedo_ignored(self):
        mock_classy = _make_mock_classy()
        with patch.dict("sys.modules", {"classy": mock_classy}):
            wave = np.linspace(0.8, 2.5, 200)
            refl = np.ones_like(wave)

            result = classify_spectrum(wave, refl, pV=float("nan"))

        assert result is not None
        # NaN albedo should not be passed
        call_kwargs = mock_classy.Spectrum.call_args.kwargs
        assert "pV" not in call_kwargs

    def test_classy_not_installed_raises(self):
        with patch.dict("sys.modules", {"classy": None}):
            wave = np.linspace(0.8, 2.5, 200)
            refl = np.ones_like(wave)

            with pytest.raises(ImportError, match="classy is required"):
                classify_spectrum(wave, refl)


# ---------------------------------------------------------------------------
# classify_asteroid (DB integration with mocked classy)
# ---------------------------------------------------------------------------


class TestClassifyAsteroid:
    def test_basic_db_classification(self, conn):
        mock_classy = _make_mock_classy()
        _insert_asteroid(conn)
        _insert_normalized_spectrum(conn)

        with patch.dict("sys.modules", {"classy": mock_classy}):
            result = classify_asteroid(4179, conn)

        assert result is True

        row = conn.execute(
            "SELECT primary_class, primary_prob, prob_vector, classifier, input_coverage "
            "FROM taxonomy WHERE asteroid_id = 4179"
        ).fetchone()

        assert row is not None
        assert row[0] == "S"
        assert row[1] == pytest.approx(0.72)
        assert row[3] == "classy_mahlke2022"
        assert row[4] == "nir_only"

        # Verify prob_vector BLOB round-trip
        prob_vec = np.frombuffer(row[2], dtype=np.float64)
        assert len(prob_vec) == 17
        s_idx = MAHLKE_CLASSES.index("S")
        assert prob_vec[s_idx] == pytest.approx(0.72)

    def test_with_albedo_from_db(self, conn):
        mock_classy = _make_mock_classy()
        _insert_asteroid(conn)
        _insert_normalized_spectrum(conn)
        _insert_albedo(conn, 4179, 0.20)

        with patch.dict("sys.modules", {"classy": mock_classy}):
            classify_asteroid(4179, conn)

        # Verify albedo was passed
        call_kwargs = mock_classy.Spectrum.call_args.kwargs
        assert call_kwargs.get("pV") == pytest.approx(0.20)

    def test_no_spectrum_returns_false(self, conn):
        _insert_asteroid(conn)

        mock_classy = _make_mock_classy()
        with patch.dict("sys.modules", {"classy": mock_classy}):
            result = classify_asteroid(4179, conn)

        assert result is False

    def test_unnormalized_spectrum_skipped(self, conn):
        _insert_asteroid(conn)
        # Insert raw (unnormalized) spectrum
        wl = np.linspace(0.8, 2.5, 200)
        refl = np.ones_like(wl)
        conn.execute(
            "INSERT INTO spectra "
            "(asteroid_id, survey, wavelengths, reflectance, wl_min, wl_max, normalized) "
            "VALUES (?, ?, ?, ?, ?, ?, FALSE)",
            (4179, "MITHNEOS", wl.tobytes(), refl.tobytes(), 0.8, 2.5),
        )
        conn.commit()

        mock_classy = _make_mock_classy()
        with patch.dict("sys.modules", {"classy": mock_classy}):
            result = classify_asteroid(4179, conn)

        assert result is False

    def test_prefers_good_quality(self, conn):
        mock_classy = _make_mock_classy()
        _insert_asteroid(conn)
        # Insert low_snr spectrum (wide)
        _insert_normalized_spectrum(conn, wl_min=0.45, wl_max=2.5, quality="low_snr")
        # Insert good quality spectrum (narrower)
        _insert_normalized_spectrum(conn, wl_min=0.8, wl_max=2.5, quality="good")

        with patch.dict("sys.modules", {"classy": mock_classy}):
            classify_asteroid(4179, conn)

        # Should have used the good quality spectrum (0.8-2.5)
        row = conn.execute(
            "SELECT input_coverage FROM taxonomy WHERE asteroid_id = 4179"
        ).fetchone()
        assert row[0] == "nir_only"

    def test_vnir_coverage_detected(self, conn):
        mock_classy = _make_mock_classy()
        _insert_asteroid(conn)
        _insert_normalized_spectrum(conn, wl_min=0.45, wl_max=2.5, quality="good")

        with patch.dict("sys.modules", {"classy": mock_classy}):
            classify_asteroid(4179, conn)

        row = conn.execute(
            "SELECT input_coverage FROM taxonomy WHERE asteroid_id = 4179"
        ).fetchone()
        assert row[0] == "vnir"

    def test_vis_only_coverage(self, conn):
        mock_classy = _make_mock_classy()
        _insert_asteroid(conn)
        _insert_normalized_spectrum(
            conn, survey="Gaia", wl_min=0.374, wl_max=0.8, quality="good"
        )

        with patch.dict("sys.modules", {"classy": mock_classy}):
            classify_asteroid(4179, conn)

        row = conn.execute(
            "SELECT input_coverage FROM taxonomy WHERE asteroid_id = 4179"
        ).fetchone()
        assert row[0] == "vis_only"

    def test_idempotent_rerun(self, conn):
        mock_classy = _make_mock_classy()
        _insert_asteroid(conn)
        _insert_normalized_spectrum(conn)

        with patch.dict("sys.modules", {"classy": mock_classy}):
            classify_asteroid(4179, conn)
            # Re-classify should overwrite (INSERT OR REPLACE)
            classify_asteroid(4179, conn)

        count = conn.execute("SELECT COUNT(*) FROM taxonomy WHERE asteroid_id = 4179").fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# classify_all (batch with mocked classy)
# ---------------------------------------------------------------------------


class TestClassifyAll:
    def test_classifies_unclassified(self, conn):
        mock_classy = _make_mock_classy()
        _insert_asteroid(conn, 4179, "Toutatis")
        _insert_asteroid(conn, 433, "Eros")
        _insert_normalized_spectrum(conn, asteroid_id=4179)
        _insert_normalized_spectrum(conn, asteroid_id=433)

        with patch.dict("sys.modules", {"classy": mock_classy}):
            count = classify_all(conn)

        assert count == 2
        rows = conn.execute("SELECT COUNT(*) FROM taxonomy").fetchone()[0]
        assert rows == 2

    def test_skips_already_classified(self, conn):
        mock_classy = _make_mock_classy()
        _insert_asteroid(conn)
        _insert_normalized_spectrum(conn)

        # Pre-populate taxonomy
        prob_vec = np.zeros(17, dtype=np.float64)
        conn.execute(
            "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, prob_vector, "
            "classifier, input_coverage) VALUES (?, ?, ?, ?, ?, ?)",
            (4179, "S", 0.72, prob_vec.tobytes(), "classy_mahlke2022", "nir_only"),
        )
        conn.commit()

        with patch.dict("sys.modules", {"classy": mock_classy}):
            count = classify_all(conn)

        assert count == 0

    def test_returns_zero_when_no_spectra(self, conn):
        mock_classy = _make_mock_classy()
        with patch.dict("sys.modules", {"classy": mock_classy}):
            count = classify_all(conn)
        assert count == 0

    def test_handles_classification_failure_gracefully(self, conn):
        """If one asteroid fails, others should still be classified."""
        mock_classy = MagicMock()
        call_count = [0]

        def make_spectrum(**kwargs):
            call_count[0] += 1
            spec = MagicMock()
            if call_count[0] == 1:
                # First call fails
                spec.is_classifiable.side_effect = RuntimeError("model error")
            else:
                spec.is_classifiable.return_value = True

                def do_classify(taxonomy="mahlke"):
                    spec.class_mahlke = "C"
                    spec.prob = 0.65
                    for c in MAHLKE_CLASSES:
                        setattr(spec, f"class_{c}", 0.01)
                    spec.class_C = 0.65

                spec.classify.side_effect = do_classify
            return spec

        mock_classy.Spectrum.side_effect = make_spectrum

        _insert_asteroid(conn, 4179, "Toutatis")
        _insert_asteroid(conn, 433, "Eros")
        _insert_normalized_spectrum(conn, asteroid_id=4179)
        _insert_normalized_spectrum(conn, asteroid_id=433)

        with patch.dict("sys.modules", {"classy": mock_classy}):
            count = classify_all(conn)

        # One should have succeeded despite the other failing
        assert count == 1


# ---------------------------------------------------------------------------
# prob_vector BLOB storage
# ---------------------------------------------------------------------------


class TestProbVectorStorage:
    def test_round_trip(self, conn):
        """Probability vector survives numpy BLOB round-trip."""
        mock_classy = _make_mock_classy()
        _insert_asteroid(conn)
        _insert_normalized_spectrum(conn)

        with patch.dict("sys.modules", {"classy": mock_classy}):
            classify_asteroid(4179, conn)

        row = conn.execute(
            "SELECT prob_vector FROM taxonomy WHERE asteroid_id = 4179"
        ).fetchone()

        prob_vec = np.frombuffer(row[0], dtype=np.float64)
        assert len(prob_vec) == len(MAHLKE_CLASSES)
        assert prob_vec.sum() > 0

        # Reconstruct dict
        probs = {c: float(p) for c, p in zip(MAHLKE_CLASSES, prob_vec)}
        assert probs["S"] == pytest.approx(0.72)
        assert probs["Q"] == pytest.approx(0.15)

    def test_vector_values_match_dict(self, conn):
        """prob_vector array indices correspond to MAHLKE_CLASSES order."""
        mock_classy = _make_mock_classy()
        _insert_asteroid(conn)
        _insert_normalized_spectrum(conn)

        with patch.dict("sys.modules", {"classy": mock_classy}):
            result = classify_spectrum(
                np.linspace(0.8, 2.5, 200),
                np.ones(200),
            )

        for i, c in enumerate(MAHLKE_CLASSES):
            assert result["prob_vector"][i] == pytest.approx(result["probabilities"][c])
