"""Tests validating the classy (Mahlke 2022) taxonomy package.

Research spike bead: near-earth-d52
These tests verify that the space-classy package installs, imports,
and produces correct probabilistic taxonomy classifications.
"""

import subprocess
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_classy():
    """Import classy, skipping if not installed or broken."""
    try:
        import classy
        return classy
    except ImportError:
        pytest.skip("space-classy not installed")
    except Exception as e:
        pytest.skip(f"classy import failed: {e}")


def _make_s_type_spectrum():
    """Synthetic S-type-like spectrum with Band I and Band II."""
    wave = np.linspace(0.45, 2.45, 200)
    refl = 0.8 + 0.3 * (wave - 0.45) / 2.0  # reddish slope
    refl -= 0.15 * np.exp(-((wave - 0.95) / 0.15) ** 2)  # Band I at 0.95 μm
    refl -= 0.10 * np.exp(-((wave - 1.95) / 0.25) ** 2)  # Band II at 1.95 μm
    return wave, refl


def _make_flat_spectrum():
    """Featureless flat spectrum (C-type analog)."""
    wave = np.linspace(0.45, 2.45, 200)
    rng = np.random.RandomState(42)
    refl = np.ones_like(wave) * 0.5 + rng.normal(0, 0.01, len(wave))
    return wave, refl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassyInstallation:
    """Verify classy package installs and imports correctly."""

    def test_import(self):
        classy = _import_classy()
        assert hasattr(classy, "Spectrum")
        assert hasattr(classy, "Spectra")

    def test_version(self):
        classy = _import_classy()
        assert classy.__version__  # non-empty version string

    def test_has_taxonomies(self):
        classy = _import_classy()
        assert hasattr(classy, "taxonomies")
        # All three taxonomy systems available
        for name in ("mahlke", "demeo", "tholen"):
            assert hasattr(classy.taxonomies, name)


class TestSpectrumCreation:
    """Verify Spectrum construction from raw arrays."""

    def test_from_arrays(self):
        classy = _import_classy()
        wave, refl = _make_s_type_spectrum()
        spec = classy.Spectrum(wave=wave, refl=refl)
        assert spec.wave.shape == wave.shape
        assert spec.refl.shape == refl.shape

    def test_with_error_bars(self):
        classy = _import_classy()
        wave, refl = _make_s_type_spectrum()
        refl_err = np.ones_like(wave) * 0.02
        spec = classy.Spectrum(wave=wave, refl=refl, refl_err=refl_err)
        assert spec.refl_err is not None

    def test_albedo_kwarg_is_pV(self):
        """CRITICAL: albedo must be passed as pV=, not albedo=."""
        classy = _import_classy()
        wave, refl = _make_flat_spectrum()
        spec_pv = classy.Spectrum(wave=wave, refl=refl, pV=0.15)
        spec_albedo = classy.Spectrum(wave=wave, refl=refl, albedo=0.15)
        assert hasattr(spec_pv, "pV")
        assert not hasattr(spec_albedo, "pV")  # albedo= does NOT set pV


class TestMahlkeClassification:
    """Verify Mahlke 2022 probabilistic taxonomy."""

    MAHLKE_CLASSES = [
        "A", "B", "C", "Ch", "D", "E", "K", "L",
        "M", "O", "P", "Q", "R", "S", "V", "X", "Z",
    ]

    def test_classify_returns_class_label(self):
        classy = _import_classy()
        wave, refl = _make_s_type_spectrum()
        spec = classy.Spectrum(wave=wave, refl=refl)
        spec.classify(taxonomy="mahlke")
        assert spec.class_mahlke in self.MAHLKE_CLASSES

    def test_probability_vector(self):
        """All 17 class probabilities should be available and sum to ~1."""
        classy = _import_classy()
        wave, refl = _make_s_type_spectrum()
        spec = classy.Spectrum(wave=wave, refl=refl)
        spec.classify(taxonomy="mahlke")

        probs = []
        for c in self.MAHLKE_CLASSES:
            p = getattr(spec, f"class_{c}")
            assert isinstance(p, float)
            assert 0.0 <= p <= 1.0
            probs.append(p)

        assert abs(sum(probs) - 1.0) < 0.05  # should sum to ~1

    def test_top_probability(self):
        classy = _import_classy()
        wave, refl = _make_flat_spectrum()
        spec = classy.Spectrum(wave=wave, refl=refl, pV=0.05)
        spec.classify(taxonomy="mahlke")
        assert 0.0 < spec.prob <= 1.0
        assert spec.prob == max(
            getattr(spec, f"class_{c}") for c in self.MAHLKE_CLASSES
        )

    def test_latent_scores(self):
        classy = _import_classy()
        wave, refl = _make_s_type_spectrum()
        spec = classy.Spectrum(wave=wave, refl=refl)
        spec.classify(taxonomy="mahlke")
        assert spec.scores_mahlke.shape == (4,)

    def test_c_type_with_low_albedo(self):
        """Flat spectrum + low albedo → high C-type probability."""
        classy = _import_classy()
        wave, refl = _make_flat_spectrum()
        spec = classy.Spectrum(wave=wave, refl=refl, pV=0.05)
        spec.classify(taxonomy="mahlke")
        assert spec.class_C > 0.5  # should strongly favor C-type

    def test_classifiable_check(self):
        classy = _import_classy()
        wave, refl = _make_s_type_spectrum()
        spec = classy.Spectrum(wave=wave, refl=refl)
        assert spec.is_classifiable("mahlke")

    def test_visible_only_classifiable(self):
        """Visible-only spectra (0.45–0.9 μm) should still be classifiable."""
        classy = _import_classy()
        wave = np.linspace(0.45, 0.9, 50)
        refl = 0.8 + 0.2 * (wave - 0.45) / 0.45
        spec = classy.Spectrum(wave=wave, refl=refl)
        assert spec.is_classifiable("mahlke")
        spec.classify(taxonomy="mahlke")
        assert spec.class_mahlke in self.MAHLKE_CLASSES


class TestDeMeoClassification:
    """Verify DeMeo 2009 taxonomy as cross-reference / fallback."""

    def test_demeo_classify(self):
        classy = _import_classy()
        wave, refl = _make_s_type_spectrum()
        spec = classy.Spectrum(wave=wave, refl=refl)
        spec.classify(taxonomy="demeo")
        assert isinstance(spec.class_demeo, str)
        assert len(spec.class_demeo) > 0

    def test_demeo_scores(self):
        classy = _import_classy()
        wave, refl = _make_s_type_spectrum()
        spec = classy.Spectrum(wave=wave, refl=refl)
        spec.classify(taxonomy="demeo")
        assert spec.scores_demeo.shape == (5,)

    def test_narrow_spectrum_not_demeo_classifiable(self):
        """DeMeo requires broader wavelength range than Mahlke."""
        classy = _import_classy()
        spec = classy.Spectrum(
            wave=np.array([0.5, 0.6]), refl=np.array([1.0, 1.1])
        )
        assert not spec.is_classifiable("demeo")


class TestPreprocessing:
    """Verify classy's internal preprocessing handles edge cases."""

    def test_resampling_preserves_original(self):
        """Classify should not permanently alter wave/refl arrays."""
        classy = _import_classy()
        wave, refl = _make_s_type_spectrum()
        spec = classy.Spectrum(wave=wave, refl=refl)
        wave_before = spec.wave.copy()
        refl_before = spec.refl.copy()
        spec.classify(taxonomy="mahlke")
        np.testing.assert_array_equal(spec.wave, wave_before)
        np.testing.assert_array_equal(spec.refl, refl_before)
