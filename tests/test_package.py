"""Smoke tests — verify package structure imports correctly."""

import importlib


def test_prospector_imports():
    """All subpackages should be importable."""
    for name in [
        "prospector",
        "prospector.ingest",
        "prospector.spectral",
        "prospector.scoring",
        "prospector.ephemeris",
    ]:
        mod = importlib.import_module(name)
        assert mod is not None


def test_version():
    import prospector

    assert prospector.__version__ == "0.1.0"
