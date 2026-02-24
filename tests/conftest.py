"""Shared pytest fixtures for the prospector test suite."""

import pathlib

import pytest


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def data_dir(project_root):
    """Return the data directory path."""
    return project_root / "data"
