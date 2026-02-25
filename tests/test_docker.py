"""Tests for Docker deployment artifacts.

These tests validate the Dockerfile and docker-compose.yml structure
without requiring Docker to be installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDockerfile:
    """Validate Dockerfile contents."""

    def test_exists(self):
        assert (PROJECT_ROOT / "Dockerfile").is_file()

    def test_uses_python_312(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "python:3.12" in content

    def test_installs_api_deps(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert '.[api]' in content or '".[api]"' in content

    def test_exposes_port_8000(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "EXPOSE 8000" in content

    def test_runs_uvicorn(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "uvicorn" in content
        assert "create_app" in content

    def test_sets_db_path_env(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "PROSPECTOR_DB_PATH" in content

    def test_healthcheck(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "HEALTHCHECK" in content
        assert "/health" in content


class TestDockerCompose:
    """Validate docker-compose.yml contents."""

    def test_exists(self):
        assert (PROJECT_ROOT / "docker-compose.yml").is_file()

    def test_mounts_data_volume(self):
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "/data" in content
        assert "volumes" in content

    def test_sets_env_vars(self):
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "PROSPECTOR_DB_PATH" in content
        assert "PROSPECTOR_API_KEYS" in content

    def test_port_mapping(self):
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "8000:8000" in content

    def test_healthcheck(self):
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "healthcheck" in content
        assert "/health" in content
