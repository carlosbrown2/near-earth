"""Tests for the mutation testing gate configuration and script."""

import os
import stat
import subprocess

import tomllib


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPROJECT = os.path.join(ROOT, "pyproject.toml")
GATE_SCRIPT = os.path.join(ROOT, "scripts", "mutmut-gate.sh")


def test_mutmut_in_dev_dependencies() -> None:
    """mutmut must be listed in [project.optional-dependencies] dev."""
    with open(PYPROJECT, "rb") as f:
        config = tomllib.load(f)
    dev_deps = config["project"]["optional-dependencies"]["dev"]
    assert any("mutmut" in dep for dep in dev_deps), (
        "mutmut not found in dev dependencies"
    )


def test_mutmut_tool_config() -> None:
    """[tool.mutmut] must be configured with correct paths and runner."""
    with open(PYPROJECT, "rb") as f:
        config = tomllib.load(f)
    mutmut_cfg = config["tool"]["mutmut"]
    assert mutmut_cfg["paths_to_mutate"] == "prospector/"
    assert "pytest" in mutmut_cfg["runner"]
    assert mutmut_cfg["tests_dir"] == "tests/"


def test_gate_script_exists_and_executable() -> None:
    """scripts/mutmut-gate.sh must exist and be executable."""
    assert os.path.isfile(GATE_SCRIPT), f"Gate script not found: {GATE_SCRIPT}"
    mode = os.stat(GATE_SCRIPT).st_mode
    assert mode & stat.S_IXUSR, "Gate script is not executable"


def test_gate_script_usage_error() -> None:
    """Gate script exits 2 with no arguments (usage error)."""
    result = subprocess.run(
        ["bash", GATE_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Usage" in result.stdout


def test_gate_script_missing_module_error() -> None:
    """Gate script exits 2 for a nonexistent module path."""
    result = subprocess.run(
        ["bash", GATE_SCRIPT, "nonexistent/fake_module.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "not found" in result.stdout
