"""Config checksum gate for scoring YAML drift detection.

Prevents agents from silently loosening distribution bounds to pass tests
by validating SHA-256 checksums on all scoring config files at load time.

Any modification to a scoring YAML without updating the approved checksum
in CONFIG_CHECKSUMS will immediately break the build.

Config changes require a dedicated bead and updated checksum.
"""

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

# Root of scoring config directory
_SCORING_DIR = Path(__file__).resolve().parent.parent.parent / "scoring"

# Approved SHA-256 checksums for scoring config files.
# Update these ONLY in a dedicated bead after intentional config changes.
CONFIG_CHECKSUMS: dict[str, str] = {
    "grade_density_priors.yaml": (
        "fb815064ef8ff7470e21288dd335e5d3eeb46eb76e69e5d0a61373170c0a9c5a"
    ),
    "recoverability_accessibility.yaml": (
        "a59048819f70a8025a4d3d0fd48adde644510e8caa1f36c8d9c79e47f28fa979"
    ),
    "confidence_config.yaml": (
        "42473d30c22e84a70982924f91a568831a3f9c8e2397cee1cddaee52ba3b75c5"
    ),
}


class ConfigDriftError(Exception):
    """Raised when a scoring config file's SHA-256 doesn't match the approved hash."""


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config_checksum(config_path: str | Path) -> None:
    """Validate that a scoring config file matches its approved checksum.

    Parameters
    ----------
    config_path : str or Path
        Path to a YAML config file.

    Raises
    ------
    ConfigDriftError
        If the file's SHA-256 hash does not match the approved checksum.
    KeyError
        If the filename is not in CONFIG_CHECKSUMS.
    """
    path = Path(config_path)
    filename = path.name

    if filename not in CONFIG_CHECKSUMS:
        raise KeyError(
            f"No approved checksum for {filename!r}. "
            f"Known configs: {list(CONFIG_CHECKSUMS.keys())}"
        )

    actual = _compute_sha256(path)
    expected = CONFIG_CHECKSUMS[filename]

    if actual != expected:
        raise ConfigDriftError(
            f"Config drift detected in {filename}!\n"
            f"  Expected SHA-256: {expected}\n"
            f"  Actual SHA-256:   {actual}\n"
            f"Config changes require a dedicated bead and updated checksum "
            f"in prospector/scoring/config_schema.py."
        )


def validate_all_configs(scoring_dir: str | Path | None = None) -> None:
    """Validate checksums of all known scoring config files.

    Parameters
    ----------
    scoring_dir : str or Path, optional
        Directory containing scoring YAML files.  Defaults to ``scoring/``.

    Raises
    ------
    ConfigDriftError
        If any file's checksum doesn't match.
    FileNotFoundError
        If a config file is missing.
    """
    if scoring_dir is None:
        scoring_dir = _SCORING_DIR
    scoring_dir = Path(scoring_dir)

    for filename in CONFIG_CHECKSUMS:
        path = scoring_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required scoring config missing: {path}")
        validate_config_checksum(path)


# ---------------------------------------------------------------------------
# Pydantic schemas for YAML structure validation
# ---------------------------------------------------------------------------


class DistributionWithSource(BaseModel):
    """A distribution entry that requires a source citation field."""

    mean: float
    std: float = Field(ge=0)
    min: float
    max: float
    source: str = Field(min_length=1)
    meteorite_analog: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def bounds_consistent(self) -> "DistributionWithSource":
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) > max ({self.max})")
        if not (self.min <= self.mean <= self.max):
            raise ValueError(f"mean ({self.mean}) outside [{self.min}, {self.max}]")
        return self


class DensityPriors(BaseModel):
    """Schema for the density_priors section."""

    unit: str
    classes: dict[str, DistributionWithSource]


class GradeSection(BaseModel):
    """Schema for a single material grade section (pgm, water, etc.)."""

    unit: str
    description: str | None = None
    classes: dict[str, DistributionWithSource] | None = None
    subgroups: dict[str, DistributionWithSource] | None = None

    @model_validator(mode="after")
    def has_classes_or_subgroups(self) -> "GradeSection":
        if self.classes is None and self.subgroups is None:
            raise ValueError("Grade section must have 'classes' or 'subgroups'")
        return self


class GradeDensityConfig(BaseModel):
    """Pydantic schema for grade_density_priors.yaml.

    Enforces that every density and grade entry has a ``source`` citation.
    """

    metadata: dict[str, Any]
    density_priors: DensityPriors
    grade_estimates: dict[str, GradeSection]
    references: dict[str, Any]
