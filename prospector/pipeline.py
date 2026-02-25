"""Pipeline orchestrator for the Asteroid Mining Prospector.

Runs the full or partial pipeline: ingest → spectral → scoring → output.

Usage:
    python -m prospector --data-dir data/ --mode earth_return --output results.csv
    python -m prospector --stages ingest spectral
    python -m prospector --stages scoring output --db-path data/prospector.db
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from prospector.db import get_connection
from prospector.schemas import ScoringMode

logger = logging.getLogger(__name__)

ALL_STAGES = ("ingest", "spectral", "scoring", "output")


# ---------------------------------------------------------------------------
# Pipeline Stage Protocol
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""

    stage: str
    counts: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class PipelineStage(Protocol):
    """Runtime-checkable protocol for pipeline stages.

    Each stage must declare preconditions (what the DB must contain before
    running), a run method that performs the work, and postconditions (what
    the DB must contain after running).
    """

    name: str

    def preconditions_met(self, db: sqlite3.Connection) -> bool:
        """Check whether this stage's prerequisites are satisfied."""
        ...

    def run(self, db: sqlite3.Connection) -> StageResult:
        """Execute the stage, returning counts of records processed."""
        ...

    def postconditions_met(self, db: sqlite3.Connection) -> bool:
        """Verify the stage produced the expected artefacts."""
        ...


def _table_has_rows(conn: sqlite3.Connection, table: str) -> bool:
    """Return True if *table* exists and has at least one row."""
    try:
        row = conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()  # noqa: S608
        return row is not None
    except sqlite3.OperationalError:
        return False


class IngestStage:
    """Ingest raw data files into the database."""

    name: str = "ingest"

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def preconditions_met(self, db: sqlite3.Connection) -> bool:
        # Ingest is the first stage — no DB prerequisites.
        return True

    def run(self, db: sqlite3.Connection) -> StageResult:
        counts = _run_ingest(db, self._data_dir)
        return StageResult(stage=self.name, counts=counts)

    def postconditions_met(self, db: sqlite3.Connection) -> bool:
        return _table_has_rows(db, "asteroids")


class SpectralStage:
    """Run spectral analysis: preprocess → classify → band analysis."""

    name: str = "spectral"

    def preconditions_met(self, db: sqlite3.Connection) -> bool:
        return _table_has_rows(db, "spectra")

    def run(self, db: sqlite3.Connection) -> StageResult:
        counts = _run_spectral(db)
        return StageResult(stage=self.name, counts=counts)

    def postconditions_met(self, db: sqlite3.Connection) -> bool:
        return _table_has_rows(db, "taxonomy")


class ScoringStage:
    """Run Monte Carlo scoring."""

    name: str = "scoring"

    def __init__(
        self,
        mode: ScoringMode = "earth_return",
        n_samples: int = 1000,
    ) -> None:
        self._mode = mode
        self._n_samples = n_samples

    def preconditions_met(self, db: sqlite3.Connection) -> bool:
        return _table_has_rows(db, "asteroids") and _table_has_rows(db, "orbits")

    def run(self, db: sqlite3.Connection) -> StageResult:
        counts = _run_scoring(db, mode=self._mode, n_samples=self._n_samples)
        return StageResult(stage=self.name, counts=counts)

    def postconditions_met(self, db: sqlite3.Connection) -> bool:
        return _table_has_rows(db, "scores")


class OutputStage:
    """Generate ranked output CSV."""

    name: str = "output"

    def __init__(
        self,
        mode: ScoringMode = "earth_return",
        output_path: str | None = None,
    ) -> None:
        self._mode = mode
        self._output_path = output_path

    def preconditions_met(self, db: sqlite3.Connection) -> bool:
        return _table_has_rows(db, "scores")

    def run(self, db: sqlite3.Connection) -> StageResult:
        counts = _run_output(db, mode=self._mode, output_path=self._output_path)
        return StageResult(stage=self.name, counts=counts)

    def postconditions_met(self, db: sqlite3.Connection) -> bool:
        path = self._output_path or f"prospector_results_{self._mode}.csv"
        return Path(path).exists()

# Data source directories and file patterns for auto-discovery
_INGEST_SOURCES = {
    "sbdb": {"subdir": "sbdb", "patterns": ["*.csv"]},
    "neowise": {"subdir": "neowise", "patterns": ["*.tab", "*.csv"]},
    "mithneos": {"subdir": "spectra/mithneos", "patterns": ["*.txt"]},
    "smass": {"subdir": "spectra/smass", "patterns": ["*.txt"]},
    "gaia": {"subdir": "spectra/gaia", "patterns": ["*.csv", "*.fits"]},
    "lcdb": {"subdir": "lcdb", "patterns": ["*.csv", "*.tab"]},
    "ssodnet": {"subdir": "ssodnet", "patterns": ["*.json"]},
    "jwst_water": {"subdir": "jwst", "patterns": ["*.csv", "*.json"]},
    "relab": {"subdir": "spectra/relab", "patterns": ["*.txt"]},
}


def discover_data_files(data_dir: Path) -> dict[str, list[Path]]:
    """Find available data files organized by ingest source.

    Parameters
    ----------
    data_dir : Path
        Root data directory to scan.

    Returns
    -------
    dict mapping source name to list of matching file paths.
    """
    found: dict[str, list[Path]] = {}
    for source, info in _INGEST_SOURCES.items():
        source_dir = data_dir / str(info["subdir"])
        if not source_dir.is_dir():
            continue
        files: list[Path] = []
        patterns = list(info["patterns"])
        for pattern in patterns:
            files.extend(sorted(source_dir.glob(pattern)))
        if files:
            found[source] = files
    return found


def _run_ingest(conn: sqlite3.Connection, data_dir: Path) -> dict[str, int]:
    """Run all available ingest stages.

    Returns dict mapping source name to rows ingested.
    """
    available = discover_data_files(data_dir)
    if not available:
        logger.warning("No data files found in %s", data_dir)
        return {}

    results: dict[str, int] = {}

    # SBDB — bulk asteroid catalog
    if "sbdb" in available:
        from prospector.ingest.sbdb import ingest_sbdb_csv

        for path in available["sbdb"]:
            logger.info("Ingesting SBDB: %s", path.name)
            n = ingest_sbdb_csv(str(path), conn)
            results["sbdb"] = results.get("sbdb", 0) + n

    # NEOWISE — diameter and albedo
    if "neowise" in available:
        from prospector.ingest.neowise import ingest_neowise

        for path in available["neowise"]:
            logger.info("Ingesting NEOWISE: %s", path.name)
            n = ingest_neowise(str(path), conn)
            results["neowise"] = results.get("neowise", 0) + n

    # MITHNEOS — NIR spectra
    if "mithneos" in available:
        from prospector.ingest.mithneos import ingest_mithneos_dir

        dirs = {p.parent for p in available["mithneos"]}
        for d in dirs:
            logger.info("Ingesting MITHNEOS dir: %s", d)
            n = ingest_mithneos_dir(str(d), conn)
            results["mithneos"] = results.get("mithneos", 0) + n

    # SMASS — visible spectra
    if "smass" in available:
        from prospector.ingest.smass import ingest_smass_dir

        dirs = {p.parent for p in available["smass"]}
        for d in dirs:
            logger.info("Ingesting SMASS dir: %s", d)
            n = ingest_smass_dir(str(d), conn)
            results["smass"] = results.get("smass", 0) + n

    # Gaia — spectral survey
    if "gaia" in available:
        from prospector.ingest.gaia import ingest_gaia_file

        for path in available["gaia"]:
            logger.info("Ingesting Gaia: %s", path.name)
            n = ingest_gaia_file(str(path), conn)
            results["gaia"] = results.get("gaia", 0) + n

    # LCDB — rotation properties
    if "lcdb" in available:
        from prospector.ingest.lcdb import ingest_lcdb

        for path in available["lcdb"]:
            logger.info("Ingesting LCDB: %s", path.name)
            n = ingest_lcdb(str(path), conn)
            results["lcdb"] = results.get("lcdb", 0) + n

    # SsODNet — supplementary properties
    if "ssodnet" in available:
        from prospector.ingest.ssodnet import ingest_ssodnet_json

        for path in available["ssodnet"]:
            logger.info("Ingesting SsODNet: %s", path.name)
            n = ingest_ssodnet_json(str(path), conn)
            results["ssodnet"] = results.get("ssodnet", 0) + n

    # JWST water detections
    if "jwst_water" in available:
        from prospector.ingest.jwst_water import ingest_jwst_water, ingest_jwst_water_json

        for path in available["jwst_water"]:
            logger.info("Ingesting JWST water: %s", path.name)
            if path.suffix == ".json":
                n = ingest_jwst_water_json(str(path), conn)
            else:
                n = ingest_jwst_water(str(path), conn)
            results["jwst_water"] = results.get("jwst_water", 0) + n

    # RELAB — meteorite reference spectra
    if "relab" in available:
        from prospector.ingest.relab import ingest_relab_dir

        dirs = {p.parent for p in available["relab"]}
        for d in dirs:
            logger.info("Ingesting RELAB dir: %s", d)
            n = ingest_relab_dir(str(d), conn)
            results["relab"] = results.get("relab", 0) + n

    return results


def _run_spectral(conn: sqlite3.Connection) -> dict[str, int]:
    """Run spectral analysis pipeline: preprocess → classify → band analysis."""
    from prospector.spectral.band_analysis import analyze_all
    from prospector.spectral.preprocessing import preprocess_all
    from prospector.spectral.taxonomy import classify_all

    results: dict[str, int] = {}

    logger.info("Preprocessing spectra...")
    results["preprocessed"] = preprocess_all(conn)

    logger.info("Classifying taxonomy...")
    results["classified"] = classify_all(conn)

    logger.info("Analyzing bands (S-complex)...")
    results["band_analyzed"] = analyze_all(conn)  # type: ignore[no-untyped-call]

    return results


def _run_scoring(
    conn: sqlite3.Connection, *, mode: ScoringMode = "earth_return", n_samples: int = 1000
) -> dict[str, int]:
    """Run Monte Carlo scoring."""
    from prospector.scoring.scorer import score_all

    logger.info("Scoring asteroids (mode=%s, n_samples=%d)...", mode, n_samples)
    n = score_all(conn, n_samples=n_samples, mode=mode)
    return {"scored": n}


def _run_output(
    conn: sqlite3.Connection, *, mode: ScoringMode = "earth_return", output_path: str | None = None
) -> dict[str, int]:
    """Generate ranked output CSV."""
    from prospector.output import generate_ranked_csv

    if output_path is None:
        output_path = f"prospector_results_{mode}.csv"
    logger.info("Writing output to %s", output_path)
    n = generate_ranked_csv(conn, output_path, mode=mode)  # type: ignore[no-untyped-call]
    return {"candidates_written": n}


def run_pipeline(
    *,
    data_dir: str | Path = "data",
    db_path: str | None = None,
    mode: ScoringMode = "earth_return",
    n_samples: int = 1000,
    output_path: str | None = None,
    stages: Sequence[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Run the full or partial prospector pipeline.

    Parameters
    ----------
    data_dir : str or Path
        Root data directory for file discovery.
    db_path : str, optional
        SQLite database path. Defaults to ``data_dir / "prospector.db"``.
    mode : {"earth_return", "in_space"}
        Scoring mode.
    n_samples : int
        Monte Carlo sample count for scoring.
    output_path : str, optional
        Path for output CSV. Defaults to ``prospector_results_{mode}.csv``.
    stages : sequence of str, optional
        Pipeline stages to run. Defaults to all stages.

    Returns
    -------
    dict mapping stage name to a dict of result counts.
    """
    data_dir = Path(data_dir)
    if db_path is None:
        db_path = str(data_dir / "prospector.db")

    if stages is None:
        stages = list(ALL_STAGES)

    for s in stages:
        if s not in ALL_STAGES:
            raise ValueError(f"Unknown stage {s!r}. Valid stages: {ALL_STAGES}")

    conn = get_connection(db_path)
    results: dict[str, dict[str, int]] = {}

    try:
        if "ingest" in stages:
            results["ingest"] = _run_ingest(conn, data_dir)

        if "spectral" in stages:
            results["spectral"] = _run_spectral(conn)

        if "scoring" in stages:
            results["scoring"] = _run_scoring(conn, mode=mode, n_samples=n_samples)

        if "output" in stages:
            results["output"] = _run_output(conn, mode=mode, output_path=output_path)
    finally:
        conn.close()

    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="prospector",
        description="Asteroid Mining Prospector — rank NEO mining candidates",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="SQLite database path (default: DATA_DIR/prospector.db)",
    )
    parser.add_argument(
        "--mode",
        choices=["earth_return", "in_space"],
        default="earth_return",
        help="Scoring mode (default: earth_return)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Monte Carlo sample count (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: prospector_results_{mode}.csv)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(ALL_STAGES),
        default=None,
        help="Pipeline stages to run (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--list-data",
        action="store_true",
        help="List discovered data files and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir)

    if args.list_data:
        found = discover_data_files(data_dir)
        if not found:
            print(f"No data files found in {data_dir}")
            return 1
        for source, files in sorted(found.items()):
            print(f"{source}:")
            for f in files:
                print(f"  {f}")
        return 0

    results = run_pipeline(
        data_dir=data_dir,
        db_path=args.db_path,
        mode=args.mode,
        n_samples=args.n_samples,
        output_path=args.output,
        stages=args.stages,
    )

    for stage, counts in results.items():
        summary = ", ".join(f"{k}={v}" for k, v in counts.items())
        logger.info("Stage %s: %s", stage, summary)

    return 0
