"""Tests for pipeline stage Protocol and pre/postcondition contracts.

Verifies that:
1. All concrete stage classes satisfy the PipelineStage Protocol.
2. Preconditions correctly detect missing prerequisites.
3. Postconditions correctly detect stage outputs.
4. Stages run in sequence with postconditions validated after each.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from prospector.db import get_connection
from prospector.pipeline import (
    IngestStage,
    OutputStage,
    PipelineStage,
    ScoringStage,
    SpectralStage,
    StageResult,
    _table_has_rows,
)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Every concrete stage must satisfy the PipelineStage Protocol."""

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (IngestStage, {"data_dir": Path(".")}),
            (SpectralStage, {}),
            (ScoringStage, {}),
            (OutputStage, {}),
        ],
    )
    def test_isinstance_check(self, cls, kwargs):
        stage = cls(**kwargs)
        assert isinstance(stage, PipelineStage)

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (IngestStage, {"data_dir": Path(".")}),
            (SpectralStage, {}),
            (ScoringStage, {}),
            (OutputStage, {}),
        ],
    )
    def test_has_name(self, cls, kwargs):
        stage = cls(**kwargs)
        assert isinstance(stage.name, str)
        assert len(stage.name) > 0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


class TestTableHasRows:
    """Unit tests for the _table_has_rows helper."""

    def test_empty_table(self):
        conn = get_connection(":memory:")
        assert not _table_has_rows(conn, "asteroids")
        conn.close()

    def test_nonexistent_table(self):
        conn = sqlite3.connect(":memory:")
        assert not _table_has_rows(conn, "no_such_table")
        conn.close()

    def test_populated_table(self):
        conn = get_connection(":memory:")
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name) VALUES (433, 'Eros')"
        )
        assert _table_has_rows(conn, "asteroids")
        conn.close()


# ---------------------------------------------------------------------------
# Precondition tests
# ---------------------------------------------------------------------------


class TestPreconditions:
    """Verify each stage's preconditions against empty and populated DBs."""

    @pytest.fixture()
    def empty_db(self):
        conn = get_connection(":memory:")
        yield conn
        conn.close()

    def test_ingest_always_ready(self, empty_db):
        stage = IngestStage(data_dir=Path("."))
        assert stage.preconditions_met(empty_db) is True

    def test_spectral_needs_spectra(self, empty_db):
        stage = SpectralStage()
        assert stage.preconditions_met(empty_db) is False

    def test_spectral_satisfied(self, empty_db):
        conn = empty_db
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name) VALUES (433, 'Eros')"
        )
        conn.execute(
            "INSERT INTO spectra (asteroid_id, survey, wavelengths, reflectance)"
            " VALUES (433, 'TEST', ?, ?)",
            (np.array([0.5, 1.0]).tobytes(), np.array([1.0, 0.9]).tobytes()),
        )
        stage = SpectralStage()
        assert stage.preconditions_met(conn) is True

    def test_scoring_needs_asteroids_and_orbits(self, empty_db):
        stage = ScoringStage()
        assert stage.preconditions_met(empty_db) is False

    def test_scoring_satisfied(self, empty_db):
        conn = empty_db
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name) VALUES (433, 'Eros')"
        )
        conn.execute(
            "INSERT INTO orbits (asteroid_id, a, e, i) VALUES (433, 1.458, 0.223, 10.83)"
        )
        stage = ScoringStage()
        assert stage.preconditions_met(conn) is True

    def test_output_needs_scores(self, empty_db):
        stage = OutputStage()
        assert stage.preconditions_met(empty_db) is False


# ---------------------------------------------------------------------------
# Postcondition tests
# ---------------------------------------------------------------------------


class TestPostconditions:
    """Verify postconditions detect outputs correctly."""

    @pytest.fixture()
    def empty_db(self):
        conn = get_connection(":memory:")
        yield conn
        conn.close()

    def test_ingest_postcondition_empty(self, empty_db):
        stage = IngestStage(data_dir=Path("."))
        assert stage.postconditions_met(empty_db) is False

    def test_ingest_postcondition_satisfied(self, empty_db):
        conn = empty_db
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name) VALUES (433, 'Eros')"
        )
        stage = IngestStage(data_dir=Path("."))
        assert stage.postconditions_met(conn) is True

    def test_spectral_postcondition_empty(self, empty_db):
        stage = SpectralStage()
        assert stage.postconditions_met(empty_db) is False

    def test_spectral_postcondition_satisfied(self, empty_db):
        conn = empty_db
        conn.execute(
            "INSERT INTO asteroids (asteroid_id, name) VALUES (433, 'Eros')"
        )
        conn.execute(
            "INSERT INTO taxonomy (asteroid_id, primary_class, primary_prob, classifier, prob_vector)"
            " VALUES (433, 'S', 0.9, 'test', ?)",
            (np.zeros(17).tobytes(),),
        )
        stage = SpectralStage()
        assert stage.postconditions_met(conn) is True

    def test_scoring_postcondition_empty(self, empty_db):
        stage = ScoringStage()
        assert stage.postconditions_met(empty_db) is False

    def test_output_postcondition_no_file(self, tmp_path):
        stage = OutputStage(output_path=str(tmp_path / "does_not_exist.csv"))
        conn = get_connection(":memory:")
        assert stage.postconditions_met(conn) is False
        conn.close()

    def test_output_postcondition_file_exists(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        csv_path.write_text("header\nrow1\n")
        stage = OutputStage(output_path=str(csv_path))
        conn = get_connection(":memory:")
        assert stage.postconditions_met(conn) is True
        conn.close()


# ---------------------------------------------------------------------------
# StageResult
# ---------------------------------------------------------------------------


class TestStageResult:
    """StageResult dataclass."""

    def test_basic_construction(self):
        r = StageResult(stage="ingest", counts={"sbdb": 100})
        assert r.stage == "ingest"
        assert r.counts == {"sbdb": 100}

    def test_default_counts(self):
        r = StageResult(stage="spectral")
        assert r.counts == {}


# ---------------------------------------------------------------------------
# Integration: sequential stage run (mocked data)
# ---------------------------------------------------------------------------


class TestSequentialRun:
    """Run stages in order on a minimal DB and verify postconditions."""

    def test_ingest_then_check(self, tmp_path):
        """IngestStage.run on data dir with an SBDB CSV populates asteroids."""
        sbdb_dir = tmp_path / "sbdb"
        sbdb_dir.mkdir()
        # Minimal SBDB CSV (matches ingest_sbdb_csv expectations)
        csv_content = (
            "full_name,spkid,neo,pha,e,a,i,om,w,ma,epoch,H,moid,diameter,diameter_sigma\n"
            "433 Eros,2000433,Y,N,0.2229,1.458,10.83,304.3,178.9,320.5,2460000.5,11.16,0.149,16.84,0.1\n"
        )
        (sbdb_dir / "test.csv").write_text(csv_content)

        conn = get_connection(":memory:")
        stage = IngestStage(data_dir=tmp_path)

        assert stage.preconditions_met(conn) is True
        result = stage.run(conn)
        assert isinstance(result, StageResult)
        assert result.stage == "ingest"
        assert stage.postconditions_met(conn) is True
        conn.close()
