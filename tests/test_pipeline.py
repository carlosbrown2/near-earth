"""Tests for prospector.pipeline — CLI entry point and orchestration."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from prospector.pipeline import (
    ALL_STAGES,
    build_parser,
    discover_data_files,
    main,
    run_pipeline,
)


class TestDiscoverDataFiles:
    """Test data file discovery logic."""

    def test_empty_dir(self, tmp_path):
        result = discover_data_files(tmp_path)
        assert result == {}

    def test_finds_sbdb_csv(self, tmp_path):
        sbdb_dir = tmp_path / "sbdb"
        sbdb_dir.mkdir()
        (sbdb_dir / "sbdb_query_results.csv").write_text("a,b\n1,2\n")
        result = discover_data_files(tmp_path)
        assert "sbdb" in result
        assert len(result["sbdb"]) == 1
        assert result["sbdb"][0].name == "sbdb_query_results.csv"

    def test_finds_multiple_sources(self, tmp_path):
        (tmp_path / "sbdb").mkdir()
        (tmp_path / "sbdb" / "data.csv").write_text("")
        (tmp_path / "neowise").mkdir()
        (tmp_path / "neowise" / "data.tab").write_text("")
        result = discover_data_files(tmp_path)
        assert "sbdb" in result
        assert "neowise" in result

    def test_ignores_non_matching_extensions(self, tmp_path):
        sbdb_dir = tmp_path / "sbdb"
        sbdb_dir.mkdir()
        (sbdb_dir / "readme.txt").write_text("")
        result = discover_data_files(tmp_path)
        assert "sbdb" not in result

    def test_spectra_subdirs(self, tmp_path):
        mithneos = tmp_path / "spectra" / "mithneos"
        mithneos.mkdir(parents=True)
        (mithneos / "a000433.txt").write_text("")
        result = discover_data_files(tmp_path)
        assert "mithneos" in result


class TestBuildParser:
    """Test CLI argument parsing."""

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.data_dir == "data"
        assert args.db_path is None
        assert args.mode == "earth_return"
        assert args.n_samples == 1000
        assert args.output is None
        assert args.stages is None
        assert args.verbose is False
        assert args.list_data is False

    def test_all_options(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data-dir", "/tmp/data",
            "--db-path", "/tmp/test.db",
            "--mode", "in_space",
            "--n-samples", "500",
            "--output", "out.csv",
            "--stages", "ingest", "scoring",
            "-v",
        ])
        assert args.data_dir == "/tmp/data"
        assert args.db_path == "/tmp/test.db"
        assert args.mode == "in_space"
        assert args.n_samples == 500
        assert args.output == "out.csv"
        assert args.stages == ["ingest", "scoring"]
        assert args.verbose is True

    def test_invalid_stage_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--stages", "bogus"])

    def test_invalid_mode_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "lunar"])


class TestRunPipeline:
    """Test pipeline orchestration."""

    def test_invalid_stage_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown stage"):
            run_pipeline(data_dir=tmp_path, stages=["bogus"])

    def test_empty_ingest(self, tmp_path):
        """Pipeline runs ingest with no data files — returns empty results."""
        result = run_pipeline(data_dir=tmp_path, stages=["ingest"])
        assert result["ingest"] == {}

    def test_all_stages_constant(self):
        assert ALL_STAGES == ("ingest", "spectral", "scoring", "output")

    def test_db_path_defaults_to_data_dir(self, tmp_path):
        """When db_path is None, database goes under data_dir."""
        with patch("prospector.pipeline.get_connection") as mock_conn:
            mock_conn.return_value = sqlite3.connect(":memory:")
            run_pipeline(data_dir=tmp_path, stages=["ingest"])
            mock_conn.assert_called_once_with(str(tmp_path / "prospector.db"))


class TestMainCLI:
    """Test the main() CLI entry point."""

    def test_list_data_empty(self, tmp_path, capsys):
        rc = main(["--data-dir", str(tmp_path), "--list-data"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "No data files found" in captured.out

    def test_list_data_shows_files(self, tmp_path, capsys):
        sbdb_dir = tmp_path / "sbdb"
        sbdb_dir.mkdir()
        (sbdb_dir / "data.csv").write_text("")
        rc = main(["--data-dir", str(tmp_path), "--list-data"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "sbdb:" in captured.out

    def test_runs_pipeline(self, tmp_path):
        """main() with ingest-only stage succeeds on empty data dir."""
        rc = main(["--data-dir", str(tmp_path), "--stages", "ingest"])
        assert rc == 0
