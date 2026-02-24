"""Tests for RELAB PDS4 meteorite spectral library ingestion."""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from prospector.db import get_connection, init_schema
from prospector.ingest.relab import (
    _parse_tab_file,
    _parse_xml_label,
    ingest_relab_dir,
    ingest_relab_file,
    is_meteorite,
)


# ---------------------------------------------------------------------------
# Helpers: create temp RELAB files
# ---------------------------------------------------------------------------

_SPECLIB_NS = "http://pds.nasa.gov/pds4/speclib/v1"
_PDS_NS = "http://pds.nasa.gov/pds4/pds/v1"


def _write_tab(tmp_path: Path, filename: str, content: str) -> Path:
    """Write a .tab file and return its path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content))
    return p


def _write_xml_label(
    tmp_path: Path,
    filename: str,
    *,
    specimen_id: str = "MB-TXH-050",
    specimen_name: str = "ALHA81002",
    specimen_type: str = "Other Meteorite",
    rock_subtypes: list[str] | None = None,
    rock_type: str | None = None,
    description: str | None = None,
    grain_min: float | None = None,
    grain_max: float | None = None,
) -> Path:
    """Write a minimal PDS4 XML label for testing."""
    if rock_subtypes is None:
        rock_subtypes = ["Carbonaceous Chondrite", "CM2"]

    subtypes_xml = "\n".join(
        f'        <speclib:rock_subtype>{s}</speclib:rock_subtype>'
        for s in rock_subtypes
    )

    rock_type_xml = (
        f'        <speclib:rock_type>{rock_type}</speclib:rock_type>'
        if rock_type else ""
    )

    desc_xml = (
        f'        <speclib:specimen_description>{description}</speclib:specimen_description>'
        if description else ""
    )

    grain_xml = ""
    if grain_min is not None:
        grain_xml += f'        <speclib:specimen_min_size unit="micrometer">{grain_min}</speclib:specimen_min_size>\n'
    if grain_max is not None:
        grain_xml += f'        <speclib:specimen_max_size unit="micrometer">{grain_max}</speclib:specimen_max_size>\n'

    xml = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="{_PDS_NS}"
                       xmlns:speclib="{_SPECLIB_NS}">
    <Observation_Area>
        <Discipline_Area>
            <speclib:Specimen_Parameters>
                <speclib:specimen_id>{specimen_id}</speclib:specimen_id>
                <speclib:specimen_name>{specimen_name}</speclib:specimen_name>
                <speclib:specimen_type>{specimen_type}</speclib:specimen_type>
{desc_xml}
{subtypes_xml}
{rock_type_xml}
{grain_xml}
            </speclib:Specimen_Parameters>
        </Discipline_Area>
    </Observation_Area>
</Product_Observational>
"""
    p = tmp_path / filename
    p.write_text(xml)
    return p


def _make_meteorite_pair(
    tmp_path: Path,
    stem: str = "c0mb50",
    **xml_kwargs,
) -> tuple[Path, Path]:
    """Create a paired .tab + .xml RELAB file set."""
    tab = _write_tab(tmp_path, f"{stem}.tab", """\
        500.0  0.0320
        600.0  0.0345
        700.0  0.0380
        800.0  0.0410
        900.0  0.0425
    """)
    xml = _write_xml_label(tmp_path, f"{stem}.xml", **xml_kwargs)
    return tab, xml


# ---------------------------------------------------------------------------
# _parse_tab_file
# ---------------------------------------------------------------------------


class TestParseTabFile:
    def test_basic_two_column(self, tmp_path):
        p = _write_tab(tmp_path, "test.tab", """\
            500.0  0.032
            600.0  0.035
            700.0  0.038
        """)
        wl, refl = _parse_tab_file(p)
        assert len(wl) == 3
        assert len(refl) == 3
        # Wavelengths converted from nm to μm
        np.testing.assert_allclose(wl, [0.500, 0.600, 0.700])
        np.testing.assert_allclose(refl, [0.032, 0.035, 0.038])

    def test_wavelength_nm_to_um_conversion(self, tmp_path):
        p = _write_tab(tmp_path, "test.tab", """\
            350.0  0.010
            2500.0 0.100
        """)
        wl, refl = _parse_tab_file(p)
        np.testing.assert_allclose(wl, [0.350, 2.500])

    def test_skips_blank_lines_and_comments(self, tmp_path):
        p = _write_tab(tmp_path, "test.tab", """\
            # Header comment
            500.0  0.032

            600.0  0.035
            # Another comment
            700.0  0.038
        """)
        wl, refl = _parse_tab_file(p)
        assert len(wl) == 3

    def test_empty_file_raises(self, tmp_path):
        p = _write_tab(tmp_path, "empty.tab", "")
        with pytest.raises(ValueError, match="No valid data rows"):
            _parse_tab_file(p)

    def test_returns_float64_arrays(self, tmp_path):
        p = _write_tab(tmp_path, "test.tab", "500.0 0.032\n600.0 0.035\n")
        wl, refl = _parse_tab_file(p)
        assert wl.dtype == np.float64
        assert refl.dtype == np.float64


# ---------------------------------------------------------------------------
# _parse_xml_label
# ---------------------------------------------------------------------------


class TestParseXmlLabel:
    def test_meteorite_with_all_fields(self, tmp_path):
        xml_path = _write_xml_label(
            tmp_path,
            "test.xml",
            specimen_id="MB-TXH-050",
            specimen_name="ALHA81002",
            specimen_type="Other Meteorite",
            rock_subtypes=["Carbonaceous Chondrite", "CM2"],
            description="Carbonaceous Chondrite, CM2",
            grain_min=0.0,
            grain_max=100.0,
        )
        meta = _parse_xml_label(xml_path)
        assert meta is not None
        assert meta["sample_id"] == "MB-TXH-050"
        assert meta["specimen_name"] == "ALHA81002"
        assert meta["specimen_type"] == "Other Meteorite"
        assert meta["meteorite_type"] == "CM2"
        assert meta["meteorite_group"] == "Carbonaceous Chondrite"
        assert meta["sample_desc"] == "Carbonaceous Chondrite, CM2"
        assert meta["grain_size_min"] == 0.0
        assert meta["grain_size_max"] == 100.0

    def test_single_rock_subtype(self, tmp_path):
        xml_path = _write_xml_label(
            tmp_path,
            "test.xml",
            rock_subtypes=["Iron"],
        )
        meta = _parse_xml_label(xml_path)
        assert meta["meteorite_type"] == "Iron"
        assert meta["meteorite_group"] == "Iron"

    def test_no_rock_subtype_falls_back_to_rock_type(self, tmp_path):
        xml_path = _write_xml_label(
            tmp_path,
            "test.xml",
            rock_subtypes=[],
            rock_type="Igneous",
        )
        meta = _parse_xml_label(xml_path)
        assert meta["meteorite_type"] == "Igneous"
        assert meta["meteorite_group"] == "Igneous"

    def test_no_grain_size(self, tmp_path):
        xml_path = _write_xml_label(tmp_path, "test.xml")
        meta = _parse_xml_label(xml_path)
        assert meta["grain_size_min"] is None
        assert meta["grain_size_max"] is None

    def test_invalid_xml_returns_none(self, tmp_path):
        p = tmp_path / "bad.xml"
        p.write_text("not valid xml <><><>")
        assert _parse_xml_label(p) is None


# ---------------------------------------------------------------------------
# is_meteorite
# ---------------------------------------------------------------------------


class TestIsMeteorite:
    def test_other_meteorite(self):
        assert is_meteorite({"specimen_type": "Other Meteorite"})

    def test_ordinary_chondrite(self):
        assert is_meteorite({"specimen_type": "Ordinary Chondrite"})

    def test_meteorite_keyword(self):
        assert is_meteorite({"specimen_type": "Meteorite"})

    def test_terrestrial_sample(self):
        assert not is_meteorite({"specimen_type": "Terrestrial Sample"})

    def test_mineral(self):
        assert not is_meteorite({"specimen_type": "Mineral"})

    def test_none_specimen_type(self):
        assert not is_meteorite({"specimen_type": None})

    def test_empty_dict(self):
        assert not is_meteorite({})


# ---------------------------------------------------------------------------
# ingest_relab_file
# ---------------------------------------------------------------------------


class TestIngestRelabFile:
    def test_ingest_meteorite(self, tmp_path):
        tab, xml = _make_meteorite_pair(
            tmp_path,
            stem="c0mb50",
            specimen_id="MB-TXH-050",
            specimen_name="ALHA81002",
            specimen_type="Other Meteorite",
            rock_subtypes=["Carbonaceous Chondrite", "CM2"],
            grain_min=0.0,
            grain_max=100.0,
        )
        conn = get_connection(":memory:")
        assert ingest_relab_file(tab, conn)
        conn.commit()

        row = conn.execute(
            "SELECT * FROM lab_spectra WHERE spectrum_key = ?", ("c0mb50",)
        ).fetchone()
        assert row is not None

        # Check columns by name
        cols = [d[0] for d in conn.execute("SELECT * FROM lab_spectra").description]
        data = dict(zip(cols, row))
        assert data["sample_id"] == "MB-TXH-050"
        assert data["meteorite_name"] == "ALHA81002"
        assert data["meteorite_type"] == "CM2"
        assert data["meteorite_group"] == "Carbonaceous Chondrite"
        assert data["grain_size_min"] == 0.0
        assert data["grain_size_max"] == 100.0
        assert data["source"] == "RELAB"

    def test_blob_roundtrip(self, tmp_path):
        tab, _ = _make_meteorite_pair(tmp_path)
        conn = get_connection(":memory:")
        ingest_relab_file(tab, conn)
        conn.commit()

        row = conn.execute(
            "SELECT wavelengths, reflectance, wl_min, wl_max FROM lab_spectra"
        ).fetchone()
        wl = np.frombuffer(row[0], dtype=np.float64)
        refl = np.frombuffer(row[1], dtype=np.float64)
        assert len(wl) == 5
        assert len(refl) == 5
        # Wavelengths should be in μm (converted from nm)
        np.testing.assert_allclose(wl[0], 0.500)
        np.testing.assert_allclose(wl[-1], 0.900)
        assert row[2] == pytest.approx(0.500)
        assert row[3] == pytest.approx(0.900)

    def test_skip_non_meteorite(self, tmp_path):
        tab = _write_tab(tmp_path, "c01ag0.tab", "500.0 0.032\n600.0 0.035\n")
        _write_xml_label(
            tmp_path,
            "c01ag0.xml",
            specimen_type="Terrestrial Sample",
            specimen_name="Basalt slab",
        )
        conn = get_connection(":memory:")
        assert not ingest_relab_file(tab, conn, meteorites_only=True)

    def test_include_non_meteorite_when_flag_false(self, tmp_path):
        tab = _write_tab(tmp_path, "c01ag0.tab", "500.0 0.032\n600.0 0.035\n")
        _write_xml_label(
            tmp_path,
            "c01ag0.xml",
            specimen_type="Terrestrial Sample",
            specimen_name="Basalt slab",
        )
        conn = get_connection(":memory:")
        assert ingest_relab_file(tab, conn, meteorites_only=False)

    def test_missing_xml_with_meteorites_only_skips(self, tmp_path):
        """Without XML label, cannot determine if meteorite → skip."""
        tab = _write_tab(tmp_path, "noxml.tab", "500.0 0.032\n600.0 0.035\n")
        conn = get_connection(":memory:")
        assert not ingest_relab_file(tab, conn, meteorites_only=True)

    def test_missing_xml_with_flag_false_ingests(self, tmp_path):
        """Without XML, ingest anyway when meteorites_only=False."""
        tab = _write_tab(tmp_path, "noxml.tab", "500.0 0.032\n600.0 0.035\n")
        conn = get_connection(":memory:")
        assert ingest_relab_file(tab, conn, meteorites_only=False)
        row = conn.execute("SELECT sample_id FROM lab_spectra").fetchone()
        assert row[0] == "noxml"  # falls back to filename stem

    def test_idempotent_reingest(self, tmp_path):
        """INSERT OR REPLACE should overwrite on same spectrum_key."""
        tab, _ = _make_meteorite_pair(tmp_path)
        conn = get_connection(":memory:")
        ingest_relab_file(tab, conn)
        ingest_relab_file(tab, conn)
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM lab_spectra").fetchone()[0]
        assert count == 1

    def test_missing_tab_file(self, tmp_path):
        conn = get_connection(":memory:")
        assert not ingest_relab_file(tmp_path / "nonexistent.tab", conn)


# ---------------------------------------------------------------------------
# ingest_relab_dir
# ---------------------------------------------------------------------------


class TestIngestRelabDir:
    def test_batch_ingest(self, tmp_path):
        # Create 3 meteorite files
        for i, (name, mtype) in enumerate([
            ("Murchison", "CM2"),
            ("Allende", "CV3"),
            ("Mundrabilla", "Iron"),
        ]):
            stem = f"c0mb{i:02d}"
            _write_tab(
                tmp_path, f"{stem}.tab",
                f"{500 + i * 10}.0  0.03{i}\n{600 + i * 10}.0  0.04{i}\n",
            )
            _write_xml_label(
                tmp_path,
                f"{stem}.xml",
                specimen_id=f"MB-TEST-{i:03d}",
                specimen_name=name,
                rock_subtypes=[mtype],
            )

        conn = get_connection(":memory:")
        count = ingest_relab_dir(tmp_path, conn)
        assert count == 3

        rows = conn.execute("SELECT COUNT(*) FROM lab_spectra").fetchone()[0]
        assert rows == 3

    def test_mixed_meteorite_and_terrestrial(self, tmp_path):
        # 1 meteorite + 1 terrestrial
        _make_meteorite_pair(
            tmp_path,
            stem="met01",
            specimen_name="Murchison",
        )
        _write_tab(tmp_path, "terr01.tab", "500.0 0.5\n600.0 0.6\n")
        _write_xml_label(
            tmp_path,
            "terr01.xml",
            specimen_type="Terrestrial Sample",
            specimen_name="Basalt",
        )

        conn = get_connection(":memory:")
        count = ingest_relab_dir(tmp_path, conn, meteorites_only=True)
        assert count == 1

    def test_nonexistent_dir_raises(self):
        conn = get_connection(":memory:")
        with pytest.raises(FileNotFoundError):
            ingest_relab_dir("/nonexistent/path", conn)

    def test_empty_dir(self, tmp_path):
        conn = get_connection(":memory:")
        count = ingest_relab_dir(tmp_path, conn)
        assert count == 0

    def test_meteorites_only_false(self, tmp_path):
        # 1 meteorite + 1 terrestrial — both should be ingested
        _make_meteorite_pair(tmp_path, stem="met01")
        _write_tab(tmp_path, "terr01.tab", "500.0 0.5\n600.0 0.6\n")
        _write_xml_label(
            tmp_path,
            "terr01.xml",
            specimen_type="Terrestrial Sample",
            specimen_name="Basalt",
        )

        conn = get_connection(":memory:")
        count = ingest_relab_dir(tmp_path, conn, meteorites_only=False)
        assert count == 2


# ---------------------------------------------------------------------------
# Schema integration
# ---------------------------------------------------------------------------


class TestSchemaIntegration:
    def test_lab_spectra_table_exists(self):
        conn = get_connection(":memory:")
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='lab_spectra'"
        ).fetchall()
        assert len(tables) == 1

    def test_indexes_exist(self):
        conn = get_connection(":memory:")
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_lab_spectra%'"
        ).fetchall()
        index_names = {row[0] for row in indexes}
        assert "idx_lab_spectra_type" in index_names
        assert "idx_lab_spectra_group" in index_names
        assert "idx_lab_spectra_sample" in index_names

    def test_no_fk_to_asteroids(self):
        """lab_spectra has no FK to asteroids — meteorite samples are independent."""
        conn = get_connection(":memory:")
        # Should be able to insert without any asteroids in the DB
        conn.execute(
            "INSERT INTO lab_spectra (sample_id, spectrum_key, source) "
            "VALUES ('TEST-001', 'test_key', 'RELAB')"
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM lab_spectra").fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# Iron meteorite specific — PGM analog matching use case
# ---------------------------------------------------------------------------


class TestIronMeteorite:
    def test_iron_meteorite_ingest(self, tmp_path):
        """Iron meteorites are key for PGM analog matching."""
        tab, _ = _make_meteorite_pair(
            tmp_path,
            stem="c0mb34",
            specimen_id="MB-TXH-034",
            specimen_name="Mundrabilla diffuse surface",
            specimen_type="Other Meteorite",
            rock_subtypes=["Iron"],
            rock_type="Igneous",
        )
        conn = get_connection(":memory:")
        ingest_relab_file(tab, conn)
        conn.commit()

        row = conn.execute(
            "SELECT meteorite_type, meteorite_group FROM lab_spectra WHERE spectrum_key = 'c0mb34'"
        ).fetchone()
        assert row[0] == "Iron"
        assert row[1] == "Iron"

    def test_query_by_meteorite_type(self, tmp_path):
        """Verify we can query lab_spectra by meteorite_type for curve matching."""
        for stem, mtype, group in [
            ("iron01", "IIIAB", "Iron"),
            ("iron02", "IVA", "Iron"),
            ("cm01", "CM2", "Carbonaceous Chondrite"),
        ]:
            _write_tab(tmp_path, f"{stem}.tab", "500.0 0.03\n600.0 0.04\n")
            _write_xml_label(
                tmp_path,
                f"{stem}.xml",
                specimen_id=f"TEST-{stem}",
                specimen_name=stem,
                rock_subtypes=[group, mtype] if group != mtype else [mtype],
            )

        conn = get_connection(":memory:")
        ingest_relab_dir(tmp_path, conn)

        # Query iron meteorites for PGM matching
        iron_rows = conn.execute(
            "SELECT spectrum_key FROM lab_spectra WHERE meteorite_group = 'Iron'"
        ).fetchall()
        assert len(iron_rows) == 2

        # Query carbonaceous chondrites for water matching
        cc_rows = conn.execute(
            "SELECT spectrum_key FROM lab_spectra WHERE meteorite_group = 'Carbonaceous Chondrite'"
        ).fetchall()
        assert len(cc_rows) == 1
