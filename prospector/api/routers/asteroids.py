"""Asteroid detail endpoint (US-002).

GET /v1/asteroids/{asteroid_id} — full detail for one asteroid,
joining all available data tables.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from prospector.api.auth import get_api_key
from prospector.api.app import get_db

router = APIRouter(prefix="/v1", dependencies=[Depends(get_api_key)])

_DETAIL_QUERY = """
    SELECT
        a.asteroid_id, a.name, a.designation, a.neo, a.pha,
        o.epoch, o.e, o.a, o.i, o.om, o.w, o.ma, o.q, o.H,
        o.moid, o.diameter AS orbit_diameter, o.diameter_sigma,
        pp.albedo_pv, pp.albedo_pv_err, pp.diameter_km AS neowise_diameter,
        pp.diameter_err, pp.beaming_eta, pp.source AS pp_source,
        t.primary_class, t.primary_prob, t.classifier, t.input_coverage,
        ba.band1_center, ba.band2_center, ba.bar,
        ba.ol_opx_ratio, ba.fa_mol_pct, ba.fs_mol_pct,
        ba.gaffey_subtype, ba.calibration,
        cm.ol_pct, cm.opx_pct, cm.cpx_pct,
        cm.fa_mol_pct AS cnn_fa, cm.fs_mol_pct AS cnn_fs, cm.wo_mol_pct,
        cm.ol_unc, cm.opx_unc, cm.cpx_unc,
        cm.classical_agreement, cm.method AS cnn_method,
        pg.pgm_tier, pg.pgm_confidence, pg.m_type_prob,
        pg.featureless_nir, pg.no_silicate_bands,
        pg.iron_analog_match, pg.iron_analog_wmse,
        pg.radar_albedo, pg.beaming_eta AS pg_beaming_eta,
        pg.s_type_metal, pg.metal_fraction_pct, pg.signal_count, pg.notes AS pgm_notes,
        rp.rotation_period, rp.period_unc, rp.amplitude,
        rp.quality_code, rp.is_monolithic, rp.is_binary_suspect,
        s.composite_score, s.estimated_mass_kg, s.grade_estimate,
        s.target_material, s.unit_value, s.accessibility, s.score_mode
    FROM asteroids a
    LEFT JOIN orbits o ON a.asteroid_id = o.asteroid_id
    LEFT JOIN physical_properties pp ON a.asteroid_id = pp.asteroid_id
    LEFT JOIN taxonomy t ON a.asteroid_id = t.asteroid_id
    LEFT JOIN band_analysis ba ON a.asteroid_id = ba.asteroid_id
    LEFT JOIN cnn_mineral cm ON a.asteroid_id = cm.asteroid_id
    LEFT JOIN pgm_convergence pg ON a.asteroid_id = pg.asteroid_id
    LEFT JOIN rotation_properties rp ON a.asteroid_id = rp.asteroid_id
    LEFT JOIN scores s ON a.asteroid_id = s.asteroid_id
    WHERE a.asteroid_id = ?
"""

_COL_NAMES = [
    "asteroid_id", "name", "designation", "neo", "pha",
    "epoch", "e", "a", "i", "om", "w", "ma", "q", "H",
    "moid", "orbit_diameter", "diameter_sigma",
    "albedo_pv", "albedo_pv_err", "neowise_diameter",
    "diameter_err", "beaming_eta", "pp_source",
    "primary_class", "primary_prob", "classifier", "input_coverage",
    "band1_center", "band2_center", "bar",
    "ol_opx_ratio", "fa_mol_pct", "fs_mol_pct",
    "gaffey_subtype", "calibration",
    "cnn_ol_pct", "cnn_opx_pct", "cnn_cpx_pct",
    "cnn_fa", "cnn_fs", "cnn_wo",
    "cnn_ol_unc", "cnn_opx_unc", "cnn_cpx_unc",
    "cnn_agreement", "cnn_method",
    "pgm_tier", "pgm_confidence", "m_type_prob",
    "featureless_nir", "no_silicate_bands",
    "iron_analog_match", "iron_analog_wmse",
    "radar_albedo", "pg_beaming_eta",
    "s_type_metal", "metal_fraction_pct", "signal_count", "pgm_notes",
    "rotation_period", "period_unc", "amplitude",
    "quality_code", "is_monolithic", "is_binary_suspect",
    "composite_score", "estimated_mass_kg", "grade_estimate",
    "target_material", "unit_value", "accessibility", "score_mode",
]


def _row_to_detail(row: tuple[Any, ...]) -> dict[str, Any]:
    """Convert a raw SQL row to a structured detail dict."""
    flat = dict(zip(_COL_NAMES, row))

    return {
        "asteroid_id": flat["asteroid_id"],
        "name": flat["name"],
        "designation": flat["designation"],
        "neo": bool(flat["neo"]) if flat["neo"] is not None else None,
        "pha": bool(flat["pha"]) if flat["pha"] is not None else None,
        "orbit": {
            "epoch": flat["epoch"],
            "e": flat["e"],
            "a": flat["a"],
            "i": flat["i"],
            "om": flat["om"],
            "w": flat["w"],
            "ma": flat["ma"],
            "q": flat["q"],
            "H": flat["H"],
            "moid": flat["moid"],
            "diameter": flat["orbit_diameter"],
            "diameter_sigma": flat["diameter_sigma"],
        } if flat["a"] is not None else None,
        "physical_properties": {
            "albedo_pv": flat["albedo_pv"],
            "albedo_pv_err": flat["albedo_pv_err"],
            "diameter_km": flat["neowise_diameter"],
            "diameter_err": flat["diameter_err"],
            "beaming_eta": flat["beaming_eta"],
            "source": flat["pp_source"],
        } if flat["albedo_pv"] is not None or flat["neowise_diameter"] is not None else None,
        "taxonomy": {
            "primary_class": flat["primary_class"],
            "primary_prob": flat["primary_prob"],
            "classifier": flat["classifier"],
            "input_coverage": flat["input_coverage"],
        } if flat["primary_class"] is not None else None,
        "band_analysis": {
            "band1_center": flat["band1_center"],
            "band2_center": flat["band2_center"],
            "bar": flat["bar"],
            "ol_opx_ratio": flat["ol_opx_ratio"],
            "fa_mol_pct": flat["fa_mol_pct"],
            "fs_mol_pct": flat["fs_mol_pct"],
            "gaffey_subtype": flat["gaffey_subtype"],
            "calibration": flat["calibration"],
        } if flat["band1_center"] is not None else None,
        "cnn_mineral": {
            "ol_pct": flat["cnn_ol_pct"],
            "opx_pct": flat["cnn_opx_pct"],
            "cpx_pct": flat["cnn_cpx_pct"],
            "fa_mol_pct": flat["cnn_fa"],
            "fs_mol_pct": flat["cnn_fs"],
            "wo_mol_pct": flat["cnn_wo"],
            "ol_unc": flat["cnn_ol_unc"],
            "opx_unc": flat["cnn_opx_unc"],
            "cpx_unc": flat["cnn_cpx_unc"],
            "agreement": flat["cnn_agreement"],
            "method": flat["cnn_method"],
        } if flat["cnn_ol_pct"] is not None else None,
        "pgm_convergence": {
            "pgm_tier": flat["pgm_tier"],
            "pgm_confidence": flat["pgm_confidence"],
            "m_type_prob": flat["m_type_prob"],
            "featureless_nir": bool(flat["featureless_nir"]) if flat["featureless_nir"] is not None else None,
            "no_silicate_bands": bool(flat["no_silicate_bands"]) if flat["no_silicate_bands"] is not None else None,
            "iron_analog_match": bool(flat["iron_analog_match"]) if flat["iron_analog_match"] is not None else None,
            "iron_analog_wmse": flat["iron_analog_wmse"],
            "radar_albedo": flat["radar_albedo"],
            "beaming_eta": flat["pg_beaming_eta"],
            "s_type_metal": bool(flat["s_type_metal"]) if flat["s_type_metal"] is not None else None,
            "metal_fraction_pct": flat["metal_fraction_pct"],
            "signal_count": flat["signal_count"],
            "notes": flat["pgm_notes"],
        } if flat["pgm_tier"] is not None else None,
        "rotation": {
            "period": flat["rotation_period"],
            "period_unc": flat["period_unc"],
            "amplitude": flat["amplitude"],
            "quality_code": flat["quality_code"],
            "is_monolithic": bool(flat["is_monolithic"]) if flat["is_monolithic"] is not None else None,
            "is_binary_suspect": bool(flat["is_binary_suspect"]) if flat["is_binary_suspect"] is not None else None,
        } if flat["rotation_period"] is not None else None,
        "score": {
            "composite_score": flat["composite_score"],
            "estimated_mass_kg": flat["estimated_mass_kg"],
            "grade_estimate": flat["grade_estimate"],
            "target_material": flat["target_material"],
            "unit_value": flat["unit_value"],
            "accessibility": flat["accessibility"],
            "score_mode": flat["score_mode"],
        } if flat["composite_score"] is not None else None,
    }


@router.get("/asteroids/{asteroid_id}")
def get_asteroid_detail(
    asteroid_id: int,
    db: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    """Return full detail for a single asteroid."""
    row = db.execute(_DETAIL_QUERY, (asteroid_id,)).fetchone()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asteroid not found",
        )
    return _row_to_detail(row)
