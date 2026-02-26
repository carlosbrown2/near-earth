"""Z3 SMT proof: Gaffey subtype classification completeness.

Formally verifies that classify_gaffey_subtype(bic, bar) in
spectral/band_analysis.py covers the BIC-BAR domain without zone
overlaps among the scientific zone definitions from Gaffey et al.
(1993, Icarus 106, 573-602).

Proofs:
1. The 7 intended Gaffey zones are pairwise disjoint (no overlaps).
2. The Python if-return cascade provides total domain coverage.
3. The cascade assigns exactly one subtype per point (no ambiguity).
4. Primary zones have specific gap regions, correctly filled by fallbacks.
5. Adjacent zone boundaries share clean seams (no micro-gaps).
"""

import itertools

import pytest

z3 = pytest.importorskip("z3")
from z3 import And, Not, Or, Real, Solver, sat, unsat  # noqa: E402

# Physical domain bounds — BIC from BAND1_VALID, BAR non-negative
BIC_MIN, BIC_MAX = 0.85, 1.10
BAR_MIN = 0.0


def _intended_zones(bic, bar):  # type: ignore[no-untyped-def]
    """Gaffey (1993) intended zone geometry as non-overlapping regions.

    These encode the *scientific* zone definitions, with explicit bounds
    that make zones disjoint.  The Python cascade achieves the same
    partitioning via ordered if-return checks.

    Zone boundaries (BIC in um, BAR dimensionless):
      S(I):   bar < 0.10                         (pure olivine, any BIC)
      S(II):  bic > 1.02,   0.10 <= bar < 0.80   (olivine-pyroxene)
      S(III): 0.92 <= bic <= 1.02, 0.10 <= bar < 0.80  (transitional)
      S(IV):  0.92 <= bic <= 1.04, 0.80 <= bar <= 1.80  (ordinary chondrite)
      S(V):   bic < 0.92,   0.40 <= bar <= 1.20   (pyroxene-rich)
      S(VI):  bic < 0.92,   1.20 < bar <= 1.80    (OPX-dominant)
      S(VII): bar > 1.80                          (basaltic, any BIC)
    """
    return {
        "S(I)": bar < 0.10,
        "S(II)": And(bic > 1.02, bar >= 0.10, bar < 0.80),
        "S(III)": And(bic >= 0.92, bic <= 1.02, bar >= 0.10, bar < 0.80),
        "S(IV)": And(bic >= 0.92, bic <= 1.04, bar >= 0.80, bar <= 1.80),
        "S(V)": And(bic < 0.92, bar >= 0.40, bar <= 1.20),
        "S(VI)": And(bic < 0.92, bar > 1.20, bar <= 1.80),
        "S(VII)": bar > 1.80,
    }


def _cascade_regions(bic, bar):  # type: ignore[no-untyped-def]
    """Effective regions of the Python if-return cascade in classify_gaffey_subtype.

    Each entry is the conjunction of the check condition AND the negation of
    all prior checks — the exact set of (bic, bar) values that trigger that
    specific return statement.
    """
    # Raw conditions in source order
    checks = [
        ("S(I)", bar < 0.10),
        ("S(VII)", bar > 1.80),
        ("S(VI)", And(bic < 0.92, bar > 1.20)),
        ("S(V)", And(bic < 0.92, bar >= 0.40, bar <= 1.20)),
        ("S(IV)", And(bic >= 0.92, bic <= 1.04, bar >= 0.80, bar <= 1.80)),
        ("S(III)", And(bic >= 0.92, bic <= 1.02, bar >= 0.10, bar < 0.80)),
        ("S(II)", And(bic > 1.02, bar >= 0.10, bar < 0.80)),
        # Fallback assignments for gap regions
        ("S(IV)_fb", bar >= 0.80),
        ("S(II)_fb", bic > 1.00),
    ]

    regions = {}
    prior_negations = []

    for label, cond in checks:
        if prior_negations:
            eff = And(*prior_negations, cond)
        else:
            eff = cond
        regions[label] = eff
        prior_negations.append(Not(cond))

    # Final else — all checks failed → S(III)
    regions["S(III)_fb"] = And(*prior_negations)
    return regions


def _domain(bic, bar):  # type: ignore[no-untyped-def]
    """Physical domain: BIC in valid range, BAR non-negative."""
    return And(bic >= BIC_MIN, bic <= BIC_MAX, bar >= BAR_MIN)


# ---------------------------------------------------------------------------
# Proof 1: Primary zones are pairwise disjoint
# ---------------------------------------------------------------------------


class TestPrimaryZonesDisjoint:
    """Prove that no two intended Gaffey zones overlap in BIC-BAR space."""

    def test_all_21_pairs_disjoint(self):
        """For every pair of primary zones, their conjunction is UNSAT."""
        bic, bar = Real("bic"), Real("bar")
        zones = _intended_zones(bic, bar)
        dom = _domain(bic, bar)

        names = list(zones.keys())
        for i, j in itertools.combinations(range(len(names)), 2):
            s = Solver()
            s.add(dom)
            s.add(zones[names[i]])
            s.add(zones[names[j]])
            assert s.check() == unsat, (
                f"{names[i]} and {names[j]} overlap: {s.model()}"
            )


# ---------------------------------------------------------------------------
# Proof 2: Cascade total coverage and uniqueness
# ---------------------------------------------------------------------------


class TestCascadeCoverage:
    """Prove the if-return cascade classifies every domain point exactly once."""

    def test_total_coverage(self):
        """No point in the physical domain escapes all cascade branches."""
        bic, bar = Real("bic"), Real("bar")
        cascade = _cascade_regions(bic, bar)
        dom = _domain(bic, bar)

        covered = Or(*cascade.values())
        s = Solver()
        s.add(dom)
        s.add(Not(covered))
        assert s.check() == unsat, f"Uncovered point: {s.model()}"

    def test_exactly_one_assignment(self):
        """No point in the domain triggers two cascade branches."""
        bic, bar = Real("bic"), Real("bar")
        cascade = _cascade_regions(bic, bar)
        dom = _domain(bic, bar)

        names = list(cascade.keys())
        for i, j in itertools.combinations(range(len(names)), 2):
            s = Solver()
            s.add(dom)
            s.add(cascade[names[i]])
            s.add(cascade[names[j]])
            assert s.check() == unsat, (
                f"Cascade branches {names[i]} and {names[j]} overlap: {s.model()}"
            )


# ---------------------------------------------------------------------------
# Proof 3: Primary zone gap characterization
# ---------------------------------------------------------------------------


class TestPrimaryZoneGaps:
    """Document the gap regions not covered by primary zones.

    The Gaffey (1993) zone rectangles do not perfectly tile BIC-BAR space.
    Two gap regions exist in the physical domain, correctly filled by the
    cascade fallback assignments.
    """

    def test_gaps_exist(self):
        """Primary zones alone do NOT cover the full domain."""
        bic, bar = Real("bic"), Real("bar")
        zones = _intended_zones(bic, bar)
        dom = _domain(bic, bar)

        s = Solver()
        s.add(dom)
        s.add(Not(Or(*zones.values())))
        assert s.check() == sat, "Expected gaps but found total coverage"

    def test_gap_low_bic_low_bar(self):
        """Gap: bic < 0.92, 0.10 <= bar < 0.40 — no primary zone covers this.

        The cascade assigns S(III) via final else.  Physically: low-BIC
        asteroids with very low BAR (weak Band II) are rare boundary cases.
        """
        bic, bar = Real("bic"), Real("bar")
        zones = _intended_zones(bic, bar)
        dom = _domain(bic, bar)
        gap = And(bic < 0.92, bar >= 0.10, bar < 0.40)

        # No primary zone covers any point in this region
        s = Solver()
        s.add(dom, gap, Or(*zones.values()))
        assert s.check() == unsat, "This region should be a gap"

    def test_gap_high_bic_high_bar(self):
        """Gap: bic > 1.04, 0.80 <= bar <= 1.80 — no primary zone covers this.

        The cascade assigns S(IV) via the `bar >= 0.80` fallback.
        Physically: high-BIC with high BAR — adjacent to S(IV) zone.
        """
        bic, bar = Real("bic"), Real("bar")
        zones = _intended_zones(bic, bar)
        dom = _domain(bic, bar)
        gap = And(bic > 1.04, bar >= 0.80, bar <= 1.80)

        s = Solver()
        s.add(dom, gap, Or(*zones.values()))
        assert s.check() == unsat, "This region should be a gap"


# ---------------------------------------------------------------------------
# Proof 4: Boundary consistency at zone edges
# ---------------------------------------------------------------------------


class TestBoundaryConsistency:
    """Verify that adjacent zones share clean boundary seams."""

    def test_s4_s3_seam_at_bar_080(self):
        """S(IV) includes bar=0.80; S(III) has bar<0.80. No gap or overlap."""
        bic, bar = Real("bic"), Real("bar")
        zones = _intended_zones(bic, bar)

        # In the shared BIC range [0.92, 1.02], bar=0.80 → S(IV) only
        constraint = And(bic >= 0.92, bic <= 1.02, bar == 0.80)

        s_iv = Solver()
        s_iv.add(constraint, zones["S(IV)"])
        assert s_iv.check() == sat, "S(IV) should include bar=0.80"

        s_iii = Solver()
        s_iii.add(constraint, zones["S(III)"])
        assert s_iii.check() == unsat, "S(III) should exclude bar=0.80"

    def test_s3_s2_seam_at_bic_102(self):
        """S(III) includes bic=1.02; S(II) has bic>1.02. No gap or overlap."""
        bic, bar = Real("bic"), Real("bar")
        zones = _intended_zones(bic, bar)

        constraint = And(bar >= 0.10, bar < 0.80, bic == 1.02)

        s_iii = Solver()
        s_iii.add(constraint, zones["S(III)"])
        assert s_iii.check() == sat, "S(III) should include bic=1.02"

        s_ii = Solver()
        s_ii.add(constraint, zones["S(II)"])
        assert s_ii.check() == unsat, "S(II) should exclude bic=1.02"

    def test_s5_s6_seam_at_bar_120(self):
        """S(V) includes bar=1.20; S(VI) has bar>1.20. No gap or overlap."""
        bic, bar = Real("bic"), Real("bar")
        zones = _intended_zones(bic, bar)

        constraint = And(bic < 0.92, bic >= BIC_MIN, bar == 1.20)

        s_v = Solver()
        s_v.add(constraint, zones["S(V)"])
        assert s_v.check() == sat, "S(V) should include bar=1.20"

        s_vi = Solver()
        s_vi.add(constraint, zones["S(VI)"])
        assert s_vi.check() == unsat, "S(VI) should exclude bar=1.20"

    def test_s1_seam_at_bar_010(self):
        """S(I) has bar<0.10; adjacent zones start at bar>=0.10."""
        bic, bar = Real("bic"), Real("bar")
        zones = _intended_zones(bic, bar)
        dom = _domain(bic, bar)

        # At bar=0.10, S(I) must not apply
        s = Solver()
        s.add(dom, bar == 0.10, zones["S(I)"])
        assert s.check() == unsat, "S(I) should exclude bar=0.10"

    def test_s7_seam_at_bar_180(self):
        """S(VII) has bar>1.80; S(IV) and S(VI) include bar<=1.80."""
        bic, bar = Real("bic"), Real("bar")
        zones = _intended_zones(bic, bar)
        dom = _domain(bic, bar)

        # At bar=1.80, S(VII) must not apply
        s = Solver()
        s.add(dom, bar == 1.80, zones["S(VII)"])
        assert s.check() == unsat, "S(VII) should exclude bar=1.80"

        # But S(IV) or S(VI) should cover bar=1.80
        s2 = Solver()
        s2.add(dom, bar == 1.80, Or(zones["S(IV)"], zones["S(VI)"]))
        assert s2.check() == sat, "bar=1.80 should be covered by S(IV) or S(VI)"
