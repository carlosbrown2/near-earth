<!-- bead-type: research -->
<!-- phase: 1 -->
<!-- output-artifact: scoring/grade_density_priors.yaml -->

# Scoring Model Review — Grade Estimates & Density Priors

## Purpose

The scoring formula in the PRD (US-005) requires concrete numerical values for `grade_estimate` and `estimated_mass` (via density priors). These are literature-derivable: published meteorite compositions and asteroid bulk density measurements provide ground truth. This task produces a lookup table of `{taxonomy_class → {material: concentration, density, uncertainty}}` with citations per value.

---

## Question 1: What are the grade estimates (material concentrations) by taxonomy class?

For each taxonomy class that maps to a target material, we need a **median concentration and range** derived from meteorite analog measurements.

### PGMs in M-type (iron meteorite analogs)

- **Sources to check:**
  - Wasson (1999) — Ir, Ga, Ge systematics in iron meteorite groups; PGM concentrations by group (IIIAB, IVA, etc.)
  - Cannon et al. (2023) — Updated PGM concentration model; non-chondritic PGM ratios at high Ir; maximum Pt, Rh, Ru values
  - Kargel (1994) — Historical PGM grade estimates (likely too optimistic; verify how much Cannon revises downward)
- **What to determine:**
  1. Median and range of total PGM content (Pt + Pd + Rh + Ru + Os + Ir) in ppm for IIIAB and IVA iron meteorites
  2. How much do Cannon et al. (2023) revise Kargel (1994) estimates downward? Provide both for comparison.
  3. What is the variance across iron meteorite groups? (i.e., is "M-type" specific enough, or do we need sub-group resolution?)

### PGMs in S-type (ordinary chondrite metal fraction)

- **Sources to check:**
  - Cannon et al. (2023) — PGM content in OC metal grains; bulk vs. separated-metal concentrations
  - Hutchison (2004), *Meteorites* — metal fraction by OC group (H ~18%, L ~8%, LL ~3%)
- **What to determine:**
  1. Bulk PGM ppm in H, L, LL chondrites (whole-rock)
  2. PGM ppm in the **metal phase** of H, L, LL chondrites (after separation)
  3. Metal mass fraction by OC group (H/L/LL) — needed to compute how much metal is available per kg of asteroid

### Water in C-type (carbonaceous chondrite analogs)

- **Sources to check:**
  - Alexander et al. (2012, Science 337, 721) — H₂O/OH content in CM and CI chondrites
  - Rivkin et al. (2002) — hydrated mineral detections on C-type asteroids
- **What to determine:**
  1. Median water content (wt%) in CI chondrites
  2. Median water content (wt%) in CM chondrites
  3. Range across samples — what is the floor and ceiling?
  4. Is there a mapping from spectral band depth (3 μm) to water wt%?

### Iron/silicates in S-type and V-type

- **Sources to check:**
  - Dunn et al. (2010) — modal abundances from OC calibration
  - Burbine et al. (2001) — V-type / HED compositions
- **What to determine:**
  1. Typical olivine, pyroxene, and metal wt% for H, L, LL analog compositions
  2. Typical pyroxene wt% for V-type (eucrite/diogenite) analogs
  3. Iron content (metallic + silicate-bound) by taxonomy class

---

## Question 2: What are the density priors by taxonomy class?

- **Primary source:** Carry (2012, P&SS 73, 98–118) — the standard reference for asteroid bulk densities
- **What to determine:**

| Taxonomy class | Expected density (g/cm³) | 1σ uncertainty | Meteorite analog basis |
|---|---|---|---|
| C-complex | ? | ? | CI/CM carbonaceous chondrites |
| M-type | ? | ? | Iron meteorites / stony-irons |
| S-complex | ? | ? | Ordinary chondrites |
| V-type | ? | ? | HED achondrites |
| A-type | ? | ? | Pallasites / brachinites |
| D-type / P-type | ? | ? | Tagish Lake / IDPs |
| E-type | ? | ? | Enstatite chondrites |

- **Additional question:** Carry (2012) reports bulk densities which include macroporosity. Should we use meteorite grain densities (higher, no porosity) or asteroid bulk densities (lower, includes rubble-pile porosity) for mass estimation? The choice affects estimated_mass significantly — document the decision and its impact.

---

## Deliverable

A YAML/JSON reference file (`scoring/grade_density_priors.yaml`) containing:
- `grade_estimates`: Per taxonomy class, per material — median concentration, unit, lower bound, upper bound, source citation
- `density_priors`: Per taxonomy class — median density (g/cm³), 1σ uncertainty, source citation, note on porosity assumption
- All values traceable to specific papers and tables

---

## Verification

- Cross-check at least 3 grade values against a second independent source
- Verify density priors against Carry (2012) Table 2 or equivalent
- Sanity check: compute estimated mass for 16 Psyche (known diameter ~226 km, M-type) and compare to published mass estimates (~2.7 × 10¹⁹ kg)
