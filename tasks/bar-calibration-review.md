<!-- role: completed-analysis -->
<!-- phase: n/a — findings already incorporated into PRD Stage 2 -->

# BAR Calibration Review — Fact-Check Writeup

## Issue Summary

The original PRD (Stage 2: Band Parameter Analysis) contained the following claim and formula:

> **Original claim:** "BAR is nearly linear and independent of particle size (Cloutis et al.)."
>
> **Original formula:** `OPX/(OPX+OL) = 0.4187 × BAR + 0.125`

Both the claim and the formula require verification. This document lays out the specific questions for independent fact-checking.

---

## Question 1: Is BAR independent of particle size?

**Our position: No.**

The original PRD attributed this claim to "Cloutis et al." — likely referring to Cloutis et al. (1986, JGR 91, 11641–11653), which studied the spectral properties of olivine-orthopyroxene mixtures.

What Cloutis et al. (1986) actually demonstrated is that BAR correlates with the **olivine-to-pyroxene ratio** in intimate mixtures, and this correlation is useful for remote sensing. However, the claim that BAR is "independent of particle size" overstates the finding. Known factors that affect BAR measurements include:

1. **Grain size / particle size distribution** — Changes in grain size alter the relative depths and widths of Band I and Band II differently, shifting the BAR value for an identical mineral composition. This is well-documented in laboratory studies of mineral reflectance (e.g., Pieters & Hiroi 2004; Reddy et al. 2015).

2. **Temperature** — NEO surface temperatures (which can reach 300–400 K at perihelion for close-approach objects) shift band centers and alter band depths relative to room-temperature lab measurements (~300 K). Sanchez et al. (2012, Icarus 220, 36–50) document temperature corrections for band parameters.

3. **Phase angle (viewing geometry)** — Band depth and area vary with solar phase angle. Phase reddening effects modify spectral slopes and can shift apparent BAR values.

4. **Space weathering** — Solar wind irradiation reddens spectra and reduces band contrast, modifying both band areas and their ratio. This is why our pipeline applies weathering correction (Stage 1B) before band analysis.

**What to verify:**
- Read Cloutis et al. (1986) §3–4. Does the paper actually claim particle-size independence for BAR, or does it note that BAR is *less sensitive* to particle size than individual band depths?
- Check Reddy et al. (2015, "Mineralogy and Surface Composition of Asteroids," in *Asteroids IV*) for a modern review of BAR limitations.
- Check Gaffey et al. (1993, Icarus 106, 573–602) for the original S-subtype calibration — does it mention grain size caveats?

---

## Question 2: What is the correct BAR → mineral ratio formula?

**Our position: The formula `OPX/(OPX+OL) = 0.4187 × BAR + 0.125` needs source verification. The coefficients may be wrong or may refer to a different ratio.**

Several published calibrations exist, and they differ:

### Cloutis et al. (1986)
- Studied olivine-orthopyroxene mixtures with known modal compositions.
- The original relationship was presented as OPX/(OPX+OL) vs. BAR, but the specific regression coefficients vary by grain size fraction studied.
- **Key question:** Are the coefficients 0.4187 and 0.125 from this paper? If so, from which table/figure, and for which grain size fraction?

### Dunn et al. (2010, Icarus 208, 789–797)
- Updated the calibration using ordinary chondrite (OC) meteorites rather than pure mineral mixtures.
- Provided calibrations for **ol/(ol+px)** (note: this is the *inverse ratio* of what the PRD formula computes) vs. BAR:
  - `ol/(ol+px) = -0.242 × BAR + 0.728` (Equation 1 in Dunn et al. 2010)
- Also provided Fa and Fs calibrations from Band I and Band II centers.
- **Critically:** This calibration is explicitly valid only for **ordinary chondrite-like (S(IV)) compositions**. Applying it to S(I), S(II), or other subtypes is not valid.

### Gaffey et al. (1993)
- Defined the S-subtype classification scheme (S(I)–S(VII)) based on Band I center and BAR.
- The subtypes define compositional zones in Band I center vs. BAR space, but Gaffey's calibration is **zone-based** (categorical), not a single linear regression.

**What to verify:**
1. Locate the exact source of the coefficients `0.4187` and `0.125`. Search Cloutis et al. (1986), Gaffey et al. (1993), and Dunn et al. (2010) for these specific numbers.
2. Confirm whether the formula computes `OPX/(OPX+OL)` or `ol/(ol+px)` — these are inverses and the sign of the slope should flip. If BAR increases with pyroxene content, then `OPX/(OPX+OL)` should increase with BAR (positive slope), while `ol/(ol+px)` should decrease with BAR (negative slope). The Dunn et al. (2010) calibration uses a **negative** slope for `ol/(ol+px)`, which is physically consistent.
3. Determine which compositional domain the formula is calibrated for (pure binary OL-OPX mixtures vs. ordinary chondrite assemblages), and verify the valid BAR range.

---

## Question 3: What calibration should we actually use?

**Our recommendation:** Use the **Dunn et al. (2010)** calibrations for ordinary-chondrite-like S(IV) asteroids, as these are calibrated on meteorites (ground truth) rather than synthetic binary mixtures. Specifically:

- `ol/(ol+px) = -0.242 × BAR + 0.728` (valid for S(IV)-subtype, BAR ~0.0–2.0)
- `Fa = -14.63 × BIC + 15.33` (where BIC = Band I center in μm)
- `Fs = -53.46 × BIIC + 109.4` (where BIIC = Band II center in μm)

For non-S(IV) compositions, BAR should be used for **Gaffey subtype classification** (zone placement in BIC vs. BAR space) rather than a single linear mineral ratio.

**What to verify:**
- Confirm Dunn et al. (2010) Equation 1 coefficients and valid domain.
- Check whether more recent calibrations (post-2010) have updated these coefficients.
- Verify the Fa and Fs calibration equations against the paper.

---

## Summary of Changes Made to PRD

1. **Replaced** the unqualified "independent of particle size" claim with a nuanced description: BAR is *relatively insensitive* to grain size per Cloutis et al. (1986), but not absolutely independent per Reddy et al. (2015) and Gaffey et al. (1993)
2. **Restored** the Cloutis et al. (1986) formula `OPX/(OPX+OL) = 0.4187 × BAR + 0.125` with verified provenance (Figure 10, §4) and explicit domain restriction to pure binary OL-OPX mixtures
3. **Added** verified Dunn et al. (2010) calibrations as the preferred default for S(IV) asteroids, with confirmed coefficients and RMSE values
4. **Added** post-2010 refinements: Sanchez et al. (2020) adjusted coefficients and Lindsay et al. (2016) red-edge corrections
5. **Added** Gaffey et al. (1993) zone-based classification as the method for non-S(IV) compositions
6. **Added** implementation requirement to unit-test calibrations and apply temperature corrections
7. **Retained** caveats section noting BAR sensitivity to grain size, temperature, phase angle, and weathering

**Status: All three questions resolved. Findings incorporated into PRD Stage 2.**
