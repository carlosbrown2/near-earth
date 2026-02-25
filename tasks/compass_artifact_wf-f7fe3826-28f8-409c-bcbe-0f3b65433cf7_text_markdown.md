<!-- role: reference -->
<!-- phase: n/a — audit findings already incorporated into PRD; ERO attribution fix still pending in §scoring-recoverability-accessibility-design -->

# Asteroid mining prospector: a technical specification audit

**The specifications describe a scientifically grounded but incompletely specified pipeline for estimating asteroid resource value from spectral data.** The BAR calibration review correctly characterizes the Cloutis et al. (1986) and Dunn et al. (2010) literature, and the core formula ol/(ol+px) = −0.242 × BAR + 0.728 is verified (R² = 0.73, RMS = 0.03). However, one critical attribution error exists — the ERO concept with Δv < 500 m/s originates from García Yárnoz et al. (2013), not Elvis (2014) — and several methodology gaps would block a clean implementation. This report provides a comprehensive evaluation across all four requested areas: dataset landscape, spectral techniques, methodology gaps, and specific fact-checks.

---

## The asteroid data landscape spans 14 major public datasets with uneven coverage

Any asteroid mining prospector must ingest data from multiple complementary sources, since no single dataset provides the full chain from orbit to composition. The datasets fall into four categories: spectral/compositional, physical properties, orbital/ephemeris, and accessibility/mission-planning.

**Spectral and taxonomic datasets** form the foundation. SMASS II (Bus & Binzel 2002) provides visible-wavelength spectra (0.44–0.93 μm) for **1,447 asteroids** with 26 taxonomic classes, but lacks the near-infrared coverage essential for mineralogy. MITHNEOS (Binzel et al. 2019) fills this gap with NIR spectra (0.8–2.5 μm) for **~1,000 NEOs** obtained via IRTF/SpeX, though thermal contamination beyond ~2 μm affects warm NEOs. Gaia DR3 (Galluccio et al. 2023) delivers low-resolution reflectance in 16 bands (374–1,034 nm) for **60,518 asteroids** — the largest spectral sample by far — but its resolution cannot resolve the mineralogical features at 1–2 μm that drive BAR analysis. The SDSS Moving Object Catalog provides five-band photometry for **~100,000 matched asteroids**, useful for broad taxonomic classification but far too coarse for mineral identification.

The Bus-DeMeo taxonomy (DeMeo et al. 2009) defines **24 classes** over 0.45–2.45 μm based on 371 asteroids and remains the standard classification scheme. The Mahlke et al. (2022) probabilistic taxonomy ("classy" tool) represents a major advance: it classifies **4,526 asteroids** across 17 classes using a mixture-of-common-factor-analysers model that outputs probability vectors rather than single labels, and critically reintroduces albedo to resolve the X-complex degeneracy (E/M/P types are spectrally indistinguishable without albedo). The classy Python package aggregates ~70,000 spectra from multiple repositories. M4AST (Popescu et al. 2012) provides >6,000 spectra with integrated analysis tools including automated meteorite matching against RELAB's **>20,000 laboratory spectra** of ~8,000 samples.

**Physical property datasets** include NEOWISE, which provides diameter and albedo measurements for **~164,000 asteroids** from thermal modeling of WISE infrared observations. The cryogenic 4-band data achieves ~10% diameter uncertainty, while post-reactivation 2-band data has ~30% uncertainty. The Asteroid Lightcurve Database (LCDB, Warner et al. 2009) contains rotation periods and amplitudes for **>24,500 asteroids**. Planetary radar observations from Goldstone (Arecibo collapsed in 2020) have characterized **>850 NEAs**, providing shapes, sizes, radar albedos, and binary detection — but no centralized public archive exists for all radar products.

**Orbital and ephemeris systems** are well-served by JPL's Small-Body Database covering **all >1.3 million known objects** with orbital elements, uncertainties, and physical parameters, accessible via robust APIs (SBDB API, Horizons API, SBDB Query API). The ESA NEO Coordination Centre independently tracks **>36,000 NEOs** with impact risk assessment. NASA's NHATS identifies NEAs accessible for human spaceflight missions, currently listing **~4,790 compliant objects** filtered by specific trajectory constraints.

A prospector tool should treat these datasets as a layered evidence system. The critical limitation is that **fewer than 5% of known NEOs have spectral characterization**, and the overlap between spectrally characterized and NHATS-accessible asteroids is even smaller.

---

## Spectral mineralogy rests on well-validated but assumption-laden techniques

The BAR calibration chain — from Cloutis et al. (1986) through Gaffey et al. (1993) to Dunn et al. (2010) — is the workhorse for S-type asteroid mineralogy. **Band I (~1 μm) absorption arises from Fe²⁺ in both olivine and pyroxene, while Band II (~2 μm) arises from Fe²⁺ exclusively in pyroxene.** The Band Area Ratio (BAR = Area_II / Area_I) therefore tracks the pyroxene-to-olivine proportion. Cloutis et al. (1986) demonstrated this relationship using synthetic olivine-orthopyroxene mixtures, establishing that BAR is nearly linear with ol/(ol+px) and largely insensitive to grain size. Gaffey et al. (1993) defined seven S-asteroid subtypes (S(I)–S(VII)) in Band I Center vs. BAR space, with S(IV) corresponding to ordinary chondrite compositions.

Dunn et al. (2010) provided the modern calibration using **48 equilibrated ordinary chondrites** (types 4–6) with XRD-measured modal abundances — the first calibration based on actual measured rather than normative abundances:

- **ol/(ol+px) = −0.242 × BAR + 0.728** (R² = 0.73, RMS = 0.03)
- **Fa = 1284.9 × BIC² − 2656.5 × BIC + 1389.1** (RMS = 1.3 mol%)
- **Fs = 879.1 × BIC² − 1824.9 × BIC + 953.0** (RMS = 1.4 mol%)

These equations were validated against Hayabusa-returned Itokawa samples with remarkable precision: derived Fa differed by only **0.6 mol%** and Fs by **1.4 mol%** from direct measurement. The H/L/LL classification boundaries in Fa-Fs-ol/(ol+px) space are: H chondrites (Fa ~16–20, Fs ~14–18), L chondrites (Fa ~22–26, Fs ~19–22), LL chondrites (Fa ~26–32, Fs ~22–26).

Beyond BAR analysis, several complementary techniques exist. The Modified Gaussian Model (Sunshine et al. 1990) decomposes spectra into individual absorption components, resolving overlapping olivine and pyroxene bands to estimate modal abundances within **5–10%**. Hapke radiative transfer modeling converts reflectance to single-scattering albedo for linear unmixing but requires knowledge of grain size and optical constants. The CNN approach of Korda et al. (2023) processes raw spectra without continuum removal, achieving **<10 percentage-point accuracy** for modal abundances and mineral chemistry of olivine-pyroxene assemblages. The 3-μm band (Rivkin et al. 2002, 2015) diagnoses hydrated minerals on C-complex asteroids but is blocked by telluric water absorption from the ground.

**Space weathering corrections** use the Brunetto et al. (2006) exponential continuum model: W(λ) = K × exp(Cs/λ), where Cs (units of μm, negative for reddening) parameterizes the degree of weathering. The critical finding for BAR-based analysis is that **space weathering does not significantly affect band center positions or BAR**, only spectral slope and band depth. Thermal corrections for NEOs use the NEATM model (Harris 1998) to subtract thermal emission contaminating Band II. Phase angle corrections follow Sanchez et al. (2012): band depths increase ~0.66%/10° (Band I) and ~0.93%/10° (Band II), but band centers and BAR remain stable.

---

## Seven critical methodology gaps would impede implementation

Based on the described specifications — a BAR calibration review, confidence score formula, grade/density priors, and recoverability/accessibility scoring — several gaps emerge that would block or degrade implementation.

**Gap 1: The red-edge problem is likely unaddressed.** Lindsay et al. (2015, 2016) demonstrated that the Dunn et al. calibration assumes a Band II red edge at **2.50 μm**, but ground-based SpeX data is reliable only to ~2.45 μm. This truncation causes systematic BAR errors that exceed the calibration RMS in **41.67% of ordinary chondrite samples**. The critical BAR change threshold is ΔBAR_crit = 0.03/0.242 = 0.124. Any implementation using real telescope data must apply a red-edge correction factor, and this correction should be explicitly specified.

**Gap 2: Temperature corrections for band parameters are essential but may be missing.** Asteroid surface temperatures vary from ~150 K (main belt) to >400 K (close NEOs), far from laboratory conditions (~300 K). Band I Center shifts systematically with temperature (Hinrichs & Lucey 2002; Sanchez et al. 2012), directly affecting derived Fa and Fs values. MacLennan et al. (2024) found BAR itself does not require temperature correction, but BIC does. All calibrations should normalize to 300 K.

**Gap 3: The featureless-spectra problem undermines half of all targets.** Roughly **50% of NEAs have featureless or near-featureless spectra** (C-complex, X-complex, D-type). The BAR calibration is applicable only to S(IV)-subtype asteroids with clear 1-μm and 2-μm absorptions. For the remaining half, mineralogical confidence is inherently much lower, relying on taxonomic priors and meteorite analog matching rather than direct spectral mineralogy. The confidence score must explicitly account for this bifurcation.

**Gap 4: Density priors have enormous within-class variance.** Carry (2012) is correctly identified as the standard reference, but the actual density ranges within single taxonomic classes are extremely broad: **C-complex spans 1.3–2.1 g/cm³, S-complex 2.0–3.5 g/cm³, and X-complex 2.5–5.0+ g/cm³**. Using point estimates without distributions would dramatically understate uncertainty. Critically, ground truth from Ryugu (bulk density 1,282 kg/m³, lower than any known meteorite) demonstrates that meteorite analog densities systematically overestimate small rubble-pile asteroid densities. A size-dependent density correction is essential: smaller asteroids have higher macroporosity (30–60% for C-complex, 15–40% for S-complex).

**Gap 5: The ERO attribution is incorrect.** The Easily Retrievable Object concept with **Δv < 500 m/s** was defined by García Yárnoz, Sánchez & McInnes (2013), not Elvis (2014). García Yárnoz identified **12 initial EROs** that could be captured to Earth-Sun L1/L2 points. Elvis (2014) separately defined the "Elvis equation" for commercially viable asteroid count using a **4.5 km/s** rendezvous Δv threshold and estimated only ~10 PGM ore-bearing NEOs. These are fundamentally different metrics: García Yárnoz's 500 m/s is a capture/retrieval cost, while Elvis's 4.5 km/s is a rendezvous Δv from LEO. Any accessibility scoring function must correctly distinguish these frameworks.

**Gap 6: Uncertainty propagation through the pipeline is likely incomplete.** The chain from spectrum → taxonomy → mineralogy → composition → mass → value involves multiplicative uncertainties at every stage. Elvis (2014) explicitly noted that all terms in his ore estimation formula are "in need of far better definition." A well-specified pipeline must propagate probability distributions (not point estimates) through every stage, ideally using Monte Carlo or Bayesian methods. The confidence score should be a composite of sub-scores: SNR, spectral coverage, taxonomic ambiguity, thermal contamination, number of observations, phase angle, and calibration-range compliance.

**Gap 7: Missing integration specifications.** The specs likely lack explicit definitions for data format ingestion (PDS4, FITS, VOTable, JSON from JPL APIs), API rate limits and error handling, how to merge data from heterogeneous sources (SMASS visible + MITHNEOS NIR + WISE albedo + LCDB rotation), and validation against ground-truth asteroids (Itokawa, Eros, Vesta, Bennu, Ryugu). Additional datasets that should be incorporated include thermal inertia data (Delbo et al., critical for regolith characterization), radar albedo for metal content estimation (>0.3 indicates metal-rich), binary asteroid flags affecting operations, Yarkovsky drift for orbital uncertainty, and OSIRIS-REx/Hayabusa2 calibration data.

---

## Fact-check results reveal one error and one imprecise attribution

The following table summarizes verification of the seven specific claims:

| Claim | Verdict | Key finding |
|---|---|---|
| Dunn et al. 2010: ol/(ol+px) = −0.242 × BAR + 0.728 | **Verified** | R² = 0.73, RMS = 0.03; validated on Itokawa returned samples |
| Carry (2012) as standard density reference | **Verified** | 287 bodies compiled; remains the most comprehensive single reference, supplemented by Hanuš et al. (2017) |
| Cannon et al. 2023 PGM revisions | **Verified** | PSS 225:105608; PGMs deviate from chondritic ratios at high Ir, reducing maximum PGM grades below prior estimates |
| Elvis (2014) ERO definition, Δv < 500 m/s | **Incorrect** | EROs defined by García Yárnoz et al. (2013), not Elvis. Elvis (2014) used 4.5 km/s rendezvous threshold |
| NHATS accessibility criteria | **Verified** | Total Δv ≤ 12 km/s, duration ≤ 450 days, stay ≥ 8 days, C3 ≤ 60 km²/s², entry speed ≤ 12 km/s, H ≤ 26.5 |
| Alexander et al. 2012 water content | **Partially correct** | Paper reported bulk H and D/H ratios, not water wt% directly; water content inferred as CI ~13–20%, CM ~3–14%, CR ~1–6% |
| Cloutis et al. 1986 BAR calibration | **Verified** | BAR linear with ol/(ol+px), grain-size insensitive; 33 parameters tested on synthetic mixtures at 0.3–2.5 μm |

The Elvis/ERO misattribution is consequential because it conflates two different Δv frameworks. The **500 m/s García Yárnoz threshold** refers to the total cost of capturing an asteroid into an Earth-Sun L1/L2 libration orbit — a retrieval scenario. Elvis's **4.5 km/s threshold** refers to rendezvous Δv from low-Earth orbit — a visit-and-return scenario. An accessibility scoring function built on the wrong framework would produce meaningless rankings.

The Alexander et al. (2012) imprecision matters less for implementation: while the paper reported hydrogen abundances and isotopic ratios rather than water weight percentages directly, the derived water content ranges (CI ~13–20 wt%, CM ~3–14 wt%, CR ~1–6 wt%) are broadly consistent with how the spec likely uses them. However, the spec should cite Brearley (2006) or Garenne et al. (2014) for direct water weight percentages.

---

## Cannon et al. 2023 materially changes PGM economics

The Cannon et al. (2023) revision deserves special attention because PGM value drives the entire economic case for metallic asteroid mining. Previous estimates (Kargel 1994, Lewis 1996) assumed PGM ratios in iron meteorites followed CI chondritic scaling linearly. Cannon et al. analyzed 83 elements across likely asteroid materials and found that **at higher iridium concentrations, other PGMs (Pt, Pd, Rh, Os, Ru) deviate from simple chondritic ratios**, such that maximum total PGM content is lower than extrapolations predicted. PGM concentrations in M-type asteroids may range from **6–230 ppm**, with statistical distributions provided for different iron meteorite groups.

This revision compounds the already pessimistic Elvis (2014) estimate of only ~10 PGM ore-bearing NEOs. The implication for a prospector tool is that metal-content grade estimates based on older PGM scaling would overstate value. The tool should use the Cannon et al. distributions as updated priors.

---

## Conclusion: a sound scientific foundation with addressable engineering gaps

The prospector's core scientific architecture — BAR-based mineralogy for S-types, taxonomic priors for composition, Carry densities, and NHATS-style accessibility — is well-grounded in the literature. The Dunn et al. calibration is the right choice for ordinary chondrite mineralogy, and the Mahlke et al. probabilistic taxonomy represents the state of the art for classification uncertainty.

Three changes would most improve the specifications. First, **correct the ERO attribution** from Elvis (2014) to García Yárnoz et al. (2013) and ensure the accessibility function correctly implements the intended Δv framework. Second, **mandate full uncertainty propagation** as probability distributions through every pipeline stage, not just a single confidence score at the end — the multiplicative nature of the uncertainty chain means point estimates will systematically overstate precision. Third, **specify explicit correction protocols** for the red-edge problem, temperature effects on BIC, and thermal tail contamination, with fallback behaviors when correction parameters are unknown.

The most impactful missing dataset is **thermal inertia** (Delbo et al.), which directly constrains regolith presence and thus mining extractability — a factor orthogonal to composition and accessibility but critical to actual recoverability. The most impactful missing validation step is calibration against the five spacecraft ground-truth asteroids: Itokawa, Eros, Vesta, Bennu, and Ryugu, which collectively span the S, V, C, and B taxonomic classes and would reveal systematic biases in the pipeline.