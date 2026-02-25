# PRD: Asteroid Mining Prospector

## 1. Introduction

Asteroid Mining Prospector is a data pipeline and scoring system that ingests publicly available asteroid survey data (orbital parameters, spectral observations, physical properties), performs mineralogical analysis, and produces a ranked list of near-Earth asteroid mining candidates. The product answers three questions: *what* is worth mining, *which* asteroids likely contain it, and *when* are they accessible.

---

## 2. Goals

- Produce a ranked candidate list of asteroids scored by estimated mineral value and mission accessibility (delta-v / approach windows).
- Ingest and unify at least three independent data sources (SBDB orbital data, spectral survey data, physical property catalogs) into a single queryable dataset.
- Classify each asteroid's likely mineralogy using spectral curve matching and taxonomic classification (Mahlke et al. 2022 taxonomy with albedo integration).
- Calculate future close-approach windows and delta-v estimates for each candidate.
- Make the scoring methodology transparent and configurable (users can weight materials by market value or strategic priority).

---

## 3. Target Materials — What to Mine

The materials below are ordered by a combination of Earth-market value, strategic scarcity, and detectability from spectral data.

### Tier 1: Highest Value — Platinum Group Metals (PGMs)
| Material | Earth price (approx.) | Spectral detectability | Asteroid association |
|---|---|---|---|
| **Platinum (Pt)** | ~$30,000/kg | Indirect — via M-class/X-class taxonomy + high albedo + metallic spectral slope | M-type asteroids (e.g., 16 Psyche); iron meteorite analogs |
| **Rhodium (Rh)** | ~$145,000/kg | Indirect — same as Pt | Co-located with Pt in Fe-Ni metal bodies |
| **Palladium (Pd)** | ~$32,000/kg | Indirect — same as Pt | Co-located with Pt |
| **Gold (Au)** | ~$75,000/kg | Not directly detectable via reflectance spectroscopy | Inferred from M-type classification + meteorite analog matching |

**Detection strategy:** PGMs cannot be detected directly from reflectance spectra. They are *inferred* by identifying metallic (M-type / M-complex per Mahlke 2022) asteroids whose spectra match iron-nickel meteorites. Key indicators:
- Featureless, reddish spectral slope (increasing reflectance with wavelength) in VNIR
- Moderate-to-high radar albedo (from radar surveys, not optical)
- Absence of silicate absorption bands at 1 μm and 2 μm
- Matching to iron meteorite lab spectra from RELAB
- High thermal inertia (Γ > 100 J/m²/s⁰·⁵/K⁻¹) from dedicated thermophysical modeling with shape/spin constraints (Matter et al. 2013; Harris & Drube 2014) — **not a standard NEOWISE catalog product**; NEOWISE provides diameter, pV, and beaming parameter η via NEATM fits, but deriving Γ requires additional modeling
- Polarimetric properties consistent with iron meteorite analogs (Belskaya et al. 2022)

**Important caveat on PGM concentrations (Cannon et al. 2023):** Older estimates of PGM grades in asteroids (Kargel 1994) assumed all PGMs vary linearly with Ir according to chondritic ratios. Updated measurements show PGMs **deviate from chondritic ratios** at high Ir concentrations — Ru, Rh, and Pt roll over, meaning maximum PGM contents are lower than previously assumed. Additionally, S-type/Q-type asteroids with ordinary chondrite mineralogy contain PGM-bearing metal grains (10–20% metal by mass) that could be separated via mineral processing, expanding the set of viable PGM candidates beyond M-types alone.

### Tier 2: High Value — Rare Earth Elements & Strategic Metals
| Material | Earth price (approx.) | Spectral detectability | Asteroid association |
|---|---|---|---|
| **Cobalt (Co)** | ~$30/kg | Indirect — co-located with Ni in Fe-Ni bodies | M-type, stony-iron analogs |
| **Nickel (Ni)** | ~$16/kg (but enormous volume) | Indirect — Fe-Ni metallic signature | M-type asteroids |
| **Rare Earth Elements** | $20–$3,000/kg depending on element | Not directly detectable in VNIR; some features in mid-IR | Inferred from C-type/carbonaceous chondrite analogs |

### Tier 3: In-Space Economy — Water & Volatiles
| Material | Value | Spectral detectability | Asteroid association |
|---|---|---|---|
| **Water/Ice (H₂O)** | Extremely high *in space* (propellant, life support) | **Direct** — 3 μm absorption band (OH/H₂O); **6 μm emission** band (unambiguous H₂O, Arredondo et al. 2024); overtones at 1.4 μm, 1.9 μm | C-type, B-type, G-type asteroids; carbonaceous chondrite analogs; also detected on S-type surfaces (Iris, Massalia) |
| **Hydrated minerals** | Supporting indicator | **Direct** — phyllosilicate features at 0.7 μm, 3 μm | Same as above |

**Note on water detection (Arredondo et al. 2024):** The first unambiguous detection of molecular H₂O on asteroid surfaces was achieved using the 6 μm emission feature (H-O-H bending mode) observed by SOFIA/FORCAST. Unlike the 3 μm band, which can be caused by OH, organics, or ammonium, the 6 μm feature is exclusively due to molecular water. Critically, water was found on *nominally anhydrous* S-type asteroids (Iris, Massalia) at abundances of ~450 μg/g — meaning water-mining candidates may extend beyond C-types. JWST is now expanding this survey to 30+ targets.

**Economic caveat on S-type water:** The ~450 μg/g water abundance detected on S-type asteroids is scientifically significant but likely too low for economic extraction. For mining scoring purposes, S-type water should be heavily penalized relative to C-type targets where hydrated mineral abundances reach 10–20 wt%. The scorer must distinguish between "detectable" and "extractable at scale."

### Tier 4: Bulk Construction — Silicates & Iron
| Material | Value | Spectral detectability | Asteroid association |
|---|---|---|---|
| **Iron (Fe)** | ~$0.10/kg on Earth, high value in space | **Direct** — Band I (1 μm) and Band II (2 μm) absorption positions | S-type asteroids, ordinary chondrite analogs |
| **Olivine** | Structural / refractories | **Direct** — broad 1 μm feature, no 2 μm band | A-type, S(I)-type asteroids |
| **Pyroxene** | Structural | **Direct** — both 1 μm and 2 μm bands | S-type subtypes S(IV)-S(VII), V-type (Vestoids) |

---

## 4. Spectral Analysis Methodology — How to Detect Materials

Based on Mahlke et al. (2022), Korda et al. (2023), Gaffey et al. (1993), Cloutis et al., and M4AST (Popescu et al.), the system implements a multi-stage classification pipeline combining classical spectral analysis with modern machine learning approaches. **Critical wavelength constraint:** the near-infrared (NIR) region from 0.8–2.5 μm is where the decisive mineralogical information lives. Visible-only data (like Gaia's 0.374–1.034 μm) enables coarse taxonomy but *cannot* perform real mineralogy.

### Why Wavelength Coverage Matters — The Hard Physics

The two diagnostic absorption features for silicate mineralogy are:

- **Band I (~1 μm):** Spans from a reflectance peak at ~0.7 μm to a second peak at 1.3–1.7 μm. The band *minimum* falls at:
  - **Orthopyroxene:** 0.911–0.956 μm (within visible range)
  - **Olivine:** 1.055–1.090 μm (**outside** Gaia's 1.034 μm cutoff)
  - **Mixtures:** Shifts between these endmembers based on OPX/OL ratio
- **Band II (~2 μm):** Present only in pyroxene. Minimum near 1.8–2.1 μm, enclosed between ~1.5 μm and ~2.4 μm. **Completely outside visible range.**

**Consequence:** Visible-only data (Gaia, SMASS II, ECAS) can detect the *onset* of the 1 μm absorption but cannot:
1. Capture the Band I minimum for olivine-rich compositions (≥1.055 μm)
2. Define the Band I area (needs the 1.3–1.7 μm continuum anchor)
3. Detect Band II at all (entirely above 1.5 μm)
4. Compute Band Area Ratio (BAR) — the key olivine/pyroxene discriminator
5. Distinguish M-type from E-type or P-type (all featureless in visible; need albedo)

**This means MITHNEOS (0.8–2.5 μm) is the irreplaceable data source for mineralogy.** Gaia provides the volume (60,518 asteroid spectra) for initial triage; MITHNEOS provides the depth (>1,000 NEOs with full VNIR) for actual composition determination.

### Stage 0: Spectral Normalization & Quality Control

Before any classification or analysis, all ingested spectra must be standardized and quality-checked. Without this stage, downstream χ² matching, band parameters, and ML outputs will be brittle.

- **Input:** Raw reflectance spectra from any survey (Gaia, MITHNEOS, SMASS, etc.)
- **Method:**
  1. Resample to a common wavelength grid via interpolation (preserving original resolution metadata)
  2. Normalize to a standard convention (e.g., reflectance = 1.0 at 550 nm)
  3. Mask known bad channels and telluric absorption regions (1.35–1.45 μm, 1.80–2.00 μm for ground-based data)
  4. Estimate per-channel SNR (from flux uncertainties where available, or from spectral smoothness heuristics)
  5. Record phase angle, airmass, and observation geometry metadata
  6. Flag spectra below a minimum SNR threshold as "low quality" (usable for taxonomy triage only, not band analysis)
- **Output:** Cleaned, normalized spectrum with quality metadata (SNR estimate, wavelength coverage flags, telluric gap flags)
- **Why it matters:** Heterogeneous survey data with different wavelength grids, normalization conventions, and noise characteristics will cause systematic errors in every downstream stage if not standardized first.

### Stage 1: Unified Probabilistic Taxonomic Classification (Mahlke et al. 2022)

This stage replaces the prior two-step approach (separate albedo disambiguation + Bus-DeMeo PCA classification) with a single unified classification that integrates albedo directly into the taxonomy. This is based on Mahlke et al. (2022, A&A 665, A26), which represents the current state-of-the-art in asteroid taxonomy.

**Why Mahlke 2022 over Bus-DeMeo:**
- **Integrates albedo directly** into the classification model via a mixture of common factor analysers (MCFA), resolving the X-complex degeneracy natively — no separate albedo disambiguation step needed
- **Probabilistic class assignments:** returns a vector of class probabilities rather than a single label, directly usable as a confidence score
- **Handles partial wavelength coverage:** visible-only, NIR-only, or combined VNIR spectra are all classified within a single model, exactly matching our heterogeneous data situation
- **Replaces X-complex with M-complex:** explicitly separates E/M/P using a Gaussian mixture model on albedo, with P-class moved to C-complex
- **10× larger training set:** built on 2983 observations of 2125 asteroids (vs. 371 in Bus-DeMeo)
- **Open-source Python tool:** the `classy` (CLAssification of a Solar System bodY) package provides command-line classification ready to integrate

**Implementation:**
- **Input:** Reflectance spectrum (any coverage: 0.45–2.45 μm full VNIR, visible-only, or NIR-only) + NEOWISE visible albedo (pV) where available
- **Method:** MCFA model projects observations into a 4-dimensional latent space, classifies using 50 Gaussian mixture components mapped to 17 taxonomic classes across three complexes (C, M, S), then applies feature-flag detection (0.7 μm hydration, 0.9 μm silicate, 0.5 μm Xe feature)
- **Output:** Probabilistic class assignment vector across 17 classes; primary class + confidence; feature flags
- **Coverage:** Classifies any asteroid with spectral data and/or albedo. Gaia DR3 (60,518 asteroids) provides volume triage; MITHNEOS (~1,000+ NEOs) provides full VNIR classification; NEOWISE albedo (>158,000 minor planets) resolves featureless spectra
- **Key advantage for PGM detection:** M-type asteroids are identified directly from the joint spectral-albedo model, not inferred from a separate albedo threshold. The probabilistic output quantifies exactly how confident we are in an M-type vs. E-type vs. P-type assignment.
- **Fallback:** For asteroids without albedo data, the classifier gracefully degrades to spectral-only classification (equivalent to Bus-DeMeo behavior) and flags the X-complex ambiguity in the probability vector

**Validation:** Burbine et al. (2024, PSJ) tested Bus-DeMeo taxonomy with ~1500 meteorite spectra and found that featureless classes (D, X) group meteorites with diverse mineralogies. Mahlke 2022's albedo integration and probabilistic output directly address this ambiguity.

**Bus-DeMeo compatibility:** Many published studies and ground-truth datasets reference Bus-DeMeo classes. The system should maintain a mapping layer — `classy` can output Bus-DeMeo-equivalent labels for cross-referencing. Taxonomy probabilities should be treated as compositional *priors*, not composition truth.

**Source:** [`classy` Python package](https://github.com/maxmahlke/classy) — actively maintained, available on PyPI

### Stage 1B: Space Weathering Correction (conditional — silicate spectra only)

For silicate-bearing spectra (S/Q/V/A-complex from Stage 1), space weathering correction is applied **before** band parameter analysis and analog matching, because weathering biases the spectral slopes and band depths those stages depend on.

- **Input:** Raw spectrum (ideally full VNIR) + Stage 1 taxonomy result
- **Applicability:** Apply **only** to silicate-bearing spectra (S, Q, V, A and related subtypes). **Do not** apply to C-complex, M/X-complex, or other featureless spectra — Brunetto-style corrections can introduce spurious features on intrinsically featureless surfaces.
- **Method:** Brunetto et al. (2006) model — compute Cs parameter via least-squares fitting of the exponential reddening/darkening function
- **Output:** De-weathered spectrum for use in Stages 2–3; original spectrum preserved for comparison
- **Why it matters:** Solar wind irradiation reddens and darkens spectra over ~10⁶ years. Fresh Q-type ordinary chondrite surfaces weather into S-type appearance. Without correction, an asteroid's true composition may be misidentified. Space weathering effects are most pronounced in the visible-NIR transition region (~0.7–1.5 μm), making this especially important for MITHNEOS data.
- **Note on ML alternatives:** The CNN mineral quantification (Stage 2B) implicitly handles some space weathering effects — Korda et al. (2023) demonstrated that their model detects the olivine depletion trend from Q-type to S-type asteroids that is attributable to weathering. For the classical pipeline (Stages 2–3), explicit correction remains necessary. Hapke modeling (Stage 3B) also treats space weathering as a free parameter (SMFe abundance), providing a more physically grounded correction.

### Stage 2: Band Parameter Analysis (requires NIR data ≥ 2.0 μm)

**This stage is only possible for asteroids with MITHNEOS or equivalent NIR spectra.** It cannot be performed with Gaia or visible-only data. For silicate-bearing spectra, this stage operates on the de-weathered spectrum from Stage 1B.

- **Input:** Continuum-removed NIR spectrum (0.7–2.5 μm minimum); de-weathered where Stage 1B was applied
- **Method:**
  1. Fit linear continua tangent to reflectance peaks at ~0.7 μm and ~1.5 μm (Band I) and ~1.5 μm and ~2.4 μm (Band II). Document chosen anchor points per spectrum for reproducibility.
  2. Divide spectrum by continua to isolate absorption bands
  3. Compute **Band I center** via polynomial fit of the absorption minimum
  4. Compute **Band II center** via polynomial fit of the absorption minimum
  5. Apply **temperature correction** to band centers (NEO surface temperatures differ from lab conditions; uncorrected band positions introduce systematic error in mineral chemistry estimates)
  6. Integrate areas under both bands → compute **BAR = BII / BI**
  7. Apply calibrated BAR-to-mineral-ratio relation (see caveats below) to estimate olivine/pyroxene ratio. **This calibration is valid only for ordinary-chondrite-like S-complex assemblages.**
  8. Use band center wavelength shifts to estimate Fe²⁺ and Ca²⁺ molar content (Band I shifts 0.035–0.045 μm from Fs₀ to Fs₁₀₀)
- **Output:** Olivine/pyroxene ratio (with systematic uncertainty), molar iron content, Gaffey S-subtype (I–VII)
- **Why it matters:** This is the *only* quantitative mineralogical measurement possible from remote sensing via classical methods.
- **Critical dependency:** Band II is diagnostic of pyroxene. Without it, you cannot distinguish pure olivine from olivine-pyroxene mixtures, and you cannot compute BAR.

**⚠️ BAR calibration — verified findings:**

**Particle-size sensitivity:** Cloutis et al. (1986) describes BAR as a "grain-size independent parameter" in the sense that the BAR-vs-mineral-ratio correlation holds broadly across tested grain size fractions (<45 μm, 45–90 μm, 90–250 μm), making BAR *relatively insensitive* compared to individual band depths or centers. However, BAR is **not absolutely independent** of particle size — modern reviews (Reddy et al. 2015, *Asteroids IV*) confirm that grain size distribution, temperature, phase angle, and space weathering all shift BAR values for the same composition. Gaffey et al. (1993) adds that comminution and regolith processes on smaller asteroids (<100 km) can further affect band parameters. Treat BAR-derived mineral ratios as **calibrated estimates with systematic uncertainty**, not physics-exact measurements.

**Calibration formulas (verified):**

1. **Cloutis et al. (1986) — binary mixtures:** `OPX/(OPX+OL) = 0.4187 × BAR + 0.125` (Figure 10, §4). Derived from linear regression across all grain size fractions of synthetic olivine-orthopyroxene mixtures. Valid BAR range ~0–2.5. **Domain: pure binary OL-OPX mixtures only** — not directly applicable to ordinary chondrite or other complex assemblages.

2. **Dunn et al. (2010) — ordinary chondrites (preferred for S(IV) asteroids):**
   - `ol/(ol+px) = -0.242 × BAR + 0.728` (Eq. 1; RMSE ~0.03; valid BAR ~0.0–2.0)
   - `Fa (mol%) = -14.63 × BIC + 15.33` (BIC = Band I center in μm; RMSE ~1.3 mol%)
   - `Fs (mol%) = -53.46 × BIIC + 109.4` (BIIC = Band II center in μm; RMSE ~1.4 mol%)
   - Derived from 48 ordinary chondrite spectra with known modal abundances. **Valid only for OC-like S(IV)-subtype compositions.**
   - Note: Dunn computes olivine fraction `ol/(ol+px)` (negative slope), while Cloutis computes pyroxene fraction `OPX/(OPX+OL)` (positive slope) — these are inverses, not contradictions.

3. **Post-2010 refinements:**
   - Sanchez et al. (2020): `ol/(ol+px) = -0.228 × BAR + 0.768` — adjusted for noisy/incomplete NIR spectra (RMSE ~0.05)
   - Lindsay et al. (2016): Conversion equations for "red edge" continuum endpoint effects on BAR measurement, aligning with Dunn's calibration framework

4. **Gaffey et al. (1993) — zone-based classification:** Does not provide a single linear formula. Uses BAR for categorical S-subtype zoning (S(I)–S(VII)) in Band I center vs. BAR space. **Use for non-S(IV) compositions** where Dunn's linear calibration does not apply.

**Implementation requirement:** Use the Dunn et al. (2010) calibration as the default for S(IV)-subtype asteroids. For non-S(IV) compositions, use Gaffey subtype classification (zone placement) rather than a single linear mineral ratio. Unit-test all calibration equations against published worked examples before deployment. Apply temperature corrections (Sanchez et al. 2012) to band centers before computing mineral chemistry.

### Stage 2B: CNN-Based Mineral Quantification (Korda et al. 2023)

This stage runs in parallel with classical band parameter analysis and provides a modern machine-learning alternative that is more robust to noise, requires no manual continuum removal, and handles space weathering effects implicitly.

**Why add this:** Classical band parameter extraction (Stage 2) requires careful continuum fitting and only works for silicate-bearing spectra with clear absorption features. CNN-based approaches (Korda et al. 2023, A&A 669, A101; Tang et al. 2025, AJ 169:201) operate directly on raw spectra and provide quantitative mineral abundances with ~10 percentage-point accuracy.

- **Input:** VNIR reflectance spectrum (0.45–2.45 μm), normalized at 550 nm
- **Method:** Convolutional neural network (2 hidden layers, 24 and 8 convolutional kernels) trained on RELAB + C-Tape lab spectra of olivine, orthopyroxene, clinopyroxene, their mixtures, and meteorites
- **Output:**
  - Modal abundances: olivine / orthopyroxene / clinopyroxene ratios
  - Chemical compositions: Fa (fayalite content in olivine), Fs (ferrosilite in OPX), Wo (wollastonite in CPX)
  - Uncertainty estimates per prediction
- **Advantages over classical approach:**
  - No continuum removal required — the CNN locks onto diagnostic band shapes and relative proportions rather than absolute values
  - Implicitly handles space weathering — detected olivine depletion in S-types vs. Q-types consistent with known weathering effects (may reduce need for separate Stage 4 correction)
  - Handles clinopyroxene as a third endmember (classical BAR analysis only distinguishes OL vs. OPX)
  - Processes spectra in milliseconds — suitable for batch processing entire survey datasets
- **Limitations:** Currently validated only for S*-complex (olivine-pyroxene-rich) asteroids. Does not apply to C-type, M-type, or other featureless spectra.
- **Source:** [GitHub repository](https://github.com/Sirrah91/Asteroid-spectra) — code and training data publicly available
- **Advanced option (Tang et al. 2025):** A Transformer-based autoencoder network (AE-Trans) achieves even higher accuracy for spectral unmixing by leveraging multi-head self-attention mechanisms. This architecture also provides albedo estimation from spectra (AAE-Net), which could supplement NEOWISE albedo for objects without thermal observations.

### Stage 3: Meteorite Analog Curve Matching

- **Input:** Asteroid spectrum (any wavelength range, de-weathered where Stage 1B was applied) + RELAB/USGS spectral libraries
- **Method:**
  - Resample lab and asteroid spectra to common wavelength grid
  - Apply free scaling and optional slope adjustment (or continuum removal) before matching — without this, albedo/weathering mismatches dominate over diagnostic band-shape similarity
  - Compute weighted mean squared error: `WMSE = (1/N) Σ (x_i - μ_i)² / σ_i²` where σ_i are per-channel uncertainties from Stage 0 (if uncertainties are unavailable, use unweighted MSE or cosine distance after normalization)
  - Compute correlation coefficient: `ρ = cov(X,M) / (σ_X × σ_M)`
  - Compute combined match quality metric. Two options are provided; **Option A is recommended** for its simplicity:
    - **Option A (recommended):** Rank by `WMSE` (lower is better) with `ρ` as a tiebreaker (higher is better). Reject matches with `ρ < 0.90`. This avoids introducing an additional metric and is straightforward to interpret.
    - **Option B (M4AST-style, Popescu et al. 2012):** Compute `Φ_comb = ρ / σ_res`, where `σ_res` is the standard deviation of the residual spectrum after scaling: `σ_res = std(X_i - α × M_i)` with `α` the best-fit scale factor. Higher `Φ_comb` indicates better matches (high correlation relative to residual scatter). This formulation originates from the M4AST tool (Popescu et al. 2012, A&A 544, A130) and is equivalent to a signal-to-noise ratio on the fit quality.
  - Filter matches to meteorite types consistent with taxonomic class, but treat taxonomy as probabilistic — retain some probability mass for surprising but plausible matches (e.g., an X-type matching a carbonaceous chondrite when Mahlke C-probability is non-trivial)
- **Output:** Top-5 best-matching meteorite analogs with fit scores
- **Why it matters:** Meteorite analogs give ground-truth mineralogy. Works with *any* wavelength range, but **full VNIR matching is far more discriminating** than visible-only matching because it uses the diagnostic absorption features.
- **Key constraint for PGM detection:** Iron meteorite spectra are featureless with a reddish slope — they look similar to many other materials in visible-only. NIR matching significantly improves confidence in iron meteorite analog identification.

### Stage 3B: Hapke Radiative Transfer Forward Modeling (high-priority candidates)

For high-priority mining candidates, simple χ² curve matching (Stage 3) can be supplemented with physics-based spectral modeling that provides quantitative mineral abundances rather than just "best-matching meteorite."

**Why add this:** Lawrence (2007, JGR) and Clark (1995, JGR) demonstrated that Hapke radiative transfer intimate mixture models can compute synthetic VNIR spectra from known mineral assemblages at arbitrary grain sizes. By inverting this process, we can solve for modal mineralogy, grain size, and space weathering state simultaneously — information that χ² matching cannot provide.

- **Input:** Asteroid VNIR spectrum + candidate mineral endmember optical constants (n, k)
- **Method:**
  1. Define mineral endmembers: olivine, pyroxene, plagioclase, troilite, Fe-Ni metal (with variable chemistry)
  2. Compute single scattering albedo (SSA) for each endmember at specified grain sizes using Hapke theory
  3. Mix SSAs weighted by cross-sectional area fractions for intimate mixtures
  4. Add submicroscopic iron (SMFe) space weathering via Hapke (2001) formulation
  5. Convert mixed SSA to bidirectional reflectance
  6. Optimize mineral abundances, grain sizes, and SMFe fraction via gradient descent to minimize residual against observed spectrum
- **Output:** Modal mineralogy estimates (wt% with uncertainty ranges), grain size distribution, SMFe abundance (space weathering state), fit quality. **Results should be presented with uncertainty ranges, not point estimates**, due to inherent model degeneracies.
- **Key advantage for PGM candidates:** The spectral effects of coarse-grained Fe-Ni metal on an asteroid surface are similar to the effects of SMFe from space weathering (Lawrence 2007). Simple curve matching cannot distinguish these two physically different scenarios. Hapke modeling treats them as separate parameters, which *can* help distinguish metal-rich surfaces from weathered ones — though degeneracies remain, especially from VNIR alone.
- **Limitations:** Computationally expensive (minutes per asteroid); requires optical constants for endmember minerals; assumes intimate mixing geometry which may not hold for all regolith configurations. **The inversion is ill-posed** — multiple combinations of grain size, mineral abundance, porosity, roughness, and SMFe can produce similar spectra. Use as a physics-consistency check and scenario-fitting tool for top candidates, not as a quantitative wt% oracle.
- **When to use:** Apply only to top-N candidates from the scoring pipeline, not to the full catalog. Particularly valuable for M-type PGM candidates and S-type asteroids where metal fraction is a key question.

### ~~Stage 4: Space Weathering Correction~~ → Moved to Stage 1B

Space weathering correction has been moved to **Stage 1B** in the pipeline, where it is applied conditionally to silicate-bearing spectra *before* band parameter analysis and analog matching. See Stage 1B above for full details.

### Stage 5: PGM Candidate Identification (multi-signal convergence)

PGMs are *never* directly detectable via reflectance spectroscopy. Identification requires convergence of multiple independent signals:

| Signal | Source | What it tells you | Weight |
|---|---|---|---|
| M-complex taxonomy (probabilistic) | Mahlke 2022 `classy` classifier | Spectrally featureless + albedo-resolved → M-type probability | **Primary** — replaces separate X-complex + albedo steps |
| Featureless NIR with reddish slope | MITHNEOS | Consistent with Fe-Ni metal surface | Strong indicator |
| Absence of 1 μm & 2 μm absorption | MITHNEOS | No significant silicates present | Strong indicator |
| Iron meteorite as best analog match | RELAB curve matching / Hapke modeling | Ground-truth composition | Strongest spectral indicator |
| High thermal inertia (Γ > 100 J/m²/s⁰·⁵/K⁻¹) | Dedicated thermophysical modeling (requires shape/spin constraints) | Metal is an excellent thermal conductor; high Γ for size indicates metal-rich surface (Matter et al. 2013) | **Strong** when available — independent of spectroscopy; **not** a standard NEOWISE catalog field |
| NEATM beaming parameter (η) | NEOWISE | η correlates loosely with thermal properties and observing geometry (Harris & Drube 2014); composition inference is weak without strong controls | Weak-to-moderate — useful as supporting indicator only, not a standalone metal detector |
| High radar albedo (>0.25) | Arecibo/Goldstone (where available) | Strong indicator of metallic surface composition — among the best remote constraints, though roughness and regolith state affect the measurement | **Strong** (but sparse) |
| Polarimetric sub-group 1 (low \|Pmin\|, low μc) | Optical polarimetry (Belskaya et al. 2022) | Consistent with iron/stony-iron meteorite analogs; distinguishes from enstatite chondrite M-types | Moderate — independent signal |
| Weak 0.9 μm feature | MITHNEOS | Some M-types show subtle silicate feature — may indicate stony-iron rather than pure metal; modifies PGM grade estimate downward | Modifies estimate |
| S-type with high metal fraction | Korda CNN or Hapke model | S/Q-type asteroids with OC mineralogy contain 10–20% Fe-Ni metal with PGM-bearing grains (Cannon et al. 2023) | **New candidate class** — requires mineral processing factor |

**Confidence tiers for PGM candidates:**
- **Tier 1 (High):** M-type (Mahlke p > 0.7) + featureless NIR + iron meteorite match + radar confirmation → ~70-90% confidence metallic
- **Tier 2 (Medium):** M-type (Mahlke p > 0.5) + featureless NIR + iron meteorite match (no radar) + consistent polarimetry; optionally supported by high thermal inertia from dedicated TPM → ~40-60% confidence
- **Tier 3 (Low):** M-type (Mahlke p > 0.3) + albedo only (no NIR spectra) → ~20-30% confidence
- **Tier 4 (Speculative):** Spectral-only X-complex (no albedo, Mahlke degrades to ambiguous) → <10% confidence, flag as "needs data"
- **Tier S (S-type metal):** S/Q-type with confirmed high metal fraction from Hapke modeling or CNN → PGM-bearing but at lower bulk grade; requires mineral processing cost factor in scoring

---

## 5. Data Sources — The Optimal Combination

The killer combination is: **Gaia DR3** (visible spectra + volume taxonomy) + **NEOWISE** (albedo + diameter + beaming parameter) + **MITHNEOS** (near-IR mineralogy) + **RELAB** (meteorite analog matching + CNN training data) + **JPL SBDB/Horizons** (orbits + ephemerides). Emerging data from **JWST** (6 μm water detection) will augment water-candidate identification.

### 5A. Primary Data Sources

| Priority | Source | What it provides | Coverage | Wavelength / Bands | Access |
|---|---|---|---|---|---|
| **★★★** | **MITHNEOS** (Binzel et al. 2019) | NIR reflectance spectra — the mineralogy layer | >1,000 NEOs, >350 Mars-crossers | **0.8–2.5 μm** (SpeX/IRTF) | [smass.mit.edu/minus.html](http://smass.mit.edu/minus.html) — public, no auth |
| **★★★** | **NEOWISE** (Mainzer et al.) | Diameter + visible albedo (pV) + beaming parameter (η) from NEATM fits — albedo input to Mahlke taxonomy; η as weak thermal indicator | >158,000 minor planets; 1,845 unique NEOs | Thermal IR (3.4, 4.6, 12, 22 μm) | [IRSA catalog](https://irsa.ipac.caltech.edu/data/WISE/NEOWISE_SB/), [PDS SBN](https://sbn.psi.edu/pds/resource/neowisediam.html) |
| **★★★** | **JPL SBDB** | Orbital elements, H magnitude, NEO/PHA flags | All known small bodies (~1.4M) | N/A | [ssd-api.jpl.nasa.gov](https://ssd-api.jpl.nasa.gov/sbdb.api), bulk CSV |
| **★★☆** | **Gaia DR3** (Tanga et al. 2022) | Visible reflectance spectra — the volume triage layer | 60,518 asteroids (16 wavelength bands) | **0.374–1.034 μm** (BP/RP) | [Gaia Archive](https://gea.esac.esa.int/archive/) |
| **★★☆** | **JPL Horizons** | Ephemerides, close-approach predictions | Any cataloged body | N/A | `astroquery.jplhorizons` (Python) |

### 5B. Why each source matters and what it *cannot* do

| Source | Can do | **Cannot do** |
|---|---|---|
| **Gaia DR3** | Coarse taxonomy via Mahlke 2022 (visible-only mode); spectral slope; huge MBA sample for statistical triage; Ch-type hydration feature at 0.7 μm. **NEO coverage caveat:** the 60,518-asteroid sample is predominantly main-belt; NEO coverage is much smaller, limiting Gaia's "volume triage" role specifically for NEO mining candidates. | Full Mahlke taxonomy without albedo (degrades to ambiguous for featureless spectra); Band I center for olivine (minimum at 1.055–1.090 μm is outside range); Band II or BAR; CNN mineral quantification |
| **NEOWISE** | Diameter; visible albedo (pV) — feeds directly into Mahlke taxonomy for M/E/P resolution; beaming parameter (η) as weak thermal indicator; IR albedo ratio (pIR/pV). **Does not provide thermal inertia (Γ) directly** — Γ requires dedicated thermophysical modeling with shape/spin constraints beyond standard NEATM fits. | Spectral features; mineralogy; detailed taxonomy; thermal inertia (without additional modeling) |
| **MITHNEOS** | Full Band I and Band II analysis; BAR computation; olivine/pyroxene ratio; Fe/Ca content; CNN mineral quantification (Korda 2023); detect subtle 0.9 μm silicate features on "featureless" M-types; full Mahlke taxonomy (when combined with visible + albedo) | Large sample coverage (only ~1,000 NEOs); albedo |
| **JPL SBDB** | Orbital elements; H magnitude; diameter (where measured); NEO/PHA classification | Composition; spectral properties |

### 5C. Laboratory Reference Spectra (for curve matching)

| Source | What it provides | Key content for mining | Access |
|---|---|---|---|
| **RELAB** (Brown University) | >15,000 spectra of meteorites, terrestrial rocks, lunar soils, synthetic minerals | Iron meteorites (IIIAB, IVA) for PGM analog matching; CM/CI carbonaceous chondrites for water; ordinary chondrites (H/L/LL) for silicate calibration | [PDS Geosciences Node](https://pds-speclib.rsl.wustl.edu/search.aspx?catalog=RELAB) |
| **USGS Spectral Library v7** | Thousands of mineral reflectance spectra (0.2–200 μm) | Pure mineral reference spectra (olivine, pyroxene, Fe-Ni metal, phyllosilicates) | [USGS](https://doi.org/10.5066/F7RR1WDJ) |
| **Klima Synthetic Pyroxene DB** | Synthetic pyroxene spectra with precisely known compositions | Ground-truth calibration of Band I/II centers vs. Fe/Ca content | [RELAB/Brown](https://sites.brown.edu/relab/synthetic-pyroxene-spectral-database/) |
| **Gaffey meteorite spectra** (via RELAB) | Directional hemispheric reflectance of meteorites | Historical reference dataset used in Gaffey et al. (1993) S-subtype calibration | [RELAB](https://sites.brown.edu/relab/gaffey-collection-adams-diffuse-0-35-2-5-microns/) |

### 5D. Supplementary Data (enhances confidence)

| Source | What it provides | Why it helps | Coverage |
|---|---|---|---|
| **Radar albedo** (Arecibo†, Goldstone) | Surface radar reflectivity | **Strong** metallic composition indicator — high radar albedo (>0.25) strongly suggests metal-dominated surface, though roughness and regolith state also affect measurements. Among the best single remote indicators for PGMs. | Sparse — scattered across individual publications; ~500 objects |
| **SMASS II** (Bus & Binzel 2002) | Visible spectra (0.44–0.92 μm) | Supplements Gaia for objects not in DR3; provides Bus taxonomy classifications | 1,341 asteroids |
| **SMASSIR** (Burbine & Binzel 2002) | NIR spectra (0.90–1.60 μm) | Partial NIR coverage — captures Band I minimum for most compositions but not Band II | 241 asteroids |
| **52-color survey** (Bell et al.) | 52 narrow-band filters (0.8–2.5 μm) | Full VNIR color coverage, though lower resolution than spectroscopy | ~300 asteroids |
| **Asteroid Lightcurve Database (LCDB)** | Rotation periods, amplitude | Fast rotators (P < 2.2 hr) are likely monolithic — easier to mine. Slow rotators with large amplitude may be rubble piles or binaries. | ~30,000 objects |
| **JWST mid-IR spectroscopy** | 6 μm emission feature (molecular H₂O); mid-IR mineralogy | **Unambiguous water detection** — 6 μm H-O-H bending mode cannot be confused with OH or organics (Arredondo et al. 2024). Expanding to 30+ targets in Cycle 2+. | Emerging — currently ~4 asteroids; growing rapidly |
| **Optical polarimetry** (various surveys) | Pmin, αinv polarimetric parameters | Distinguishes metallic M-type sub-populations: iron/stony-iron vs. enstatite chondrite analogs (Belskaya et al. 2022). Independent compositional constraint for PGM candidate confirmation. | Sparse — ~50 M-type asteroids with polarimetric data |
| **Mahlke 2022 taxonomy catalog** | Pre-computed probabilistic classifications for 4,526 asteroids | Ready-to-use classifications in the Mahlke taxonomy for asteroids already observed; reduces classification compute for known objects | 6,038 observations of 4,526 asteroids |

† Arecibo collapsed in 2020; no new radar data from this facility. Goldstone continues operations.

### 5E. Data Ingestion Specifications

The data sources above are described by *what they contain*, but not *how to programmatically ingest them*. This section specifies file formats, query methods, and the cross-survey entity resolution strategy required for implementation.

#### File Formats & Query Methods

| Source | Format | Ingestion method | Key columns / fields | Notes |
|---|---|---|---|---|
| **JPL SBDB** | JSON (API) or CSV (bulk export) | REST API: `ssd-api.jpl.nasa.gov/sbdb_query.api` with field selection; bulk: `ssd.jpl.nasa.gov/data/sbdb/` CSV dumps | `spkid`, `full_name`, `neo`, `pha`, `e`, `a`, `i`, `om`, `w`, `ma`, `epoch`, `H`, `diameter`, `albedo`, `moid` | API returns JSON with configurable field lists; bulk CSV is ~200 MB uncompressed; rate limit: ~5 req/s |
| **JPL Horizons** | Ephemeris tables (ASCII or JSON) | `astroquery.jplhorizons` Python API (already in use) | RA, DEC, delta, r, V, solar phase angle, per epoch | Query by SPK-ID or designation; batch via `id_type='smallbody'` |
| **MITHNEOS** | ASCII two-column text files (wavelength in μm, relative reflectance) | HTTP download from `smass.mit.edu/minus.html`; no API — scrape the index page or use the downloadable tar archive if available | wavelength (col 1), reflectance (col 2); some files include a 3rd column (uncertainty) | Files are named by designation (e.g., `a004179.sp05.txt` for 4179 Toutatis); **designation-based naming requires mapping to SBDB numbers** |
| **NEOWISE** | IPAC table format (pipe-delimited ASCII) or FITS | IRSA TAP service: `irsa.ipac.caltech.edu/TAP`; table `neowiser_p1bs_psd` for single-exposure data; pre-computed diameters from PDS SBN | `desig`, `wmean_w1`, `wmean_w2`, `diam_km`, `pv` (albedo), `beaming`, `diam_err` | Cross-match by packed/unpacked MPC designation; ~158,000 objects |
| **Gaia DR3** | VOTable or CSV via ADQL query | Gaia Archive TAP: `gea.esac.esa.int/tap-server/tap`; table `gaiadr3.sso_reflectance_spectrum` | `source_id`, `number_mp`, `denomination`, 16 reflectance bands (0.374–1.034 μm) | Gaia uses its own `source_id`; `number_mp` maps to MPC number when available; many MBAs lack MPC numbers → match by `denomination` |
| **RELAB** | ASCII spectral files (wavelength vs. reflectance) with associated metadata catalogs | PDS Geosciences Node: `pds-speclib.rsl.wustl.edu`; bulk download via PDS archive; individual spectra via web search interface (not a programmatic API — requires scraping or pre-downloading the catalog) | Wavelength, reflectance; metadata: sample ID, meteorite name, type, grain size range | **No bulk download API** — the practical approach is to download the full RELAB PDS archive (~2 GB) and index it locally. Meteorite type/group comes from the sample catalog, not the spectral files. |
| **SMASS II** | ASCII (wavelength, reflectance, uncertainty) | `smass.mit.edu/smass.html`; bulk download as tar archive | wavelength (μm), reflectance, uncertainty | Similar naming convention to MITHNEOS; 1,447 asteroids |

#### Cross-Survey Entity Resolution

**This is a real engineering problem.** Different surveys use different identifier schemes:

| Survey | Primary identifier | Secondary identifier | Example |
|---|---|---|---|
| JPL SBDB | SPK-ID (integer) | IAU number, designation, name | SPK `2004179`, number `4179`, desig `1989 FB`, name `Toutatis` |
| MITHNEOS | MPC designation (in filename) | — | `a004179.sp05.txt` → number `4179` |
| NEOWISE | Packed MPC designation | — | `J89F00B` → `1989 FB` |
| Gaia DR3 | Gaia `source_id` | `number_mp` (when assigned) | `number_mp = 4179` |
| RELAB | Internal sample ID | Meteorite name + type | `TB-TJM-090` → `Murchison CM2` |
| SMASS II | MPC designation | — | Similar to MITHNEOS |

**Resolution strategy:**

1. **Canonical key:** Use **IAU asteroid number** (integer) as the primary join key. ~640,000 numbered asteroids exist; all major surveys include this for numbered objects.
2. **Unnumbered objects:** Fall back to **packed MPC designation** (7-character string). Use the MPC's packed ↔ unpacked conversion algorithm. The `astropy` or `sbpy` libraries provide this.
3. **Gaia special case:** Gaia `source_id` is unique to Gaia and has no external meaning. Join via `number_mp` for numbered asteroids. For unnumbered Gaia objects, join by `denomination` after normalizing whitespace and formatting.
4. **RELAB (no asteroid identity):** RELAB spectra are meteorite samples, not asteroid observations. The join is conceptual, not by ID — meteorite type/group is mapped to asteroid taxonomy class via the asteroid-meteorite connection table (§10).
5. **Ambiguity handling:** Some asteroids have multiple designations (pre-numbering provisional designations that were later linked). Use JPL SBDB's `alt_des` field or the MPC's orbit identification catalog to resolve aliases. Flag and log any unresolved matches.

**Implementation requirement (FR-1 update):** The ingestion layer must include an `entity_resolver` module that normalizes identifiers across surveys to a canonical form before any data joins. Unit-test entity resolution against a set of 50+ known multi-survey asteroids (e.g., Itokawa = `25143` = `1998 SF36`; Bennu = `101955` = `1999 RQ36`).

---

## 6. User Stories

### US-001: Unified Asteroid Database
**Description:** As a researcher, I want all asteroid data (orbital elements, physical properties, spectral observations, taxonomic classifications) unified into a single queryable dataset so that I can filter and analyze across all dimensions.

**Acceptance Criteria:**
- [ ] Ingest SBDB orbital/physical data, at least one spectral survey (SMASS II), and Mahlke 2022 taxonomy catalog into a unified schema
- [ ] Each asteroid record links orbital data to any available spectral observations via entity resolution across identifier schemes (IAU number, MPC designation, Gaia source_id — see §5E)
- [ ] Entity resolution handles aliases, unnumbered objects, and ambiguous designations with logging
- [ ] Missing data fields are explicitly marked as null, not dropped
- [ ] Schema supports multiple spectral observations per asteroid (different surveys, different epochs)
- [ ] Dataset can be queried by taxonomy class, orbital parameters, and data availability

### US-002: Taxonomic Classifier
**Description:** As a researcher, I want the system to classify asteroids into Mahlke 2022 taxonomy classes from their spectra and albedo so that I can filter by composition category with quantified uncertainty.

**Acceptance Criteria:**
- [ ] Integrates the `classy` Python package (Mahlke et al. 2022) for MCFA-based classification
- [ ] Handles partial spectral coverage (visible-only, NIR-only, combined) and optional albedo input
- [ ] Returns probabilistic class assignment vector across all 17 classes
- [ ] Results match published Mahlke 2022 classifications for asteroids in their catalog (>85% agreement)
- [ ] For asteroids with NEOWISE albedo, M/E/P resolution is applied automatically

### US-003: Meteorite Analog Matching
**Description:** As a researcher, I want each asteroid's spectrum matched against RELAB meteorite spectra so that I can infer detailed mineralogy.

**Acceptance Criteria:**
- [ ] Ingests RELAB meteorite spectral library
- [ ] Computes weighted MSE (or cosine distance when uncertainties unavailable) and correlation coefficient between asteroid and lab spectra, with free scaling/slope adjustment applied before matching
- [ ] Returns top-5 best-matching meteorite analogs with fit metrics
- [ ] Filters matches to only meteorite types consistent with the asteroid's taxonomic class

### US-004: Mineral Composition Analysis
**Description:** As a researcher, I want the system to determine olivine/pyroxene ratios and chemical compositions from silicate-bearing asteroid spectra using both classical band analysis and CNN-based methods so that I can quantify mineralogy with cross-validated results.

**Acceptance Criteria:**
- [ ] **Classical path:** Performs continuum removal, computes band centers via polynomial fitting, computes BAR (Band II area / Band I area), calculates OPX/(OPX+OL) ratio using Cloutis formula, classifies S-type asteroids into Gaffey subtypes S(I)–S(VII)
- [ ] **CNN path:** Integrates Korda et al. (2023) neural network to derive modal abundances (OL/OPX/CPX) and chemical compositions (Fa, Fs, Wo) directly from VNIR spectra
- [ ] When both paths are available, reports agreement/disagreement between classical and CNN results as a quality indicator
- [ ] For high-priority candidates, supports Hapke radiative transfer modeling to derive modal mineralogy, grain size, and space weathering state

### US-005: Mining Value Scorer
**Description:** As a user, I want each asteroid assigned a composite mining score based on estimated recoverable value, material confidence, and mission accessibility so that I can prioritize targets.

**Scoring formula:**
```
Expected Mining Utility ≈ estimated_mass × grade_estimate × recoverability_factor × unit_value × confidence × accessibility
```

Where:
- **estimated_mass** = diameter-derived volume × density prior by taxonomic class (with propagated uncertainty from diameter measurement error and density assumption). Density priors specified in [`tasks/scoring-grade-and-density-review.md`](scoring-grade-and-density-review.md).
- **grade_estimate** = estimated concentration of target material (from taxonomy + analog match + CNN/Hapke mineralogy). Concrete values per taxonomy class and material specified in [`tasks/scoring-grade-and-density-review.md`](scoring-grade-and-density-review.md).
- **recoverability_factor** = fraction of material that is practically extractable (e.g., PGMs in S-type metal grains require concentration/processing → lower factor than PGMs in pure M-type metal; water at ~450 μg/g on S-types → near-zero factor vs. 10–20 wt% hydrated minerals on C-types → high factor). Concrete values and design rationale specified in [`tasks/scoring-recoverability-accessibility-design.md`](scoring-recoverability-accessibility-design.md).
- **unit_value** = configurable commodity price (supports both Earth-market and in-space-utility modes)
- **confidence** = mineralogical assessment confidence (from Mahlke probability vector entropy + number of independent converging signals + wavelength coverage + SNR). Formula, normalization, and weights specified in [`tasks/scoring-confidence-formula-design.md`](scoring-confidence-formula-design.md).
- **accessibility** = mission accessibility score (delta-v, approach frequency, time-of-flight). Function form and parameters specified in [`tasks/scoring-recoverability-accessibility-design.md`](scoring-recoverability-accessibility-design.md).

**Acceptance Criteria:**
- [ ] Score includes asteroid mass estimate derived from diameter + taxonomy-based density prior, with propagated uncertainty
- [ ] Score includes recoverability factor per resource type (e.g., bulk metal vs. trace PGMs vs. hydrated minerals vs. trace water)
- [ ] Scorer supports two modes: **Earth-return value** and **in-space utility value** (water is nearly worthless on Earth but extremely valuable in space)
- [ ] PGM grade estimates use updated Cannon et al. (2023) data (non-chondritic PGM ratios at high Ir) rather than Kargel (1994) assumptions
- [ ] S-type asteroids with confirmed high metal fraction are scored as PGM candidates with a mineral processing cost factor applied
- [ ] Material values are configurable without code changes (YAML/JSON config)
- [ ] Output is a ranked CSV/table with asteroid ID, name, taxonomy (with probability), top analog match, estimated composition, estimated mass, PGM confidence tier, value score, accessibility score, composite score, and next close-approach date
- [ ] Score methodology is documented and reproducible

### US-006: Approach Window Calculator
**Description:** As a user, I want to know when each candidate asteroid will be accessible for a rendezvous mission so that I can plan around launch windows.

**Acceptance Criteria:**
- [ ] Computes ephemerides for candidate asteroids over a configurable time horizon (default: 10 years)
- [ ] Performs Lambert transfer search over launch windows and time-of-flight bounds to identify feasible transfer opportunities
- [ ] Reports for each window: launch date, arrival date, time of flight, departure Δv (from LEO), arrival Δv (for rendezvous), total mission Δv, and optionally Earth-return Δv
- [ ] **Reference orbit:** Δv is computed from a defined LEO parking orbit (default: 200 km circular). The reference must be explicitly stated in all outputs.
- [ ] Supports fast prefiltering by MOID, inclination, synodic period, and Earth-relative encounter velocity before running full Lambert optimization
- [ ] Uses existing `distance.py` Horizons integration as foundation
- [ ] Outputs approach date, minimum distance, and estimated delta-v for each window

---

## 7. Technical Context

### Existing Code
| File | Purpose | Reuse plan |
|---|---|---|
| `get_bodies.py` | Queries JPL SBDB API for asteroid data | Expand to bulk-ingest; add spectral data joins |
| `kepler.py` | Solves Kepler's equation, computes orbital positions | Reuse for rough distance estimates; supplement with Horizons for precision |
| `distance.py` | Computes ephemerides via `astroquery.jplhorizons` | Reuse as-is for US-006; refactor `Distance` class for batch processing |
| `too_close.py` | Plots distance to celestial bodies | Reference for visualization patterns |
| `sbdb_query_results.csv` | SBDB bulk export (orbital elements, diameters, NEO flags) | Primary orbital data source |
| `EDA.ipynb` | Exploratory analysis of SBDB data | Reference for data quality issues (missing `moid`, diameter) |

### Proposed Architecture
```
near-earth/
├── data/                       # Raw + processed data
│   ├── sbdb/                   # JPL SBDB exports
│   ├── spectra/                # Spectral survey data (SMASS, S3OS2, etc.)
│   ├── relab/                  # RELAB meteorite lab spectra
│   └── processed/              # Unified dataset outputs
├── prospector/                 # Core library
│   ├── ingest/                 # Data ingestion modules per source
│   │   ├── sbdb.py
│   │   ├── smass.py
│   │   └── relab.py
│   ├── spectral/               # Spectral analysis pipeline
│   │   ├── preprocessing.py    # Stage 0: normalization, QC, telluric masking, SNR
│   │   ├── taxonomy.py         # Stage 1: Mahlke 2022 classifier (wraps classy)
│   │   ├── weathering.py       # Stage 1B: Space weathering correction (conditional)
│   │   ├── band_analysis.py    # Stage 2: Classical band parameter extraction
│   │   ├── cnn_mineral.py      # Stage 2B: CNN mineral quantification (Korda 2023)
│   │   ├── curve_match.py      # Stage 3: WMSE / correlation matching
│   │   └── hapke_model.py      # Stage 3B: Hapke radiative transfer forward modeling
│   ├── scoring/                # Mining value & accessibility scoring
│   │   ├── material_value.py
│   │   └── accessibility.py
│   └── ephemeris/              # Orbital mechanics (refactored from existing)
│       ├── kepler.py
│       └── horizons.py
├── notebooks/                  # Analysis notebooks
├── tasks/                      # PRDs
├── distance.py                 # (existing, to be refactored into prospector/)
├── kepler.py                   # (existing, to be refactored into prospector/)
└── get_bodies.py               # (existing, to be refactored into prospector/)
```

### Tech Stack
- **Python 3.10+** (already in use)
- **NumPy / SciPy** (already in use — Newton solver, array math)
- **Pandas** (already in use — data manipulation)
- **Astroquery** (already in use — JPL Horizons)
- **scikit-learn** (new — PCA for taxonomy fallback, curve fitting)
- **classy** (new — Mahlke 2022 taxonomic classification; wraps MCFA model)
- **PyTorch or TensorFlow** (new — CNN mineral quantification via Korda 2023 model; Hapke model optimization)
- **Matplotlib** (already in use — visualization)
- **SQLite or DuckDB** (new — unified queryable dataset; lightweight, no server)

---

## 8. Functional Requirements

- **FR-0:** The system must normalize and quality-check all ingested spectra (common wavelength grid, standard normalization, telluric masking, SNR estimation) before any downstream analysis.
- **FR-1:** The system must ingest SBDB data, NEOWISE albedo/thermal data, and at least two spectral surveys into a unified schema keyed by asteroid number/designation. The ingestion layer must include entity resolution logic that normalizes identifiers across surveys (SPK-ID, IAU number, packed/unpacked MPC designation, Gaia source_id) to a canonical form before data joins (see §5E).
- **FR-2:** The system must classify asteroids using the Mahlke et al. (2022) taxonomy via the `classy` package, producing probabilistic class assignments from available spectral data and albedo, supporting partial wavelength coverage. Bus-DeMeo-equivalent labels must be available for cross-referencing.
- **FR-2B:** The system must apply space weathering correction (Brunetto et al. 2006) conditionally to silicate-bearing spectra (S/Q/V/A-complex) before band parameter analysis and analog matching, and must not apply corrections to featureless or C-complex spectra.
- **FR-3:** The system must match asteroid spectra against RELAB meteorite spectra using properly labeled fit metrics (weighted MSE or cosine distance, not mislabeled χ²) with free scaling/slope adjustment, and return ranked analog matches.
- **FR-4:** For asteroids with silicate absorption features, the system must compute olivine/pyroxene ratios using both classical band parameter analysis (with temperature correction and documented calibration domain) and CNN-based mineral quantification (Korda et al. 2023), reporting agreement between methods. BAR-derived mineral ratios must include systematic uncertainty estimates.
- **FR-4B:** For high-priority candidates, the system must support Hapke radiative transfer modeling to derive modal mineralogy estimates with uncertainty ranges, grain size, and space weathering state.
- **FR-5:** The system must assign a composite mining score using the formula: `estimated_mass × grade × recoverability_factor × unit_value × confidence × accessibility`. The scorer must support both Earth-return and in-space-utility economic models. S-type asteroids with high metal fraction must be scored as PGM candidates with mineral processing cost factor.
- **FR-6:** The system must compute future approach windows via Lambert transfer search with defined reference orbit (default: 200 km LEO), reporting departure Δv, arrival Δv, total Δv, and time of flight.
- **FR-7:** Material value weights, density priors by taxonomy, and recoverability factors must be configurable without code changes (e.g., via a YAML/JSON config file).
- **FR-8:** The system must output a ranked candidate list as CSV with all scoring components, estimated mass, PGM confidence tier, Mahlke taxonomy probabilities, and supporting data.

---

## 9. Non-Goals

- **Real-time data streaming** — this is a batch analysis tool, not a live monitoring system.
- **Mission trajectory optimization** — we estimate delta-v and approach windows, but do not plan actual mission trajectories.
- **Full mid-infrared spectral analysis** — initial scope is VNIR (0.35–2.5 μm) plus targeted use of NEOWISE thermal parameters (albedo, beaming parameter) and JWST 6 μm water detections where available. Comprehensive mid-IR spectral modeling (e.g., Christiansen feature analysis) is out of scope.
- **Dedicated thermophysical modeling (TPM)** — deriving thermal inertia (Γ) from NEOWISE data requires shape/spin constraints and is out of initial scope. Where published Γ values exist in literature, they can be ingested as supplementary data.
- **Proprietary or paywalled data** — all data sources must be publicly accessible.
- **Web UI** — initial product is a Python library + CLI + notebooks. A web frontend is a future phase.
- **Asteroid mass estimation from gravitational perturbation** — out of scope; we use diameter + assumed density from taxonomy.

---

## 10. Design Considerations

### Spectral Data Heterogeneity
Different surveys cover different wavelength ranges at different resolutions. The system must:
- Normalize all spectra to a common wavelength grid via interpolation
- Track which wavelength range is available per observation (metadata)
- Degrade classification gracefully when only partial coverage exists (e.g., visible-only spectra can still do Bus taxonomy but not band parameter analysis)

### Confidence Scoring
Not all asteroids have equal data quality. The confidence score should account for:
- Mahlke taxonomy probability vector entropy (low entropy = high-confidence classification)
- Number of independent spectral observations
- Wavelength coverage (visible-only < visible+NIR < full VNIR)
- Signal-to-noise ratio of the spectra
- Agreement between taxonomy and analog matching
- Agreement between classical band analysis and CNN mineral quantification (when both available)
- Availability of supplementary data (albedo, radar, thermal inertia, polarimetry)
- For PGM candidates: number of converging signals from the multi-signal table (Stage 5)

### Asteroid-Meteorite Connection
The spectral match to a meteorite type gives the strongest compositional constraint. The key mappings are:

| Taxonomy | Meteorite analog | Inferred materials |
|---|---|---|
| M-type / Xk | Iron meteorites (IIIAB, IVA) | Fe-Ni metal, PGMs (Cannon et al. 2023: non-chondritic ratios), Co |
| S-type (high metal) | H/L ordinary chondrites | Fe-Ni metal (10–20%), olivine, pyroxene; **PGM-bearing metal grains** separable via mineral processing (Cannon et al. 2023) |
| C-type / B-type | CI/CM carbonaceous chondrites | Water (up to 20 wt%), organics, clays |
| V-type | HED achondrites (eucrites/diogenites) | Pyroxene-rich, limited metal |
| A-type | Pallasites, brachinites | Olivine-dominated, some Fe-Ni |
| D-type / P-type | Tagish Lake, IDPs | Organics, volatiles |

---

## 11. Risks and Open Questions

### Risks
- **Spectral coverage gaps:** Many NEOs lack spectral observations entirely. The scored candidate list may be biased toward well-studied asteroids. Mitigation: rank unclassified NEOs separately as "unknown — needs observation."
- **PGM detection is always indirect:** No reflectance spectroscopy method can directly confirm PGMs. All M-type identifications are probabilistic. The system must communicate this uncertainty clearly. Cannon et al. (2023) show PGM concentrations may be lower than historically assumed.
- **M-type heterogeneity:** Not all M-type asteroids are metallic. Rivkin et al. (2000) found hydration features on 10/27 M-types; Fornasier et al. (2010) and Hardersen et al. (2011) found silicate features on many. Polarimetry (Belskaya et al. 2022) suggests at least two sub-populations with different meteorite analogs. Multi-signal convergence is essential.
- **Data format variability:** Each spectral survey uses slightly different file formats, wavelength calibrations, and normalization. Robust parsing and validation is critical.
- **CNN model domain limits:** Korda et al. (2023) CNN is validated only for silicate-rich (S*-complex) spectra. Applying it to C-type, M-type, or featureless spectra will produce unreliable results. The system must enforce domain checks.
- **Hapke model degeneracy:** Multiple combinations of grain size, mineral abundance, and SMFe can produce similar spectra. Results should be presented with uncertainty ranges, not point estimates.
- **Third-party dependency viability:** Two critical pipeline components depend on external open-source packages that must be validated before committing to the architecture:
  - **`classy` (Mahlke 2022 taxonomy):** Requires verification that the package installs on Python 3.10+, that its MCFA model weights are bundled or downloadable, and that its API accepts raw spectrum arrays (not just internal catalog lookups). If `classy` is unmaintained or incompatible, the fallback is a simpler PCA + Gaussian mixture classifier trained on Bus-DeMeo data using `scikit-learn`.
  - **Korda et al. (2023) CNN (`Asteroid-spectra` repo):** Requires verification that the trained model weights are included in the repository, that the code runs on current PyTorch/TensorFlow versions, and what input format the model expects (raw spectrum array? specific wavelength grid? normalization convention?). The repository may be research-grade code without a stable API. If it cannot be integrated as a library, the fallback is to re-implement the architecture (2-layer CNN) and retrain on RELAB data, which adds significant scope.
  - **Mitigation:** Validate both packages in Phase 1 (see §12) before building pipeline stages that depend on them. Document the actual API signatures, input/output formats, and any required preprocessing.

### Open Questions

**Scoring model specification (prerequisite tasks — must be resolved before implementation):**
- **~~What are the grade estimates and density priors by taxonomy class?~~** → Specified in [`tasks/scoring-grade-and-density-review.md`](scoring-grade-and-density-review.md). Literature review task: derive concentrations from meteorite data (Wasson 1999, Cannon et al. 2023, Alexander et al. 2012, Carry 2012).
- **~~What recoverability factors and accessibility function should we use?~~** → Specified in [`tasks/scoring-recoverability-accessibility-design.md`](scoring-recoverability-accessibility-design.md). Design task: define recoverability table and delta-v → score function, reviewing Sonter 1997, Cannon & Britt 2019, NHATS criteria.
- **~~What is the confidence score formula?~~** → Specified in [`tasks/scoring-confidence-formula-design.md`](scoring-confidence-formula-design.md). Design task: define formula combining taxonomy entropy, wavelength coverage, SNR, method agreement, and supplementary data availability.
- **~~How should we score S-type asteroids for PGMs?~~** → Addressed across grade review (bulk vs. metal-phase PGM concentrations) and recoverability design (mineral processing cost factor). See both tasks above.

**Remaining open questions:**
- **Should we incorporate radar albedo data?** Radar data from Arecibo/Goldstone would significantly improve M-type confirmation, but the data is scattered across individual publications rather than a single catalog.
- **How should we weight in-space utility vs. Earth-return value?** Water is nearly worthless on Earth but extremely valuable in space. Should the default scoring assume an Earth-return or in-space-use economic model? (Addressed in recoverability/accessibility task as a design decision.)
- **Should we include main-belt asteroids or restrict to NEOs?** NEOs are more accessible but fewer in number. Main-belt asteroids are abundant but require higher delta-v.
- **What minimum spectral S/N threshold should we require** for classification to be considered reliable? (Addressed in confidence formula task as `SNR_good` parameter.)
- **Should we incorporate JWST 6 μm data as it becomes available?** JWST Cycle 2+ proposals will expand unambiguous water detection to 30+ asteroids. This data could significantly improve water-candidate scoring but requires mid-IR processing capabilities.
- **~~Should we use Mahlke 2022 taxonomy exclusively or maintain Bus-DeMeo compatibility?~~** **Resolved:** Use Mahlke 2022 as primary taxonomy with Bus-DeMeo-equivalent labels maintained via `classy` mapping layer for cross-referencing with published literature.

---

## 12. Implementation Phases

The pipeline described above has ~8 stages, 3 ML/modeling approaches, and 6+ data sources. Attempting to build everything at once is infeasible. This section defines a phased delivery where each phase produces a usable, testable artifact.

### Phase 1 — Minimum Viable Pipeline (MVP)

**Goal:** Produce a ranked list of NEO mining candidates from publicly available data, using the simplest defensible version of each pipeline component.

**Scope:**

| Component | What's included | What's deferred |
|---|---|---|
| **Data ingestion** | SBDB (bulk CSV), MITHNEOS (ASCII spectra), NEOWISE (albedo + diameter from PDS SBN pre-computed table) | Gaia DR3, SMASS II, RELAB bulk ingest, supplementary sources |
| **Entity resolution** | IAU number + packed MPC designation matching (covers >95% of MITHNEOS + NEOWISE objects) | Gaia source_id resolution, alias deduplication |
| **Spectral preprocessing** | Resample to common grid, normalize at 550 nm, mask telluric bands | SNR estimation heuristics, phase angle metadata |
| **Taxonomy** | `classy` package — run Mahlke 2022 classification on available spectra + NEOWISE albedo | Feature flag detection, Bus-DeMeo mapping layer |
| **Mineralogy** | Classical band parameter analysis (Stage 2): continuum removal, Band I/II centers, BAR, Dunn et al. calibration for S(IV) | CNN (Korda), Hapke modeling, space weathering correction |
| **Scoring** | Simplified: `estimated_mass × grade × unit_value × accessibility`. Grade from taxonomy lookup table (§scoring-grade-and-density-review). Accessibility from precomputed MOID + Tisserand parameter as proxy (no Lambert solver). | Full confidence scoring, recoverability factors, Lambert transfer search, PGM convergence |
| **Output** | Ranked CSV with: asteroid ID, name, taxonomy class, BAR-derived mineralogy (where available), estimated mass, simple score, MOID, next close approach | Full scoring breakdown, PGM tiers, approach windows |

**Acceptance criteria for MVP:**
- [ ] Ingests SBDB + MITHNEOS + NEOWISE into a unified SQLite/DuckDB database
- [ ] Classifies ≥500 NEOs via `classy`
- [ ] Computes BAR and mineral ratios for S-complex asteroids with MITHNEOS spectra
- [ ] Produces a ranked list of ≥100 scored candidates
- [ ] Validates against known asteroids: Itokawa (S-type, high confidence) and Bennu (B-type) should rank sensibly

**Key risk to retire in Phase 1:** Validate that `classy` and its dependencies install and run on Python 3.10+ (see §11, Risks). If `classy` is broken or unmaintained, the fallback is a simpler PCA-based Bus-DeMeo classifier from `scikit-learn`.

#### Phase 1 Task Dependencies

The dependency graph below defines the execution order for Phase 1 beads. Tasks at the same depth can run in parallel. Each task produces one module or one config artifact.

```
Phase 0 (infrastructure):
  project-setup ─────────────────────────────────────┐
  (pyproject.toml, prospector/ package layout,        │
   tests/, pytest config, .gitignore updates)         │
                                                      ▼
Phase 1a (data layer):                          schema-design
  ┌─────────────────────────────────────────────(defines tables)
  │                                                   │
  ▼                    ▼                    ▼          │
sbdb-ingest      neowise-ingest     validate-classy   │
  │                    │                    │          │
  └────────┬───────────┘                    │          │
           ▼                                │          │
     entity-resolver                        │          │
           │                                │          │
           ▼                                ▼          │
     mithneos-ingest ──────────► spectral-preprocessing│
                                            │          │
Phase 1b (analysis):                        ▼          │
                                   taxonomy-module     │
                                            │          │
                                   band-analysis       │
                                            │          │
Phase 1c (scoring):                         ▼          │
                              scoring-config (YAML)◄───┘
                                            │
                                       scorer
                                            │
                                       output-csv
```

**Task type annotations:**
- `project-setup`, `schema-design`: implementation (infrastructure)
- `validate-classy`: research (spike — install, test API, document findings)
- `sbdb-ingest`, `neowise-ingest`, `mithneos-ingest`, `entity-resolver`: implementation (data engineering)
- `spectral-preprocessing`, `taxonomy-module`, `band-analysis`: implementation (scientific computing)
- `scoring-config`: research (populate YAML from literature — see `scoring-grade-and-density-review.md`)
- `scorer`, `output-csv`: implementation (application logic)

#### Unified Database Schema (Phase 1)

All ingest modules write to this shared schema. SQLite or DuckDB — chosen at `project-setup` time.

```sql
-- Core identity table. All other tables join here.
CREATE TABLE asteroids (
    asteroid_id    INTEGER PRIMARY KEY,  -- IAU number (canonical key)
    designation    TEXT,                  -- packed MPC designation (for unnumbered objects)
    name           TEXT,                  -- common name (Itokawa, Bennu, etc.)
    full_name      TEXT,                  -- SBDB full_name field
    neo            BOOLEAN,
    pha            BOOLEAN
);

-- Orbital elements from SBDB
CREATE TABLE orbits (
    asteroid_id    INTEGER PRIMARY KEY REFERENCES asteroids(asteroid_id),
    epoch          REAL,                  -- Julian date
    e              REAL,                  -- eccentricity
    a              REAL,                  -- semi-major axis (AU)
    i              REAL,                  -- inclination (deg)
    om             REAL,                  -- longitude of ascending node (deg)
    w              REAL,                  -- argument of perihelion (deg)
    ma             REAL,                  -- mean anomaly (deg)
    q              REAL,                  -- perihelion distance (AU)
    H              REAL,                  -- absolute magnitude
    moid           REAL,                  -- Earth MOID (AU)
    diameter       REAL,                  -- diameter (km), NULL if unknown
    diameter_sigma REAL
);

-- Physical properties from NEOWISE
CREATE TABLE physical_properties (
    asteroid_id    INTEGER PRIMARY KEY REFERENCES asteroids(asteroid_id),
    albedo_pv      REAL,                  -- visible geometric albedo
    albedo_pv_err  REAL,
    diameter_km    REAL,                  -- NEOWISE-derived diameter
    diameter_err   REAL,
    beaming_eta    REAL,                  -- NEATM beaming parameter
    source         TEXT                   -- 'NEOWISE', 'IRAS', etc.
);

-- Spectral observations (one asteroid can have many)
CREATE TABLE spectra (
    spectrum_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    asteroid_id    INTEGER REFERENCES asteroids(asteroid_id),
    survey         TEXT NOT NULL,          -- 'MITHNEOS', 'SMASS', 'Gaia', etc.
    wavelengths    BLOB,                  -- numpy array serialized
    reflectance    BLOB,                  -- numpy array serialized
    uncertainty    BLOB,                  -- numpy array serialized, NULL if unavailable
    wl_min         REAL,                  -- min wavelength (μm) for quick filtering
    wl_max         REAL,                  -- max wavelength (μm)
    normalized     BOOLEAN DEFAULT FALSE, -- has Stage 0 been applied?
    snr_estimate   REAL,
    quality_flag   TEXT                   -- 'good', 'low_snr', 'partial'
);

-- Taxonomy results from classy
CREATE TABLE taxonomy (
    asteroid_id    INTEGER PRIMARY KEY REFERENCES asteroids(asteroid_id),
    primary_class  TEXT,                  -- e.g., 'S', 'M', 'C'
    primary_prob   REAL,                  -- probability of primary class
    prob_vector    BLOB,                  -- full 17-class probability vector (JSON or numpy)
    classifier     TEXT,                  -- 'classy_mahlke2022', 'bus_demeo_fallback'
    input_coverage TEXT                   -- 'vnir', 'vis_only', 'nir_only', 'albedo_only'
);

-- Band analysis results (S-complex only)
CREATE TABLE band_analysis (
    asteroid_id    INTEGER PRIMARY KEY REFERENCES asteroids(asteroid_id),
    band1_center   REAL,                  -- Band I center (μm)
    band2_center   REAL,                  -- Band II center (μm)
    bar            REAL,                  -- Band Area Ratio
    ol_opx_ratio   REAL,                  -- ol/(ol+px) from Dunn calibration
    fa_mol_pct     REAL,                  -- fayalite mol%
    fs_mol_pct     REAL,                  -- ferrosilite mol%
    gaffey_subtype TEXT,                  -- S(I) through S(VII)
    calibration    TEXT                   -- 'dunn2010', 'gaffey1993_zone'
);

-- Scoring output
CREATE TABLE scores (
    asteroid_id       INTEGER PRIMARY KEY REFERENCES asteroids(asteroid_id),
    estimated_mass_kg REAL,
    grade_estimate    REAL,               -- concentration of target material
    target_material   TEXT,               -- 'PGM', 'water', 'iron', etc.
    unit_value        REAL,
    accessibility     REAL,               -- 0-1 score from MOID/Tisserand proxy
    composite_score   REAL,
    score_mode        TEXT                -- 'earth_return' or 'in_space'
);
```

**Conventions:** BLOB columns store numpy arrays via `array.tobytes()` / `np.frombuffer()`. For DuckDB, use native array types instead. The schema is deliberately flat for Phase 1 — normalization and views can be added in Phase 2.

### Phase 2 — Full Spectral Pipeline + Scoring

**Goal:** Add all spectral analysis methods and the full scoring formula.

**Scope (additions over Phase 1):**
- Gaia DR3 + SMASS II ingestion (volume triage layer)
- RELAB bulk ingest + meteorite analog curve matching (Stage 3)
- CNN mineral quantification via Korda et al. 2023 (Stage 2B) — requires validating the Korda model runs
- Space weathering correction (Stage 1B)
- Full confidence scoring formula (per `scoring-confidence-formula-design.md`)
- Recoverability factors (per `scoring-recoverability-accessibility-design.md`)
- Lambert transfer search for approach windows (US-006, using `distance.py` + Horizons as foundation)
- PGM multi-signal convergence table (Stage 5)

**Acceptance criteria for Phase 2:**
- [ ] Confidence score produces sensible orderings (Itokawa > random Gaia-only MBA)
- [ ] Classical and CNN mineral results reported side-by-side for S-complex asteroids
- [ ] Lambert solver identifies approach windows for top-20 candidates
- [ ] Full scoring formula applied with all six components

### Phase 3 — Advanced Modeling & Extensions

**Goal:** Add computationally expensive or research-grade capabilities for high-priority candidates.

**Scope (additions over Phase 2):**
- Hapke radiative transfer modeling (Stage 3B) for top-N candidates
- Phase curve analysis (H, G1, G2) from MPC photometry (per commercial cross-reference recommendations)
- Thermal depletion filter for water targets (Toliou et al. 2021)
- Spin-barrier filter from LCDB rotation periods
- SsODNet ingestion as supplementary data source
- Debiased population prior (Granvik et al. 2018) for uncharacterized NEOs
- JWST 6 μm water detection integration (as data becomes available)

**Phase 3 is explicitly aspirational** — it depends on Phase 2 being complete and validated, and individual items can be prioritized independently.
