<!-- role: reference -->
<!-- phase: n/a — recommendations feed Phase 2/3 scope, not a direct task source -->

# Commercial Asteroid Mining: Cross-Reference of Industry Approaches vs. Our Prospector Pipeline

## Executive Summary

This addendum expands the prior technical specification audit by mapping what active asteroid mining companies are publicly signaling about their target identification and characterization methods against our prospector pipeline design. The commercial sector divides into two distinct camps — PGM-return-to-Earth players (AstroForge) and water-for-ISRU players (Karman+, TransAstra) — and each camp uses fundamentally different prospecting strategies. Our pipeline is well-positioned relative to both, but has gaps that the commercial sector's hard-won lessons expose.

---

## 1. The Active Commercial Landscape (as of early 2026)

Five companies have active programs or missions relevant to asteroid prospecting. The defunct Planetary Resources and Deep Space Industries provide additional lessons-learned.

### AstroForge (Huntington Beach, CA) — PGMs from M-type NEAs

AstroForge is targeting platinum group metals from metallic near-Earth asteroids. Their target selection methodology, to the extent publicly disclosed, works as follows:

- **Target pool:** They maintain a shortlist of approximately five M-type NEA candidates (20–300m diameter), selected from the ~3–5% of NEAs estimated to be M-type.
- **Selection criteria for 2022 OB5 (Odin target):** Selected "in part due to its suspected metallic composition, and its proximity to Earth during the spacecraft encounter." The metallic classification appears to be inferred from existing ground-based spectral data and albedo, not from proprietary observations.
- **Characterization approach:** The Odin mission (launched Feb 2025, contact lost ~20 hours after deployment) was equipped only with an optical camera — no spectrometer. The mission goal was to "determine if the asteroid is metallic" from high-resolution imagery, including surface morphology and crater characteristics. This is a coarse characterization method compared to spectroscopy.
- **Meteorite analog work:** AstroForge partners with Colorado School of Mines on meteorite analysis, determining that metallic-type asteroids contain "up to 16,000 times more PGM concentrations than found here on Earth." This figure appears to use pre-Cannon (2023) scaling.
- **AI tracking:** They use AI-based autonomous optical navigation to track and lock onto asteroids during approach. This is an operations capability, not a prospecting technique.
- **Secrecy posture:** CEO Matt Gialich has explicitly stated they may not disclose future high-value targets: "When we find this mythical asteroid that's purely platinum…am I going to tell the world which one it is? Probably not." This creates an asymmetry where the best commercially validated targets may be invisible to open science.

**Key insight for our pipeline:** AstroForge is operating with minimal remote characterization — essentially taxonomy + albedo + orbital accessibility, then fly there with a camera. Their Vestri mission (planned 2026) aims to land on a target asteroid and take composition measurements in situ. This suggests they view remote spectral mineralogy as insufficient for M-type characterization and are betting on direct measurement. Our pipeline should account for this reality: for M-type candidates, the confidence ceiling from remote sensing is inherently low.

### Karman+ (Denver, CO) — Water from carbonaceous NEAs

Karman+ represents the closest analog to what our pipeline is attempting to build, and their publicly disclosed methodology is remarkably detailed.

- **Meta-survey approach:** They explicitly describe building "a meta-survey that brings disparate surveys and observations together" combined with "advanced statistical inference techniques" to create "a probabilistic model of all near-Earth asteroids." This is architecturally identical to our pipeline design.
- **The Compono leaderboard:** Dr. Lauri Siltala, their asteroid characterization specialist, has published details of their target selection tool. It queries JPL SBDB, SsODNet (the IMCCE Solar System Open Database Network), and the astorb database, cross-referencing orbital parameters, physical properties, taxonomic classifications, NEOWISE albedos, radar observations (Goldstone schedule), and lightcurve data. When centralized databases are incomplete, they manually query prolific survey teams for unpublished data.
- **Probabilistic rather than deterministic targeting:** Their stated philosophy is "instead of spending years identifying and analyzing one specific asteroid, we operate probabilistically by targeting unknown objects that fit a certain statistical profile." This is a deliberate choice to optimize for expected value across a class rather than certainty about a single target.
- **Phase curve analysis:** They developed a custom Python tool ("Phazon") that fetches all available photometric observations from the Minor Planet Center for arbitrary asteroids, corrects for distance, and computes phase functions. Phase curve parameters (G1, G2 in the H,G1,G2 system) constrain albedo, surface roughness, and taxonomy without spectroscopy — a technique our pipeline does not currently include.
- **Water depletion modeling:** They apply the Toliou et al. (2021) probabilistic model for near-Sun dwell times to exclude NEAs whose orbits bring them close enough to the Sun that water would have been thermally depleted. This is a critical filter that our pipeline's recoverability/accessibility design does not address.
- **Rotation period filter:** They require rotation periods slower than the ~2.2 hour spin barrier to maximize the likelihood of finding loose unconsolidated regolith, which is essential for excavation. Fast rotators tend to be monolithic and have lost their loose surface material.
- **Hyperspectral and ML:** Their co-founder Daynan Crull has a remote sensing and ML background and has discussed applying hyperspectral infrared signatures for water detection, spectral classification approaches using ML, and working with the 3 μm water feature — though specifics of their ML models are not public.

**Key insight for our pipeline:** Karman+ has operationalized several techniques that our specs don't include: phase curve analysis for albedo/taxonomy refinement, thermal depletion modeling for water targets, and the spin-barrier filter for operational feasibility. These should be added. Their reliance on SsODNet (a European aggregation service that centralizes asteroid physical properties from multiple literature sources) is also notable — our pipeline should ingest this in addition to SBDB.

### TransAstra (Los Angeles, CA) — Optical Mining + Sutter Survey

TransAstra's prospecting approach centers on detection capability rather than characterization:

- **Sutter Survey telescopes:** A proprietary telescope system using "Optimized Matched Filter Tracking" (OMFT) designed to detect very faint, fast-moving objects. Their stated goal is to "prospect thousands of near-Earth asteroids every year" and specifically to find small (5–10m diameter) targets that current surveys miss. This is discovery-focused, not composition-focused.
- **Optical mining compatibility:** They target carbonaceous asteroids for water extraction via concentrated solar energy ("optical mining"), and specifically avoid M-types due to their harder surfaces. Their target characterization needs are less demanding than PGM prospecting — they primarily need to confirm the target is carbonaceous (taxonomy) and slow-rotating (lightcurve) with accessible orbits.

### Origin Space (Shenzhen, China) — NEA Survey Infrastructure

China's Origin Space has deployed two relevant assets:

- **Yangwang-1:** A visible/UV space telescope launched in 2021, described as "the only large-field ultraviolet space probe in orbit around the world." This provides NEA detection and potentially broadband color classification from space, avoiding atmospheric limitations.
- **NEO-1:** A technology demonstration satellite testing capture and observation capabilities in LEO.
- **CNSA Tianwen-2:** While government-led (not commercial), China's Tianwen-2 mission (launched May 2025) targets asteroid Kamo'oalewa with a visible/near-infrared imaging spectrometer, thermal emission spectrometer, and other instruments. This will produce the kind of detailed compositional data our pipeline would ingest.

### Planetary Resources (defunct, IP now public domain) — Lessons Learned

Planetary Resources' approach provides a cautionary template:

- **Arkyd-100 design:** 40-color hyperspectral imaging in visible to near-infrared, plus midwave-infrared (MWIR, 3–5 μm) imaging. The MWIR capability was specifically designed for water detection on asteroids — the 3 μm OH/H₂O band falls squarely in this range. This was the most scientifically ambitious prospecting sensor suite any commercial entity has proposed.
- **Arkyd-6 demonstration:** Successfully operated the first commercial MWIR imager in space (launched Jan 2018), validating the sensor technology before the company folded.
- **Arkyd-301 concept:** Multiple spacecraft dispatched to different asteroids from a single launch, each carrying MWIR imagers and miniprobes that would burrow into the asteroid surface.
- **IP now public:** ConsenSys made all Planetary Resources intellectual property public domain in May 2020, meaning their sensor designs and characterization algorithms are theoretically available.

**Key insight:** Planetary Resources was the only commercial entity to design a multi-band space-based spectrometer specifically optimized for asteroid resource characterization. Their bankruptcy occurred due to funding, not technical failure. The MWIR approach for water detection was validated in orbit.

### Asteroid Mining Corporation (London, UK) — Robotic Prospecting

AMC is developing a satellite to prospect NEAs for mining candidates with plans to "commercialise this data set." Their SCAR-E walking robot is designed for physical surface exploration. Their near-term approach is hardware-focused (robotics) rather than remote-characterization-focused.

---

## 2. Commercial Methods vs. Our Pipeline: Gap Analysis

### What the industry is doing that we're not

| Commercial technique | Who uses it | Our pipeline status | Priority to add |
|---|---|---|---|
| Phase curve analysis (H,G1,G2) for albedo/taxonomy | Karman+ | **Missing** | High — cheap to implement, large data availability from MPC |
| Thermal depletion modeling (near-Sun dwell time) | Karman+ | **Missing** | High for water targets — eliminates false positives |
| Spin-barrier filter (ω > 2.2 hr = monolithic) | Karman+ | **Missing** | High for operational feasibility |
| SsODNet data aggregation | Karman+ | **Missing** — only SBDB specified | Medium — superior European data aggregation |
| MWIR / 3–5 μm water detection | Planetary Resources (defunct) | N/A (no space sensor) | Informational — validates 3 μm importance |
| In-situ visual classification from flyby imagery | AstroForge | **Out of scope** | N/A — our pipeline is remote-sensing only |
| Probabilistic population-level targeting | Karman+ | **Partially covered** via Mahlke taxonomy | Medium — extend to full Bayesian expected-value framework |
| Autonomous optical navigation for target acquisition | AstroForge, Karman+ | **Out of scope** | N/A — mission operations, not prospecting |
| Meteorite analog laboratory analysis | AstroForge + Colorado School of Mines | **Covered** via RELAB matching | — |
| Space-based UV/visible telescope surveys | Origin Space (Yangwang-1) | **Could ingest** | Low — proprietary data, limited availability |

### What we're doing that the industry isn't (or isn't disclosing)

| Our pipeline technique | Industry analog | Assessment |
|---|---|---|
| Full BAR calibration chain (Dunn et al. 2010) | None disclosed | **Unique advantage** — no commercial entity has published using quantitative mineral ratio estimation |
| CNN-based mineral quantification (Korda et al. 2023) | None disclosed | **Unique advantage** — ML mineral chemistry from raw spectra |
| Multi-signal PGM convergence table | None disclosed | **Unique advantage** — systematic multi-evidence PGM scoring |
| Explicit confidence scoring with uncertainty propagation | Karman+ does probabilistic targeting | **Partial overlap** — our approach is more formalized |
| Weathering correction (Brunetto model) | None disclosed | **Standard in academia but not commercial** |
| Mahlke 2022 probabilistic taxonomy | Karman+ likely uses something similar | **State-of-the-art alignment** |

### What neither we nor industry adequately addresses

1. **The M-type characterization problem.** M-type asteroids are the highest-value PGM targets, but they are spectrally featureless — meaning our entire BAR/band-parameter pipeline is inapplicable. Radar albedo (>0.25 suggests metallic) and thermal inertia are the primary discriminators, but radar data exists for only ~850 NEAs (and Arecibo is gone). AstroForge's strategy of "fly there and look" reflects this reality. Our pipeline needs to explicitly score M-type candidates using a different sub-pipeline: taxonomy + albedo + radar albedo + thermal inertia + polarimetry, with a fundamentally lower confidence ceiling than S-type mineralogy.

2. **The small-asteroid problem.** Commercial targets are 20–300m diameter. At these sizes, most NEAs have no spectral data at all — they're too faint for ground-based VNIR spectroscopy. Karman+ addresses this by "targeting unknown objects that fit a certain statistical profile." Our pipeline should incorporate the Granvik et al. (2018) debiased NEA population model to assign probabilistic taxonomy to uncharacterized objects based on their orbital source regions (different escape routes from the main belt produce different compositional mixes).

3. **Near-Sun thermal processing.** Both our specs and most commercial approaches underweight the effect of perihelion distance on composition. NEAs with low perihelion (q < 0.5 AU) experience surface temperatures >600K, which dehydrates phyllosilicates and can thermally process metallic surfaces. Toliou et al. (2021) provides a framework, but this needs to be integrated into both the grade estimates and the confidence scoring.

---

## 3. Recommended Additions to the Pipeline Specs

Based on this cross-reference, the following additions would make the specs both more scientifically rigorous and better aligned with commercial best practices:

### High Priority

**A. Phase curve module.** Add a pre-processing stage that computes H, G1, G2 parameters from MPC photometric archives for every target asteroid. This provides independent albedo estimates (cross-validating NEOWISE), taxonomic constraints, and surface roughness indicators. Karman+'s Phazon tool demonstrates this is implementable with MPC data alone.

**B. Thermal depletion filter.** For water/volatile targets, implement the Toliou et al. (2021) near-Sun dwell time model as a multiplicative penalty on the recoverability factor. Asteroids with extended low-perihelion histories should have their water grade estimates reduced or zeroed.

**C. Spin-barrier filter.** Ingest rotation periods from LCDB and apply a binary or graded filter: asteroids with P < 2.2 hours are likely monolithic (no loose regolith), making surface mining operationally much harder. This should factor into recoverability, not accessibility.

**D. Dual sub-pipeline for M-type vs. S-type.** Formalize what's implicit: the BAR/mineralogy pipeline applies only to S-complex asteroids. M-type candidates need a separate scoring path using taxonomy probability + radar albedo + thermal inertia + polarimetry. The confidence ceiling for M-type remote characterization should be explicitly capped lower than S-type.

### Medium Priority

**E. SsODNet ingestion.** Add the IMCCE SsODNet service as a data source alongside SBDB. It aggregates physical properties from a broader set of European literature sources and provides a more complete picture of available characterization data per asteroid.

**F. Debiased population prior.** For asteroids with no spectral data, apply the Granvik et al. (2018) NEA population model to assign probabilistic taxonomy based on orbital source region. This extends coverage from the ~5% of NEAs with spectra to the entire known population.

**G. Vera Rubin Observatory readiness.** The Legacy Survey of Space and Time (LSST) at Vera Rubin Observatory will begin operations in ~2025–2026 and will detect an estimated 5+ million solar system objects with 6-band (ugrizy) photometry. The pipeline should be designed to ingest LSST photometry and produce taxonomic classifications at scale — this will be the dominant data source within 2–3 years.

### Lower Priority

**H. Commercial target tracking.** Maintain a watchlist of asteroids that commercial entities have publicly disclosed as targets (AstroForge's 2022 OB5, Karman+'s High Frontier target when disclosed). These asteroids will accumulate more characterization data than average and represent validation opportunities.

**I. Secrecy-adjusted scoring.** The emerging commercial norm of target secrecy (AstroForge explicitly reserves the right to withhold high-value targets) means that the highest-scoring asteroids in our pipeline might attract competitive interest. Consider whether the scoring output should include a "competitive risk" flag for targets that have been publicly claimed.

---

## 4. The Fundamental Strategic Question

The commercial sector reveals a philosophical split that our pipeline should acknowledge:

- **AstroForge's approach:** Minimal remote characterization → fly there → confirm in situ. This works if mission costs are low enough to tolerate Type I errors (visiting the wrong asteroid).
- **Karman+'s approach:** Maximal probabilistic characterization → target a statistical class → accept residual uncertainty. This works if the target material (water) is common enough in the target class (C-type) that any compliant target is likely viable.
- **Our pipeline's implicit approach:** Detailed remote characterization → rank by expected value → present a prioritized list. This is the traditional prospecting model, and it's most valuable for the PGM case where targets are rare and the cost of error is highest.

All three are valid strategies, but they have different information requirements. Our pipeline is best suited to the third — which is also the hardest. The specs should make this strategic position explicit and document the confidence limitations honestly, particularly for M-type targets where remote characterization hits a hard ceiling.

---

## 5. One Operational Lesson from Commercial Failures

Both AstroForge missions (Brokkr-1 and Odin) experienced critical failures early in their operational lives — communications loss, attitude control problems, and electromagnetic interference. Planetary Resources folded due to funding before reaching asteroid targets. Deep Space Industries pivoted entirely. Origin Space's NEO-1 was a technology test, not a science mission.

The implication for our pipeline: **no commercial entity has yet successfully performed in-situ spectral characterization of an asteroid for mining purposes.** Every commercial target selection decision made to date has been based entirely on the same ground-based and archival remote sensing data that our pipeline ingests. This means our pipeline, if implemented correctly, would be operating at the current frontier of what is practically achievable for asteroid mining target selection — not behind it.
