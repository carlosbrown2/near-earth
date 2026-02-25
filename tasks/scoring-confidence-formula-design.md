<!-- bead-type: research -->
<!-- phase: 2 -->
<!-- output-artifact: scoring/confidence_config.yaml -->

# Scoring Model Design — Confidence Score Formula

## Purpose

The scoring formula in the PRD (US-005) includes a `confidence` term that quantifies how certain we are about the mineralogical assessment. The PRD (§10, Confidence Scoring) lists six qualitative inputs but provides no formula to combine them into a number. This task defines the concrete formula, validates it against known asteroids, and specifies the implementation.

---

## Inputs (from PRD §10)

The PRD identifies these confidence factors:

1. **Mahlke taxonomy probability vector entropy** — low entropy = high-confidence classification
2. **Number of independent spectral observations** — more observations = more reliable
3. **Wavelength coverage** — visible-only < visible+NIR < full VNIR
4. **Signal-to-noise ratio** — higher SNR spectra yield more reliable parameters
5. **Agreement between taxonomy and analog matching** — concordance = higher confidence
6. **Agreement between classical band analysis and CNN mineral quantification** — when both available
7. **Availability of supplementary data** — albedo, radar, thermal inertia, polarimetry
8. **For PGM candidates:** number of converging signals from Stage 5 multi-signal table

---

## Question 1: What is the functional form for combining these inputs?

### Option A: Weighted geometric mean

```
confidence = ∏ᵢ (cᵢ)^(wᵢ)     where Σ wᵢ = 1
```

Each input `cᵢ` is normalized to [0, 1]. Geometric mean ensures that a single zero-confidence factor drives the overall score toward zero (a multiplicative "veto" property).

**Pros:** Intuitive — one terrible input can't be hidden by good inputs elsewhere.
**Cons:** Requires careful normalization of each input to [0, 1]; true zeros collapse the score entirely.

### Option B: Weighted arithmetic mean

```
confidence = Σᵢ (wᵢ × cᵢ)     where Σ wᵢ = 1
```

**Pros:** Simpler; more forgiving of missing data (a missing input can default to 0.5 without collapsing the score).
**Cons:** A high score on one axis can compensate for a completely unknown axis — less physically meaningful.

### Option C: Tiered / rule-based

Define discrete confidence tiers (e.g., High / Medium / Low / Insufficient) based on which data is available:

| Tier | Minimum data required | Score |
|---|---|---|
| High (0.8–1.0) | Full VNIR + albedo + taxonomy agrees with analog match + CNN/classical agree | 0.9 |
| Medium (0.5–0.8) | NIR-only or visible+albedo + taxonomy assigned + analog match available | 0.65 |
| Low (0.2–0.5) | Visible-only taxonomy OR albedo-only | 0.35 |
| Insufficient (<0.2) | No spectral data; orbital/physical only | 0.1 |

Then multiply by continuous modifiers (SNR factor, entropy factor) within each tier.

**Pros:** Transparent; easy to explain; wavelength coverage (the dominant factor) is handled directly.
**Cons:** Less granular; tier boundaries are arbitrary.

### Design decision to make

Which option (or hybrid) best serves the product? Consider that the primary consumer is a ranked list — what matters most is that the confidence score produces **sensible orderings**, not that it has a precise probabilistic interpretation.

---

## Question 2: How should each input be normalized to [0, 1]?

For each input, define the mapping to a 0–1 scale:

### Taxonomy probability entropy

- **Metric:** Shannon entropy of the Mahlke 17-class probability vector: `H = -Σ pᵢ log₂(pᵢ)`
- **Max entropy:** log₂(17) ≈ 4.09 (uniform distribution — no information)
- **Min entropy:** 0 (100% certain on one class)
- **Proposed normalization:** `c_taxonomy = 1 - H / log₂(17)`
- **What to verify:** Does `classy` output a full 17-class probability vector, or only top-N classes? If top-N, entropy computation needs adjustment.

### Wavelength coverage

- **Proposed mapping:**

| Coverage | Score |
|---|---|
| Full VNIR (0.45–2.45 μm) + albedo | 1.0 |
| Full VNIR without albedo | 0.85 |
| NIR-only (0.8–2.5 μm) + albedo | 0.7 |
| NIR-only without albedo | 0.6 |
| Visible + albedo (Gaia + NEOWISE) | 0.4 |
| Visible-only | 0.25 |
| Albedo-only | 0.15 |
| No spectral/albedo data | 0.05 |

- **What to verify:** Are these relative weights sensible for mineralogical discrimination? The key jump is at "has NIR" vs. "no NIR" — that's where Band II and BAR become available.

### SNR factor

- **Proposed normalization:** `c_snr = min(1, SNR / SNR_good)` where `SNR_good` is the threshold for "good quality" spectra
- **What to determine:** What is `SNR_good`? Check MITHNEOS data quality — what is the typical SNR range? What SNR does Korda et al. (2023) require for reliable CNN output?

### Method agreement

- **When both classical band analysis and CNN are available:** Compare OL/(OL+PX) from both methods. `c_agree = 1 - |classical - CNN| / max_plausible_disagreement`
- **When only one method is available:** Default to 0.5 (neutral)
- **What to determine:** What is `max_plausible_disagreement`? Check Korda et al. (2023) validation — what is the typical spread between CNN and classical results?

### Taxonomy–analog concordance

- **Proposed:** Binary or graded check — does the best analog match's meteorite type correspond to the assigned taxonomy class?
  - Full concordance (e.g., S-type + OC analog): 1.0
  - Partial concordance (e.g., S-type + pallasite): 0.5
  - Discordance (e.g., S-type + iron meteorite): 0.2
- **Requires:** A concordance lookup table mapping {taxonomy_class → expected_meteorite_types}

### Supplementary data availability

- **Proposed:** Count of available supplementary signals (albedo, radar albedo, thermal inertia, polarimetry, lightcurve) normalized by maximum possible: `c_supp = n_available / n_max`
- **Or:** Weighted count, since radar albedo is much more informative than lightcurve for PGM candidates

### PGM convergence (Stage 5 only)

- **Proposed:** Count of converging signals from the Stage 5 multi-signal table (max ~9 signals listed), weighted by signal strength column: `c_pgm = Σ(signal_present × signal_weight) / Σ(signal_weight)`

---

## Question 3: What weights should each factor receive?

Propose initial weights and determine how to validate them:

| Factor | Proposed weight | Rationale |
|---|---|---|
| Wavelength coverage | 0.30 | Dominant factor — determines which analyses are possible at all |
| Taxonomy entropy | 0.20 | Core classification confidence |
| SNR | 0.15 | Directly affects all spectral measurements |
| Taxonomy–analog concordance | 0.15 | Cross-validation of two independent methods |
| Method agreement (classical vs. CNN) | 0.10 | Available only for S-complex with full VNIR |
| Supplementary data | 0.10 | Incremental confirmation |

**What to verify:** Run the formula on 10–20 well-studied asteroids with known compositions and check whether the confidence ranking is sensible. Specifically:
- Itokawa (S-type, Hayabusa sample return — should be highest confidence)
- Bennu (B-type, OSIRIS-REx — should be highest confidence)
- 16 Psyche (M-type, MITHNEOS + radar + thermal — should be high)
- A random Gaia-only MBA (visible-only, no albedo — should be low)

---

## Deliverable

1. Chosen formula with all normalization functions and weights
2. Concordance lookup table: `{taxonomy_class → [expected_meteorite_types]}`
3. Worked examples for ≥5 asteroids spanning the confidence range
4. Implementation spec: function signature, input schema, output schema
5. Config values in YAML/JSON (`scoring/confidence_config.yaml`)

---

## Verification

- Compute confidence scores for Itokawa, Bennu, 16 Psyche, Eros, and 2 poorly-characterized NEOs
- Verify that sample-return targets (ground-truth composition known) score highest
- Verify that asteroids with only Gaia visible-only data score substantially lower than those with full VNIR
- Check that the confidence score distribution across a test set of ~100 asteroids is not degenerate (not all clustered at one value)
