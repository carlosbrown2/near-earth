<!-- bead-type: research -->
<!-- phase: 2 -->
<!-- output-artifact: scoring/recoverability_accessibility.yaml -->

# Scoring Model Design — Recoverability Factors & Accessibility Function

## Purpose

The scoring formula in the PRD (US-005) requires design decisions for `recoverability_factor` and `accessibility`. Unlike grade estimates and density priors, these are not single "correct" values from the literature — they are modeling choices informed by engineering and economic assumptions. This task defines concrete values, documents the rationale, and reviews prior art in asteroid mining economic models.

---

## Question 1: What recoverability factors should we assign per material type?

The `recoverability_factor` represents the fraction of a material present in an asteroid that could realistically be extracted and utilized. It accounts for extraction technology, mineral processing difficulty, and physical accessibility.

### Prior art to review

- **Sonter (1997, Acta Astronautica 41, 637–647):** Early economic model for asteroid mining; defines a "mass payback ratio" and extraction efficiency
- **Andrews et al. (2015, NASA/TM-2015-218756):** Technical assessment of asteroid mining approaches; extraction yields by method
- **Cannon & Britt (2019, Icarus 317, 470–478):** Quantitative model of asteroid mining profitability; distinguishes beneficiation from direct extraction
- **Kargel (1994):** Historical economic model (superseded by Cannon et al. 2023 for PGM grades, but extraction methodology discussion may still be relevant)

### Values to determine

| Material | Asteroid type | Key extraction challenge | Factor to assign (0–1) | Source/rationale |
|---|---|---|---|---|
| PGMs — bulk metal | M-type | Must process Fe-Ni metal to separate trace PGMs (~10–50 ppm). Metal itself is easy to identify; PGM separation is hard. | ? | Cannon & Britt 2019 |
| PGMs — OC metal grains | S-type (H/L/LL) | Must first separate metal grains (10–20% by mass) from silicate matrix, then extract PGMs from metal. Two-stage process. | ? (should be lower than M-type) | Cannon et al. 2023 |
| Water — hydrated minerals | C-type (CM/CI) | Thermal extraction (heating to ~500°C releases bound water from phyllosilicates). Well-studied process. | ? | Andrews et al. 2015 |
| Water — molecular H₂O | S-type (Iris-like) | ~450 μg/g is far below any practical extraction threshold. | ~0 (flag but don't score) | Arredondo et al. 2024 |
| Iron — metallic | M-type | Bulk metal; minimal processing needed for in-space use | ? (high) | — |
| Iron — silicate-bound | S-type | FeO in olivine/pyroxene; requires smelting to reduce | ? (low-moderate) | — |
| Olivine / pyroxene | S-type, V-type, A-type | Useful as structural/refractory material; available in bulk | ? (high for in-space) | — |

### Design decisions to make

1. Should recoverability be a single number per {material, asteroid_type} pair, or should it depend on asteroid size (larger = more infrastructure-friendly)?
2. Should we define separate recoverability factors for Earth-return vs. in-space-use scenarios? (e.g., water recoverability is the same, but the *value* differs; PGM recoverability for Earth-return requires additional Earth-entry considerations)
3. What is the minimum concentration threshold below which recoverability should be set to 0? (e.g., S-type water at 450 μg/g)

---

## Question 2: What accessibility function should map delta-v to a score?

The `accessibility` component converts mission parameters (delta-v, approach frequency, time-of-flight) into a 0–1 score. This requires choosing a functional form.

### Options to evaluate

1. **Inverse linear:** `accessibility = max(0, 1 - Δv/Δv_max)` where `Δv_max` is a cutoff (e.g., 10 km/s). Simple but creates a hard cliff at the cutoff.

2. **Exponential decay:** `accessibility = exp(-Δv / Δv_scale)` where `Δv_scale` is a characteristic scale (e.g., 5 km/s). Smooth, no hard cutoff, penalizes high delta-v progressively.

3. **Sigmoid / logistic:** `accessibility = 1 / (1 + exp((Δv - Δv_mid) / k))`. Creates a soft threshold centered at `Δv_mid` with steepness `k`. Useful if there's a natural "feasibility boundary."

4. **Multi-factor composite:** Combine delta-v with approach frequency and synodic period. An asteroid with low delta-v but rare windows (long synodic period) is less accessible than one with low delta-v and frequent windows.

### Reference values to determine

- **Δv_min (easiest NEOs):** ~3.5–4.5 km/s for the most accessible NEOs (comparable to lunar surface)
- **Δv_max (practical cutoff):** What total mission Δv makes mining infeasible? ~7 km/s? ~10 km/s? ~15 km/s?
- **Approach frequency weight:** How much should we penalize asteroids with rare approach windows (e.g., synodic period >5 years)?
- **Time-of-flight factor:** Should longer missions (e.g., >2 years) be penalized?

### Prior art

- **Benner (2023, NHATS):** NASA's Near-Earth Object Human Space Flight Accessible Targets Study defines accessibility criteria (Δv ≤ 12 km/s, mission duration ≤ 450 days, stay time ≥ 8 days)
- **Shoemaker & Helin (1978):** Original delta-v accessibility calculations for NEOs
- **Elvis (2014, P&SS 91, 20–26):** Defines "Easily Retrievable Objects" (EROs) with Δv < 500 m/s for capture to Earth orbit — extremely restrictive but useful as a "best-case" tier

### Design decisions to make

1. Which functional form (linear, exponential, sigmoid, composite)?
2. What are the scale parameters?
3. Should approach frequency be a multiplier on the delta-v score, or a separate factor?
4. Should we define accessibility tiers (e.g., Tier 1: ERO-class <500 m/s, Tier 2: easy <6 km/s, Tier 3: feasible <12 km/s, Tier 4: hard >12 km/s)?

---

## Deliverable

A design document specifying:
1. Recoverability factor table: `{material, asteroid_type} → factor` with rationale per entry
2. Accessibility function: chosen form, parameters, and worked examples
3. Minimum concentration thresholds per material
4. Both values encoded in YAML/JSON for the scoring config (`scoring/recoverability_accessibility.yaml`)

---

## Verification

- Compute composite scores for 5 known asteroids spanning the range (e.g., 16 Psyche, Bennu, Itokawa, Apophis, 1943 Anteros) and verify the ranking is physically sensible
- Compare accessibility scores against NHATS accessibility classifications for sanity
- Verify that the recoverability factors produce non-degenerate rankings (i.e., not all asteroids score ~0 or ~1)
