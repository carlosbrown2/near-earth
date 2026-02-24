# validate-classy: Research Spike Findings

**Bead:** near-earth-d52
**Date:** 2026-02-23
**Status:** VIABLE — classy works on Python 3.12. Mahlke 2022 taxonomy produces
probabilistic 17-class assignments. No fallback needed.

---

## 1. Installation

**PyPI package name:** `space-classy` (not `classy` — that's a cosmology package)

```bash
pip install space-classy tf-keras 'tensorflow-probability==0.24.*'
```

**Import name:** `import classy`

**Version tested:** space-classy 0.8.8

**Dependencies installed:**
- tensorflow 2.16.2 (latest available for Python 3.12/macOS)
- tensorflow-probability 0.24.0 (must match TF version; 0.25 requires TF ≥ 2.18)
- tf-keras 2.16.0 (required by tfp 0.24)
- mcfa 0.1.6 (Mixture of Common Factor Analyzers — the core model)
- space-rocks 1.9.15 (asteroid name resolution; needs cache dir)
- numpy 1.26.4 (space-classy pins numpy < 2.0)

**Known issues:**
- numpy downgrade: space-classy requires numpy < 2.0, which conflicts with other packages
  needing numpy ≥ 2.0. Solution: use a dedicated venv for the pipeline.
- space-rocks cache: First run prompts for cache directory interactively. Set
  `ROCKS_CACHE_DIR` env var to avoid the prompt in automated pipelines.
- CWD sensitivity: Importing from certain directories (e.g., a numpy source tree)
  can break. Not an issue in normal usage.

---

## 2. Python API

### Creating a Spectrum

```python
import classy
import numpy as np

spec = classy.Spectrum(
    wave=wavelength_array,   # np.ndarray, wavelengths in μm
    refl=reflectance_array,  # np.ndarray, reflectance values
    refl_err=error_array,    # np.ndarray, optional uncertainty
    pV=0.15,                 # float, visual geometric albedo (CRITICAL: use pV, NOT albedo)
)
```

**CRITICAL:** The albedo kwarg must be `pV=value`. Using `albedo=value` silently sets an
unused attribute — the Mahlke classifier ignores it. The `pV` attribute is checked by
`taxonomies.mahlke.preprocess()`.

### Classifying

```python
spec.classify(taxonomy='mahlke')   # Mahlke et al. 2022 (MCFA-based, probabilistic)
spec.classify(taxonomy='demeo')    # DeMeo et al. 2009 (PCA-based)
spec.classify(taxonomy='tholen')   # Tholen 1984
```

### Accessing Results

```python
# Top class and probability
spec.class_mahlke  # str: e.g., 'S', 'C', 'M'
spec.prob           # float: probability of top class

# Full probability vector (17 classes)
spec.class_A   # float: probability of A-type
spec.class_B   # float: probability of B-type
spec.class_C   # float: probability of C-type
# ... etc for all 17 classes

# Latent factor scores (4-dimensional)
spec.scores_mahlke  # np.ndarray, shape (4,)

# Full classified data (DataFrame)
spec.data_classified  # pd.DataFrame with all columns
```

### 17 Mahlke 2022 Classes

`A, B, C, Ch, D, E, K, L, M, O, P, Q, R, S, V, X, Z`

### Checking Classifiability

```python
spec.is_classifiable('mahlke')  # True/False
spec.is_classifiable('demeo')   # True/False — requires wider wavelength range
```

---

## 3. Preprocessing Pipeline (Internal)

When `spec.classify(taxonomy='mahlke')` is called, the following happens automatically:

1. **Resample** spectrum to 53 standard wavelength bins (0.45–2.45 μm at 0.025–0.05 μm steps)
2. **Normalize** using Gaussian mixture normalization ("mixnorm" method)
3. **Albedo handling:**
   - If `spec.pV` exists → log-transform: `pV = log10(pV)`
   - If `spec.target` is a `rocks.Rock` → use cataloged albedo
   - Otherwise → `pV = NaN` (MCFA handles missing values via imputation)
4. **MCFA model** computes responsibility matrix (cluster probabilities) and latent scores
5. **Decision tree** maps MCFA output to 17 taxonomy classes
6. **Feature detection** (e.g., 0.7 μm hydration band for Ch class)

**Key:** The MCFA model supports missing values. Partial wavelength coverage works —
out-of-range wavelengths become NaN and are imputed. This means visible-only spectra
can still be classified.

---

## 4. MCFA Model Details

- **Input features:** 53 wavelength bins + 1 albedo (pV) = 54 features
- **Wavelength grid:** 0.45, 0.475, ..., 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05,
  1.1, 1.15, ..., 2.45 μm (finer sampling in visible, coarser in NIR)
- **Latent factors:** 4 (n_factors)
- **Missing data:** Handled via MCFA imputation — any feature can be NaN
- **Model weights:** Bundled with the package (no separate download needed)

---

## 5. DeMeo Fallback

The DeMeo 2009 taxonomy (`spec.classify(taxonomy='demeo')`) works as a fallback:

- **Method:** PCA-based (no tensorflow dependency for classification itself)
- **Preprocessing:** Normalize at 0.55 μm, remove slope, resample to DeMeo grid
- **Output:** `spec.class_demeo` (single label), `spec.scores_demeo` (5 PCA scores)
- **Limitation:** Requires broader wavelength range than Mahlke; not probabilistic
  (returns single class, no probability vector)

**Recommendation:** Use Mahlke as primary (probabilistic output matches our Bayesian
architecture). DeMeo is viable as cross-reference but not as primary classifier.

---

## 6. Integration Recommendations for taxonomy.py

```python
# Wrapper pattern for prospector/spectral/taxonomy.py
MAHLKE_CLASSES = ['A', 'B', 'C', 'Ch', 'D', 'E', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S', 'V', 'X', 'Z']

def classify_spectrum(wave, refl, refl_err=None, pV=None):
    """Classify a spectrum using Mahlke 2022 taxonomy.

    Parameters
    ----------
    wave : np.ndarray — wavelengths in μm
    refl : np.ndarray — reflectance values
    refl_err : np.ndarray, optional — reflectance uncertainties
    pV : float, optional — visual geometric albedo

    Returns
    -------
    dict with keys:
        'class': str — top class label
        'prob': float — probability of top class
        'probabilities': dict[str, float] — {class: probability} for all 17 classes
        'scores': np.ndarray — 4-dim MCFA factor scores
    """
    import classy

    kwargs = {}
    if pV is not None:
        kwargs['pV'] = pV

    spec = classy.Spectrum(wave=wave, refl=refl, refl_err=refl_err, **kwargs)

    if not spec.is_classifiable('mahlke'):
        return None

    spec.classify(taxonomy='mahlke')

    return {
        'class': spec.class_mahlke,
        'prob': spec.prob,
        'probabilities': {c: getattr(spec, f'class_{c}') for c in MAHLKE_CLASSES},
        'scores': spec.scores_mahlke,
    }
```

---

## 7. Dependency Considerations

| Concern | Status | Mitigation |
|---|---|---|
| numpy < 2.0 pin | Active conflict | Use isolated venv; pin numpy~=1.26 in pyproject.toml |
| tensorflow 2.16 (macOS) | Works | TF 2.18+ not available for macOS/Python 3.12 yet |
| tfp version matching | Resolved | Pin tensorflow-probability==0.24.* alongside TF 2.16 |
| space-rocks cache | Minor | Set ROCKS_CACHE_DIR env var in pipeline config |
| Package maintenance | Active | Last release 0.8.8; author responds to issues |

---

## 8. Conclusion

**classy is viable** for the prospector pipeline. The Mahlke 2022 taxonomy provides
exactly what we need: probabilistic 17-class assignments from spectra + albedo, with
support for partial wavelength coverage. The probability vector feeds directly into the
Bayesian scoring architecture (Granvik prior × spectral likelihood).

**No fallback to PCA-based Bus-DeMeo needed** — classy works on Python 3.12 with
compatible dependency versions. DeMeo classification is also available within classy
as a cross-reference.
