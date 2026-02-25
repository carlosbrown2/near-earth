# Task Document Manifest

This file tells a bead generator which documents to read and how to use them.

## Bead sources (generate tasks from these)

| Document | Type | Phase | Output artifact | Description |
|---|---|---|---|---|
| `prd-asteroid-mining-prospector.md` | implementation | 1, 2, 3 | Full pipeline | Main PRD. §12 defines phased scope. Phase 1 dependency graph defines task order. Schema definition provides shared data contract. |
| `scoring-grade-and-density-review.md` | research | 1 | `scoring/grade_density_priors.yaml` | Research task: find PGM concentrations, water content, and density priors from published literature. Populate YAML with concrete values. |
| `scoring-confidence-formula-design.md` | research | 2 | `scoring/confidence_config.yaml` | Design task: choose formula, define normalization functions, set weights. Not needed for Phase 1 (MVP uses simplified scoring). |
| `scoring-recoverability-accessibility-design.md` | research | 2 | `scoring/recoverability_accessibility.yaml` | Design task: choose delta-v function form, set recoverability factors. Not needed for Phase 1 (MVP uses MOID proxy). |

## Reference only (do not generate tasks from these)

| Document | Why it's reference-only |
|---|---|
| `bar-calibration-review.md` | Completed analysis. All findings already incorporated into PRD Stage 2. |
| `commercial-asteroid-mining-cross-reference.md` | Industry analysis. Recommendations feed Phase 2/3 scope but are not direct task sources. |
| `compass_artifact_wf-*.md` | External technical audit. Findings incorporated into PRD. One pending fix: ERO attribution (Garcia Yarnoz vs. Elvis) in recoverability doc. |

## Bead sizing guidance

- Each bead should produce **one module** (one `.py` file) or **one config artifact** (one `.yaml` file).
- If a bead would touch more than 3 files, split it.
- Research beads output a document or config file with concrete values and citations.
- Implementation beads output code with at least one test.
- Every bead must have a **done criterion** that is mechanically checkable (test passes, file exists with required keys, etc.).
