# Asteroid Mining Prospector

Bayesian data pipeline + Monte Carlo scoring system to rank NEO mining candidates from public asteroid survey data.

## Task Tracking — Beads (bd)

This project uses [Beads](https://github.com/steveyegge/beads) for task tracking. Use the `bd` CLI for all work management.

```bash
bd ready              # Show unblocked tasks you can work on
bd show <id>          # View task details and dependencies
bd update <id> --claim  # Claim a task (sets assignee + in_progress)
bd close <id> --reason "description"  # Mark complete
bd list               # See all tasks with dependency info
bd dep tree <id>      # Visualize dependency chain
bd blocked            # See what's stuck
```

**Commit convention:** Include bead ID in commit messages: `feat: description (near-earth-xxx)`

## Ralph Loop

This project uses the Ralph loop for autonomous development. See `prompt.md` for per-iteration instructions.

- **Task tracker:** `bd ready` / `bd list` — beads manages the dependency DAG
- **Progress log:** `progress.txt` — append-only log with codebase patterns at top
- **Task specs:** `tasks/` directory — PRD and research task documents
- **Manifest:** `tasks/MANIFEST.md` — bead sizing rules and document index
- **Loop runner:** `ralph.sh` — bash loop that spawns fresh Claude instances

## Project Structure

```
near-earth/
├── prospector/              # Core library (Python package)
│   ├── ingest/              # Data ingestion per source (sbdb, neowise, mithneos)
│   ├── spectral/            # Spectral analysis pipeline (preprocessing, taxonomy, band_analysis)
│   ├── scoring/             # Mining value scoring (scorer, config)
│   └── ephemeris/           # Orbital mechanics
├── tests/                   # pytest test suite (mirrors prospector/ structure)
├── data/                    # Raw + processed data (gitignored: .csv, .db, .fits)
│   ├── sbdb/                # JPL SBDB exports
│   ├── spectra/             # Spectral survey data
│   └── processed/           # Pipeline outputs
├── scoring/                 # Config artifacts (YAML)
├── notebooks/               # Analysis notebooks
├── archive/                 # Legacy research scripts (read-only reference)
├── tasks/                   # PRD and task specifications
├── .beads/                  # Beads database (bd CLI)
├── prompt.md                # Ralph iteration instructions
└── progress.txt             # Append-only progress log
```

## Tech Stack

- Python 3.10+ (venv at `env/`, Python 3.12)
- NumPy, SciPy, Pandas, Astroquery
- classy (Mahlke 2022 taxonomy)
- SQLite or DuckDB (chosen at project-setup time)
- pytest for testing

## Branch

- Working branch: `prospector`
- Main branch: `master`

## Bead Dependency Graph

15 Phase 1 beads in dependency order. Run `bd dep tree near-earth-bv8` for the full tree.

**Dependency paths:**
```
Path A (spectral):  project-setup → schema → sbdb-ingest → entity-resolver → mithneos → preprocessing → taxonomy → band-analysis
Path B (prior):     schema → sbdb-ingest → granvik-prior
Path C (scoring):   schema → scoring-config
Convergence:        scorer (blocked by scoring-config + granvik-prior)
Output:             output-csv (blocked by scorer + band-analysis + evoi)
EVOI:               evoi (blocked by scorer)
```

After schema-design completes, sbdb-ingest, neowise-ingest, validate-classy, and scoring-config all unblock in parallel.

## Key Technical Decisions

- **Mahlke 2022 taxonomy** via `classy` package (`pip install space-classy`) — probabilistic 17-class assignments, not single labels. Albedo MUST be passed as `pV=` kwarg, NOT `albedo=`. See `tasks/validate-classy-findings.md` for full API docs.
- **Bayesian scoring:** Granvik 2018 prior gives every NEO a taxonomy distribution from orbital source regions. Monte Carlo scorer samples from distributions. Confidence = posterior width.
- **Entity resolution:** Canonical key is IAU asteroid number (integer). Fallback: packed MPC designation.
- **Database schema:** Defined in PRD §12. BLOB columns store numpy arrays via `.tobytes()` / `np.frombuffer()`.
- **Spectral data:** MITHNEOS is the irreplaceable NIR data source for mineralogy (0.8-2.5 μm).

## Critical Constraints

- PGM grades: Use **Cannon et al. (2023)**, not Kargel (1994) — PGMs deviate from chondritic ratios at high Ir
- ERO attribution: **Garcia Yarnoz et al. (2013)**, not Elvis (2014)
- BAR calibration: **Dunn et al. (2010)** for S(IV); Gaffey zones for non-S(IV)
- Band analysis: Must include temperature correction (Sanchez 2012) and red-edge correction (Lindsay 2015/2016)
- Density priors: **Carry (2012)** — porosity matters for mass estimates
- Scoring config values: All as distributions `{mean, std, min, max}` for Monte Carlo sampling
- CNN model (Korda 2023): Only valid for S*-complex silicate spectra — enforce domain checks

## Conventions

- One module (`.py`) or one config (`.yaml`) per bead — max 3 files touched
- Every bead produces at least one test
- Research beads output a document or config file with concrete values and citations
- Implementation beads output code with tests
- Done criterion must be mechanically checkable (test passes, file exists with required keys)

## Codebase Patterns

_Populated by Ralph iterations. Check `progress.txt` for the latest patterns._

### Design by Contract (deal)

Add `@deal.pre` / `@deal.post` decorators to new public functions in `prospector/`:

```python
import deal

@deal.pre(lambda diameter_km, density_gcm3: diameter_km >= 0, message="diameter must be non-negative")
@deal.post(lambda result: result >= 0)
def estimate_mass_kg(diameter_km: float, density_gcm3: float) -> float:
    ...
```

- Use `deal.pre` for input domain constraints (physical bounds, array shapes)
- Use `deal.post` for output invariants (range bounds, finiteness)
- Add `deal.cases(func, count=N)` in `tests/test_properties.py` to auto-generate tests from contracts
- Contracts are active in tests, can be disabled in production via `deal.disable(permanent=True)`
