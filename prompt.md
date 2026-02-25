# Ralph Agent Instructions — Bead Loop

You are an autonomous coding agent building the Asteroid Mining Prospector pipeline. Each iteration, you complete ONE bead (task unit) and commit your work.

## Your Task

1. Run `bd ready --json` to find unblocked tasks
2. Read `progress.txt` (check **Codebase Patterns** section first)
3. Check you're on the `prospector` branch. If not, check it out.
4. Pick the **highest priority ready bead** from `bd ready`
5. Run `bd show <id>` to read the full bead description
6. Read the bead's referenced spec file (in `tasks/`) for full requirements
7. Read `tasks/MANIFEST.md` for bead sizing rules
8. Claim the bead: `bd update <id> --claim`
9. Implement that single bead
10. Run quality checks: `python -m pytest tests/ && python -m mypy prospector/`
11. If checks pass, commit ALL changes: `git commit -m "feat: bead-name — short description (near-earth-xxx)"`
12. Close the bead: `bd close <id> --reason "what was done"`
13. Append your progress to `progress.txt`
14. Update `CLAUDE.md` if you discover reusable patterns

## Bead Selection Rules

- Run `bd ready` — it shows only unblocked tasks (all dependencies satisfied)
- Pick the **highest priority** (lowest P number) from the ready list
- If multiple beads have equal priority, pick the first one listed
- If NO beads are ready (all blocked or complete), check the stop condition

## Reading the Spec

Each bead description references a section in the PRD or a standalone task document:
- **Implementation beads** (most beads): read `tasks/prd-asteroid-mining-prospector.md` and find the referenced section. The PRD §12 dependency graph, §7 Proposed Architecture, and the Unified Database Schema are essential context.
- **Research beads** (validate-classy, scoring-config): read the standalone task doc (e.g., `tasks/scoring-grade-and-density-review.md`). Output is a config file or document with concrete values and citations.
- **Bead sizing rule** (from MANIFEST): one module (`.py`) or one config (`.yaml`) per bead. Max 3 files changed. Every bead needs at least one test.

## Key Architecture Context

- **Package layout:** `prospector/` is the core library. Subpackages: `ingest/`, `spectral/`, `scoring/`, `ephemeris/`
- **Database:** SQLite or DuckDB. Schema defined in PRD §12. All ingest modules write to shared schema.
- **Bayesian scoring:** Every NEO gets a taxonomy distribution (Granvik prior). Monte Carlo scorer samples from distributions. Confidence = posterior width.
- **Existing code:** `archive/` has research-phase scripts (kepler.py, distance.py, etc.) — reference only, do not modify
- **Data files:** `.csv`, `.db`, `.fits` are gitignored. Use `data/` directory structure from PRD §7.
- **Virtual env:** `env/` (Python 3.12)

## Technical Constraints (from project audit)

- ERO attribution: Garcia Yarnoz et al. (2013), NOT Elvis (2014)
- PGM grades: Cannon et al. (2023), not Kargel (1994) — grades are lower than historical estimates
- BAR calibration: Dunn et al. (2010) for S(IV); Gaffey zones for non-S(IV)
- Band analysis must include red-edge correction (Lindsay 2015/2016) and temperature correction (Sanchez 2012)
- Density priors: Carry (2012); porosity matters for mass estimates
- Scoring config: all values as distributions `{mean, std, min, max}` for Monte Carlo sampling

## Progress Report Format

APPEND to `progress.txt` (never replace, always append):

```
## [Date] — <bead-id>: bead-name
- What was implemented
- Files changed
- Tests added/modified
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
  - Useful context for downstream beads
---
```

The learnings section is critical — it helps future iterations avoid repeating mistakes.

## Consolidate Patterns

If you discover a **reusable pattern**, add it to the `## Codebase Patterns` section at the TOP of `progress.txt` (create if absent):

```
## Codebase Patterns
- Example: Use `prospector.db.get_connection()` for all database access
- Example: All spectra stored as numpy arrays via .tobytes() / np.frombuffer()
- Example: Entity resolver canonical key is IAU asteroid number (integer)
```

Only add patterns that are **general and reusable**, not bead-specific details.

## Update CLAUDE.md

Before committing, check if you discovered something future iterations should know:
- API patterns or conventions
- Non-obvious requirements or gotchas
- Dependencies between modules
- Testing approaches

Add genuinely reusable knowledge to the relevant section of `CLAUDE.md`. Do NOT add bead-specific implementation details.

## Quality Requirements

- ALL commits must pass `python -m pytest tests/ && python -m mypy prospector/`
- Property-based tests (Hypothesis) and Pydantic schema contracts provide additional backpressure
- Do NOT commit broken code
- Keep changes focused — one module or config per bead (MANIFEST rule)
- Follow existing code patterns in `prospector/`
- Each bead must have at least one test

## Mutation Testing Gate

After closing a bead that touches `prospector/` modules, run the mutation testing gate on each changed module:

```bash
scripts/mutmut-gate.sh prospector/path/to/module.py
```

The gate requires **<10% surviving mutants**. If it fails:
1. Review the surviving mutants: `mutmut results`
2. Show a specific mutant: `mutmut show <id>`
3. Add targeted tests to kill the surviving mutants
4. Re-run the gate until it passes

This is mandatory for beads that produce or modify `prospector/` Python modules. Research beads (output config/docs only) and tooling beads are exempt.

## Stop Condition

After completing a bead, run `bd ready`. If there are no more ready tasks AND `bd list` shows all tasks closed, reply with:
<promise>COMPLETE</promise>

If there are still open beads, end your response normally (another iteration will pick up the next bead).

## Important

- Work on ONE bead per iteration
- Claim before starting: `bd update <id> --claim`
- Close after committing: `bd close <id> --reason "..."`
- Commit after each bead with bead ID in message
- Keep tests green
- Read Codebase Patterns in progress.txt before starting
- Read CLAUDE.md for project context before starting
