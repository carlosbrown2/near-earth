# Asteroid Mining Prospector

Bayesian data pipeline and Monte Carlo scoring system to rank near-Earth object (NEO) mining candidates from public asteroid survey data.

## Overview

The prospector ingests data from multiple asteroid surveys (JPL SBDB, NEOWISE, MITHNEOS, SMASS, SsODNet, RELAB, LCDB, Gaia, JWST), resolves entities across catalogs, performs spectral analysis to infer mineralogy, and scores each NEO's mining value using Bayesian priors and Monte Carlo sampling.

**Key capabilities:**

- **Multi-source ingestion** — entity-resolved catalog of ~36,000 NEOs from 9+ data sources
- **Spectral analysis** — taxonomy classification (Mahlke 2022), band analysis, mineral identification, space weathering correction
- **Bayesian scoring** — Granvik (2018) orbital priors give every NEO a taxonomy distribution, even without spectra
- **Monte Carlo valuation** — samples from parameter distributions rather than point estimates; confidence = posterior width
- **EVOI ranking** — expected value of information identifies which uncharacterized NEOs to observe next
- **REST API** — FastAPI service for querying rankings, asteroid details, and EVOI recommendations

## Installation

Requires Python 3.10+.

```bash
# Clone and set up virtual environment
git clone <repo-url> && cd near-earth
python -m venv env
source env/bin/activate

# Install core dependencies
pip install -e .

# Install dev dependencies (testing, type checking, mutation testing)
pip install -e ".[dev]"

# Install API dependencies (FastAPI + Uvicorn)
pip install -e ".[api]"
```

## Running the Pipeline

```bash
# Full pipeline: ingest → spectral → scoring → output
python -m prospector --data-dir data/ --mode earth_return --output results.csv

# Run specific stages
python -m prospector --stages ingest spectral
python -m prospector --stages scoring output --db-path data/prospector.db
```

## Running the API

```bash
# Direct
uvicorn prospector.api.app:create_app --factory --host 0.0.0.0 --port 8000

# Docker
docker compose up --build
```

The API serves on `http://localhost:8000` with endpoints for asteroid rankings, search, and EVOI recommendations. Health check at `/health`.

## Running Tests

```bash
# Run test suite
python -m pytest tests/

# With coverage
python -m pytest tests/ --cov=prospector

# Type checking
python -m mypy prospector/

# Mutation testing (per module)
scripts/mutmut-gate.sh prospector/path/to/module.py
```

## Project Structure

```
near-earth/
├── prospector/              # Core library
│   ├── ingest/              # Data ingestion (sbdb, neowise, mithneos, smass, ...)
│   ├── spectral/            # Spectral analysis (preprocessing, taxonomy, band_analysis)
│   ├── scoring/             # Mining value scoring (scorer, granvik_prior, evoi)
│   ├── ephemeris/           # Orbital mechanics (Lambert transfers)
│   ├── api/                 # FastAPI REST service
│   ├── db.py                # Database access layer
│   ├── schemas.py           # Pydantic data models
│   ├── pipeline.py          # Pipeline orchestrator
│   └── output.py            # Result export
├── scoring/                 # Scoring config YAMLs (checksum-locked)
├── tests/                   # pytest suite (mirrors prospector/ structure)
├── data/                    # Raw + processed data (gitignored)
├── notebooks/               # Analysis notebooks
└── archive/                 # Legacy research scripts (read-only reference)
```

## Key References

- **Granvik et al. (2018)** — Debiased orbit and size distributions of NEOs; source-region taxonomy priors
- **Mahlke et al. (2022)** — Probabilistic 17-class asteroid taxonomy via the `classy` package
- **Cannon et al. (2023)** — PGM grade estimates for asteroid mining (replaces Kargel 1994)
- **Garcia Yarnoz et al. (2013)** — Easily retrievable objects (ERO) for Earth-return missions
- **Carry (2012)** — Asteroid density compilation with porosity corrections
- **Dunn et al. (2010)** — Band area ratio calibration for S(IV) silicate mineralogy
- **Sanchez et al. (2012)** — Temperature corrections for NIR band centers
- **Lindsay et al. (2015, 2016)** — Red-edge spectral corrections

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
