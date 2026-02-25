# PRD: REST API Deployment

## 1. Introduction

Expose the asteroid mining prospector's pre-computed scores and rankings as a read-only REST API using FastAPI. The API serves data from the existing SQLite database — no pipeline execution at request time. Deployed as a Docker container with API key authentication.

## 2. Goals

- Serve ranked mining candidates, asteroid details, and scoring breakdowns over HTTP/JSON
- Sub-100ms response times for typical queries (pre-computed data, no MC scoring at request time)
- Auto-generated OpenAPI/Swagger docs at `/docs`
- Containerized deployment with a single `docker compose up`
- API key gating to control access and enable future rate limiting

## 3. User Stories

### US-001: Ranked Candidates Endpoint

**Description:** As a client, I want to fetch ranked asteroid mining candidates so that I can display or analyze the top targets.

**Acceptance Criteria:**
- [ ] `GET /v1/rankings` returns paginated list of scored asteroids ordered by `composite_score` DESC
- [ ] Supports query params: `mode` (earth_return|in_space), `limit` (default 50, max 500), `offset`, `min_score`, `material` (filter by target_material)
- [ ] Response includes: rank, asteroid_id, name, designation, composite_score, estimated_mass_kg, grade_estimate, target_material, unit_value, accessibility, score_mode
- [ ] Returns 200 with `{"data": [...], "total": N, "limit": N, "offset": N}`
- [ ] Tests pass

### US-002: Asteroid Detail Endpoint

**Description:** As a client, I want to fetch full details for a specific asteroid so that I can see its orbital, physical, taxonomic, compositional, and scoring data in one call.

**Acceptance Criteria:**
- [ ] `GET /v1/asteroids/{asteroid_id}` returns all available data for one asteroid
- [ ] Response joins: asteroids, orbits, physical_properties, taxonomy (class + prob, not raw BLOB), band_analysis, cnn_mineral, pgm_convergence, rotation_properties, scores
- [ ] Returns 404 with `{"detail": "Asteroid not found"}` for unknown IDs
- [ ] Tests pass

### US-003: Asteroid Search Endpoint

**Description:** As a client, I want to search asteroids by name, designation, or orbital properties so that I can find specific targets.

**Acceptance Criteria:**
- [ ] `GET /v1/asteroids` with query params: `q` (text search on name/designation), `neo` (bool), `pha` (bool), `min_diameter`, `max_diameter`, `min_moid`, `max_moid`, `taxonomy_class`, `limit`, `offset`
- [ ] Text search is case-insensitive prefix match on name or designation
- [ ] Returns paginated results: `{"data": [...], "total": N, "limit": N, "offset": N}`
- [ ] Tests pass

### US-004: EVOI Observation Priority Endpoint

**Description:** As a client, I want to fetch the EVOI ranking to see which uncharacterized asteroids would benefit most from new observations.

**Acceptance Criteria:**
- [ ] `GET /v1/evoi` returns pre-computed EVOI results if available in the database, or computes on-the-fly for a limited set
- [ ] Supports `mode`, `limit`, `offset` params
- [ ] Response includes: asteroid_id, name, score_mean, score_std, evoi_vnir, evoi_vis, evoi_radar, evoi_albedo, best_observation, best_evoi
- [ ] Tests pass

### US-005: API Key Authentication

**Description:** As an operator, I want API key authentication so that I can control who accesses the data.

**Acceptance Criteria:**
- [ ] All `/v1/*` endpoints require `X-API-Key` header or `api_key` query param
- [ ] API keys configured via environment variable `PROSPECTOR_API_KEYS` (comma-separated list)
- [ ] Invalid/missing key returns 401 `{"detail": "Invalid or missing API key"}`
- [ ] `/health` and `/docs` are unauthenticated
- [ ] Tests pass

### US-006: Docker Deployment

**Description:** As an operator, I want to run the API with `docker compose up` so that deployment is reproducible.

**Acceptance Criteria:**
- [ ] `Dockerfile` builds a production image with uvicorn
- [ ] `docker-compose.yml` configures the service with env vars for API keys and DB path
- [ ] Database file is mounted as a volume (not baked into the image)
- [ ] Health check endpoint at `GET /health` returns `{"status": "ok", "asteroids_count": N, "scores_count": N}`
- [ ] Container starts and responds within 5 seconds
- [ ] Tests pass

## 4. Technical Context

**Existing code to build on:**
- `prospector/db.py:get_connection()` — SQLite connection factory with WAL mode
- `prospector/output.py:_fetch_candidates()` — existing multi-table JOIN query for ranked output (lines 72–136). This is the core query for the rankings endpoint.
- `prospector/output.py:generate_ranked_dicts()` — returns list[dict], can be adapted directly
- `prospector/schemas.py` — Pydantic models already defined (ScoringResult, BandAnalysisResult, EVOIResultSchema). Extend these for API response models.
- `prospector/scoring/evoi.py:rank_all()` — EVOI computation for observation prioritization

**New files:**
- `prospector/api/` — new package for API layer
  - `__init__.py`
  - `app.py` — FastAPI application factory
  - `routers/rankings.py` — rankings endpoint
  - `routers/asteroids.py` — asteroid search + detail endpoints
  - `routers/evoi.py` — EVOI endpoint
  - `auth.py` — API key dependency
  - `models.py` — Pydantic response models (extend existing schemas)
- `Dockerfile`
- `docker-compose.yml`

**Dependencies to add to `pyproject.toml`:**
- `fastapi>=0.110`
- `uvicorn[standard]>=0.27`

**Key constraint:** The database is read-only at the API layer. The connection should be opened with `create=False` (no schema migration) and queries should never write. Use a single connection per-process (SQLite WAL supports concurrent reads).

## 5. Functional Requirements

- FR-1: All endpoints return JSON with `Content-Type: application/json`
- FR-2: Pagination uses `limit`/`offset` pattern. Default limit=50, max limit=500.
- FR-3: The API reads from an existing `prospector.db` file. The DB path is configurable via `PROSPECTOR_DB_PATH` env var (default: `data/prospector.db`).
- FR-4: BLOB columns (prob_vector, wavelengths, reflectance) are NOT exposed in API responses. Taxonomy is returned as `{primary_class, primary_prob}` only.
- FR-5: Numeric values are returned as native JSON numbers, not formatted strings (unlike the CSV output module).
- FR-6: The API must handle an empty database gracefully (0 results, not 500 errors).

## 6. Non-Goals

- No write endpoints (ingestion, scoring, pipeline triggers)
- No WebSocket or streaming
- No user accounts, roles, or per-user permissions (just shared API keys)
- No caching layer (Redis, etc.) — SQLite with WAL is fast enough for read-only queries
- No rate limiting implementation (just the auth gate for now)
- No frontend / dashboard

## 7. Design Considerations

**Database concurrency:** SQLite WAL mode supports multiple concurrent readers. For a read-only API, a connection-per-request or connection-pool approach both work. A single shared connection with `check_same_thread=False` is simplest for FastAPI's async model, since all operations are reads.

**Response model reuse:** The existing Pydantic models in `schemas.py` validate internal pipeline data. API response models should be separate — they control the public contract and may omit internal fields (like raw BLOB data) or rename fields for clarity.

## 8. Risks and Open Questions

- **SQLite at scale:** If concurrent request volume exceeds what SQLite WAL handles (~50-100 concurrent readers), would need to migrate to PostgreSQL. Acceptable for initial deployment.
- **EVOI compute cost:** If EVOI data isn't pre-computed in the DB, on-the-fly computation is expensive (~seconds per asteroid). The endpoint should return only what's pre-stored, or limit on-the-fly computation to a small batch.
