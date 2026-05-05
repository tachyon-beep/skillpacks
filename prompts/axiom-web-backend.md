# Refresh: axiom-web-backend

**Verdict:** MEDIUM / M effort. Structural content ages well; framework code examples need targeted refresh.

## Context

- Pack path: `/home/john/skillpacks/plugins/axiom-web-backend/`
- Full review: `/tmp/skillpack-refresh-review/axiom-web-backend.md`
- Purpose: web API design, framework patterns, integration, microservices, auth, queues.

## Why refresh

Reviewer flagged specific deprecated/EOL items in code examples:

- **Pydantic v1 idioms** in some sheets (FastAPI users have moved to v2).
- **Deprecated `@app.on_event`** — replaced by lifespan context managers.
- **`aioredis` recommendations** — `aioredis` was merged into `redis-py`.
- **EOL `apollo-server` v3** — superseded by Apollo Server 4.
- **Deprecated `apollo-server-testing`** — testing moved into core.
- **Unmaintained `databases` library** — recommend SQLAlchemy 2.0 async or asyncpg directly.
- **OpenAPI 3.0 references** — current is 3.1 (JSON Schema 2020-12 alignment).

## Scope — DO

1. **FastAPI sheet.** Migrate examples to Pydantic v2, lifespan event handlers, SQLAlchemy 2.0 async syntax.
2. **Async stack.** Replace `aioredis` with `redis.asyncio`. Replace `databases` with SQLAlchemy 2.0 async or asyncpg.
3. **GraphQL sheet.** Update Apollo Server 4 examples; testing patterns from core.
4. **OpenAPI.** Update examples to OpenAPI 3.1 where used (FastAPI emits 3.1 by default in current versions).
5. **Express / Node sheet.** Spot-check for similar staleness; update if found.

## Scope — DO NOT

- Do not rewrite the structural content (decision matrices, OAuth2/REST/queue patterns) — it ages well.
- Do not switch frameworks recommended (the framework selection logic is sound).

## Acceptance criteria

1. Zero Pydantic v1 syntax in examples.
2. Zero `@app.on_event` examples (lifespan only).
3. Zero `aioredis` recommendations.
4. Zero `databases` library recommendations.
5. Apollo Server 4 in GraphQL examples.
6. OpenAPI 3.1 noted where examples reference the spec.
7. `plugin.json` version bumped (minor).

## Process

1. Read `/tmp/skillpack-refresh-review/axiom-web-backend.md`.
2. `grep -rn "on_event\|aioredis\|apollo-server@3\|databases.Database" plugins/axiom-web-backend/` — find every staleness marker.
3. Edit each occurrence. Verify code examples actually run.
4. Bump version.

## Constraints

- Every code example must compile / run against the named framework version.
- Specify framework version explicitly in examples (e.g. "FastAPI 0.115+").
- No fabrication of API signatures.
