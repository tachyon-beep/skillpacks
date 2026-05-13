---
name: boundary-and-when-to-leave
description: Use when an embedded-database layer is showing strain and the question is whether to fix it or replace it. Covers observable leave signals, migration paths to Postgres and MySQL/MariaDB, what changes operationally after migration, and the over-leave anti-pattern — most "we outgrew SQLite" stories are actually "we misconfigured SQLite" stories.
---

# Boundary and When to Leave

**SQLite is fast, simple, and unfit for a small list of well-defined workloads. Knowing when to leave is a discipline — pre-commit to the leave criteria before the leave is forced.**

Most migration decisions happen under pressure: a production incident, an architecture review, a new requirement that arrived after the system was already in production. That is the wrong moment to reason about database choices — by then, the question has become "what do we do right now" rather than "what is actually the right move." This sheet names the leave criteria in advance, so that when load grows or requirements change, the decision has already been made structurally rather than reactively.

The criteria are observable and specific. Vague signals like "it feels too slow" or "the team is worried about scale" are not leave signals — they are audit triggers. Run the audit in `pragma-discipline.md`, `transactions-and-isolation.md`, and `concurrent-access-patterns.md` before interpreting them as leave signals. A quarter-long migration project is the wrong fix for a missing `PRAGMA journal_mode = WAL`.

This sheet covers both engines in the pack: SQLite and DuckDB. DuckDB's boundary is narrower than SQLite's in one direction and broader in another. DuckDB's leave signals are simpler: DuckDB is single-process (one process can have a read-write connection; multiple processes can have read-only connections), it has no built-in network server mode for production multi-client use, and it is designed for analytical queries rather than OLTP point-writes. If the workload needs multi-process writes or a client-server deployment model, DuckDB's boundary is reached immediately — it was never designed for that shape. The migration path from DuckDB is the same as from SQLite: Postgres for OLTP, or a purpose-built analytical warehouse (ClickHouse, BigQuery) for OLAP at scale.

## What this pack does NOT cover

- **Cross-host replication.** SQLite's locking model is designed for a single host's filesystem. Sharing a `.db` file over a network filesystem across hosts is not a supported configuration — it is a correctness boundary, not a performance limit. Load `axiom-web-backend` for the API layer and a server database for the shared store.
- **Row-level access control.** SQLite has no roles and no built-in row-level security. Enforcing which users may read which rows is an application concern; the application implements it by filtering its queries. If the access-control model is complex enough that it needs to live in the database itself — declarative policies, role hierarchies, audit-of-access enforcement — that complexity is a leave signal. A simple application-layer check is not.
- **Cross-region failover.** SQLite does not replicate. Failover across regions requires a server database with a replication stream, a shared distributed store, or an application-level write fanout. None of these fit the embedded model.
- **Multiple simultaneous writers at sustained throughput.** SQLite is single-writer at the WAL level — one write transaction proceeds at a time. A write queue in the application buys headroom by serialising writes before they reach SQLite, but it does not add throughput: it trades latency spikes for a queue that can drain at the rate the single writer can commit. Queueing buys time; it does not increase the per-second write ceiling.
- **DDL changes coordinated across many app instances without downtime.** Rolling schema migrations across multiple processes sharing one database file are possible with strict discipline (`schema-migrations.md`), but each additional instance adds coordination surface. If you are running more than two or three writer processes against the same file and rolling migrations are becoming operationally expensive, that is a leave signal.

## The leave signals

These are concrete, observable thresholds. Measure before deciding.

| Signal | Threshold | Notes |
|--------|-----------|-------|
| Sustained write rate | > ~10K small transactions/sec on a fast NVMe SSD, WAL + `synchronous=NORMAL` | This is the practical ceiling on NVMe hardware; HDD and cloud-volume storage will be lower. Measure with realistic transaction size — a 10-row batch commit and a 1-row commit are not the same. |
| Writer processes | > 1 host needs to write | Once writes come from more than one machine, the embedded model is structurally wrong. There is no threshold to tune past. |
| Schema migrations under live traffic | Multi-instance deploy with rolling migration | Each additional instance sharing the database file multiplies the coordination surface. One instance is fine; two is manageable with `BEGIN IMMEDIATE` discipline; more is fragile. |
| Access control requirements | RBAC, audit-of-access, column-level encryption in the database layer | None of these belong in SQLite. Application-layer enforcement is the embedded pattern; if the requirement explicitly needs database-layer enforcement, leave. |
| Database size | > 100 GB and growing | SQLite handles large files, but operational costs grow: `PRAGMA integrity_check` takes minutes to hours, `VACUUM INTO` backup time lengthens, migration time on a large file grows proportionally. Measure the actual operation times before treating size as a leave trigger — "100 GB SQLite" is not inherently a problem, but "100 GB SQLite with 10-minute backup windows and 30-minute integrity checks" may be. |
| Cross-host access | Any pattern requiring cross-host shared state | The NFS caveat in `concurrent-access-patterns.md` and `pragma-discipline.md` is not a performance warning — it is a correctness prohibition. |

**The over-leave anti-pattern**: if the signal is write-rate or SQLITE_BUSY, audit `pragma-discipline.md` (WAL enabled? `busy_timeout` set?) and `transactions-and-isolation.md` (`BEGIN IMMEDIATE` on all write paths?) first. If the signal is corruption, audit `backup-restore-and-corruption.md` and `concurrent-access-patterns.md`. If the signal is slow reads, audit `duckdb-for-analytics.md` — the OLAP workload may belong in DuckDB, not in a new server database. The section below on the over-leave anti-pattern expands on this.

### DuckDB leave signals

DuckDB's leave criteria are structurally different from SQLite's because DuckDB is not designed for OLTP or multi-process concurrent writes.

| Signal | Leave destination |
|--------|------------------|
| Multiple processes need a read-write connection simultaneously | Any server database (Postgres, MySQL) — DuckDB enforces a single read-write connection per file |
| The workload is OLTP (many small point-writes, per-row reads) | SQLite for single-host; Postgres for multi-host |
| The analytical dataset exceeds single-machine RAM + disk capacity | Purpose-built warehouse: ClickHouse, BigQuery, Snowflake |
| A network-accessible SQL endpoint for multiple clients is required | ClickHouse, Postgres with `pg_analytics` / `timescaledb`, or DuckDB MotherDuck (managed cloud offering) — DuckDB's in-process model has no production multi-tenant server mode |

DuckDB does not have a leave-signal equivalent to SQLite's "you misconfigured it." DuckDB's constraints are architectural, not configuration-tunable. If the workload is multi-process writes, there is no PRAGMA to set — it simply isn't the right tool.

For the hybrid OLTP+OLAP pattern, the correct architecture is SQLite for writes and DuckDB for reads — not DuckDB replacing SQLite for both. Reach the DuckDB leave signal by needing cross-host OLAP at warehouse scale, not by pushing it into a write path it was never designed for.

## Migration paths

### SQLite → PostgreSQL

PostgreSQL is the canonical migration target from SQLite for applications that need multi-writer concurrency, RBAC, or large-dataset operations.

**`pgloader`** is the standard migration tool. It reads a SQLite database file directly and loads it into Postgres, handling the type-mapping pass automatically. Install via the system package manager (`apt install pgloader`) or build from source.

```bash
pgloader sqlite:///path/to/database.db postgresql://user:pass@host/dbname
```

pgloader handles most common cases automatically but review these specifically:

| SQLite construct | Postgres equivalent | Notes |
|------------------|---------------------|-------|
| `INTEGER PRIMARY KEY` (rowid alias) | `BIGSERIAL PRIMARY KEY` or `GENERATED ALWAYS AS IDENTITY` | pgloader emits `BIGINT` + sequence; verify your ORM or query layer matches |
| `TEXT` with date/time values | `TIMESTAMP WITHOUT TIME ZONE` or `TIMESTAMPTZ` | SQLite stores dates as text; Postgres types them strictly — add explicit casts in the pgloader transform step |
| `REAL` floating point | `DOUBLE PRECISION` | Direct mapping; no action needed |
| `BLOB` | `BYTEA` | Direct mapping; pgloader handles the encoding |
| `json_extract(col, '$.field')` queries | `col->'field'` or `col->>'field'` (for `JSONB`) | Replace `JSON1` columns with `JSONB`; rewrite `json_extract` expressions as Postgres JSON operators |
| FTS5 virtual tables | `tsvector` / `tsquery` + `GIN` index | No direct migration; rewrite as Postgres full-text-search columns. FTS5 has richer tokenizer/stemmer options — verify ranking equivalence |
| `PRAGMA user_version` as migration register | A `schema_migrations` table (standard Rails/Flyway pattern) or Postgres-native migration runner | The integer register works, but Postgres migration tooling generally uses a table |

After loading, run `ANALYZE` on all tables and verify query plans — Postgres's query planner differs from SQLite's and previously fast queries may need indexes that were not needed in SQLite.

**Runner migration**: if you are using a custom migration runner keyed on `PRAGMA user_version` (as described in `schema-migrations.md`), rewrite it against a `schema_migrations` table with a `version` integer column or adopt a Postgres-native runner (Flyway, Liquibase, Alembic) at migration time. Do not carry the `user_version` pattern into Postgres — it works but is invisible to standard tooling.

**Optimistic locking carries over.** If the application uses version columns and CAS-style `WHERE version = $expected` updates (as described in `optimistic-locking-and-leases.md`), that pattern is valid in Postgres and MySQL without change. The version column and the update guard are application logic; they do not depend on SQLite's file-lock model. Keep them after migration — they close the same lost-update race in a server database.

**Lease tables carry over.** A claim-lease table that grants exclusive access to a job or resource via a `claimed_until` timestamp and a CAS insert pattern continues to work in Postgres. The difference is that Postgres `INSERT ... ON CONFLICT DO NOTHING` (or `INSERT IGNORE` in MySQL) replaces SQLite's `INSERT OR IGNORE`. Rewrite the conflict clause; the logical shape is identical.

### SQLite → MySQL / MariaDB

The migration shape is similar to Postgres but with some differences:

- Type mapping is broadly the same. `TEXT`, `INTEGER`, `REAL`, and `BLOB` all have direct MySQL equivalents.
- JSON: MySQL 5.7+ and MariaDB 10.2+ have a `JSON` type with a `->` / `->>` path operator similar to Postgres's `json` type. Fewer functions than Postgres `JSONB` but the migration is straightforward.
- Full-text search: MySQL and MariaDB both have `FULLTEXT` indexes (InnoDB). They work; they are less configurable than FTS5 (limited tokenizer control, simpler BM25-variant ranking, no custom stemmer plugins), but they are not absent. Rewrite FTS5 virtual tables as `FULLTEXT` indexed columns and rewrite FTS5 queries as `MATCH ... AGAINST` expressions.
- pgloader supports MySQL as a target. Alternatively, `sqlite3` `.dump` output piped through a type-substitution pass works for smaller databases.

### Hybrid: keep SQLite for one component, move another to Postgres

This is the most common real outcome. An application typically has one write-heavy component that is hitting the single-writer ceiling and several read-heavy or low-volume components that are well within SQLite's envelope.

Before migrating the entire application:

1. Identify which component is actually generating the leave signal.
2. Move that component to Postgres.
3. Leave the others on SQLite.

The result is a system where the configuration service, the feature-flag store, and the local audit log continue running on SQLite (simple, no operational cost, no network hop) while the high-write transactional core runs on Postgres. This is not a transitional architecture — it is a permanent design where each component uses the store that fits its actual workload.

The coordination cost is that two database layers exist; the payoff is that neither is mismatched to its workload.

**When the hybrid is not appropriate**: if the leave signal is cross-host replication — not write-rate — then all components that need shared state must move together. A hybrid that keeps some components on SQLite while others share a Postgres instance is fine only if the SQLite components are genuinely local-only (no shared state across hosts). If any component needs state visible on more than one machine, it must be in the shared store.

### Migration sequencing

Regardless of the target database, the migration sequence follows the same shape:

1. **Stand up the target database alongside the existing SQLite store.** Do not switch traffic until the migration is complete and verified.
2. **Run `pgloader` (or equivalent) against a copy of the production database.** Never run the initial load against a live writer.
3. **Replay in-flight writes** to the target database using a dual-write pass (write to both SQLite and Postgres during the cutover window) or accept a brief write-hold during final cutover.
4. **Verify row counts and spot-check data** before redirecting traffic. Run `ANALYZE` (Postgres) or `ANALYZE TABLE` (MySQL) on all migrated tables.
5. **Redirect traffic** to the new database. Keep the SQLite file available in read-only mode for at least one release cycle for rollback.
6. **Remove the dual-write pass and decommission the SQLite file** after the rollback window expires.

This sequencing applies equally to the hybrid case — the component being migrated follows the same steps; the unchanged components continue on SQLite without interruption.

## What changes after you leave

Moving to a server database changes the operational model in concrete ways that are worth naming explicitly before the migration begins. The table below summarises the key differences; each row is explained in the paragraphs that follow.

| Concern | SQLite (embedded) | Postgres / MySQL (server) |
|---------|------------------|--------------------------|
| Connection cost | Microseconds, in-process | Tens of milliseconds, TCP socket to server process |
| Connection pooling | Optional (per-thread pattern suffices) | Mandatory (PgBouncer, pgpool-II, HikariCP) |
| Default isolation | Snapshot isolation (WAL readers) + serial writes | `READ COMMITTED` (Postgres), `REPEATABLE READ` (MySQL) |
| Write contention signal | `SQLITE_BUSY` | Deadlock (`40P01`), lock timeout (`55P03`), serialisation failure |
| Backup mechanism | File operation: `VACUUM INTO`, Online Backup API | `pg_dump` / `pg_basebackup` / `pgBackRest`; `mysqldump` / `xtrabackup` |
| Concurrency mechanism | OS file locks + WAL | MVCC |
| DDL during traffic | Requires migration window (no concurrent writes) | `CREATE INDEX CONCURRENTLY`; transactional DDL |
| Schema version register | `PRAGMA user_version` (integer, per-file) | `schema_migrations` table; Flyway / Liquibase / Alembic |

**Connection pooling becomes mandatory.** SQLite connections are in-process; opening a new connection costs microseconds and the only risk is per-thread contention. A Postgres or MySQL connection is a TCP socket to a separate process; opening one costs tens of milliseconds and the server has a connection limit. Every application that moves to a server database must introduce a connection pool (PgBouncer, pgpool-II, `asyncpg` pool, HikariCP). There is no equivalent concern with SQLite.

**Transaction isolation model changes.** SQLite in WAL mode provides snapshot isolation for readers and serialised writes — readers see a consistent snapshot pinned at their `BEGIN`, and writes are serialised through the WAL writer lock. Postgres defaults to `READ COMMITTED` (each statement sees the latest committed state) and offers `REPEATABLE READ` and `SERIALIZABLE` as explicit upgrades. MySQL defaults to `REPEATABLE READ` with MVCC. The result: queries that behaved as if under snapshot isolation in SQLite may produce different results in Postgres under the default `READ COMMITTED` level. Audit transaction-boundary assumptions after migration.

**`busy_timeout` is replaced by lock-wait timeouts.** SQLite's `busy_timeout` retries internally for a configured duration before returning `SQLITE_BUSY`. Postgres and MySQL use a `lock_timeout` (fail fast) and `statement_timeout` (maximum wall-clock time for a statement). The application's retry logic may need to be rewritten against the server database's error codes (`lock_timeout` raises `55P03` in Postgres; deadlocks raise `40P01`).

**Backups are database-server-driven.** SQLite backup is a file operation: `VACUUM INTO`, the Online Backup API, or a checkpoint-and-copy sequence. On Postgres, the backup toolchain is `pg_dump` (logical), `pg_basebackup` (physical), or WAL archiving with `pgBackRest` or `Barman`. On MySQL, `mysqldump` (logical) or `xtrabackup` (physical). The file-level backup discipline from `backup-restore-and-corruption.md` does not apply to server databases.

**Isolation is now via MVCC instead of file locks.** SQLite uses OS-level file locks to serialise writers; Postgres and MySQL use Multi-Version Concurrency Control. The failure modes differ: SQLite produces `SQLITE_BUSY`; Postgres produces deadlocks, serialisation failures, or silent phantom reads depending on isolation level. Application error handling must be updated to handle server-database concurrency errors, not SQLite lock errors.

**DDL is online.** Postgres supports `CREATE INDEX CONCURRENTLY`, transactional DDL, and partial table rewrites without exclusive locks (depending on the operation and version). SQLite schema migrations are atomic per migration but require the application to not write during the migration window. After migration, the DDL ceremony from `schema-migrations.md` remains correct in spirit (version the schema, never run migrations without a transaction, keep a forward-only track), but the implementation changes to a Postgres-native runner.

**The single-file abstraction disappears.** One of SQLite's operational virtues is that the database is a single file — it can be copied, inspected with the `sqlite3` CLI, diffed with `sqldiff`, and carried between environments without a server. After migration, the database is a server-managed cluster of files in a data directory; the `sqlite3` CLI does not apply; inspection requires `psql` or `mysql`; "copy the database" means `pg_dump` + `pg_restore`. This is not a deficiency of server databases — it is a deliberate trade-off. Name it explicitly so the operations team knows what to plan for.

## When NOT to leave

The most expensive migration mistake is migrating because of a problem that was actually a misconfiguration. Most "we outgrew SQLite" stories, when traced to their root cause, are one of these four patterns:

**SQLITE_BUSY under concurrent writes** is almost never a capacity problem. It is almost always `BEGIN DEFERRED` on a write path, missing `busy_timeout`, or WAL mode disabled. A common pattern: an application that ran fine in development hits `SQLITE_BUSY` under any production load because the development environment never had concurrent processes. Before treating this as a leave signal, verify: `PRAGMA journal_mode` returns `wal`; `PRAGMA busy_timeout` returns a value ≥ 5000; all write-path transactions use `BEGIN IMMEDIATE`. If all three are true and `SQLITE_BUSY` persists at a rate that `busy_timeout` cannot absorb, then write rate is the actual problem. Audit `transactions-and-isolation.md` and `pragma-discipline.md` before concluding.

**Slow aggregate queries** are almost never a SQLite-vs-Postgres problem. They are a DuckDB-vs-SQLite problem. Before migrating an analytics workload from SQLite to Postgres, read `duckdb-for-analytics.md`. A columnar in-process engine that takes 200ms to aggregate 10 million rows is a better fit than a Postgres server that takes 300ms. DuckDB can query SQLite databases directly via the `sqlite_scanner` extension, making the evaluation zero-migration-cost.

**Large database files** are not automatically a problem. SQLite has been deployed with databases exceeding a terabyte in research contexts. The question is not the file size but the operation times: integrity check duration, backup window, migration time. Measure those before treating size as a leave trigger.

**Multiple writers** requires care about what "multiple" means. Multiple threads within one process, each with their own connection, are fine — they are serialised through the WAL writer lock and `busy_timeout`. Multiple processes on the same host writing to the same file are fine — same mechanism. Multiple processes on different hosts — that is the correctness boundary. Host count is the leave trigger, not process count.

Before committing to a migration, work through this checklist:

| Check | Tool / sheet | Leave if… |
|-------|-------------|-----------|
| WAL mode enabled, `synchronous=NORMAL`, `busy_timeout` ≥ 5000 | `pragma-discipline.md`, `/audit-sqlite-discipline` | Write-rate ceiling still hit after correct PRAGMA config |
| All write paths use `BEGIN IMMEDIATE` | `transactions-and-isolation.md` | `SQLITE_BUSY` persists after fix |
| Slow queries are OLAP shape (aggregates over large tables) | `duckdb-for-analytics.md`, `/profile-sqlite-workload` | DuckDB cannot close the gap (requires cross-host writes) |
| Writer count | `concurrent-access-patterns.md` | More than one host must write |
| Access-control requirement | `encryption-with-sqlcipher.md`, `parameterized-sql-only.md` | Requirement is database-layer RBAC, not application-layer filtering |
| Backup operation times | `backup-restore-and-corruption.md` | Backup window or integrity-check time exceeds operational SLA |

If the evidence supports migration, identify which component generates the signal and consider the hybrid path before migrating everything.

The cost of an unnecessary migration is not only the engineering time. It is the ongoing operational cost of running a server database — connection pooling infrastructure, server lifecycle management, backup tooling, monitoring, and the latency floor that a network hop introduces on every query. Embedded databases earn their cost by eliminating that entire operational tier. Leaving them before the leave signals are clearly met means paying that cost without a corresponding benefit.

## Cross-references

All twelve preceding sheets close back here because the leave decision depends on understanding the embedded model fully:

- `sqlite-fundamentals.md` — the in-process model that defines why cross-host sharing is a correctness boundary, not a performance limit.
- `pragma-discipline.md` — audit WAL and `synchronous` before interpreting write-rate complaints as leave signals.
- `schema-migrations.md` — migration-runner discipline that carries forward (in spirit) to any server database; the leave signal for DDL coordination complexity.
- `transactions-and-isolation.md` — audit `BEGIN` flavour before interpreting `SQLITE_BUSY` as a leave signal; the isolation-model differences after migration are named here.
- `concurrent-access-patterns.md` — the NFS prohibition and single-writer model that make cross-host access a hard boundary.
- `optimistic-locking-and-leases.md` — application-layer conflict detection patterns that survive migration to any server database.
- `parameterized-sql-only.md` — parameterisation discipline applies identically after migration; no change needed.
- `json1-and-structured-data.md` — `JSON1` → `JSONB` migration story and the `json_extract` → JSON operator rewrite.
- `fts5-full-text-search.md` — FTS5 → `tsvector`/`tsquery` migration story and the tokenizer-equivalence caveat.
- `duckdb-for-analytics.md` — OLAP leave signals belong here before considering a server database; DuckDB often closes the gap without a migration.
- `encryption-with-sqlcipher.md` — encryption/access-control requirements that exceed what SQLCipher + application-layer RBAC can close are a leave signal toward a server database with native access control.
- `backup-restore-and-corruption.md` — backup operation times at large database sizes; the backup model changes entirely after migration.

External packs:

- `axiom-web-backend` — for the API surface after migration; connection pooling, ORM configuration, and endpoint design in the context of a server database.
- `axiom-audit-pipelines` — if the audit trail moves to a server database separately from the application store, the chain-construction discipline in that pack applies regardless of the physical store.
- `ordis-security-architect` — for the threat-model handoff when encryption or access-control requirements are the leave driver: data-at-rest classification, key management, and what the new system's security posture needs to cover that SQLCipher did not.
