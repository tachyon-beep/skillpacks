---
name: using-embedded-database
description: Use when running **SQLite or DuckDB inside an application process** as the durable store — not as a development convenience but as the production database. Use when scaling an SQLite layer that worked at low concurrency and is now hitting SQLITE_BUSY, WAL bloat, lock contention, schema-migration ceremony, or correctness gaps under multi-process writers. Use when introducing DuckDB as an OLAP complement to an OLTP SQLite store, or when picking between the two for a new component. Pairs with `/web-backend` (the API surface above the DB) and `/audit-pipelines` (when the DB is also the audit trail). Do not load for server databases (Postgres, MySQL), key-value stores, or ORM choice in isolation.
---

# Using Embedded Database

## Overview

**An embedded database is not a database with the network removed. It is a different discipline — the application owns concurrency, the process owns the lock, and the file owns the durability contract.**

Server databases (Postgres, MySQL) separate concerns across process boundaries: a dedicated server process serialises writes, mediates locks, and answers client connections from an independent runtime. When that runtime dies, a supervising process restarts it. When a client misbehaves, the server can forcibly disconnect it. The durability contract runs in a process the application does not own.

Embedded databases invert every one of these properties. The application is the lock manager. The application is the crash recovery agent. The write serialiser is whatever the application does before calling `sqlite3_exec`. When the application dies mid-write, the database file is what the OS left behind — and the database's own WAL or journal is the only recovery path. There is no separate supervisor; the application and the durability contract live in the same address space.

SQLite and DuckDB fail differently from each other, and both fail differently from server databases. SQLite is single-writer at the file level: WAL mode improves read concurrency but one writer still holds the write lock; a second writer sees `SQLITE_BUSY` and must retry or fail. DuckDB is column-oriented and optimised for analytical scans; it does not belong in a multi-process OLTP write path. Using either one outside its design envelope produces failure modes that a "just use a database" mental model does not predict.

This pack addresses five failure modes that recur in embedded-database production deployments:

1. **PRAGMA mis-configuration** — SQLite ships with safe-but-slow defaults (`journal_mode=DELETE`, `synchronous=FULL`, `cache_size` sized for 1985). Applications that never touch PRAGMAs are paying a performance tax in exchange for defaults they didn't choose.
2. **Schema-migration mis-design** — ad-hoc `CREATE TABLE IF NOT EXISTS` at startup, unversioned `ALTER TABLE`, no migration runner, no rollback path.
3. **Transaction-isolation mis-choice** — `BEGIN` used interchangeably with `BEGIN IMMEDIATE` and `BEGIN EXCLUSIVE`; deferred transactions that collide at first write; SERIALIZABLE isolation assumed where SNAPSHOT semantics apply.
4. **Concurrent-write mis-coordination** — multi-process writers without WAL, NFS mounts, advisory lock ceremonies that are skipped under load, optimistic updates with no version check.
5. **Encryption mis-attribution** — SQLCipher added to satisfy a compliance checkbox, keyed with a static string in source, with no analysis of what threat model it actually closes.

## When to Use

Use this pack when:

- You are building a new SQLite-backed component and want to get PRAGMA configuration, schema migration, transaction discipline, and parameterisation right before the first production deploy.
- You are scaling an existing SQLite layer that worked fine at low concurrency and is now producing `SQLITE_BUSY`, WAL file bloat, or unexplained data races under multi-process writers.
- You are introducing DuckDB as an OLAP complement to an OLTP SQLite store — or deciding whether to use DuckDB instead of SQLite for a new component.
- You are debugging `SQLITE_BUSY`, WAL accumulation, database corruption, or unexplained write failures and need a systematic root-cause checklist.
- You need to add full-text search, JSON storage, or encryption to an existing SQLite store and want to avoid the virtual-table traps.
- You are choosing between SQLite and DuckDB for a new workload and the right answer depends on read/write ratio, query shape, and concurrency model.

Do **not** use this pack when:

- Your database is Postgres or MySQL — load `/web-backend` instead.
- Your question is purely about ORM library selection in isolation — load `/web-backend` or the relevant language-engineering pack.
- You need distributed concurrency across multiple hosts — sheet 13 (`boundary-and-when-to-leave.md`) explains exactly why embedded databases fail here and where to go instead.
- Your question is database theory (CAP, consensus, serializability proofs) without an implementation anchor.
- You are designing the cryptographic chain for an audit trail — even if storage is SQLite, load `/audit-pipelines` for the chain; this pack covers the storage layer only.

## Start Here

If your input is "we have (or want) an embedded database and need it to be production-grade," and you have not run this pack before:

1. [`sqlite-fundamentals.md`](sqlite-fundamentals.md) — the SQLite execution model: connection lifecycle, the VFS layer, WAL vs rollback journal, the type-affinity system, and the connection-pool shape that survives multi-threaded access.
2. [`pragma-discipline.md`](pragma-discipline.md) — PRAGMA configuration is the application: `journal_mode`, `synchronous`, `cache_size`, `temp_store`, `foreign_keys`, `busy_timeout` — choose them deliberately, not by default.
3. [`schema-migrations.md`](schema-migrations.md) — version the schema before building anything else: migration runner discipline, `user_version` / `application_id` as the version register, forward-only vs rollback strategies, the SQLite `ALTER TABLE` constraint surface.
4. [`transactions-and-isolation.md`](transactions-and-isolation.md) — choose `BEGIN` flavour deliberately: DEFERRED vs IMMEDIATE vs EXCLUSIVE, the lock-upgrade surprise, SAVEPOINT for nested scopes, isolation semantics in WAL mode.
5. [`concurrent-access-patterns.md`](concurrent-access-patterns.md) — single-writer is a design, not a workaround: WAL reader/writer coexistence, multi-process safe patterns, shared-cache mode pitfalls, NFS and network-filesystem prohibitions.
6. [`optimistic-locking-and-leases.md`](optimistic-locking-and-leases.md) — turn write races into testable application logic: version columns, CAS-style update patterns, lease tables, conflict detection at the application layer rather than the lock layer.
7. [`parameterized-sql-only.md`](parameterized-sql-only.md) — one rule with no exceptions: every SQL statement uses bound parameters; `executescript()` with string interpolation is never acceptable regardless of input provenance.
8. [`json1-and-structured-data.md`](json1-and-structured-data.md) — JSON columns without losing schema discipline: `json_extract` indexed expressions, partial-JSON update patterns, when JSON beats a normalised schema and when it doesn't.
9. [`fts5-full-text-search.md`](fts5-full-text-search.md) — when full-text search is a first-class feature: FTS5 virtual table creation, content-table sync triggers, porter tokeniser vs unicode61, snippet and rank functions, the rebuild/integrity-check lifecycle.
10. [`duckdb-for-analytics.md`](duckdb-for-analytics.md) — when DuckDB beats SQLite: columnar scan performance, the SQLite extension for reading `.db` files, ATTACH patterns for OLAP-over-OLTP, and when not to introduce DuckDB.
11. [`encryption-with-sqlcipher.md`](encryption-with-sqlcipher.md) — match the threat model before adding encryption: SQLCipher key derivation, PBKDF2 iteration count, re-keying discipline, what SQLCipher does and does not protect against.
12. [`backup-restore-and-corruption.md`](backup-restore-and-corruption.md) — the backup you never tested is not a backup: Online Backup API vs file-copy semantics, WAL checkpoint before backup, corruption detection with `PRAGMA integrity_check`, the `sqlite3_recover` path.
13. [`boundary-and-when-to-leave.md`](boundary-and-when-to-leave.md) — read last, recall when scaling: the signals that mean embedded storage has reached its envelope and a server database is the right next move.

## Sheet Index

| Sheet | Role |
|-------|------|
| [`sqlite-fundamentals.md`](sqlite-fundamentals.md) | Connection model, VFS, WAL vs rollback journal, type affinity, connection-pool shape |
| [`pragma-discipline.md`](pragma-discipline.md) | Production PRAGMA configuration: journal mode, sync level, cache, foreign keys, busy timeout |
| [`schema-migrations.md`](schema-migrations.md) | Versioned schema evolution: migration runner, `user_version`, ALTER TABLE constraints, rollback strategy |
| [`transactions-and-isolation.md`](transactions-and-isolation.md) | BEGIN flavour selection, lock-upgrade semantics, SAVEPOINT, WAL-mode isolation model |
| [`concurrent-access-patterns.md`](concurrent-access-patterns.md) | Multi-reader/single-writer discipline, WAL coexistence, NFS prohibitions, shared-cache pitfalls |
| [`optimistic-locking-and-leases.md`](optimistic-locking-and-leases.md) | Version columns, CAS updates, lease tables, application-layer conflict detection |
| [`parameterized-sql-only.md`](parameterized-sql-only.md) | Bound parameters everywhere; `executescript` + interpolation is never acceptable |
| [`json1-and-structured-data.md`](json1-and-structured-data.md) | JSON column discipline, indexed `json_extract`, schema-vs-JSON tradeoffs |
| [`fts5-full-text-search.md`](fts5-full-text-search.md) | FTS5 virtual tables, content-table sync, tokeniser choice, rank and snippet functions |
| [`duckdb-for-analytics.md`](duckdb-for-analytics.md) | DuckDB columnar scan, SQLite ATTACH, OLAP-over-OLTP patterns, when not to use DuckDB |
| [`encryption-with-sqlcipher.md`](encryption-with-sqlcipher.md) | SQLCipher key derivation, threat model alignment, re-keying, static-key prohibition |
| [`backup-restore-and-corruption.md`](backup-restore-and-corruption.md) | Online Backup API, WAL checkpoint discipline, integrity check, corruption recovery |
| [`boundary-and-when-to-leave.md`](boundary-and-when-to-leave.md) | Envelope signals, migration path from SQLite to a server database, DuckDB limits |

## Anti-Patterns This Pack Closes

1. **Opening one connection per request and inheriting it across threads.** SQLite connections are not thread-safe by default; sharing a connection across threads without explicit serialisation produces data races that `SQLITE_BUSY` does not fully describe. *(sqlite-fundamentals)*

2. **Default `journal_mode=DELETE` in a multi-reader workload.** Rollback-journal mode takes a shared lock on every read to prevent a writer from starting mid-read; WAL mode avoids this entirely and is the correct default for any workload with concurrent readers. *(pragma-discipline)*

3. **`ALTER TABLE … DROP COLUMN` ignoring SQLite's pre-3.35 limitation.** SQLite did not support `DROP COLUMN` until 3.35.0 (2021-03-12); migrations written against that assumption silently fail or corrupt schema state on older installs. *(schema-migrations)*

4. **`BEGIN DEFERRED` followed by a write, then `SQLITE_BUSY` surprise at first INSERT.** A deferred transaction acquires no lock at `BEGIN`; the write-lock upgrade races with any concurrent writer, and the upgrade fails with `SQLITE_BUSY` rather than blocking. Use `BEGIN IMMEDIATE` when the transaction will write. *(transactions-and-isolation)*

5. **Filesystem-level locking on NFS or SMB.** SQLite's locking protocol depends on POSIX `fcntl` advisory locks working correctly; NFS and SMB implementations routinely violate these semantics, producing silent corruption or spurious lock failures. NFS-backed SQLite databases are unsupported and dangerous. *(concurrent-access-patterns)*

6. **`UPDATE x SET y = … WHERE id = …` with no version check in a multi-writer context.** Without a version column and a CAS-style `WHERE version = $expected` guard, two concurrent writers both read the same row and both update it; the second write silently discards the first. *(optimistic-locking-and-leases)*

7. **`executescript()` with f-string interpolation.** `executescript()` bypasses the parameter binding layer entirely; any string interpolation into SQL text is a SQL injection surface regardless of where the values originate. *(parameterized-sql-only)*

8. **JSON column with no indexed expression and a `json_extract` WHERE clause.** A `WHERE json_extract(col, '$.field') = ?` without a corresponding expression index forces a full table scan; at scale this is indistinguishable from a missing index on a normal column. *(json1-and-structured-data)*

9. **FTS5 virtual table out of sync with its content table (no trigger).** An FTS5 content-table configuration defers index maintenance to the application; without `AFTER INSERT / UPDATE / DELETE` triggers on the content table, the full-text index silently diverges from the data it is supposed to index. *(fts5-full-text-search)*

10. **OLAP scan over a million rows in SQLite when DuckDB would do it in 1/50th the time.** SQLite is a row-store optimised for OLTP point queries; aggregate scans over large result sets pay row-at-a-time overhead that DuckDB's columnar engine eliminates structurally. *(duckdb-for-analytics)*

11. **SQLCipher keyed with a static string in source code.** A static key in source is a key in every build artefact, every log that prints the connection string, and every developer's laptop. SQLCipher provides at-rest encryption; a static embedded key turns it into obfuscation. *(encryption-with-sqlcipher)*

12. **Copying the `.db` file as backup while a writer is mid-transaction.** A filesystem-level copy of a live SQLite database copies whatever page state the OS has, including pages in the middle of an uncommitted write. The resulting file is structurally corrupt. Use the Online Backup API or checkpoint-then-copy. *(backup-restore-and-corruption)*

13. **Treating SQLite as a multi-host shared store.** SQLite's locking model is designed for a single host's filesystem; sharing a `.db` file over a network filesystem across hosts combines anti-patterns 5 and 12 and adds clock-skew. This is not a scaling limitation — it is a correctness boundary. *(boundary-and-when-to-leave)*

## Boundary

This pack does **not** cover:

- **Server databases (Postgres, MySQL, MariaDB)** — load `/web-backend` for the API layer or the appropriate framework-specific guidance.
- **ORM library choice in isolation** — which ORM to use, how to configure it, migrations via ORM tooling rather than raw SQL — load `/web-backend` or the language-engineering pack (`python-engineering`, `rust-engineering`).
- **Database theory at the formal level** (CAP theorem, serialisability proofs, consensus protocols) — out of scope; this pack is operational, not theoretical.
- **Driver and binding implementation** — how to call `rusqlite`, `sqlx`, `sqlite3`, or `aiosqlite` correctly is the language-engineering pack's job (`axiom-rust-engineering`, `axiom-python-engineering`); this pack covers the database discipline those drivers invoke.
- **Web-API design over the database** — schema design for an HTTP API, REST resource mapping, query-to-response shaping — load `/web-backend`.
- **Audit-trail cryptographic chain** — even when an SQLite table is the physical store for an audit log, the append-only chain, fingerprint construction, and signed-export discipline belong to `/audit-pipelines`. This pack covers only the storage layer underneath.

## Routing by Symptom

### "I'm getting SQLITE_BUSY errors under load"

**Symptoms**: `SQLITE_BUSY` or `database is locked` errors; writes fail intermittently; error rate increases under concurrent writers; `busy_timeout` set to zero or not set.

**Route to**: [`transactions-and-isolation.md`](transactions-and-isolation.md) first, then [`pragma-discipline.md`](pragma-discipline.md).

**Why**: `SQLITE_BUSY` at write time usually means a `BEGIN DEFERRED` transaction encountering a live writer at the lock-upgrade step. Setting `busy_timeout` buys retry time; switching to `BEGIN IMMEDIATE` eliminates the surprise. If busy errors persist after both fixes, read [`concurrent-access-patterns.md`](concurrent-access-patterns.md) — the problem may be WAL mode not enabled, or filesystem-level locking on a network mount.

### "The WAL file is growing without bound"

**Symptoms**: `.db-wal` file reaching gigabytes; reads slow as the WAL length grows; checkpoint never seems to complete; `PRAGMA wal_checkpoint` returns unexpected row counts.

**Route to**: [`pragma-discipline.md`](pragma-discipline.md), specifically the `wal_autocheckpoint` and `PRAGMA wal_checkpoint(TRUNCATE)` sections.

**Why**: The WAL grows when readers hold open snapshots that prevent checkpointing. Long-running read transactions, read connections never closed, or `synchronous=OFF` causing premature returns — all prevent the WAL from shrinking. Setting `wal_autocheckpoint` and ensuring readers close promptly is the fix.

### "Schema migration is breaking on upgrade"

**Symptoms**: `ALTER TABLE` fails; `CREATE TABLE IF NOT EXISTS` silently ignores schema changes; migration applied twice or skipped; no way to roll back a bad deploy.

**Route to**: [`schema-migrations.md`](schema-migrations.md).

**Why**: SQLite's `ALTER TABLE` is more constrained than Postgres. `DROP COLUMN` requires 3.35+; `ADD COLUMN` is allowed but constrained. A proper migration runner keyed on `PRAGMA user_version` and locked with `BEGIN IMMEDIATE` is the only safe shape.

### "I want to add full-text search to an existing table"

**Symptoms**: `LIKE '%query%'` scans are too slow; planning to add an FTS index; unsure whether FTS5 or FTS3 is appropriate; FTS results out of date relative to the content table.

**Route to**: [`fts5-full-text-search.md`](fts5-full-text-search.md).

**Why**: FTS5 with a content table and sync triggers is the only production-ready shape. FTS without triggers produces stale results silently; FTS3 is deprecated; the tokeniser choice affects unicode handling and multilingual ranking.

### "Should I use SQLite or DuckDB for this component?"

**Symptoms**: deciding storage for a new analytics feature; `GROUP BY` + aggregation queries dominating load; write volume low, read volume high; existing SQLite store being queried for reports.

**Route to**: [`duckdb-for-analytics.md`](duckdb-for-analytics.md).

**Why**: SQLite is optimised for OLTP point queries. If the workload is dominated by full-table aggregations, window functions, or columnar scans over millions of rows, DuckDB's columnar engine eliminates the row-at-a-time overhead structurally. The sheet gives the decision criteria and the ATTACH pattern for reading SQLite data from DuckDB.

### "I added SQLCipher but I'm not sure it's doing what I think it is"

**Symptoms**: SQLCipher integrated; key is a string literal in the source file or an environment variable; unsure what threat model it addresses; compliance requirement cited but not specified.

**Route to**: [`encryption-with-sqlcipher.md`](encryption-with-sqlcipher.md).

**Why**: SQLCipher protects data at rest from a physical attacker who has the file but not the key. It does not protect against an attacker who has the process, the binary, or the environment. A static key in source provides no meaningful protection — it is in every build artefact. The sheet gives the correct PBKDF2-derived key derivation pattern and the threat model that justifies encryption at all.

### "My database file got corrupted after a crash"

**Symptoms**: `SQLITE_CORRUPT` errors; `PRAGMA integrity_check` reports errors; hot journal or WAL file present; application crashed mid-write.

**Route to**: [`backup-restore-and-corruption.md`](backup-restore-and-corruption.md).

**Why**: SQLite's WAL and rollback journal are the first recovery tools — `PRAGMA integrity_check` tells you whether the file is recoverable. If not, `sqlite3_recover` (3.41+) can extract readable rows from a corrupt database. The sheet covers both recovery path and the backup discipline that makes recovery unnecessary.

## Pipeline Position

```
axiom-embedded-database (this pack)          axiom-web-backend (API layer)
  storage discipline: PRAGMA config,  ←-→   endpoint design, ORM config,
  schema migration, transaction             connection pooling from the
  isolation, concurrency model,             application framework side;
  encryption, backup, corruption            SQL is executed here against
  ────────────────────────────────────────────────────────────────
       The embedded store is not the API. web-backend governs what
       the application layer above the store looks like; this pack
       governs what the store itself does. Load both for a complete
       treatment of an application that has an embedded DB and an API.

axiom-embedded-database (this pack)          axiom-audit-pipelines (evidence)
  SQLite table as physical audit store ←-→  append-only chain, fingerprint
  — page layout, WAL durability,            construction, canonical encoding,
  backup integrity, corruption              signed export, retention policy
  ────────────────────────────────────────────────────────────────
       The physical storage is this pack's concern; the chain on top
       of it is audit-pipelines' concern. A SQLite-backed audit log
       uses both: this pack governs that the pages commit durably and
       the backup is valid; audit-pipelines governs that the rows form
       a verifiable chain.

axiom-embedded-database (this pack)          axiom-python-engineering / axiom-rust-engineering
  database discipline: which PRAGMA  ←-→   driver idioms: how to open a
  to set, what BEGIN flavour to use,        connection in Python or Rust,
  how to version the schema                 how to use the library's API,
  ────────────────────────────────────────────────────────────────
       This pack teaches what to ask the database to do; the language-
       engineering packs teach how to make the driver do it. A Rust
       crate using rusqlite needs both: this pack for the DB discipline,
       rust-engineering for the crate structure and binding correctness.
```

## How to Access Reference Sheets

All reference sheets are in the same directory as this `SKILL.md`. When you see a link like [`pragma-discipline.md`](pragma-discipline.md), read the file from the same directory as this file.

## Quick Reference

| Symptom / Need | Sheet |
|----------------|-------|
| `SQLITE_BUSY` at write time | `transactions-and-isolation.md`, `pragma-discipline.md` |
| WAL file growing without bound | `pragma-discipline.md` |
| Schema migration fails or loops | `schema-migrations.md` |
| Multi-threaded connection sharing | `sqlite-fundamentals.md`, `concurrent-access-patterns.md` |
| Two writers, silent data loss | `optimistic-locking-and-leases.md` |
| SQL injection concern | `parameterized-sql-only.md` |
| FTS results stale or missing | `fts5-full-text-search.md` |
| JSON WHERE clause causes full scan | `json1-and-structured-data.md` |
| Aggregate queries over millions of rows too slow | `duckdb-for-analytics.md` |
| SQLite vs DuckDB choice | `duckdb-for-analytics.md` |
| SQLCipher key management | `encryption-with-sqlcipher.md` |
| Backup while writer is active | `backup-restore-and-corruption.md` |
| `SQLITE_CORRUPT` after crash | `backup-restore-and-corruption.md` |
| NFS or SMB mount | `concurrent-access-patterns.md`, `boundary-and-when-to-leave.md` |
| "When should we move to Postgres?" | `boundary-and-when-to-leave.md` |

## Commands and Agents

The pack ships two slash commands and one agent.

**Commands:**

- `/scaffold-embedded-db` — scaffold a production-grade SQLite layer: PRAGMA configuration, migration runner skeleton, connection-pool shape, parameterised query helpers, and an integrity-check cron stub. Aligned to a declared language target (Python `sqlite3`/`aiosqlite` or Rust `rusqlite`/`sqlx`).
- `/audit-sqlite-layer` — sweep an existing SQLite layer for anti-patterns: PRAGMA configuration gaps, migration runner discipline, transaction flavour misuse, raw string interpolation in SQL, missing indexed expressions on JSON columns, FTS trigger absence. Produces a structured findings list with sheet references.

**Agent:**

- **`embedded-db-reviewer`** — reviews an application's embedded-database usage for correctness and production-readiness gaps. Sweeps source against all 13 sheets and the 13 anti-patterns; reports findings with severity and the sheet that closes each gap. Follows the SME Agent Protocol (Confidence Assessment, Risk Assessment, Information Gaps, Caveats).

## Cross-References

- `axiom-web-backend` — for the API layer above the embedded store: endpoint design, ORM configuration, connection pooling from the application framework's perspective.
- `axiom-audit-pipelines` — when the database is also an append-only audit trail: fingerprint chains, canonical encoding, signed exports, and retention policy sit in that pack even if the physical store is SQLite.
- `axiom-python-engineering` — for Python driver discipline: `sqlite3` module idioms, `aiosqlite` async patterns, `duckdb` Python API, connection lifecycle in async frameworks.
- `axiom-rust-engineering` — for Rust driver discipline: `rusqlite` connection management, `sqlx` compile-time query verification, `duckdb-rs` API surface, feature-flag discipline for bundled vs system SQLite.
- `ordis-security-architect` — for the threat model that SQLCipher is supposed to close: data-at-rest threat modelling, key management, what encryption actually protects against in the context of the broader system.
