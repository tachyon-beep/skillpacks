---
description: SQLite and DuckDB as application-embedded databases at production scale - PRAGMA discipline, schema migrations under SQLite's ALTER constraints, WAL tuning, transactions and isolation (DEFERRED/IMMEDIATE/EXCLUSIVE), single-writer concurrency, optimistic locking and claim leases, parameterized-SQL enforcement, JSON1, FTS5, DuckDB for OLAP, SQLCipher, online backup and corruption recovery, and the boundary at which an in-process database stops earning its cost
---

# Embedded Database Routing

**An embedded database is not a database with the network removed. It is a different discipline - the application owns concurrency, the process owns the lock, and the file owns the durability contract. For server databases (Postgres/MySQL) use `/web-backend` instead.**

Use the `using-embedded-database` skill from the `axiom-embedded-database` plugin to route to the right specialist sheet. Content authority lives in `plugins/axiom-embedded-database/skills/using-embedded-database/SKILL.md` - this wrapper is a thin pointer.

## Sheets

- **sqlite-fundamentals** - in-process execution model, connection lifecycle, ACID in the embedded context, thread/process concurrency rules
- **pragma-discipline** - production PRAGMA block: `journal_mode`, `synchronous`, `cache_size`, `temp_store`, `foreign_keys`, `busy_timeout`, `wal_autocheckpoint`
- **schema-migrations** - versioned schema evolution via `application_id` / `user_version`, migration runner discipline, SQLite ALTER constraints, 12-step rebuild-table pattern
- **transactions-and-isolation** - BEGIN DEFERRED vs IMMEDIATE vs EXCLUSIVE, lock-upgrade semantics, SAVEPOINT, SQLITE_BUSY retry with backoff
- **concurrent-access-patterns** - WAL reader/writer coexistence, multi-process safe patterns, NFS/SMB prohibitions, portable cross-process locking
- **optimistic-locking-and-leases** - version columns, CAS-style updates, claim-lease tables with double-predicate guard, heartbeats for long jobs
- **parameterized-sql-only** - bound parameters everywhere, identifier allowlists, PRAGMA-as-identifier exception, `executescript` hazards, AST-based CI enforcement
- **json1-and-structured-data** - JSON column discipline, indexed `json_extract` expressions, schema-vs-JSON tradeoffs
- **fts5-full-text-search** - FTS5 virtual tables, content-table sync triggers, tokeniser choice, rank and snippet functions
- **duckdb-for-analytics** - DuckDB columnar engine, SQLite ATTACH for OLAP-over-OLTP, when not to introduce DuckDB
- **encryption-with-sqlcipher** - SQLCipher key derivation, PBKDF2 iterations, threat model alignment, re-keying discipline
- **backup-restore-and-corruption** - Online Backup API, VACUUM INTO, WAL checkpoint discipline, `PRAGMA integrity_check`, `sqlite3_recover`
- **boundary-and-when-to-leave** - envelope signals, the over-leave anti-pattern, migration paths from SQLite to Postgres/MySQL via pgloader

## Commands

- `/axiom-embedded-database:scaffold-sqlite-schema` - emit a connection helper, migration runner, optional claim-lease helpers, and a smoke test aligned to a declared workload shape
- `/axiom-embedded-database:audit-sqlite-discipline` - static sweep against the 13-sheet specification with structured findings JSON; optionally dispatches the reviewer agent for narrative synthesis
- `/axiom-embedded-database:profile-sqlite-workload` - EXPLAIN QUERY PLAN sweep, index hit-rate via `sqlite_stat1`, WAL-size sampling, slow-query ranking

## Agents

- `sqlite-schema-architect` - forward design: workload elicitation, schema and indexing strategy, claim-lease shape, migration sequence
- `embedded-database-reviewer` - brownfield audit: sweeps source against all 13 sheets and 13 anti-patterns, produces findings with severity and the sheet that closes each gap

## Cross-references

- API layer over the embedded store → `/web-backend`
- Append-only audit chain over a SQLite table → `/audit-pipelines`
- Python driver discipline (`sqlite3`, `aiosqlite`, `duckdb`) → `/python-engineering`
- Threat model that SQLCipher is supposed to close → `/security-architect`
