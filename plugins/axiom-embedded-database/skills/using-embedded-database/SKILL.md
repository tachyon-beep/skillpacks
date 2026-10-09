---
name: using-embedded-database
description: "Use when SQLite or DuckDB application behavior depends on connection configuration, write concurrency, atomic claims, migrations, backup/restore, FTS or query/storage boundaries."
---

# Embedded database operations

Check the installed database version, connection lifecycle and filesystem before applying recipes. The relevant evidence is the behavior of the actual configured connections.

## Work from the affected contract

1. Identify workload, database build/version, connection ownership, storage/filesystem and durability requirement. Confirm the workload fits the embedded engine.
2. Inspect effective connection/file PRAGMAs and transaction ownership. Configure connection-scoped settings on every connection, with the required isolation and busy behavior.
3. Trace read-modify-write and claim paths. Use atomic predicates/transactions and expiry/fencing semantics for concurrent workers; do not hold write locks across network or user think-time.
4. Use parameterized values and explicit identifier allowlists. Inspect query plans and index/FTS/JSON maintenance on representative data.
5. Verify affected contention, migration rollback, crash durability, backup and restore behavior. Use file-backed tests for promises involving locking or WAL.

## Scope and completion

Deliver the affected configuration/transaction contract and observed checks. A query repair does not require database redesign. Apply migration procedures only when stored state must survive; follow the project retention/reset policy. Encryption and retention promises need independent deployment checks.

Use the user’s existing intent and authorization. Ask only for missing information that materially changes the result; use additional reviewers when they address a concrete uncertainty. Treat unavailable checks as gaps rather than successful verification.

## Focused references

Read only the relevant sections. These are optional technical references, not a required reading sequence or a checklist of artifacts to manufacture. Verify version-specific recipes against the installed toolchain.

| Concern | Reference |
|---|---|
| Backup, Restore, and Corruption | [backup-restore-and-corruption.md](backup-restore-and-corruption.md) |
| Boundary and When to Leave | [boundary-and-when-to-leave.md](boundary-and-when-to-leave.md) |
| Concurrent Access Patterns | [concurrent-access-patterns.md](concurrent-access-patterns.md) |
| DuckDB for Analytics | [duckdb-for-analytics.md](duckdb-for-analytics.md) |
| Encryption with SQLCipher | [encryption-with-sqlcipher.md](encryption-with-sqlcipher.md) |
| FTS5 Full-Text Search | [fts5-full-text-search.md](fts5-full-text-search.md) |
| JSON1 and Structured Data | [json1-and-structured-data.md](json1-and-structured-data.md) |
| Optimistic Locking and Claim Leases | [optimistic-locking-and-leases.md](optimistic-locking-and-leases.md) |
| Parameterized SQL Only | [parameterized-sql-only.md](parameterized-sql-only.md) |
| PRAGMA Discipline | [pragma-discipline.md](pragma-discipline.md) |
| Schema Migrations | [schema-migrations.md](schema-migrations.md) |
| SQLite Fundamentals | [sqlite-fundamentals.md](sqlite-fundamentals.md) |
| Transactions and Isolation | [transactions-and-isolation.md](transactions-and-isolation.md) |

## Optional task entry points

- [audit-sqlite-discipline](../../commands/audit-sqlite-discipline.md): Audit sqlite discipline for the affected embedded database contract, with scoped source and verification evidence.
- [profile-sqlite-workload](../../commands/profile-sqlite-workload.md): Profile sqlite workload for the affected embedded database contract, with scoped source and verification evidence.
- [scaffold-sqlite-schema](../../commands/scaffold-sqlite-schema.md): Scaffold sqlite schema for the affected embedded database contract, with scoped source and verification evidence.

Use a specialist agent for a bounded independent investigation or review when useful. Available roles: [embedded-database-reviewer](../../agents/embedded-database-reviewer.md), [sqlite-schema-architect](../../agents/sqlite-schema-architect.md). No fixed reviewer count is required.
