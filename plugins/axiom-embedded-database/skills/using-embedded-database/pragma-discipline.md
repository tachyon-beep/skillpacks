---
name: pragma-discipline
description: Use when configuring a SQLite database for production — covers journal_mode, synchronous, busy_timeout, foreign_keys, cache_size, mmap_size, temp_store, application_id, user_version, and the critical distinction between connection-scoped and database-scoped settings. The correct shape for a production PRAGMA block is shown in Python and Rust.
---

# PRAGMA Discipline

**PRAGMA settings are not tuning — they are the application's contract with the database engine. The defaults are wrong for most production workloads, and they are silently wrong.**

SQLite ships with settings chosen for maximum compatibility across every possible deployment environment, including 1980s-era hardware, hostile filesystems, and embedded systems with 256 KB of RAM. An application that never touches PRAGMAs is using those defaults. It is paying for durability guarantees it didn't choose and concurrency behaviour it didn't understand, getting performance it didn't ask for, and skipping enforcement it assumed was on.

This sheet covers every PRAGMA that matters in a production SQLite deployment — what it does, what it should be set to, why, and what happens when it is wrong.

## When this earns its cost

Read this sheet when:

- You are seeing `SQLITE_BUSY` under concurrent read and write load and have not yet confirmed `journal_mode=WAL` is set.
- You are deciding whether to use `synchronous=NORMAL` or `synchronous=FULL` and need the exact durability difference spelled out.
- You have experienced a corruption event and are not sure whether `synchronous` was a contributing factor.
- You cannot remember which PRAGMAs persist in the database file versus which must be re-applied every time a connection is opened — and a connection pool or test suite is applying them inconsistently.
- You are writing or auditing the `connect()` helper that gates all access to the database.

`sqlite-fundamentals.md` covers the connection model that PRAGMAs configure; read that first if you have not.

## The PRAGMA catalogue

### `journal_mode=WAL`

**What it does.** Controls how SQLite handles uncommitted writes. The default, `DELETE`, uses a rollback journal: before modifying a database page, SQLite writes the original page to a separate journal file, then modifies the database page directly. On commit, the journal is deleted. The consequence is that any writer must hold an EXCLUSIVE lock on the entire database file — no readers can proceed while a writer is active, and no writer can proceed while any reader holds a SHARED lock.

`WAL` (Write-Ahead Log) inverts this. Writers append new page versions to a separate `.db-wal` file rather than modifying database pages directly. Readers see a consistent snapshot of the database as of their transaction's start by reading pages from either the main file or the WAL, whichever is newer. The result: readers and one writer proceed concurrently without blocking each other.

**Production-correct value.** `PRAGMA journal_mode = WAL`

**Scope.** Database-scoped — persisted in the database file header. Set it once when the database is first created; it survives connection close and process restart. Verify it by reading `PRAGMA journal_mode` — if the database was previously in DELETE mode, the WAL transition happens on the first connection that issues the command.

**Failure mode if mis-set.** With the default `DELETE` mode and any multi-reader or multi-writer workload: readers block writers and writers block readers. A long-running read transaction causes every write attempt to return `SQLITE_BUSY` immediately. The application appears to have a lock contention problem; the root cause is a missing one-line PRAGMA.

**Caveat.** WAL mode is unsafe on network filesystems (NFS, SMB, CIFS). The WAL protocol requires shared-memory coordination via the `.db-shm` file, which uses POSIX memory-mapped locking semantics that NFS implementations routinely violate. On NFS, WAL mode can produce silent corruption. If the filesystem is a network mount, use `DELETE` mode with `busy_timeout` and accept the concurrency constraints — or move the database off the network filesystem.

**WAL sidecar files.** In WAL mode, SQLite creates two additional files alongside the database: `.db-wal` (the write-ahead log) and `.db-shm` (a shared-memory coordination file). These are not independent backups. A backup that copies only the `.db` file while the database is open is corrupt. The Online Backup API handles this correctly; see `backup-restore-and-corruption.md`.

---

### `synchronous=NORMAL`

**What it does.** Controls when SQLite calls `fsync` (or `fdatasync`) to flush OS page cache to durable storage. Three levels matter:

| Value | fsync behaviour | Durability guarantee |
|-------|-----------------|---------------------|
| `OFF` | Never | Data in OS cache; any OS crash or power loss loses recent writes |
| `NORMAL` | At WAL checkpoint | Committed transactions survive OS crash; last transaction may be lost on power loss |
| `FULL` | On every commit and at WAL checkpoint | Committed transactions survive OS crash and power loss |

**Production-correct value.** `PRAGMA synchronous = NORMAL` for most workloads paired with WAL mode. Use `PRAGMA synchronous = FULL` when you cannot accept any committed-transaction loss on power failure — financial records, safety-critical state, compliance-mandated durability.

**Scope.** Connection-scoped — must be re-set on every connection open.

**Failure mode if mis-set.** `synchronous=OFF` with any crash scenario: the OS page cache holds writes that have been committed from SQLite's perspective but have not been flushed to disk. A power loss silently discards them. The application retried the operation successfully; the database never saw the write land. There is no error, no warning, and no way to detect the loss after the fact.

**The `NORMAL` vs `FULL` boundary.** With `journal_mode=WAL` and `synchronous=NORMAL`, SQLite does not fsync the WAL file on every commit — it writes the WAL pages and the WAL header update, but does not flush them to the storage medium. On a clean OS shutdown, all committed transactions survive. On power loss between the WAL write and the storage-medium flush, the most recent committed transaction may be lost. `synchronous=FULL` adds an fsync after each WAL commit, closing this window at a write-latency cost. Neither setting causes unbounded WAL growth — that is `wal_autocheckpoint=0`.

---

### `busy_timeout`

**What it does.** Sets the number of milliseconds SQLite will wait for a lock before returning `SQLITE_BUSY`. Default is 0 — any lock contention raises `SQLITE_BUSY` immediately. With a non-zero timeout, SQLite retries with internal backoff until the timeout expires or the lock is acquired.

**Production-correct value.** `PRAGMA busy_timeout = 5000` for human-interactive workloads (5 seconds). Long-running batch processes may want 30000 or higher. The exact value depends on your worst-case write transaction duration and how long a blocked caller can wait.

**Scope.** Connection-scoped — must be re-set on every connection open.

**Failure mode if mis-set.** With `busy_timeout = 0` (the default) and any concurrent writers: every cross-connection write collision produces an immediate `SQLITE_BUSY`. In a web application under even moderate load, this surfaces as random 500 errors or failed job enqueues with no explanation — the database is "locked" for microseconds and the application crashes the request rather than waiting.

**This is not a substitute for `BEGIN IMMEDIATE`.** `busy_timeout` buys wait time on lock acquisition. It does not fix a `BEGIN DEFERRED` transaction that races to upgrade to a write lock at first INSERT. The correct fix for deferred-to-write races is `BEGIN IMMEDIATE`; see `transactions-and-isolation.md`. Use `busy_timeout` as a backstop, not a solution.

---

### `foreign_keys=ON`

**What it does.** Enables enforcement of `FOREIGN KEY` constraints. Default is `OFF` — a historical accident that exists for backwards compatibility with applications that created foreign key columns before SQLite enforced them. With `foreign_keys=OFF`, `INSERT INTO child (parent_id) VALUES (999)` succeeds silently even when no row with `id=999` exists in `parent`.

**Production-correct value.** `PRAGMA foreign_keys = ON`

**Scope.** Connection-scoped — must be re-set on every connection open, on every connection in a pool, and on every test connection. There is no mechanism to persist this in the file.

**Failure mode if mis-set.** Referential integrity violations accumulate silently. A parent row is deleted; orphaned child rows remain. An application reads the child rows and dereferences the parent; it gets NULL or a missing-row error far from the insertion site. The insertion succeeded with no error. If FK enforcement is then switched on later — during a maintenance window or after a migration — `PRAGMA integrity_check` reveals thousands of constraint violations in a database that has been "clean" for months.

---

### `cache_size`

**What it does.** Sets the number of pages the connection will hold in its in-process page cache. Positive values are page counts; negative values are kibibytes. Default is `-2000` (approximately 2 MiB with the default 4096-byte page size).

**Production-correct value.** `PRAGMA cache_size = -64000` (64 MiB) for a database with active reads over a large dataset. Adjust based on available memory and working set size.

**Scope.** Connection-scoped — must be re-set on every connection open.

**Failure mode if mis-set.** With 2 MiB of cache and a database that has more than a few thousand rows in its hot tables: every query that does not fit in cache produces disk reads. On SSDs this is tolerable; on spinning disk or network storage it dominates query latency. The application "gets slow as the database grows" — the root cause is a cache too small for the working set.

**How to size it.** `PRAGMA page_stats` returns per-page-type counts. A rough rule: set cache to hold the hot tables' total pages comfortably. For an analytical workload, 256 MiB is not excessive. For a small embedded store on a memory-constrained device, the default may be correct. Measure with `PRAGMA cache_spill`.

---

### `mmap_size`

**What it does.** Enables memory-mapped I/O for reads. When set to a non-zero value (bytes), SQLite maps up to that many bytes of the database file into the process address space and uses pointer arithmetic for reads rather than `read(2)` system calls.

**Production-correct value.** `PRAGMA mmap_size = 268435456` (256 MiB) for large analytical read workloads on 64-bit platforms. Leave at 0 (default, disabled) for small databases or 32-bit platforms where address space is constrained.

**Scope.** Connection-scoped — must be re-set on every connection open.

**Failure mode if mis-set.** For workloads that read large sequential ranges of the database (analytics, exports, integrity checks), leaving `mmap_size` at 0 means those reads go through `read(2)` system call overhead and an extra copy through the kernel buffer. The loss is in latency per byte, not in correctness. Enabling it on a 32-bit platform with a large database causes address space exhaustion.

**WAL interaction.** Memory-mapped reads apply to the main database file only; WAL pages are always accessed via `read(2)`. In a WAL-mode database under active write load, a significant fraction of hot pages are in the WAL, not the main file. `mmap_size` provides less benefit during heavy write bursts.

---

### `temp_store=MEMORY`

**What it does.** Controls where SQLite stores temporary tables and indexes created during query execution (for ORDER BY, GROUP BY, subqueries, and explicit `CREATE TEMP TABLE`). Default is `DEFAULT`, which typically means on-disk temp files. `MEMORY` stores them in RAM.

**Production-correct value.** `PRAGMA temp_store = MEMORY` for any workload with complex queries.

**Scope.** Connection-scoped — must be re-set on every connection open.

**Failure mode if mis-set.** With temp files on disk and a query that produces large intermediate results (a sort over a million rows, a multi-way join with no covering index): the temp file creation and I/O dominate the query. On a machine with sufficient RAM, the same query with `temp_store=MEMORY` runs in a fraction of the time. The failure mode is silent performance regression on complex queries.

---

### `application_id`

**What it does.** Stores a 32-bit integer in the database file header. Used to identify which application owns the file. The `file` utility and `sqlite3_file_control(SQLITE_FCNTL_PRAGMA)` can read it from a file without opening it as a SQLite database. Default is 0.

**Production-correct value.** A fixed, unique integer that identifies your application. Choose one that is unlikely to collide: a CRC32 of your application name, a value registered in the SQLite application ID registry, or a value derived from a UUID. Document it in your project.

**Scope.** Database-scoped — persisted in the file header. Set once at database creation via the migration runner.

**Failure mode if mis-set.** With `application_id = 0`: when recovering database files from a crashed system, all files look identical to `file`. If you have multiple applications each using SQLite, you cannot distinguish them. If your application opens a file that belongs to a different application, it will attempt to read it as its own schema and fail in confusing ways rather than detecting the mismatch at open time.

---

### `user_version`

**What it does.** Stores a 32-bit integer in the database file header. By convention, this is the application schema version. The migration runner reads it at startup to determine which migrations need to be applied, increments it after each migration commits, and uses it as the gate for "has this migration already run."

**Production-correct value.** Managed by the migration runner — not set manually. Read with `PRAGMA user_version`; set with `PRAGMA user_version = N`.

**Scope.** Database-scoped — persisted in the file header.

**Failure mode if mis-set.** With no migration runner using `user_version`: migrations are either re-applied on every startup (idempotent-if-you're-lucky) or applied via `CREATE TABLE IF NOT EXISTS` (which silently ignores schema changes). A deployment that applies the wrong migration twice or skips a migration has a schema that does not match the application. The failure surfaces as runtime errors long after the deploy. See `schema-migrations.md` for the complete migration runner pattern.

---

### Other PRAGMAs to know

**`secure_delete`** — default `OFF`. When enabled, overwrites deleted page content with zeroes. Relevant when the database file may be forensically examined. High write amplification for large deletes; do not enable unless the threat model requires it.

**`auto_vacuum`** — default `NONE`. `FULL` shrinks the database file after each delete by moving pages to fill gaps; this causes page fragmentation and write amplification. `INCREMENTAL` allows manual compaction via `PRAGMA incremental_vacuum`. Most workloads are better served by periodic offline compaction than by enabling `auto_vacuum`. Must be set before any data is written — changing it on an existing database requires `VACUUM`.

**`journal_size_limit`** — caps the rollback journal or WAL file at a given number of bytes. `PRAGMA journal_size_limit = 67108864` (64 MiB) prevents unbounded WAL growth under write bursts. A complement to `wal_autocheckpoint`, not a substitute. If the WAL reaches the limit and cannot checkpoint because a reader holds an open snapshot, writes block.

**`wal_autocheckpoint`** — default `1000` (pages). SQLite automatically checkpoints — copies WAL pages back into the main database file — when the WAL reaches this many pages. Set to 0 to disable automatic checkpointing entirely and checkpoint manually. Disabling auto-checkpoint is correct when you want explicit control over checkpoint timing (e.g., during a batch import). It is the direct cause of unbounded WAL growth if you forget to checkpoint.

## Connection-scoped vs database-scoped

This is the most operationally critical distinction in PRAGMA discipline.

**Database-scoped PRAGMAs** are stored in the database file header. They persist across connection close, process restart, and all future connections to the same file. Set them once when the database is created:

| PRAGMA | Stored in |
|--------|-----------|
| `journal_mode` | File header |
| `application_id` | File header |
| `user_version` | File header |
| `auto_vacuum` | File header (must set before first write) |

**Connection-scoped PRAGMAs** are not stored anywhere. They apply to the current connection only and reset to their defaults the next time that connection is opened:

| PRAGMA | Default | Must re-set on every open |
|--------|---------|--------------------------|
| `synchronous` | `FULL` | Yes |
| `busy_timeout` | `0` | Yes |
| `foreign_keys` | `OFF` | Yes |
| `cache_size` | `-2000` | Yes |
| `mmap_size` | `0` | Yes |
| `temp_store` | `DEFAULT` | Yes |

**The operational implication.** Connection-scoped PRAGMAs must be set on every connection that enters the database — in application code, in tests, in one-off scripts, in migration tooling. The only reliable mechanism is a `connect()` helper function that is the **only** path to a connection and that applies all PRAGMAs before returning. No caller should open a raw connection and skip the helper.

## The standard production PRAGMA block

A `setup_pragmas` function that is idempotent, runs on every connection open, and is the only way the application obtains a connection.

**Python (`sqlite3`):**

```python
import sqlite3
import threading
from pathlib import Path

_local = threading.local()
_DB_PATH: str = ""


def configure(db_path: str | Path) -> None:
    """Set the database path. Call once at startup before any thread touches the DB."""
    global _DB_PATH
    _DB_PATH = str(db_path)


def _setup_pragmas(conn: sqlite3.Connection) -> None:
    """Apply all connection-scoped PRAGMAs. Called on every connection open.

    journal_mode is database-scoped and persists; issuing it here is
    idempotent and ensures any new database gets WAL mode set immediately.
    """
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")    # durable on OS crash; last txn may be lost on power loss
    conn.execute("PRAGMA foreign_keys = ON")       # off by default; must be set per connection
    conn.execute("PRAGMA busy_timeout = 5000")     # 5 s retry window before SQLITE_BUSY raises
    conn.execute("PRAGMA cache_size = -64000")     # 64 MiB page cache
    conn.execute("PRAGMA temp_store = MEMORY")


def connection() -> sqlite3.Connection:
    """Return the per-thread SQLite connection, opening and configuring one if needed.

    This is the only function that opens a connection. All callers go through here.
    """
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_DB_PATH, isolation_level=None)
        conn.row_factory = sqlite3.Row
        _setup_pragmas(conn)
        _local.conn = conn
    return conn
```

**Rust (`rusqlite`):**

```rust
use rusqlite::{Connection, Result};

fn setup_pragmas(conn: &Connection) -> Result<()> {
    // journal_mode is database-scoped but idempotent to set on every open.
    conn.execute_batch("PRAGMA journal_mode = WAL")?;
    conn.execute_batch("PRAGMA synchronous = NORMAL")?;
    conn.execute_batch("PRAGMA foreign_keys = ON")?;
    conn.execute_batch(&format!("PRAGMA busy_timeout = {}", 5_000))?;
    conn.execute_batch("PRAGMA cache_size = -64000")?;
    conn.execute_batch("PRAGMA temp_store = MEMORY")?;
    Ok(())
}

/// Open a connection and apply production PRAGMAs.
///
/// This is the only function that opens a connection. All callers use this.
pub fn open(db_path: &std::path::Path) -> Result<Connection> {
    let conn = Connection::open(db_path)?;
    setup_pragmas(&conn)?;
    Ok(conn)
}
```

Both examples apply all connection-scoped PRAGMAs before returning the connection. The caller cannot obtain a misconfigured connection by bypassing the helper.

## Anti-patterns

- **Setting connection-scoped PRAGMAs once at app boot and assuming they persist.** `busy_timeout`, `foreign_keys`, `synchronous`, `cache_size`, and `temp_store` are per-connection, not per-file. Setting them on the first connection does nothing for the second connection opened in a different thread. Every connection must go through `setup_pragmas` on open.

- **`journal_mode=WAL` on a network filesystem.** WAL mode depends on shared-memory coordination via `.db-shm`, which requires POSIX locking semantics. NFS and SMB do not implement these reliably. The result is silent corruption, not an error on open. If the database lives on a network mount, use `DELETE` journal mode and accept the concurrency constraints.

- **`synchronous=OFF` for performance.** Disabling fsync means the OS page cache holds your committed writes. Any OS crash or power event loses them. There is no error, no warning, and no recovery path — the writes simply did not happen from the database's perspective. `synchronous=NORMAL` with `journal_mode=WAL` provides strong durability for ordinary OS crash scenarios and is the correct performance trade-off.

- **Setting `busy_timeout` but not handling `SQLITE_BUSY` when it still surfaces.** A `busy_timeout` of 5000 ms does not guarantee lock acquisition — it guarantees SQLite will retry for 5 seconds before giving up. If the wait is exhausted, `SQLITE_BUSY` propagates to the caller. Application code must handle it: log it, surface it as a meaningful error, and — if it is happening frequently — diagnose whether the timeout is too short or the underlying lock contention is a structural problem.

- **Mismatched PRAGMAs between application code and test code.** Tests that open a raw connection without `setup_pragmas` run with `foreign_keys=OFF`, `synchronous=FULL`, `busy_timeout=0`, and 2 MiB of cache. They are not testing the same database configuration that runs in production. Tests must use the same `connection()` helper as the application, or call `setup_pragmas` explicitly on every test connection.

- **Using an in-memory database in tests that disables the PRAGMA surface entirely.** An in-memory database does not have a WAL, does not exercise locking, and does not write to disk. It cannot validate whether `journal_mode=WAL` is set, whether `busy_timeout` is sufficient, or whether `foreign_keys=ON` is preventing schema violations in the actual schema. Use a temp-file-backed database for at least the integration layer of the test suite.

## Cross-references

- [`sqlite-fundamentals.md`](sqlite-fundamentals.md) — the connection model that PRAGMAs configure: in-process execution, ACID semantics, thread/process safety rules.
- [`schema-migrations.md`](schema-migrations.md) — `user_version` drives the migration runner; `application_id` is registered there at database creation.
- [`concurrent-access-patterns.md`](concurrent-access-patterns.md) — WAL locking semantics, reader/writer coexistence, NFS prohibitions, checkpoint discipline.
- [`backup-restore-and-corruption.md`](backup-restore-and-corruption.md) — `synchronous` interacts directly with corruption recovery; the WAL sidecar files must be included in any backup.
- [`boundary-and-when-to-leave.md`](boundary-and-when-to-leave.md) — when PRAGMA tuning is no longer sufficient and a server database is the right move.
