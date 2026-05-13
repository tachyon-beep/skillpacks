---
description: Scaffold a versioned, PRAGMA-configured SQLite schema skeleton aligned to the `axiom-embedded-database` discipline. Emits a `db.py` (or `db.rs`) connection helper that applies the production PRAGMA block on every `connect()`, an `application_id` + `user_version` migration runner, an empty migrations directory with a v1 starter, optional claim-lease helpers if the workload is a job queue, and a smoke test that round-trips a transaction. Cross-validates the layout against `axiom-embedded-database:pragma-discipline.md` and `axiom-embedded-database:schema-migrations.md`.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[db_path]"
---

# Scaffold SQLite Schema Command

You are scaffolding a versioned SQLite schema skeleton aligned to the `axiom-embedded-database` discipline. The output is implementation scaffolding — a connection helper, a migration runner, a migrations directory with a v1 starter, and a smoke test — consistent with the `using-embedded-database` skill's production shape. This command does NOT replace the design discipline; it implements the agreed shape. For design without scaffolding, load `using-embedded-database` directly.

The emitted code is complete and ready to run: no `# TODO` markers, no placeholder bodies. Every PRAGMA value, migration runner detail, and claim-lease predicate is taken from the discipline sheets (`pragma-discipline.md`, `schema-migrations.md`, `optimistic-locking-and-leases.md`) and must match them exactly. Do not invent alternatives.

## Invocation Path

`/scaffold-sqlite-schema` is a Claude Code slash command. It detects the target language from the project (Python via `pyproject.toml` or `setup.cfg`, Rust via `Cargo.toml`) and emits the appropriate file variants. If neither is found, it asks. It then asks for the DB path and the workload shape before scaffolding anything.

Use `using-embedded-database` for the design discipline. Use `/scaffold-sqlite-schema` to emit the implementation skeleton once the design is settled.

## Preconditions

### Resolve the argument

```bash
INPUT="${ARGUMENTS}"
# INPUT is the optional db_path argument, e.g. "data/app.db" or ":memory:".
# If empty, default to "app.db" in the project root after confirming with the user.
```

### Detect target language

```bash
if [ -f pyproject.toml ] || [ -f setup.cfg ]; then
  LANG=python
elif [ -f Cargo.toml ]; then
  LANG=rust
else
  LANG=unknown
fi
```

If `LANG=unknown`, use `AskUserQuestion` to ask: "No `pyproject.toml` or `Cargo.toml` found. Which language? (python / rust)"

### Elicit workload shape via AskUserQuestion

Ask: "What is the primary workload shape?
  1. general — a standard OLTP store (default)
  2. job-queue — multiple workers compete to claim and process tasks
  3. audit-log — append-only ledger; reads are infrequent, every write counts
  4. olap-companion — heavy analytical reads alongside a separate write path"

Record the answer as `WORKLOAD_SHAPE`. Only `job-queue` triggers the claims helper scaffold (step 5 below).

### Check for existing artifacts

```bash
ls db.py claims.py migrations/ 2>/dev/null
```

If any exist, use `AskUserQuestion` to decide:

1. **Augment** — fill in missing pieces; leave existing files (with `.scaffold-suggested` siblings for files that already exist).
2. **Replace** — archive existing to `.backup-<timestamp>/`, scaffold fresh.
3. **Validate only** — skip scaffolding; spot-check the existing layout against the discipline sheets and report gaps.

## Optional pre-pass: gap analysis

For brownfield projects — where a `.db` file already exists — offer to dispatch the `embedded-database-reviewer` agent (defined in Task 19) before scaffolding:

> "A `*.db` file was found. Run the `embedded-database-reviewer` agent first to analyse the existing schema for PRAGMA gaps, missing migration versioning, and FK discipline violations? (y/n)"

If yes, dispatch `embedded-database-reviewer` and incorporate its findings before emitting any files. The reviewer may surface issues (e.g., WAL mode not set, user_version still 0) that change what the scaffold needs to emit or patch.

## Scaffold steps

### Step 1 — Detect or create the target directory

```bash
TARGET_DIR="${TARGET_DIR:-.}"   # default to current directory
mkdir -p "${TARGET_DIR}/migrations"
```

Report the resolved paths before writing any files.

### Step 2 — Emit the connection helper

**Python: `db.py`**

```python
"""SQLite connection helper — production PRAGMA block.

This module is the only path to a database connection. All callers go through
`connection()`. Direct `sqlite3.connect()` calls bypass the PRAGMA block and
produce misconfigured connections.

application_id: 0x2A7F1C3E  # derived at project init: crc32(b"myapp") ^ random_4bytes
"""
import sqlite3
import threading
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

_local = threading.local()
_DB_PATH: str = ""

# 32-bit magic number unique to this application.
# Chosen at scaffold time: zlib.crc32(b"<package>") ^ random 4 bytes, masked to
# fit a 32-bit signed integer. Do not change after the database is created.
_APPLICATION_ID: int = 0x2A7F1C3E


def configure(db_path: str | Path) -> None:
    """Set the database path. Call once at startup before any thread accesses the DB."""
    global _DB_PATH
    _DB_PATH = str(db_path)


# ── PRAGMA setup ──────────────────────────────────────────────────────────────

def _setup_pragmas(conn: sqlite3.Connection) -> None:
    """Apply all connection-scoped PRAGMAs. Called on every connection open.

    journal_mode is database-scoped and persists in the file header; issuing it
    here is idempotent and ensures any new database gets WAL mode set immediately.
    """
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")   # durable on OS crash; last txn may be lost on power loss
    conn.execute("PRAGMA foreign_keys = ON")      # off by default; must be set per connection
    conn.execute("PRAGMA busy_timeout = 5000")    # 5 s retry window before SQLITE_BUSY raises
    conn.execute("PRAGMA cache_size = -64000")    # 64 MiB page cache
    conn.execute("PRAGMA temp_store = MEMORY")


# ── Connection pool ───────────────────────────────────────────────────────────

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


def close() -> None:
    """Close the current thread's connection, if open."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
```

**Rust: `db.rs`** — follows the same shape using `rusqlite`. The `open()` function calls `setup_pragmas()`, which issues the same six PRAGMAs via `execute_batch` (and asserts the WAL mode row returns `"wal"`). See `pragma-discipline.md` for the full Rust template.

### Step 3 — Emit the migrations directory and v1 starter

**`migrations/__init__.py`** (Python) — empty, marks the directory as a package.

**`migrations/v001_initial.py`**

```python
"""Migration v001 — initial schema.

Contract with the migration runner (apply_migration in db_migrate.py):
  - Do NOT issue BEGIN, COMMIT, or ROLLBACK. The runner owns the transaction.
  - Do NOT issue PRAGMA foreign_keys. The runner has already set it to OFF
    before BEGIN and will restore it to ON after COMMIT.
  - Do NOT issue PRAGMA user_version. The runner bumps the version inside the
    same transaction after this function returns.

For changes that require rebuild-table (NOT NULL constraints, changed types,
changed CHECK, changed FK, dropped columns on SQLite < 3.35):
  - Follow the 12-step rebuild-table pattern from schema-migrations.md.
  - Steps 1, 2, 11, 12 are owned by the runner; this function covers steps 3–10.
  - Run PRAGMA foreign_key_check after the rebuild; raise RuntimeError on any row.

Required tests for every migration version N:
  (a) Forward test: schema at N-1 with realistic data → apply_migration(N) → assert data.
  (b) Fresh-schema test: build at N directly → assert schema matches.
  (c) Backwards-equivalence: migrated path == fresh path (compare sqlite_schema rows).
"""
import sqlite3


def migrate_v1(conn: sqlite3.Connection) -> None:
    """Create the initial schema.

    Replace the placeholder table below with your actual schema.
    Explicit NOT NULL and type constraints are preferred over SQLite's
    permissive defaults — use them unless there is a specific reason not to.
    """
    conn.execute("""
        CREATE TABLE items (
            id         INTEGER PRIMARY KEY,
            name       TEXT    NOT NULL,
            created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
    """)
    # Add further CREATE TABLE / CREATE INDEX statements here.
    # No data migration is needed for v1 (the database is empty).
```

### Step 4 — Emit the migration runner

**`db_migrate.py`**

```python
"""Migration runner — reads user_version, applies pending migrations atomically.

Usage:
    from db_migrate import migrate
    from db import connection, configure

    configure("app.db")
    migrate(connection())   # call once at startup, before any application queries
"""
import sqlite3
from collections.abc import Callable

from db import _APPLICATION_ID, connection
from migrations.v001_initial import migrate_v1

# ── Registry ──────────────────────────────────────────────────────────────────

LATEST_VERSION: int = 1

MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    1: migrate_v1,
}

# ── Runner ────────────────────────────────────────────────────────────────────

def migrate(conn: sqlite3.Connection) -> None:
    """Apply all pending migrations up to LATEST_VERSION.

    Idempotent: calling this on a database already at LATEST_VERSION is a no-op.
    Raises RuntimeError if the database is newer than the running code.
    """
    _ensure_application_id(conn)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current == LATEST_VERSION:
        return
    if current > LATEST_VERSION:
        raise RuntimeError(
            f"Database at schema version {current} is newer than "
            f"code at {LATEST_VERSION}. Deploy the newer code first."
        )
    for v in range(current + 1, LATEST_VERSION + 1):
        _apply_migration(conn, v)


def _apply_migration(conn: sqlite3.Connection, v: int) -> None:
    """Apply a single migration and bump user_version atomically.

    The runner owns the transaction. The migrate_fn performs DDL and data work
    but must NOT issue BEGIN, COMMIT, ROLLBACK, or PRAGMA foreign_keys — the
    runner has already arranged the envelope.

    PRAGMA user_version = N cannot use parameter binding (SQLite does not allow
    binding values in PRAGMA statements). The f-string is safe: v is an int
    produced by range() from LATEST_VERSION, never from user input. The PRAGMA
    participates in transactional rollback, so it belongs inside BEGIN…COMMIT
    with the schema change, ensuring both land atomically.
    """
    migrate_fn = MIGRATIONS[v]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            migrate_fn(conn)
            conn.execute(f"PRAGMA user_version = {v}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")

    actual = conn.execute("PRAGMA user_version").fetchone()[0]
    if actual != v:
        raise RuntimeError(
            f"user_version write failed after migration v{v}: "
            f"expected {v}, got {actual}"
        )


def _ensure_application_id(conn: sqlite3.Connection) -> None:
    """Set application_id if this is a new database (id == 0).

    application_id is database-scoped and persists in the file header. Set it
    once when the database is first created; it should never change. A mismatch
    between the running code's expected ID and the file's ID indicates the
    wrong file has been opened.
    """
    existing = conn.execute("PRAGMA application_id").fetchone()[0]
    if existing == 0:
        conn.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
    elif existing != _APPLICATION_ID:
        raise RuntimeError(
            f"application_id mismatch: file has {hex(existing)}, "
            f"code expects {hex(_APPLICATION_ID)}. Wrong database file?"
        )
```

### Step 5 — Emit claim-lease helpers (job-queue workload only)

Emitted only when `WORKLOAD_SHAPE == job-queue`. Requires SQLite 3.35+ for `RETURNING`.

**`claims.py`**

```python
"""Claim-lease helpers for job-queue workloads.

At-most-one semantics: SQLite's single-writer model serialises concurrent
UPDATE statements, so exactly one caller wins the claim predicate even when
multiple workers race on the same job row. The predicate on every write is the
guarantee — do not remove or weaken it.

Requires SQLite 3.35+ for RETURNING. For older builds, see
optimistic-locking-and-leases.md (BEGIN IMMEDIATE + changes() fallback).

Schema (add to your v001 migration or a later version):

    CREATE TABLE jobs (
        id               INTEGER PRIMARY KEY,
        payload          TEXT    NOT NULL,
        status           TEXT    NOT NULL DEFAULT 'pending',
        assignee         TEXT,
        claim_expires_at INTEGER,          -- Unix milliseconds; NULL = unclaimed
        priority         INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX idx_jobs_claimable
        ON jobs (priority DESC)
     WHERE status = 'pending'
        OR (status = 'running' AND claim_expires_at IS NOT NULL);
"""
import sqlite3
import time
from typing import Optional

LEASE_DURATION_MS: int = 30_000   # 30 seconds; tune to at least 3× expected job duration
HEARTBEAT_INTERVAL_S: float = 10.0  # heartbeat every lease / 3


def claim_next(conn: sqlite3.Connection, worker_id: str) -> Optional[sqlite3.Row]:
    """Atomically find and claim the highest-priority available job.

    Returns the claimed row (with id, payload, priority), or None if the queue
    is empty. The subquery + outer WHERE predicate is evaluated as a single
    statement — no gap exists between "found the row" and "claimed the row".
    """
    now_ms = int(time.time() * 1000)
    expires_at = now_ms + LEASE_DURATION_MS

    rows = conn.execute(
        """
        UPDATE jobs
           SET assignee         = ?,
               claim_expires_at = ?,
               status           = 'running'
         WHERE id = (
             SELECT id
               FROM jobs
              WHERE status = 'pending'
                 OR (status = 'running' AND claim_expires_at < ?)
              ORDER BY priority DESC
              LIMIT 1
         )
           AND (status = 'pending' OR claim_expires_at < ?)
        RETURNING id, payload, priority
        """,
        (worker_id, expires_at, now_ms, now_ms),
    ).fetchall()

    return rows[0] if rows else None


def heartbeat(conn: sqlite3.Connection, job_id: int, worker_id: str) -> bool:
    """Extend the lease while the job is still running.

    Returns False if the lease expired and another worker has already reclaimed
    the job (zombie path). The caller must stop processing and not complete the
    job if this returns False.

    Both AND predicates are load-bearing:
      AND assignee = ?          — identity check: we still own this job
      AND claim_expires_at >= ? — expiry check: our lease has not yet expired
    """
    now_ms = int(time.time() * 1000)
    new_expires_at = now_ms + LEASE_DURATION_MS

    cursor = conn.execute(
        """
        UPDATE jobs
           SET claim_expires_at = ?
         WHERE id               = ?
           AND assignee         = ?
           AND claim_expires_at >= ?
        """,
        (new_expires_at, job_id, worker_id, now_ms),
    )
    return cursor.rowcount == 1


def complete(conn: sqlite3.Connection, job_id: int, worker_id: str) -> bool:
    """Mark the job done and release the lease.

    Returns False if the lease had already expired before completion. Design
    jobs to be idempotent: if False, a side effect may have already been applied
    by the new owner of the job. See optimistic-locking-and-leases.md.
    """
    now_ms = int(time.time() * 1000)

    cursor = conn.execute(
        """
        UPDATE jobs
           SET status           = 'done',
               assignee         = NULL,
               claim_expires_at = NULL
         WHERE id               = ?
           AND assignee         = ?
           AND claim_expires_at >= ?
        """,
        (job_id, worker_id, now_ms),
    )
    return cursor.rowcount == 1


def abandon(conn: sqlite3.Connection, job_id: int, worker_id: str) -> bool:
    """Return a claimed job to pending status (clean worker shutdown).

    Use on graceful shutdown to avoid waiting for the lease to expire naturally.
    Returns False if the lease had already expired (another worker may have
    claimed the job — do not re-abandon).
    """
    now_ms = int(time.time() * 1000)

    cursor = conn.execute(
        """
        UPDATE jobs
           SET status           = 'pending',
               assignee         = NULL,
               claim_expires_at = NULL
         WHERE id               = ?
           AND assignee         = ?
           AND claim_expires_at >= ?
        """,
        (job_id, worker_id, now_ms),
    )
    return cursor.rowcount == 1
```

### Step 6 — Emit the smoke test

**`tests/test_db_smoke.py`**

```python
"""Smoke test — opens the database, applies migrations, round-trips a row.

Run with: pytest tests/test_db_smoke.py -v

This test uses a temp-file-backed database (not :memory:) because an in-memory
database does not exercise WAL mode, locking, or fsync paths. See
pragma-discipline.md: "Using an in-memory database in tests disables the PRAGMA
surface entirely."
"""
import sqlite3
import pytest

import db as _db
import db_migrate


@pytest.fixture
def tmp_db(tmp_path):
    """Yield a configured, migrated database connection backed by a temp file."""
    db_path = tmp_path / "smoke.db"
    _db.configure(str(db_path))
    conn = _db.connection()
    db_migrate.migrate(conn)
    yield conn
    _db.close()


def test_pragmas_applied(tmp_db):
    """Verify the production PRAGMA block is in effect."""
    assert tmp_db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert tmp_db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert tmp_db.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_schema_version_at_latest(tmp_db):
    """user_version must equal LATEST_VERSION after migrate()."""
    version = tmp_db.execute("PRAGMA user_version").fetchone()[0]
    assert version == db_migrate.LATEST_VERSION


def test_application_id_set(tmp_db):
    """application_id must match the code constant."""
    app_id = tmp_db.execute("PRAGMA application_id").fetchone()[0]
    assert app_id == _db._APPLICATION_ID


def test_round_trip_transaction(tmp_db):
    """Insert a row and read it back inside an explicit transaction."""
    tmp_db.execute("BEGIN IMMEDIATE")
    tmp_db.execute(
        "INSERT INTO items (name) VALUES (?)",
        ("smoke-item",),
    )
    tmp_db.execute("COMMIT")

    row = tmp_db.execute(
        "SELECT name FROM items WHERE name = ?",
        ("smoke-item",),
    ).fetchone()
    assert row is not None
    assert row["name"] == "smoke-item"


def test_integrity_check(tmp_db):
    """PRAGMA integrity_check must return a single 'ok' row."""
    results = tmp_db.execute("PRAGMA integrity_check").fetchall()
    assert len(results) == 1
    assert results[0][0] == "ok"


def test_migrate_idempotent(tmp_db):
    """Running migrate() a second time on an up-to-date database is a no-op."""
    db_migrate.migrate(tmp_db)   # second call; must not raise or change version
    version = tmp_db.execute("PRAGMA user_version").fetchone()[0]
    assert version == db_migrate.LATEST_VERSION
```

## Emitted file inventory

Before writing any files, report the full list to the user for confirmation:

| File | Always / Conditional |
|------|---------------------|
| `db.py` (or `db.rs`) | Always |
| `migrations/__init__.py` | Always (Python only) |
| `migrations/v001_initial.py` (or `.rs`) | Always |
| `db_migrate.py` (or `db_migrate.rs`) | Always |
| `claims.py` (or `claims.rs`) | Job-queue workload only |
| `tests/test_db_smoke.py` (or `tests/test_db_smoke.rs`) | Always |

Ask for confirmation before writing if running interactively. In a non-interactive context, write and report.

## Verification

After scaffolding, run the smoke test:

```bash
python -m pytest tests/test_db_smoke.py -v
```

A passing run confirms:
- The connection helper applies all six PRAGMAs.
- The migration runner advances `user_version` to `LATEST_VERSION`.
- `application_id` is set correctly.
- A round-trip INSERT + SELECT succeeds inside a transaction.
- `PRAGMA integrity_check` returns `"ok"`.

If the smoke test fails, report the failure output and do not declare scaffolding complete.

**Rust.** For Rust projects, the equivalent verification is:

```bash
cargo test db_smoke
```

The Rust test uses `rusqlite::Connection` via the `open()` helper in `db.rs`, which calls `setup_pragmas()`, and mirrors the five assertions above.

## Cross-references

- `using-embedded-database` — the discipline router for this pack; start here for design decisions before reaching for this command.
- `pragma-discipline.md` — the authoritative source for PRAGMA values, connection-scoped vs database-scoped rules, and the standard production block emitted in step 2.
- `schema-migrations.md` — the 12-step rebuild-table pattern, versioning discipline, and the three required tests per migration version that the v001 starter docstring references.
- `optimistic-locking-and-leases.md` — the full claim-lease contract (`claim_next`, `heartbeat`, `complete`, `abandon`), dead-worker recovery, and the required race-condition test for job-queue schemas.
