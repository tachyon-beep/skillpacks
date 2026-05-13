---
name: sqlite-fundamentals
description: Use when standing up a new SQLite-backed component, inheriting an SQLite layer with surprising failure modes, or defending an SQLite choice against "just use Postgres". Covers the in-process execution model, connection lifecycle, ACID semantics in the embedded context, and thread/process concurrency rules. Foundation for every other sheet in the pack.
---

# SQLite Fundamentals

**SQLite is not a stripped-down server database — it is a different shape: the application is the database process, the file is the durability boundary, and the connection is the unit of isolation.**

## When this earns its cost

Read this sheet when you are:

- Standing up a new SQLite-backed component and want to understand the execution model before configuring PRAGMAs, writing migrations, or choosing a transaction flavour.
- Inheriting an SQLite layer that produces `SQLITE_BUSY`, silent data loss, or corruption under concurrent writers — the root cause is almost always a misunderstanding of what a connection _is_ in this model.
- Defending an SQLite choice against "just use Postgres" — the right defence is not "SQLite is simpler" but "SQLite's envelope includes this workload, and here is why."

The central mistake practitioners bring from server-database experience is treating a connection as a thin network socket to a shared server. In SQLite, a connection is an independent actor with its own lock state, its own transaction context, and its own page cache. Two connections to the same file are not collaborating through a server; they are competing through the OS filesystem locking layer. Everything that follows is a consequence of that inversion.

Target length for comprehension: one reading session. Forward references to other sheets are explicit — do not read ahead unless the cross-reference is relevant to your immediate problem.

## The in-process model

### Connection-as-isolation-unit

Each `sqlite3_open()` (and every wrapper around it — `sqlite3.connect()` in Python, `Connection::open()` in rusqlite) returns an **independent transaction context**. There is no shared lock manager coordinating connections — each connection races to acquire filesystem-level locks when it needs to read or write.

Concretely: two connections from the same process to the same file are as independent as two connections from different processes. They share nothing except the underlying pages on disk (and, in WAL mode, the WAL file).

**Thread safety**: SQLite's default compile (`SQLITE_THREADSAFE=1`, which is the default for all major distributions) serialises access to a single connection's internal state, so passing one connection across threads will not corrupt SQLite's C-level structures. It will, however, interleave your application's transactions in ways that are almost never what you want. The safe rule:

> One connection per thread. Open at thread start; close at thread end. Never share a connection across threads unless you have explicitly serialised all callers through a `Mutex` (or equivalent) and have tested the resulting transaction interleaving.

Python's `sqlite3` module enforces this by raising `ProgrammingError` when you call a connection from a thread that did not create it, unless you pass `check_same_thread=False`. That parameter disables the _check_, not the _hazard_.

```python
import sqlite3
import threading

_thread_local = threading.local()


def get_connection(db_path: str) -> sqlite3.Connection:
    """Return the per-thread connection, opening one if needed.

    Opening is intentionally deferred to first use so that threads that
    never touch the database do not pay the open cost.
    """
    conn = getattr(_thread_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # PRAGMAs that must be set per-connection, per-open.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        _thread_local.conn = conn
    return conn
```

This pattern amortises PRAGMA cost (paid once per connection, not once per query), avoids shared-connection cross-thread hazards, and is safe to call from any thread.

### No separate database process

Server databases isolate concerns through process boundaries: Postgres runs `postmaster`, which spawns per-connection workers; those workers serialise writes, manage the shared buffer pool, and enforce access control. When a client misbehaves, the server can kill its backend. When the server crashes, the OS reaps its process and a supervisor restarts it without affecting the files.

SQLite has none of these. Consequences:

- **No network round-trip cost.** A `SELECT` executes in the calling thread's address space; the only I/O is filesystem reads. For a warm page cache, many queries cost single-digit microseconds.
- **No separate authentication surface.** There is no username/password check, no role system, no TCP port to firewall. File-system permissions are the access control layer. If a process can `open(2)` the file, it has full access. This is a feature (simplicity) and a risk (no granularity below process-level).
- **No concurrent DDL mediation.** A schema migration that runs `ALTER TABLE` or `CREATE INDEX` does so while holding a write lock on the file. Other writers see `SQLITE_BUSY`. There is no background DDL path, no `LOCK NOWAIT`, no DDL visibility through a shared catalog server.
- **No replication built in.** SQLite has no WAL shipping, no streaming replication, no logical replication. The only multi-host options are application-level (copy the file, use the Online Backup API, or use Litestream/rqlite as a separate layer). The embedded model assumes one host.

### The file is the database

SQLite's durability boundary is the file (and, in WAL mode, the two sidecar files: `.db-wal` and `.db-shm`). This produces several non-obvious properties:

**Corruption surface.** The file is a structured binary artefact. A partial write — caused by power loss, OS crash, or a buggy `O_DIRECT` implementation — leaves a file whose pages are in a state SQLite's recovery logic must handle. WAL mode reduces this surface (the WAL is append-only; the main file is never written mid-transaction), but does not eliminate it.

**Atomic-rename caveats.** A common deployment pattern is to construct a new database in a temp file and `rename(2)` it into place. `rename(2)` is atomic at the directory-entry level on POSIX filesystems, but the _data_ in the new file is not fsynced by `rename` itself. If you need the renamed file to be durable across a crash, call `fsync` on the file and its parent directory before the rename. Wrappers like Python's `pathlib.Path.rename()` do not do this for you.

**Power-loss behaviour.** With `journal_mode=WAL` and `synchronous=NORMAL`, SQLite guarantees that committed transactions survive power loss _most_ of the time, but not absolutely — there is a narrow window between the WAL write and the WAL header update where the last committed transaction can be lost. `synchronous=FULL` eliminates this window at a write-latency cost. `synchronous=OFF` is not a production choice. See [`pragma-discipline.md`](pragma-discipline.md) for the complete matrix.

## ACID and what's actually guaranteed

### Atomicity

`BEGIN … COMMIT` is atomic: either all writes in the transaction land or none do. This is true for any transaction that commits inside a **single connection**. Cross-connection atomicity does not exist — there is no distributed transaction coordinator, no two-phase commit across connections.

`SAVEPOINT` / `RELEASE` / `ROLLBACK TO` provide nested scopes within a single transaction. They are the correct tool when you want partial rollback inside a batch operation. See [`transactions-and-isolation.md`](transactions-and-isolation.md) for the full treatment.

### Consistency

SQLite's consistency guarantee is: "the database satisfies the constraints that were checked." By default, very few constraints are checked:

- **`FOREIGN_KEYS` is OFF** unless you enable it with `PRAGMA foreign_keys = ON` per connection per open. This is the single most surprising default. A child row referencing a non-existent parent inserts silently if `foreign_keys` is off.
- `CHECK` constraints are enforced.
- `NOT NULL` is enforced.
- `UNIQUE` and `PRIMARY KEY` are enforced.

`foreign_keys = ON` must be set after every connection open. It is not persisted in the database file. It is not inherited across connections. If you have a connection pool, every connection in the pool must set it.

### Isolation

SQLite's isolation level is **SERIALIZABLE** for a single connection's transaction: statements inside a transaction see a consistent snapshot of the database as of the transaction's start (in WAL mode) or as of the first read (in rollback-journal mode). No dirty reads, no non-repeatable reads, no phantom reads — within a single connection.

Cross-connection isolation is enforced through **locking**, not through MVCC snapshots:

- In rollback-journal mode: a writer holds an EXCLUSIVE lock; readers must wait. Readers hold SHARED locks; a writer cannot acquire EXCLUSIVE while any reader holds SHARED.
- In WAL mode: readers see a consistent snapshot regardless of concurrent writers (true MVCC for reads); only one writer can hold the write lock at a time.

The cross-connection isolation level you get depends on the journal mode. See [`concurrent-access-patterns.md`](concurrent-access-patterns.md) for the full table.

### Durability

Durability is the product of two settings, both of which must be chosen deliberately:

| `journal_mode` | `synchronous` | Durability guarantee |
|----------------|---------------|----------------------|
| `DELETE` (default) | `FULL` | Full — survives OS crash and power loss |
| `WAL` | `NORMAL` | High — survives OS crash; last transaction may be lost on power loss |
| `WAL` | `FULL` | Full — survives OS crash and power loss; slower |
| `WAL` | `OFF` | None — data in OS cache; power loss loses recent writes |
| `MEMORY` | any | None — no on-disk journal |

The production default for most workloads is `journal_mode=WAL` + `synchronous=NORMAL`. For safety-critical data or compliance requirements, use `synchronous=FULL`. Never use `synchronous=OFF` in production. See [`pragma-discipline.md`](pragma-discipline.md) for the `fsync` and `fdatasync` path that each combination exercises.

## Threads, processes, and connections

| Pattern | Verdict | Why |
|----------------------------------------------|-----------------|-----------------------------------------------------|
| One connection, one thread | Safe | The intended model |
| One connection, multiple threads | Hazardous | Only safe with explicit serialisation through `Mutex`; transaction interleaving is easy to create |
| Multiple connections, one process | Safe | The standard pattern for thread pools |
| Multiple connections, multiple processes | Safe with WAL | Requires `journal_mode=WAL`; see [`concurrent-access-patterns.md`](concurrent-access-patterns.md) |
| Shared connection across `fork()` | UB / corruption | The child inherits the connection's internal state; always re-open after fork |

The `fork()` case deserves emphasis. Any connection open in the parent at the time of `fork()` is cloned into the child's address space with the exact same lock state, transaction state, and page cache. If both parent and child then read or write, they are operating on separate copies of a shared data structure with no coordination. The result is undefined behaviour at the SQLite level and potential file corruption. **Always close all SQLite connections before forking, or re-open fresh connections in the child after fork.**

Pre-forking servers (gunicorn, uWSGI, many worker-process models) commonly trigger this pattern. The fix: do not open any database connections before the fork point; open them lazily in the worker after fork.

## Worked example

A minimal per-thread connection manager in Python, with PRAGMA setup, context-managed transactions, and docstrings that explain each choice.

```python
import sqlite3
import threading
from contextlib import contextmanager
from typing import Generator

_local = threading.local()
_DB_PATH: str = ""  # Set at application startup via configure()


def configure(db_path: str) -> None:
    """Set the database path for this process.

    Call once at application startup, before any thread accesses the database.
    Not thread-safe itself — call from the main thread before spawning workers.
    """
    global _DB_PATH
    _DB_PATH = db_path


def _open() -> sqlite3.Connection:
    """Open and configure a fresh connection for the calling thread."""
    conn = sqlite3.connect(_DB_PATH, isolation_level=None)  # isolation_level=None = autocommit; all transaction boundaries are explicit BEGIN/COMMIT
    conn.row_factory = sqlite3.Row
    # Most PRAGMAs are session-level (not stored); foreign_keys, synchronous, and cache settings must be re-applied on every open.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")       # Off by default — always enable.
    conn.execute("PRAGMA busy_timeout = 5000")     # 5 s retry window before SQLITE_BUSY.
    conn.execute("PRAGMA cache_size = -8000")      # 8 MiB page cache per connection.
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def connection() -> sqlite3.Connection:
    """Return the per-thread connection, opening one if not yet open."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _open()
        _local.conn = conn
    return conn


def close() -> None:
    """Close the calling thread's connection. Call at thread shutdown."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None


@contextmanager
def transaction(
    mode: str = "IMMEDIATE",
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that wraps body in BEGIN <mode> … COMMIT/ROLLBACK.

    Args:
        mode: "DEFERRED" (default SQLite), "IMMEDIATE" (acquires write lock
              at BEGIN — the correct default for write transactions), or
              "EXCLUSIVE" (blocks all other connections).

    Use IMMEDIATE for any transaction that will write. Use DEFERRED only for
    read-only transactions where you are certain no write will occur.
    """
    conn = connection()
    conn.execute(f"BEGIN {mode}")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
```

Usage:

```python
configure("/var/lib/myapp/data.db")

# Read-only: DEFERRED is fine.
with transaction("DEFERRED") as conn:
    rows = conn.execute("SELECT id, name FROM items WHERE active = 1").fetchall()

# Write: use IMMEDIATE to acquire the write lock at BEGIN.
with transaction("IMMEDIATE") as conn:
    conn.execute(
        "INSERT INTO events (type, payload) VALUES (?, ?)",
        ("user.created", '{"id": 42}'),
    )
```

## Anti-patterns

- **One connection shared across a thread pool.** Each thread's statements interleave at the SQLite level. A `BEGIN` from thread A and a `COMMIT` from thread B produce transaction boundaries that neither thread intended. Use per-thread connections.

- **Opening and closing a connection per request without amortising PRAGMA cost.** `PRAGMA journal_mode`, `PRAGMA foreign_keys`, `PRAGMA busy_timeout`, and others must be re-issued on every new connection. If connections are short-lived, the PRAGMA overhead compounds. Use a per-thread or per-worker connection that lives for the duration of the thread, not the duration of the request.

- **Using `:memory:` in tests but file-backed in production without testing both paths.** `:memory:` databases do not exercise the WAL, the locking layer, the filesystem path, or concurrent-access behaviour. Tests that only run against `:memory:` are not testing the code that runs in production. Use a temporary file-backed database in at least a subset of tests.

- **Sharing a connection across `fork()` without re-opening.** Pre-forking servers (gunicorn, uWSGI) clone parent-process connections into workers. Both parent and worker then hold the same connection state against the same file. Close all connections before the fork point, or open them lazily in the worker after fork.

- **Relying on `foreign_keys` being on without setting it.** SQLite ships with `PRAGMA foreign_keys = OFF`. FK violations insert silently. Any code that depends on FK enforcement without explicitly setting `foreign_keys = ON` per connection is relying on a guarantee SQLite does not provide by default.

- **Treating `SQLITE_BUSY` as a transient error without fixing its cause.** `SQLITE_BUSY` means a lock race. Retrying with exponential backoff treats the symptom. The causes are: `busy_timeout` not set, `BEGIN DEFERRED` used for write transactions, WAL mode not enabled for multi-writer workloads, or long-running read transactions blocking checkpointing. Fix the cause; use `busy_timeout` as a backstop, not a solution.

## Cross-references

- [`pragma-discipline.md`](pragma-discipline.md) — the configuration is the application: `journal_mode`, `synchronous`, `cache_size`, `foreign_keys`, `busy_timeout` — choose them deliberately, not by default.
- [`transactions-and-isolation.md`](transactions-and-isolation.md) — what `BEGIN` flavour to pick: DEFERRED vs IMMEDIATE vs EXCLUSIVE, the lock-upgrade surprise, SAVEPOINT for nested scopes.
- [`concurrent-access-patterns.md`](concurrent-access-patterns.md) — WAL and multi-process access: reader/writer coexistence, shared-cache pitfalls, NFS prohibitions, advisory lock discipline.
- [`backup-restore-and-corruption.md`](backup-restore-and-corruption.md) — corruption recovery: Online Backup API, `PRAGMA integrity_check`, `sqlite3_recover`, WAL checkpoint before backup.
- [`boundary-and-when-to-leave.md`](boundary-and-when-to-leave.md) — when SQLite stops being the right choice: multi-host writes, sustained high-concurrency write contention, relational features that require a server database.
