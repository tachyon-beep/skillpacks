---
name: concurrent-access-patterns
description: Use when multiple threads or processes write to the same SQLite database; when you have seen SQLITE_CORRUPT after a cross-platform deploy; when choosing between a write-coordinator process and SQLite's built-in locks; or when reasoning about portalocker, fcntl, WAL checkpointing, or NFS hazards. Covers the WAL contract, per-thread connection patterns, multi-process access, cross-platform file locking, and the single-writer queue pattern.
---

# Concurrent Access Patterns

**SQLite under WAL is single-writer-many-readers — not a limitation, a model. The application's job is to fit its writes into a single-writer shape, and to make that shape correct across processes, threads, and platforms.**

## When this earns its cost

Read this sheet when:

- Multiple processes write to the same database file and you need to understand how SQLite coordinates them — or when you have discovered it is not coordinating them correctly.
- You have encountered `SQLITE_CORRUPT` after a cross-platform deployment and need to determine whether the filesystem or a concurrent-access misconfiguration is the cause.
- You are choosing between routing all writes through a coordinator process versus relying on SQLite's built-in single-writer WAL model, and need to understand what the built-in model actually guarantees.
- You are deploying to an environment (Docker, network share, CI volume) where filesystem lock semantics are unknown and you need a principled policy.

`sqlite-fundamentals.md` establishes the connection-as-isolation-unit model that this sheet builds on. `transactions-and-isolation.md` covers the `BEGIN` flavour choice; this sheet covers how those locks interact across threads, processes, and operating systems.

## The WAL contract

Write-Ahead Logging reorders the sequence of I/O without weakening its guarantees. Understanding the contract prevents both over-trusting it (running WAL on a network filesystem) and under-trusting it (adding unnecessary application-level locking on a single host).

**How WAL works.** When a write transaction commits, SQLite appends the new page versions to the `.db-wal` file rather than modifying the main database file in place. Readers determine which version of each page to use by consulting the WAL header: if the WAL contains a newer version of a page than the main file, the reader uses the WAL version. Each reader sees a consistent snapshot as of the WAL frame it identified at the start of its transaction — later writes appending to the WAL are invisible to it.

**Reader/writer coexistence.** Readers and one writer proceed concurrently without blocking each other. A WAL-mode reader never blocks a writer, and a writer never blocks WAL-mode readers. This is the primary operational improvement over rollback-journal mode, where a writer holds an EXCLUSIVE lock and every reader must wait.

**The single-writer constraint.** WAL still serialises writers. Only one connection can hold the RESERVED lock at a time — only one write transaction can be in progress simultaneously. This is not a deficiency to work around; it is the model. Multiple writers are serialised through the lock, and the correct application response to high write contention is not to break the model but to batch writes or funnel them through a queue (see the single-writer thread pattern below).

**Checkpoint.** Periodically, the WAL is folded back into the main database file in a process called checkpointing. By default, SQLite triggers an automatic checkpoint when the WAL reaches 1000 pages (`wal_autocheckpoint`). Checkpointing is driven by the writer (it happens as part of commit-path logic), but can also be triggered manually.

The default checkpoint mode is PASSIVE: SQLite copies as many WAL frames as possible into the main file without blocking any reader or writer. If a reader holds a snapshot that pins early WAL frames, those frames are left in place and the checkpoint makes partial progress. PASSIVE never blocks; it simply skips frames that are in use. The WAL file shrinks only up to the point no reader currently holds a snapshot over. Manual checkpoint modes — FULL, RESTART, TRUNCATE — can block new writers until all readers have released their snapshots, but these are not triggered by `wal_autocheckpoint`. For most workloads, PASSIVE is the right mode and requires no application changes.

**The NFS caveat.** WAL requires two processes (or threads in different processes) to coordinate through the `.db-shm` shared-memory file, which uses POSIX memory-mapped locking semantics. NFS, SMB, CIFS, and many FUSE implementations do not implement these semantics correctly or at all. Enabling WAL mode on a network filesystem is not a configuration choice that degrades gracefully — it produces silent corruption. The discipline is binary: WAL is for local-host filesystems only. If you must use a network filesystem, use `DELETE` journal mode with `busy_timeout` and accept the read/write serialisation.

## Multi-thread, single-process

The recommended pattern is one connection per thread. Each thread opens its own connection and applies the full PRAGMA block at open time. The threads share no connection state; coordination happens at the SQLite lock level through the WAL.

```python
import sqlite3
import threading

_thread_local = threading.local()
_DB_PATH: str = ""


def configure(db_path: str) -> None:
    """Set the database path. Call from the main thread before spawning workers."""
    global _DB_PATH
    _DB_PATH = db_path


def connection() -> sqlite3.Connection:
    """Return the per-thread connection, opening and configuring one if needed.

    PRAGMAs are applied on every open — they are connection-scoped and do not
    persist across close/reopen. This is the only path to a connection; callers
    do not call sqlite3.connect() directly.
    """
    conn = getattr(_thread_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_DB_PATH, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA cache_size = -8000")
        conn.execute("PRAGMA temp_store = MEMORY")
        _thread_local.conn = conn
    return conn


def close() -> None:
    """Close the calling thread's connection. Call at thread shutdown."""
    conn = getattr(_thread_local, "conn", None)
    if conn is not None:
        conn.close()
        _thread_local.conn = None
```

**`check_same_thread=False`.** Python's `sqlite3` module raises `ProgrammingError` when a connection is used from a thread other than the one that opened it, unless `check_same_thread=False` is passed at open time. That parameter disables the check, not the hazard. A connection shared across threads without explicit serialisation will produce interleaved transactions — a `BEGIN` from one thread and a `COMMIT` from another creating transaction boundaries neither intended. Using `check_same_thread=False` is only safe when all callers are serialised through an application-level `threading.Lock` or `threading.RLock` that you own. The per-thread connection pattern above is the simpler, safer default: let each thread own its connection and let the WAL serialise concurrent writers.

## Multi-process, single-host

Multiple processes accessing the same WAL-mode SQLite database on the same host is the supported multi-process case. Each process opens its own connection; SQLite uses POSIX file locks (via the `unix` VFS) to coordinate them. Readers see concurrent-snapshot isolation; at most one writer holds RESERVED at a time.

There are no application-level adjustments required to make this work. The two anti-patterns that break it:

**Anti-pattern: sharing a connection across `fork()`.** When a process forks, the child inherits the parent's file descriptors, address space, and all SQLite connection state including lock state and transaction state. The parent and child then hold separate in-memory copies of shared data structures that reference the same file. If both read or write, the result is undefined behaviour at the SQLite level and potential file corruption. The rule is unconditional: any connection open at the time of `fork()` must be closed before the fork, or a new connection must be opened in the child after the fork. Pre-forking servers (gunicorn, uWSGI, multiprocessing-based worker pools) commonly trigger this pattern. Open connections lazily in the worker process, not in the parent before the fork.

**Anti-pattern: using a network filesystem.** See the WAL contract section. NFS, SMB, and FUSE with unreliable lock semantics make multi-process WAL unsafe. The WAL's shared-memory coordination depends on POSIX file locks that network filesystems do not guarantee. Use a local filesystem. If you are in a container environment (Docker, Kubernetes), a bind-mounted host path is a local filesystem from the kernel's perspective; a volume mounted from a network-backed storage driver may not be.

## Cross-platform file locking

SQLite handles locking of the database file, WAL file, and `.db-shm` file internally using the appropriate OS API: `fcntl(2)` with `F_SETLK`/`F_GETLK` on Linux and macOS, `LockFileEx`/`UnlockFileEx` on Windows. Applications do not need to replicate this for the database file itself.

The case where applications do need cross-platform file locking is **application-level coordination**: "only one process may run migrations at a time", "the backup job must hold an exclusive lock while it copies", "this scheduled task must not overlap with itself." For these, you cannot portably use `flock(2)` directly: `flock` is not inherited by child processes in the way `fcntl` locks are, and its semantics differ between Linux and macOS. On Windows, neither `flock` nor `fcntl` is available.

**Use `portalocker` in Python.** `portalocker` wraps `fcntl` on POSIX and `LockFileEx` on Windows behind a consistent API. It is the canonical cross-platform lock library for Python.

```python
import portalocker

LOCK_FILE = "/var/run/myapp/migration.lock"


def run_migrations_with_lock(db_path: str) -> None:
    """Run schema migrations, ensuring only one process runs them at a time.

    portalocker.Lock acquires an exclusive advisory lock on LOCK_FILE.
    The `timeout` argument causes it to raise portalocker.LockTimeout rather
    than blocking forever if another process holds the lock.
    """
    with portalocker.Lock(LOCK_FILE, timeout=30, fail_when_locked=False) as fh:
        # fh is a file object; the lock is released when the context exits.
        _apply_pending_migrations(db_path)
```

`portalocker.Lock` uses `LOCK_EX` (exclusive) by default. Pass `flags=portalocker.LOCK_SH` for a shared (reader) lock. The `timeout` parameter is in seconds; `LockTimeout` is raised if the lock cannot be acquired within the window.

**In Rust**, use the `fd-lock` or `fs2` crate. Both wrap `fcntl` on POSIX and `LockFileEx` on Windows:

```rust
use std::fs::File;
use fs2::FileExt;

fn run_migrations_with_lock(lock_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let lock_file = File::create(lock_path)?;
    // try_lock_exclusive returns Err immediately if the lock is held.
    // lock_exclusive blocks until acquired.
    lock_file.try_lock_exclusive()?;
    // Lock is released when lock_file is dropped at end of scope.
    apply_pending_migrations()?;
    Ok(())
}
```

Advisory file locks are cooperative: they only work if every participant acquires the lock before proceeding. A process that skips the lock check bypasses the coordination. Document the locking protocol and ensure every entry point — application code, migration CLI, test fixtures, scheduled jobs — goes through it.

## When the OS lies about locks

File locking has a catalogue of known unreliable environments:

**NFS (all versions).** NFS has a long history of broken `fcntl` lock semantics. NLM (the Network Lock Manager used by NFSv3) has well-documented failure modes in partitioned networks. NFSv4 improved but did not eliminate them. The practical rule: do not put a SQLite database on an NFS mount. If an existing deployment does this, validate behaviour with a multi-writer stress test on the target NFS server and kernel version before trusting it in production.

**SMB/CIFS.** Windows file shares have similar lock-semantic issues when accessed from Linux via `cifs`. SMB byte-range locks and POSIX `fcntl` locks are not the same abstraction. Avoid.

**FUSE.** FUSE filesystems vary by implementation. Most FUSE drivers do not implement POSIX lock semantics at all. Encrypted filesystems (EncFS, gocryptfs), user-space network filesystems (SSHFS), and synthetic filesystems are all suspect.

**Docker bind mounts on macOS.** macOS-hosted Docker (Docker Desktop) uses a hypervisor layer; bind mounts from the macOS host to the Linux container go through `virtio-fs` or `grpcfuse`. Lock semantics and `fsync` guarantees on these mounts are not equivalent to a native Linux filesystem. This is a common development-environment trap where SQLite works correctly in CI (Linux) but produces occasional corruption in local development (macOS + Docker volume).

**Discipline.** If you do not control the filesystem where the database will live, validate behaviour explicitly. A multi-writer stress test — two processes each doing 1000 write transactions with integrity-check verification at the end — will expose lock semantic violations faster than any documentation review. Do not rely on "it worked in testing" if testing used a different filesystem than production.

## The single-writer thread pattern

When write contention becomes the bottleneck — SQLITE_BUSY appearing despite `busy_timeout`, WAL growing without checkpoint progress, or profiling showing threads spending most of their time waiting for RESERVED — the application-level response is to route all writes through a single thread (or single process) and have competing callers submit write *requests* via a queue.

This is an application pattern, not a SQLite primitive. SQLite does not know about the queue; it continues to enforce its own single-writer model at the lock level. The queue's purpose is to:

1. Eliminate the busy-wait loop between competing writers — callers submit and wait for a result rather than spinning on lock acquisition.
2. Give the single writer a chance to batch multiple enqueued writes into one transaction, amortising commit overhead.
3. Provide backpressure: if the queue grows, the application knows writes are outpacing the writer's throughput — a clear signal that the workload is approaching the embedded database boundary.

Readers continue to use their own direct connections; there is no benefit to routing reads through the writer queue.

## Worked example

A `WriteQueue` class in Python that owns the only write connection, accepts callable jobs, and returns results via `concurrent.futures.Future`. Callers submit work and block until the writer executes it; the writer thread processes jobs one at a time (or in batches).

```python
import sqlite3
import threading
import queue
import concurrent.futures
from contextlib import contextmanager
from typing import Callable, Any, Generator


class WriteQueue:
    """Routes all SQLite writes through a single background thread.

    Readers use their own per-thread connections (see connection() in
    sqlite-fundamentals.md). Only writes go through this queue.

    Usage:
        wq = WriteQueue("/var/lib/myapp/data.db")
        wq.start()

        future = wq.submit(lambda conn: conn.execute(
            "INSERT INTO events (type) VALUES (?)", ("user.login",)
        ))
        future.result()  # blocks until the write completes

        wq.stop()
    """

    def __init__(self, db_path: str, maxsize: int = 0) -> None:
        self._db_path = db_path
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread: threading.Thread | None = None
        self._conn: sqlite3.Connection | None = None

    def start(self) -> None:
        """Start the writer thread. Call once at application startup."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="sqlite-writer")
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Flush pending writes and stop the writer thread."""
        self._queue.put(None)  # sentinel
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def submit(self, job: Callable[[sqlite3.Connection], Any]) -> concurrent.futures.Future:
        """Submit a write job and return a Future for its result.

        job receives the write connection and may execute any number of
        statements inside an already-open BEGIN IMMEDIATE transaction.
        The queue commits the transaction after job returns; on exception,
        it rolls back and the Future raises the exception.
        """
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((job, future))
        return future

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA cache_size = -8000")
        conn.execute("PRAGMA temp_store = MEMORY")
        return conn

    def _run(self) -> None:
        """Writer thread main loop. Processes jobs one at a time."""
        self._conn = self._open()
        while True:
            item = self._queue.get()
            if item is None:
                break  # stop sentinel
            job, future = item
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    result = job(self._conn)
                    self._conn.execute("COMMIT")
                    future.set_result(result)
                except Exception as exc:
                    self._conn.execute("ROLLBACK")
                    future.set_exception(exc)
            except Exception as exc:
                # BEGIN IMMEDIATE itself failed (e.g. SQLITE_BUSY after busy_timeout)
                future.set_exception(exc)
        self._conn.close()
```

Usage in a web handler:

```python
_write_queue = WriteQueue("/var/lib/myapp/data.db")
_write_queue.start()


def record_event(event_type: str, payload: str) -> int:
    """Insert an event row and return its new rowid."""

    def _insert(conn: sqlite3.Connection) -> int:
        cursor = conn.execute(
            "INSERT INTO events (type, payload) VALUES (?, ?)",
            (event_type, payload),
        )
        return cursor.lastrowid

    return _write_queue.submit(_insert).result(timeout=10.0)
```

**Batching.** The queue can be extended to drain multiple pending jobs into one transaction before committing, amortising commit overhead:

```python
def _run_batched(self) -> None:
    self._conn = self._open()
    while True:
        item = self._queue.get()
        if item is None:
            break
        batch = [item]
        # Drain any additional items that arrived while we were processing the first.
        try:
            while True:
                batch.append(self._queue.get_nowait())
        except queue.Empty:
            pass

        self._conn.execute("BEGIN IMMEDIATE")
        results: list[tuple[Any, concurrent.futures.Future]] = []
        try:
            for job, future in batch:
                if job is None:
                    # sentinel inside a batch: finish the transaction first
                    break
                result = job(self._conn)
                results.append((result, future))
            self._conn.execute("COMMIT")
            for result, future in results:
                future.set_result(result)
        except Exception as exc:
            self._conn.execute("ROLLBACK")
            for _, future in results:
                if not future.done():
                    future.set_exception(exc)
```

Batching trades latency (the submitter waits slightly longer for the batch to fill) for throughput (fewer fsync calls per write). Tune the batch-drain strategy based on the observed arrival rate.

## Anti-patterns

- **WAL on NFS or SMB.** WAL mode's shared-memory coordination relies on POSIX lock semantics that network filesystems do not implement correctly. The failure mode is silent corruption, not an error on open. Do not put a WAL-mode SQLite database on a network filesystem, regardless of what the filesystem vendor claims.

- **One connection shared across multiple threads without serialisation.** Passing `check_same_thread=False` to Python's `sqlite3.connect()` disables the guard, not the hazard. Interleaved transactions from two threads sharing one connection produce transaction boundaries that neither thread intended. Use per-thread connections.

- **Using `flock()` directly for application-level coordination across platforms.** `flock` semantics differ between Linux and macOS, and it is unavailable on Windows. Use `portalocker` (Python) or `fs2`/`fd-lock` (Rust) which wrap the correct OS API on each platform.

- **Spinning on SQLITE_BUSY without backoff.** A tight retry loop that immediately re-attempts `BEGIN IMMEDIATE` wastes CPU, may cause livelock-like contention, and does not give the current holder time to commit. Always sleep between retries. Exponential backoff with jitter distributes retry pressure. The queue pattern eliminates this entirely by ensuring only one writer attempts writes at a time.

- **A long-running write transaction while readers wait for the next snapshot.** In WAL mode, a writer holding an open write transaction has committed to a WAL frame; readers at the next `BEGIN DEFERRED` see that frame. But if the write transaction runs for minutes (large batch import, slow migration), it delays WAL checkpointing and bloats the WAL file. Break large write workloads into bounded transaction sizes. The queue pattern supports this naturally — each submitted job is one transaction.

- **Application-level locking via the database itself.** SQLite does not have `SELECT ... FOR UPDATE`. `LOCK TABLE` does not exist in SQLite. Advisory application locks cannot be implemented via a "locks" table with `UPDATE ... WHERE holder IS NULL`, because two concurrent readers can both read `NULL` before either writes. Application-level coordination requires either `BEGIN IMMEDIATE` (letting SQLite's own lock serialise writers) or an external mechanism like `portalocker`. Do not attempt to build a lock table inside SQLite.

- **Inheriting a connection across `fork()` without re-opening.** The forked child inherits the parent's connection state. Both processes then modify the same in-memory SQLite structures against the same file with no coordination. Close all connections before forking; open fresh connections in the child.

## Cross-references

- [`sqlite-fundamentals.md`](sqlite-fundamentals.md) — connection lifecycle: what a connection is, why two connections to the same file compete rather than collaborate, per-thread pattern.
- [`pragma-discipline.md`](pragma-discipline.md) — `busy_timeout` and `wal_autocheckpoint`, which directly affect how concurrent access behaves; the WAL sidecar file caveat for backups.
- [`transactions-and-isolation.md`](transactions-and-isolation.md) — `BEGIN IMMEDIATE` vs `BEGIN DEFERRED`: the lock-upgrade race that concurrent writers cause, and when to use each flavour.
- [`optimistic-locking-and-leases.md`](optimistic-locking-and-leases.md) — an alternative coordination strategy for long-lived workflows: detect concurrent modification with a version column rather than holding a write lock across user think-time.
- [`backup-restore-and-corruption.md`](backup-restore-and-corruption.md) — corruption from misconfigured concurrent access: what `PRAGMA integrity_check` catches, the Online Backup API as the correct concurrent-safe backup path.
- [`boundary-and-when-to-leave.md`](boundary-and-when-to-leave.md) — when the single-writer model is the right signal that the workload has grown beyond the embedded database envelope.
