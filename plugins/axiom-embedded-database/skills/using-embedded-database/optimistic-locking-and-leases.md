---
name: optimistic-locking-and-leases
description: Use when multiple workers compete to claim the same row — job queues, distributed task runners, claim-and-process patterns — and you need at-most-one semantics without holding a write lock across user think-time or network I/O. Covers version-column compare-and-swap, claim leases with expiry, atomic find-and-claim via RETURNING, heartbeat discipline, and the dead-worker recovery pattern.
---

# Optimistic Locking and Claim Leases

**When two workers compete for the same row, the question is not "who acquires the lock" but "who notices they didn't" — optimistic locking and claim leases turn race conditions into testable, observable application logic.**

## When this earns its cost

Read this sheet when:

- Multiple worker processes or threads pull jobs from a shared queue stored in SQLite. Two workers must not both claim the same job.
- A claim-and-process pattern requires "at most one winner": whichever worker loses the race must discover this immediately, not after doing the work.
- A write transaction spans user think-time, a network call, or any operation too slow to hold a `BEGIN IMMEDIATE` lock across. The lock would block every other writer for the duration.
- You need distributed failure detection: a worker can die mid-job, and other workers must be able to recover ownership after a timeout.
- You want to turn "two threads raced" from a silent correctness bug into a testable, observable event.

`transactions-and-isolation.md` covers `BEGIN IMMEDIATE` — the mechanism that prevents races entirely by holding the RESERVED lock. That is correct when the write path is short. This sheet covers the alternative: detect the race after the fact, rather than preventing it by holding a lock. `concurrent-access-patterns.md` covers the WAL contract and single-writer thread pattern that underpin both approaches.

## Optimistic locking: version-as-precondition

Optimistic locking is a compare-and-swap pattern applied at the row level. Every mutable row carries a `version INTEGER NOT NULL` column. Reads are cheap and lock-free. Writes carry the version they read and fail atomically if that version has changed.

**Schema pattern:**

```sql
CREATE TABLE documents (
    id      INTEGER PRIMARY KEY,
    content TEXT    NOT NULL,
    version INTEGER NOT NULL DEFAULT 0
);
```

**Read phase** — ordinary `SELECT`, no transaction needed:

```python
row = conn.execute(
    "SELECT id, content, version FROM documents WHERE id = ?",
    (doc_id,),
).fetchone()
# row["version"] is now the precondition for any subsequent write.
```

**Write phase** — the version column is the predicate:

```python
cursor = conn.execute(
    """
    UPDATE documents
       SET content = ?,
           version = version + 1
     WHERE id      = ?
       AND version = ?
    """,
    (new_content, doc_id, row["version"]),
)
if cursor.rowcount == 0:
    # Another writer updated between our read and our write.
    # rowcount == 0 means the WHERE predicate did not match.
    raise StaleVersionError(f"Document {doc_id} was modified concurrently")
# rowcount == 1: we won. version is now row["version"] + 1.
```

`cursor.rowcount` is the observable outcome of the race:

| Value | Meaning |
|-------|---------|
| `1` | Predicate matched. This writer won. |
| `0` | Predicate did not match. Another writer incremented `version` first. |

There is no middle ground. SQLite either applied the UPDATE (one row matched) or it did not (zero rows matched). The version column is the compare; the UPDATE is the swap; `rowcount` is the result.

**Application protocol:**

1. Read the row. Record `version`.
2. Do whatever work informs the write (compute, validate, prompt the user).
3. Write with `AND version = ?` as a predicate.
4. Check `rowcount`. If 0, the read is stale — decide whether to retry from step 1 or surface an error to the caller.

Retry policy is application-specific. A queue worker retrying is usually correct; a user-facing edit conflict usually needs surfacing to the human.

**Important**: the `UPDATE` itself still needs to be inside a proper transaction on the write connection. What optimistic locking removes is the need to hold that transaction open during the *read* and *think* phases. The write transaction can be as tight as a single statement.

## Claim leases: ownership with expiry

A claim lease adds two columns to a row: who owns it, and when that ownership expires. The lease is the key mechanism for the dead-worker problem: if a worker claims a job and then dies, the lease expires and another worker can take over without human intervention.

**Schema addition:**

```sql
CREATE TABLE jobs (
    id               INTEGER PRIMARY KEY,
    payload          TEXT    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'pending',  -- pending | running | done | failed
    assignee         TEXT,                                 -- NULL means unclaimed
    claim_expires_at INTEGER,                              -- Unix milliseconds; NULL means unclaimed
    priority         INTEGER NOT NULL DEFAULT 0
);
```

`claim_expires_at` stores Unix milliseconds as `INTEGER`. SQLite stores integers up to 64-bit signed, so there is no overflow concern for Unix-millisecond timestamps in any foreseeable timeframe.

**Claim operation:**

```python
import time

LEASE_DURATION_MS = 30_000  # 30 seconds

def claim_job(conn: sqlite3.Connection, job_id: int, worker_id: str) -> bool:
    """Attempt to claim job_id for worker_id. Returns True if the claim succeeded."""
    now_ms = int(time.time() * 1000)
    expires_at = now_ms + LEASE_DURATION_MS

    cursor = conn.execute(
        """
        UPDATE jobs
           SET assignee         = ?,
               claim_expires_at = ?,
               status           = 'running'
         WHERE id               = ?
           AND (assignee IS NULL OR claim_expires_at < ?)
        """,
        (worker_id, expires_at, job_id, now_ms),
    )
    return cursor.rowcount == 1
```

The `WHERE` clause is the gate:

- `assignee IS NULL` — row is unclaimed.
- `claim_expires_at < now_ms` — row was claimed but the lease expired (dead worker).

Either condition admits the claim. The `rowcount` check confirms exactly one row was updated. If two workers race on the same `job_id`, SQLite's single-writer model serialises the two `UPDATE` statements — one sees the row in a claimable state and wins; the other sees it already claimed with a fresh expiry and loses (rowcount 0).

**Heartbeat — extending the lease while working:**

```python
def heartbeat(conn: sqlite3.Connection, job_id: int, worker_id: str) -> bool:
    """Extend the lease. Returns False if we no longer own the job (expired + reclaimed)."""
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
```

The `AND assignee = ?` predicate is not optional. Without it, a heartbeat could extend the lease of a job that has already expired and been reclaimed by another worker — silently giving both workers a valid lease on the same job. The `AND claim_expires_at >= now_ms` guard prevents a worker that woke up after its own lease expired from squatting on a job that another worker has already taken.

**Complete operation:**

```python
def complete_job(conn: sqlite3.Connection, job_id: int, worker_id: str) -> bool:
    """Mark the job done. Returns False if the lease had expired before completion."""
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
```

If `rowcount == 0`, the lease expired before the worker finished. Another worker may have picked up the job already. The application must handle this: was the job idempotent? Was any side effect already applied? This is not an error in the locking protocol — it is a signal that the lease duration is too short relative to job duration, or the worker is too slow.

## Claim-next: atomic find-and-claim

`claim_next` — find the highest-priority unclaimed job and claim it in one step — is a common pattern that naive implementations get wrong.

**The naive race:**

```python
# WRONG: two workers can both SELECT the same row before either UPDATE runs.
row = conn.execute(
    "SELECT id FROM jobs WHERE assignee IS NULL ORDER BY priority DESC LIMIT 1"
).fetchone()
if row:
    conn.execute("UPDATE jobs SET assignee = ? WHERE id = ?", (worker_id, row["id"]))
```

Two workers execute the `SELECT` simultaneously, both see the same highest-priority row, both issue the `UPDATE`, and both succeed — because neither `UPDATE` checks that the row is still unclaimed.

**Correct: `UPDATE ... WHERE id = (subquery) RETURNING`:**

```sql
UPDATE jobs
   SET assignee         = ?,
       claim_expires_at = ?,
       status           = 'running'
 WHERE id = (
     SELECT id
       FROM jobs
      WHERE assignee IS NULL
         OR claim_expires_at < ?
      ORDER BY priority DESC
      LIMIT 1
 )
   AND (assignee IS NULL OR claim_expires_at < ?)
RETURNING id, payload, priority
```

This is atomic because the subquery and the `UPDATE` execute as a single statement. SQLite evaluates the subquery to identify the row, applies the `UPDATE`, and returns the row in one operation — no gap exists between "found the row" and "claimed the row" in which another writer can intervene. The `AND (assignee IS NULL OR claim_expires_at < ?)` guard in the outer `WHERE` is a belt-and-suspenders check: even if the subquery found a row, the outer predicate must still hold on the row actually being updated.

`RETURNING` was added in SQLite 3.35.0 (released 2021-03-12). If you are running an older build, use `SELECT` after `UPDATE` inside the same `BEGIN IMMEDIATE` transaction — `last_insert_rowid()` does not apply to `UPDATE`, so a separate `SELECT` is the only option:

```python
# Pre-3.35 workaround
conn.execute("BEGIN IMMEDIATE")
try:
    now_ms = int(time.time() * 1000)
    expires_at = now_ms + LEASE_DURATION_MS
    conn.execute(
        """
        UPDATE jobs
           SET assignee = ?, claim_expires_at = ?, status = 'running'
         WHERE id = (
             SELECT id FROM jobs
              WHERE assignee IS NULL OR claim_expires_at < ?
              ORDER BY priority DESC LIMIT 1
         )
           AND (assignee IS NULL OR claim_expires_at < ?)
        """,
        (worker_id, expires_at, now_ms, now_ms),
    )
    if conn.execute("SELECT changes()").fetchone()[0] == 1:
        row = conn.execute(
            "SELECT id, payload FROM jobs WHERE assignee = ? AND claim_expires_at = ?",
            (worker_id, expires_at),
        ).fetchone()
    else:
        row = None
    conn.execute("COMMIT")
except Exception:
    conn.execute("ROLLBACK")
    raise
```

The `BEGIN IMMEDIATE` + `changes()` approach is safe because `IMMEDIATE` ensures no other writer can modify the table between the `UPDATE` and the `SELECT`. Prefer the `RETURNING` version on SQLite 3.35+; it is cleaner and the intent is explicit.

## Heartbeats and the dead-worker problem

**Lease duration.** The lease must be long enough that a healthy worker can complete the job (or send a heartbeat) before it expires. It must be short enough that a dead worker's job is not stuck for an unreasonable time. A rule of thumb: set lease duration to at least 3× the expected job duration, and heartbeat every `lease_duration / 3`.

If the typical job takes 5 seconds, a 30-second lease with 10-second heartbeats is reasonable. If jobs vary widely in duration, use a longer lease and rely on heartbeats to keep it alive.

**Heartbeat interval = lease / 3.** This gives three heartbeat opportunities per lease period. A worker that misses one heartbeat due to GC pause, network hiccup, or scheduler jitter still has two more before the lease expires. A worker that misses all three is likely genuinely dead.

**The dead-worker problem.** When a worker dies mid-job, its lease eventually expires. Other workers will see the job as claimable again (the `OR claim_expires_at < now_ms` condition in the `WHERE`). This is the intended recovery path.

Two hazards in the recovery:

1. **The zombie worker.** A worker that was declared dead (lease expired) may not be dead — it may have been paused by a GC, an OS scheduler, or a container freeze, and resumes after another worker has already reclaimed the job. The zombie then calls `heartbeat()` or `complete()`. Because both operations include `AND assignee = ?` and `AND claim_expires_at >= ?`, they will find `rowcount == 0` — the lease is now held by the new worker. The zombie must detect `rowcount == 0` as a signal that it has lost ownership and stop processing.

2. **Idempotency.** If the zombie worker already performed a side effect (sent an email, charged a card, wrote to an external API) before its lease expired, and the new worker does the same, the side effect is doubled. Optimistic locking cannot prevent this — it can only tell you that it happened. Design jobs to be idempotent, or use an idempotency key to detect and ignore duplicate execution.

**Observe expired leases.** A monitoring pass can surface dead-worker events:

```python
def expired_leases(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    now_ms = int(time.time() * 1000)
    return conn.execute(
        """
        SELECT id, assignee, claim_expires_at
          FROM jobs
         WHERE status = 'running'
           AND claim_expires_at < ?
        """,
        (now_ms,),
    ).fetchall()
```

Log or alert on these rows. A cluster of expired leases indicates either jobs taking longer than the lease duration (tune the lease) or workers crashing (investigate the worker).

## Testing race conditions

Testing that exactly one worker wins a race is non-negotiable for any claim-and-process pattern. The test is simple: spawn N workers all racing to claim the same row, assert exactly one wins.

```python
import sqlite3
import threading
import concurrent.futures
import time
import pytest


def make_test_db(path: str) -> None:
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("""
        CREATE TABLE jobs (
            id               INTEGER PRIMARY KEY,
            status           TEXT    NOT NULL DEFAULT 'pending',
            assignee         TEXT,
            claim_expires_at INTEGER
        )
    """)
    conn.execute("INSERT INTO jobs (id) VALUES (1)")
    conn.commit()
    conn.close()


def claim_attempt(db_path: str, worker_id: str, lease_ms: int = 30_000) -> bool:
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    now_ms = int(time.time() * 1000)
    cursor = conn.execute(
        """
        UPDATE jobs
           SET assignee = ?, claim_expires_at = ?, status = 'running'
         WHERE id = 1
           AND (assignee IS NULL OR claim_expires_at < ?)
        """,
        (worker_id, now_ms + lease_ms, now_ms),
    )
    won = cursor.rowcount == 1
    conn.close()
    return won


def test_exactly_one_worker_wins(tmp_path):
    db_path = str(tmp_path / "race.db")
    make_test_db(db_path)

    n_workers = 20
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(claim_attempt, db_path, f"worker-{i}")
            for i in range(n_workers)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    winners = [r for r in results if r]
    assert len(winners) == 1, f"Expected exactly 1 winner, got {len(winners)}"
```

Run this test with `-v` to confirm it passes reliably. A failing test (0 or 2+ winners) indicates a broken predicate in the claim `UPDATE`. Vary `n_workers` upward to stress-test; 50–100 workers should still yield exactly one winner.

## Worked example

A complete worker-pool job runner using the claim lease pattern.

```python
import sqlite3
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

LEASE_DURATION_MS = 60_000   # 60 seconds
HEARTBEAT_INTERVAL_S = 20    # ~lease / 3


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id               INTEGER PRIMARY KEY,
            payload          TEXT    NOT NULL,
            status           TEXT    NOT NULL DEFAULT 'pending',
            assignee         TEXT,
            claim_expires_at INTEGER,
            priority         INTEGER NOT NULL DEFAULT 0
        )
    """)


def claim_next(conn: sqlite3.Connection, worker_id: str) -> Optional[sqlite3.Row]:
    """Atomically find and claim the highest-priority available job.

    Returns the claimed row, or None if no jobs are available.
    Requires SQLite 3.35+ for RETURNING. For older builds, use BEGIN IMMEDIATE
    + UPDATE + SELECT changes() inside a transaction (see sheet body).
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
    """Extend the lease. Returns False if we no longer own the job."""
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


def complete_job(conn: sqlite3.Connection, job_id: int, worker_id: str) -> bool:
    """Mark the job done. Returns False if the lease had already expired."""
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


def abandon_job(conn: sqlite3.Connection, job_id: int, worker_id: str) -> bool:
    """Release a claimed job back to pending. Used on clean worker shutdown."""
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


def run_worker(db_path: str, worker_id: str) -> None:
    """Main worker loop. Claim jobs, process them, heartbeat, complete."""
    conn = open_db(db_path)
    logger.info("%s: started", worker_id)

    while True:
        job = claim_next(conn, worker_id)
        if job is None:
            time.sleep(1)
            continue

        logger.info("%s: claimed job %d", worker_id, job["id"])
        last_heartbeat = time.monotonic()

        try:
            # Simulate work with periodic heartbeats.
            for step in range(10):
                time.sleep(1)                         # actual work goes here

                if time.monotonic() - last_heartbeat >= HEARTBEAT_INTERVAL_S:
                    if not heartbeat(conn, job["id"], worker_id):
                        logger.warning("%s: lost lease on job %d — stopping", worker_id, job["id"])
                        break                          # zombie path: stop processing
                    last_heartbeat = time.monotonic()
            else:
                if complete_job(conn, job["id"], worker_id):
                    logger.info("%s: completed job %d", worker_id, job["id"])
                else:
                    logger.warning("%s: lease expired before completion of job %d", worker_id, job["id"])

        except Exception:
            logger.exception("%s: error on job %d", worker_id, job["id"])
            abandon_job(conn, job["id"], worker_id)
```

## Anti-patterns

**SELECT-then-UPDATE without a version predicate or assignee check.** Reading a row and updating it in two separate statements with no predicate tying the update to what was read is a silent race condition. Two workers read the same unclaimed row; both update it; both believe they won. The application has two workers processing the same job with no awareness of the collision. The version column or the `AND assignee IS NULL` guard is the fix; skipping it silently corrupts the queue.

**Leases that never expire.** Setting `claim_expires_at` to `NULL` (or to a date far in the future) as a shortcut means a dead worker wedges its jobs permanently. No other worker can ever take them over. The queue drains to zero apparent pending work while running jobs never complete. Always set a finite expiry. The expiry duration can be generous — an hour for long jobs is fine — but it must be finite.

**Heartbeat without the owner guard.** A heartbeat that updates `claim_expires_at` without `AND assignee = ?` and `AND claim_expires_at >= ?` is dangerous: a zombie worker that resumes after its lease expired will successfully extend a lease that now belongs to another worker. Two workers then hold what they each believe is a valid lease. The `AND assignee = ?` predicate is the identity check; the `AND claim_expires_at >= ?` predicate is the expiry check. Both are required.

**Using `BEGIN EXCLUSIVE` instead of a version column for long-lived workflows.** `BEGIN EXCLUSIVE` acquires the exclusive lock for the entire duration of the transaction. If the transaction spans a network call, a file operation, or user interaction, every other writer is blocked for that duration. Under WAL mode, readers proceed, but all other writers queue up or return `SQLITE_BUSY`. A 30-second job holding `BEGIN EXCLUSIVE` makes the database effectively unavailable for writes for 30 seconds. Use version columns and lease predicates instead: they turn a held lock into a query predicate, and the lock is released immediately after each write.

**Heartbeat interval too long relative to lease duration.** If the lease is 60 seconds and the heartbeat fires every 50 seconds, a single missed heartbeat causes the lease to expire. A GC pause, a slow disk flush, or a scheduler hiccup of more than 10 seconds loses the job. The heartbeat interval should be at most `lease_duration / 3`, giving three chances per lease period. Tune lease duration up rather than heartbeat interval down — more frequent heartbeats mean more write load; a longer lease means more dead-worker recovery time. The trade-off is explicit.

**Comparing expiry times across mismatched clock sources.** `claim_expires_at < ?` is only correct when both sides are from the same clock source and scale. If one side is Unix milliseconds and the other is Unix seconds, the comparison will always find leases expired (or never expired, depending on the direction of the mismatch). Store everything in one unit — the schema and every caller must agree. This sheet uses Unix milliseconds throughout as the canonical form. Do not mix `time.time()` (seconds, float) and `int(time.time() * 1000)` (milliseconds, integer) in the same table. Do not compare wall-clock `time.time()` to `time.monotonic()` — monotonic clocks are relative to process start and cannot be stored as absolute timestamps.

## Cross-references

- [`transactions-and-isolation.md`](transactions-and-isolation.md) — when a single-row compare-and-swap isn't enough, escalate to a `BEGIN IMMEDIATE` transaction; the write-then-read pattern and lock upgrade race that optimistic locking avoids.
- [`concurrent-access-patterns.md`](concurrent-access-patterns.md) — the WAL contract and single-writer model that makes claim-lease `UPDATE` statements safely atomic; the soft alternative to OS locks.
- [`parameterized-sql-only.md`](parameterized-sql-only.md) — version values, worker IDs, and timestamps must be parameterised; a version number concatenated into SQL is a SQL injection vulnerability.
- [`pragma-discipline.md`](pragma-discipline.md) — `busy_timeout` affects how long a claim `UPDATE` waits for the RESERVED lock under contention; set it consistently on every connection.
- [`boundary-and-when-to-leave.md`](boundary-and-when-to-leave.md) — when claim-lease contention at the SQLite layer signals the queue workload has grown beyond the embedded database envelope.
