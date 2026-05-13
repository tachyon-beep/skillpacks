---
name: transactions-and-isolation
description: Use when choosing between BEGIN DEFERRED, IMMEDIATE, and EXCLUSIVE; when SQLITE_BUSY appears under concurrent load; when a read inside a transaction unexpectedly sees a partial commit; or when a write-then-read pattern produces stale data under contention. Covers the three BEGIN flavours, the lock ladder, SQLITE_BUSY retry discipline, savepoints, read-only snapshot semantics, and the write-then-read anti-pattern. Assumes WAL mode is configured (pragma-discipline.md).
---

# Transactions and Isolation

**SQLite's transaction grammar gives you three BEGIN flavours — DEFERRED, IMMEDIATE, EXCLUSIVE — and the wrong choice does not error, it surprises. The correct flavour is determined by what the transaction will do, not by what's convenient.**

## When this earns its cost

Read this sheet when:

- You are seeing `SQLITE_BUSY` under concurrent read and write load and `busy_timeout` is already set — the likely cause is `BEGIN DEFERRED` on a write path, not an insufficient timeout.
- A read inside a transaction is seeing a partial commit: another connection wrote some rows but not all, and your SELECT is pulling mixed state. This indicates you are reading without a transaction at all (auto-commit mode, statement by statement) rather than inside a snapshot-pinning BEGIN.
- You are choosing between `DEFERRED` and `IMMEDIATE` on a write-heavy path and want to understand the actual concurrency trade-off, not just pick the default.
- A `SELECT` followed by an `UPDATE` inside the same transaction produces lost updates or write conflicts under any load.

`sqlite-fundamentals.md` covers the connection-as-isolation-unit model that makes these flavours meaningful; read that first. `pragma-discipline.md` covers `busy_timeout`, which is the correct backstop for short lock collisions — this sheet covers when collisions happen and how to prevent the structural ones.

## The three BEGIN flavours

| Flavour | First lock acquired | Subsequent upgrade? | Best for |
|-------------|-------------------------------|-------------------------------|-------------------------------------------|
| `DEFERRED` | None — SHARED on first read | Yes, RESERVED on first write — can produce `SQLITE_BUSY` at upgrade | Pure-read transactions; writes only if you accept retry logic |
| `IMMEDIATE` | RESERVED at BEGIN | No upgrade needed | Read-then-write transactions; the standard write path |
| `EXCLUSIVE` | EXCLUSIVE at BEGIN | n/a | Schema migrations, full-table rewrites; see WAL note below |

Plain `BEGIN` with no keyword is `BEGIN DEFERRED`. That default is the source of most unexpected `SQLITE_BUSY` events in SQLite applications.

### The lock ladder

SQLite's locking protocol is a four-step ladder. Understanding it explains every concurrency surprise.

**SHARED** — acquired when a connection first reads a page. Multiple connections can hold SHARED simultaneously. In rollback-journal mode, a writer cannot acquire EXCLUSIVE while any SHARED is held — reads block writes. In WAL mode, SHARED readers see a snapshot and do not block writers at all.

**RESERVED** — acquired by a writer when it is ready to begin modifying pages but has not yet done so. Only one connection can hold RESERVED at a time. New SHARED readers are still admitted and can read the current (pre-write) state of the database. RESERVED is the lock that `BEGIN IMMEDIATE` acquires at transaction open. It signals "I will write" without yet excluding readers.

**PENDING** — an intermediate state that `RESERVED` transitions through on the way to `EXCLUSIVE`. Once PENDING, no new SHARED locks are admitted, but connections already holding SHARED can finish their current reads. PENDING waits for in-flight readers to drain. This state is transient under light load; under sustained read concurrency it is the pause that `busy_timeout` is absorbing.

**EXCLUSIVE** — the only lock level under which write operations can modify database pages (in rollback-journal mode). No other connection holds any lock. All readers must wait. In **WAL mode**, `EXCLUSIVE` does not block readers — WAL readers see their snapshot from the WAL file regardless of the writer's lock. In WAL mode, `BEGIN EXCLUSIVE` and `BEGIN IMMEDIATE` are functionally equivalent from a concurrency perspective: both acquire RESERVED at BEGIN, and WAL readers proceed concurrently in both cases. `BEGIN EXCLUSIVE` adds value only in rollback-journal mode, where it prevents new SHARED readers for the transaction's duration.

**Practical implication.** In a WAL-mode production database, `BEGIN IMMEDIATE` is the correct default for all write transactions. `BEGIN EXCLUSIVE` is not "more safe" — it is the same as IMMEDIATE under WAL, and in the rare case where you switch to rollback journal (e.g., database file on a network share that can't use WAL), it serialises readers unnecessarily.

## SQLITE_BUSY: the polite collision

`SQLITE_BUSY` is not a failure — it is the expected signal that two connections wanted the same lock at the same time and SQLite resolved the conflict by telling one of them to wait or retry. Under a correctly-designed concurrent workload, `SQLITE_BUSY` should be rare but not surprising.

**Where it comes from:**

- **Lock upgrade in DEFERRED transactions.** `BEGIN DEFERRED` acquires nothing at BEGIN. The first `SELECT` acquires SHARED. The first `INSERT`, `UPDATE`, or `DELETE` attempts to acquire RESERVED. If another connection already holds RESERVED (it is in the middle of writing), the upgrade fails with `SQLITE_BUSY` immediately — SQLite does not wait. `busy_timeout` retries this, but the correct fix is `BEGIN IMMEDIATE`, which acquires RESERVED at the start and never needs an upgrade.

- **PENDING waiting for SHARED to drain.** A writer has advanced to PENDING (no new readers admitted, waiting for existing readers to finish). If the `busy_timeout` window expires before readers release their SHARED locks — because a long-running read transaction is holding open — `SQLITE_BUSY` propagates to the writer. The fix here is eliminating long-lived read transactions (see the read-only transaction section below), not lengthening the timeout indefinitely.

`busy_timeout` is the right tool for absorbing brief, transient collisions — a few milliseconds of overlap between a writer finishing and the next writer starting. It is not the right tool for a structural problem where reads and writes are routinely competing for seconds at a time.

**Retry loop with bounded exponential backoff** (for cases where `busy_timeout` is insufficient or you need application-level control over retry policy):

```python
import sqlite3
import time
import random


def execute_with_retry(
    conn: sqlite3.Connection,
    fn,
    max_retries: int = 5,
    base_delay: float = 0.05,
) -> None:
    """Execute fn(conn) inside a BEGIN IMMEDIATE, retrying on SQLITE_BUSY.

    fn must not issue BEGIN/COMMIT/ROLLBACK; this wrapper owns the transaction.
    Raises the last OperationalError if all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                fn(conn)
                conn.execute("COMMIT")
                return
            except Exception:
                conn.execute("ROLLBACK")
                raise
        except sqlite3.OperationalError as e:
            if "database is locked" not in str(e) or attempt == max_retries - 1:
                raise
            jitter = random.uniform(0, base_delay)
            time.sleep(base_delay * (2 ** attempt) + jitter)
```

The jitter term (random fraction of the base delay) prevents retry thundering-herd when multiple threads or processes all back off on the same schedule. Cap max_retries and surface a clean error — spinning forever is not an option.

## Savepoints

Savepoints are nested transaction scopes: `SAVEPOINT name`, `RELEASE name`, `ROLLBACK TO name`. They are issued inside an existing transaction (or in auto-commit mode, where the first `SAVEPOINT` implicitly opens an outer transaction).

**When to use them:**

- A library or framework component needs its own rollback boundary without disrupting the caller's transaction context.
- A migration that processes rows in batches needs to roll back one bad batch without losing the committed batches before it.
- A multi-step operation wants partial rollback on a sub-failure while keeping the outer transaction's work.

```python
conn.execute("BEGIN IMMEDIATE")
try:
    conn.execute("INSERT INTO audit_log (event) VALUES ('import_start')")

    conn.execute("SAVEPOINT batch_1")
    try:
        conn.executemany("INSERT INTO items (sku, qty) VALUES (?, ?)", batch_1_rows)
        conn.execute("RELEASE batch_1")      # "commit" the savepoint — it merges into the outer txn
    except sqlite3.IntegrityError:
        conn.execute("ROLLBACK TO batch_1")  # discard only this batch; outer txn survives
        conn.execute("RELEASE batch_1")      # must release after rollback to remove the savepoint name

    conn.execute("SAVEPOINT batch_2")
    try:
        conn.executemany("INSERT INTO items (sku, qty) VALUES (?, ?)", batch_2_rows)
        conn.execute("RELEASE batch_2")
    except sqlite3.IntegrityError:
        conn.execute("ROLLBACK TO batch_2")
        conn.execute("RELEASE batch_2")

    conn.execute("COMMIT")
except Exception:
    conn.execute("ROLLBACK")
    raise
```

`ROLLBACK TO name` rewinds to the state at the time of the `SAVEPOINT` — it does not remove the savepoint. `RELEASE name` removes it. The pattern is always: on failure, `ROLLBACK TO` then `RELEASE`; on success, just `RELEASE`.

Savepoints nest arbitrarily deep but share the outer transaction's lock: a savepoint inside a `BEGIN DEFERRED` transaction still holds only SHARED until the first write.

## Read-only transactions

A bare `SELECT` in auto-commit mode (Python's default, or any session without an open `BEGIN`) executes as its own single-statement transaction. Each statement acquires and releases SHARED independently. Two consecutive `SELECT` statements with no surrounding `BEGIN` see whatever state existed at their respective execution times — there is no snapshot pinning them together.

Under WAL mode, a `BEGIN DEFERRED` followed by reads only pins a snapshot at the moment of the first read. All `SELECT` statements inside that transaction see the same consistent view of the database, regardless of what other connections write while the transaction is open. This is WAL's reader-snapshot guarantee.

**Python's `isolation_level` wrinkle.** Python's `sqlite3` module, when used with the default `isolation_level=""` (not `None`), auto-begins a transaction around DML statements (`INSERT`, `UPDATE`, `DELETE`, `REPLACE`) but **not** around `SELECT`. This means:

```python
# With isolation_level="" (the default):
conn = sqlite3.connect("myapp.db")
row1 = conn.execute("SELECT count(*) FROM events").fetchone()  # no implicit BEGIN; reads current state
row2 = conn.execute("SELECT count(*) FROM events").fetchone()  # another standalone read; may see new rows
# row1 and row2 are not guaranteed to be equal even if no application code ran between them.
```

For consistent multi-statement reads, use explicit `BEGIN DEFERRED`:

```python
conn = sqlite3.connect("myapp.db", isolation_level=None)  # autocommit; all boundaries are explicit
conn.execute("BEGIN DEFERRED")
try:
    row1 = conn.execute("SELECT count(*) FROM events").fetchone()
    row2 = conn.execute("SELECT sum(amount) FROM events").fetchone()
    # row1 and row2 see the same snapshot.
    conn.execute("COMMIT")
finally:
    # If an exception caused BEGIN but no COMMIT, clean up.
    # In practice, wrap in a context manager (see worked example).
    pass
```

The recommendation throughout this pack is `isolation_level=None` (autocommit, explicit `BEGIN`/`COMMIT` everywhere) rather than relying on Python's implicit transaction management, precisely because the implicit mode's behaviour around `SELECT` is a source of non-obvious inconsistencies.

## The write-then-read pattern

A common mistake: open `BEGIN DEFERRED`, execute a `SELECT` to decide what to write, then execute an `UPDATE` or `INSERT`. Under any concurrent write load, this is a race condition.

```python
# INCORRECT under contention:
conn.execute("BEGIN")  # DEFERRED — acquires nothing
row = conn.execute("SELECT balance FROM accounts WHERE id = ?", (acct_id,)).fetchone()
# SHARED acquired here — balance is 100.
# Another connection writes balance = 50 and commits HERE.
# This connection has SHARED; the other had RESERVED and is now committed.
# Lock upgrade: SHARED → RESERVED — may SQLITE_BUSY here, or may succeed.
# If it succeeds, the UPDATE below uses a balance (100) that is now stale.
conn.execute("UPDATE accounts SET balance = ? WHERE id = ?", (row["balance"] - 20, acct_id))
conn.execute("COMMIT")
```

The problem: `BEGIN DEFERRED` held only SHARED during the `SELECT`. Another writer could acquire RESERVED, commit, and change the row before this connection attempts to write. If the write does succeed (after `busy_timeout` retries the lock upgrade), it is working with stale data read before the other writer's commit.

**Solution: `BEGIN IMMEDIATE`.** Acquire RESERVED at the start of the transaction. No other writer can acquire RESERVED while this transaction is open. The SELECT and the UPDATE see a consistent write context:

```python
# CORRECT:
conn.execute("BEGIN IMMEDIATE")  # RESERVED acquired — no other writer can proceed
try:
    row = conn.execute("SELECT balance FROM accounts WHERE id = ?", (acct_id,)).fetchone()
    conn.execute(
        "UPDATE accounts SET balance = ? WHERE id = ?",
        (row["balance"] - 20, acct_id),
    )
    conn.execute("COMMIT")
except Exception:
    conn.execute("ROLLBACK")
    raise
```

For workflows where the SELECT and UPDATE are far apart in the code — or where the application wants to detect concurrent modification rather than prevent it — see `optimistic-locking-and-leases.md`.

## Worked example

A production-quality retry-on-busy wrapper for Python, with sensible jitter, a max-retries cap, and a clean error on exhaustion. This is the pattern to copy into a database utility module; do not scatter ad-hoc retry loops across the codebase.

```python
import sqlite3
import time
import random
import logging
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)

_MAX_RETRIES = 8
_BASE_DELAY_S = 0.025   # 25 ms starting point
_MAX_DELAY_S = 2.0      # cap any single sleep


@contextmanager
def write_transaction(
    conn: sqlite3.Connection,
    mode: str = "IMMEDIATE",
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for a write transaction with automatic BUSY retry.

    Args:
        conn:  A connection opened with isolation_level=None.
        mode:  "IMMEDIATE" (default, correct for writes) or "EXCLUSIVE"
               (schema migrations; under WAL, identical to IMMEDIATE).

    Yields the connection for use within the transaction body. On COMMIT
    failure due to SQLITE_BUSY, retries with exponential backoff up to
    _MAX_RETRIES times. On final exhaustion, raises OperationalError.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            conn.execute(f"BEGIN {mode}")
        except sqlite3.OperationalError as e:
            # BEGIN IMMEDIATE can itself return SQLITE_BUSY if RESERVED is held.
            if "database is locked" not in str(e):
                raise
            if attempt == _MAX_RETRIES - 1:
                raise
            _backoff(attempt)
            continue

        try:
            yield conn
            conn.execute("COMMIT")
            return
        except sqlite3.OperationalError as e:
            conn.execute("ROLLBACK")
            if "database is locked" not in str(e) or attempt == _MAX_RETRIES - 1:
                raise
            logger.debug("SQLITE_BUSY on attempt %d; retrying", attempt + 1)
            _backoff(attempt)
        except Exception:
            conn.execute("ROLLBACK")
            raise


def _backoff(attempt: int) -> None:
    delay = min(_BASE_DELAY_S * (2 ** attempt), _MAX_DELAY_S)
    jitter = random.uniform(0, delay * 0.2)
    time.sleep(delay + jitter)
```

Usage:

```python
with write_transaction(conn) as c:
    c.execute(
        "UPDATE counters SET value = value + 1 WHERE name = ?",
        ("page_views",),
    )
```

The wrapper handles both `BEGIN IMMEDIATE` refusing to acquire the lock (lock held by another writer at BEGIN time) and `COMMIT` being refused (rare in WAL mode, but possible). The max-retries cap ensures the caller always gets a response — it either succeeds or raises a clean `OperationalError` that the application can log and surface.

## Anti-patterns

**`BEGIN` (plain / DEFERRED) before a write in a contended workload.** This is `BEGIN DEFERRED`. The transaction acquires SHARED on the first read, then attempts to upgrade to RESERVED on the first write. Under any concurrent write load, the upgrade races. If it succeeds, the SELECT result may be stale (another writer committed between the SELECT and the write). If it fails, `SQLITE_BUSY` surfaces mid-transaction. The fix is `BEGIN IMMEDIATE`.

**Nested `BEGIN` without `SAVEPOINT`.** SQLite does not support nested `BEGIN` statements. A second `BEGIN` inside an open transaction raises an error. If you need a nested rollback scope, use `SAVEPOINT`/`RELEASE`/`ROLLBACK TO`. Libraries that manage their own transaction context (ORMs, migration runners) must use savepoints when there may already be an outer transaction in progress.

**Relying on auto-commit for multi-statement consistency.** Any sequence of statements issued without an enclosing `BEGIN` executes as separate single-statement transactions. Each statement sees the database state at its own execution time. A two-query "read then insert" pattern without `BEGIN` is not atomic and is not isolated — another writer can land between the two statements. Wrap any sequence that needs consistency in an explicit transaction.

**Using `BEGIN EXCLUSIVE` as a blanket "be safe" default.** Under WAL mode, `EXCLUSIVE` is functionally the same as `IMMEDIATE` — it does not provide additional isolation and does not block readers. Under rollback-journal mode, it serialises all readers for the duration of the transaction, unnecessarily preventing concurrent reads during a long write. `IMMEDIATE` is the correct default for write transactions in WAL mode. Reserve `EXCLUSIVE` for the specific cases where rollback-journal semantics and reader exclusion are both intentional.

**Spinning on `SQLITE_BUSY` without backoff.** A tight retry loop that hammers `BEGIN IMMEDIATE` as fast as the CPU allows under lock contention can hold a CPU core at 100% and produce livelock-like behaviour where the winning connection is whichever one happens to be scheduled at the right microsecond. Always sleep between retries. Exponential backoff with jitter distributes retry pressure and converges faster than constant-interval polling.

**A read-only transaction held open across application "thinking time".** In WAL mode, a `BEGIN DEFERRED` that is not committed pins a read snapshot for the duration of the transaction. WAL checkpointing — the process of writing WAL pages back into the main database file — cannot advance past pages that an open reader snapshot references. A transaction that opens at snapshot position X and stays open for minutes or hours prevents the WAL from checkpointing beyond X. The WAL grows without bound until the reader commits or rolls back. The fix: commit or rollback read transactions as soon as the reads are complete. Do not hold them open across HTTP response serialisation, user think-time, or any other unbounded wait.

**`SQLITE_BUSY` treated as a permanent error rather than a retry signal.** `SQLITE_BUSY` means "try again." Applications that catch it and return an HTTP 500 (or equivalent) without retrying are surfacing a recoverable condition as a failure. The correct response is retry with backoff; the configurable response is how many retries and at what intervals. Reserve the error escalation path for when retries are exhausted.

## Cross-references

- [`sqlite-fundamentals.md`](sqlite-fundamentals.md) — the connection-as-isolation-unit model: why two connections to the same file compete rather than collaborate, and how isolation is enforced through the lock ladder rather than a shared lock manager.
- [`pragma-discipline.md`](pragma-discipline.md) — `busy_timeout` as the backstop for short collisions; why it is not a substitute for `BEGIN IMMEDIATE`; the production PRAGMA block that must be applied on every connection open.
- [`concurrent-access-patterns.md`](concurrent-access-patterns.md) — the single-writer WAL model; reader/writer coexistence under WAL; checkpoint discipline and how open reader snapshots gate WAL compaction.
- [`optimistic-locking-and-leases.md`](optimistic-locking-and-leases.md) — the alternative for long-lived workflows: detect concurrent modification with a version column rather than holding a write lock across user think-time.
- [`boundary-and-when-to-leave.md`](boundary-and-when-to-leave.md) — when transaction contention under SQLite's single-writer model indicates the workload has grown beyond the embedded database envelope.
