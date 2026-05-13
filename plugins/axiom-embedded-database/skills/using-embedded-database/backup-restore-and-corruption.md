---
name: backup-restore-and-corruption
description: Use when setting up a backup procedure for a SQLite database, when your current backup is a plain file copy, when integrity_check has surfaced corruption, or before running any migration that mutates data. Covers VACUUM INTO, the sqlite3_backup Online Backup API, checkpoint-and-copy, filesystem snapshots, integrity_check vs quick_check, corruption recovery via sqlite3-recover, and a complete Python backup script.
---

# Backup, Restore, and Corruption

**A backup that wasn't tested by restore is not a backup. A corruption that wasn't anticipated by the recovery plan is downtime.**

## When this earns its cost

Read this sheet when:

- You have no documented backup procedure and "hope it doesn't break" is your current strategy.
- Your backup procedure is `cp database.db backup.db` and you're not sure whether that races with an active writer.
- `PRAGMA integrity_check` returned errors and you need a recovery path.
- You are about to run a schema migration that mutates data and need a safe fallback point.
- You are building a production application that must demonstrate recovery capability under audit.

`pragma-discipline.md` covers WAL mode and `wal_autocheckpoint`, which are prerequisites for understanding why backup method selection matters. `schema-migrations.md` covers the "back up before migrating" pattern.

## Why cp is wrong (under WAL)

A plain file copy of a WAL-mode database is not a consistent backup. The reason is structural.

In WAL mode, SQLite maintains three files: `database.db` (the main file), `database.db-wal` (the write-ahead log), and `database.db-shm` (the shared-memory coordination file). Committed pages may live in the WAL, not yet checkpointed into the main file. A `cp database.db` that copies only the main file while the WAL contains committed data produces a backup that is missing those commits. The backup is internally consistent as of an earlier checkpoint — but it is not the current state.

The problem is worse when a write is in flight. `cp` reads the file in chunks. If a writer commits between two chunks, the backup contains a mix of pre-commit and post-commit pages from different transactions. SQLite will detect this as corruption on the next open because the page checksums won't match the state the header describes.

Even copying all three files (`db`, `db-wal`, `db-shm`) does not save you, because the copy of each file is taken at a different moment. The triplet is not atomic.

**Rule**: Never use `cp` as the backup mechanism for a WAL-mode database that is open for writes. Use one of the four methods below.

## The four backup methods

### VACUUM INTO 'backup.db'

`VACUUM INTO` (added in SQLite 3.27.0, 2019-02-07) creates a new database file that is a transactionally consistent, fully compacted copy of the current database as of the moment the statement runs. It reads inside an implicit read transaction, so it sees a stable snapshot even if writers are active. It writes directly to the destination path.

```sql
VACUUM INTO '/backups/database_20260513.db';
```

From Python:

```python
con.execute("VACUUM INTO '/backups/database_20260513.db'")
```

**Why it is the standard choice.** It is online (no exclusive lock), it handles WAL correctly, and the output file is defragmented — smaller than the source if the source has free pages. It works without any cooperation from the surrounding application code.

**Caveat.** `VACUUM INTO` rebuilds the database page by page. For a large database this takes longer than the Online Backup API because it performs work proportional to the *used* data, not just the *pages*. On a 10 GB database, expect seconds to minutes depending on storage speed. The source database remains fully usable throughout.

**Another caveat.** The destination path must not already exist. `VACUUM INTO` refuses to overwrite an existing file. Use a timestamp in the filename or write to a temp path and rename.

---

### sqlite3_backup / .backup (Online Backup API)

The `sqlite3_backup` C API copies a database page by page into a destination connection. It holds a read lock while copying each page and handles WAL correctly — it reads through the WAL to present a consistent snapshot. In Python, `sqlite3.Connection.backup()` wraps this API; it was added in Python 3.7.

```python
import sqlite3

src = sqlite3.connect("database.db")
dst = sqlite3.connect("backup.db")
with dst:
    src.backup(dst, pages=-1, progress=None)
dst.close()
src.close()
```

`pages=-1` copies all pages in a single pass. A positive integer copies that many pages per step, yielding between steps to allow writers to proceed — useful for very large databases where you want to reduce write latency impact.

**When to prefer it over VACUUM INTO.** Very large databases where you need fine-grained control over copy pace, or where you need to copy into an already-open in-memory database. For most production workloads, `VACUUM INTO` is simpler.

**Using the shell.** The `sqlite3` command-line tool's `.backup` command uses the same API:

```
sqlite3 database.db ".backup backup.db"
```

---

### Checkpoint-and-copy

Run `PRAGMA wal_checkpoint(TRUNCATE)` to flush all WAL frames into the main file and truncate the WAL to zero bytes, then immediately copy the main file.

```python
result = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
# result is (busy, log_frames, checkpointed_frames)
busy, log_frames, checkpointed = result
if busy != 0 or log_frames != checkpointed:
    raise RuntimeError(f"Checkpoint incomplete: {result}")
# Only now is a plain copy safe
import shutil
shutil.copy2("database.db", "backup.db")
```

**When this is acceptable.** Low-write workloads where you control all writers and can pause new writes during the checkpoint-and-copy window. If `busy != 0`, an active reader held the WAL open — the checkpoint is incomplete and the copy must not proceed.

**Why it is fragile.** Between the checkpoint and the copy, a new write can begin and append to the WAL again. On any workload with concurrent access, checkpoint-and-copy requires holding an external write lock for the duration. Without that lock, the procedure has a race window. Prefer `VACUUM INTO` or the Online Backup API.

---

### Filesystem snapshot

LVM, ZFS, and BTRFS offer block-level point-in-time snapshots. A snapshot is consistent if the file is durable on disk at the moment the snapshot is taken.

`PRAGMA wal_checkpoint(FULL)` flushes committed WAL frames to the main file without truncating the WAL. Running it immediately before triggering the snapshot ensures the main file contains all committed data.

```python
con.execute("PRAGMA wal_checkpoint(FULL)")
# Then trigger the filesystem snapshot via the OS/storage API
```

**Requirement.** The filesystem must guarantee that the snapshot is a consistent point-in-time view of the block device at the moment it is triggered — not a smear across time. Most LVM/ZFS/BTRFS implementations provide this. Verify your storage stack's snapshot semantics before relying on this method.

**`wal_checkpoint(FULL)` vs `TRUNCATE`.** Use `FULL` here (not `TRUNCATE`) because TRUNCATE rewrites the WAL header, which is a write — and you want the on-disk state stable for the snapshot. After the snapshot is complete, a separate `TRUNCATE` checkpoint can be run to reclaim WAL space.

## Restore

The restore procedure is:

1. Back up the current (potentially corrupt or outdated) database before touching it.
2. Copy the backup file to the production path (not in place — to a staging path first).
3. Open the backup at the staging path and run `PRAGMA integrity_check`.
4. Verify expected row counts on key tables.
5. If checks pass, atomically rename the staging file over the production path.
6. Restart or reconnect the application.

```python
import sqlite3, os, shutil

def restore(backup_path: str, prod_path: str) -> None:
    staging = prod_path + ".restoring"
    shutil.copy2(backup_path, staging)

    con = sqlite3.connect(staging)
    rows = con.execute("PRAGMA integrity_check").fetchall()
    if rows != [("ok",)]:
        con.close()
        os.unlink(staging)
        raise RuntimeError(f"Backup failed integrity check: {rows}")
    con.close()

    # Atomic swap — on POSIX, os.rename is atomic within the same filesystem
    os.rename(staging, prod_path)
```

The rename must be on the same filesystem as the production path for atomicity. On Windows, use `os.replace` which is defined to be atomic on the same volume.

**Never point the application at a backup file that has not been verified by `integrity_check`.** Restoring a corrupt backup extends downtime instead of ending it.

## integrity_check and quick_check

SQLite provides two pragmas for verifying database health. They are not interchangeable.

**`PRAGMA integrity_check`** walks every page in the database, verifies every b-tree structure, recounts every row in every index against the corresponding table, and checks all foreign key references. It returns either a single row containing `"ok"` on success, or one or more rows each containing an error description. It is slow on large databases — proportional to total data size.

```python
rows = con.execute("PRAGMA integrity_check").fetchall()
if rows != [("ok",)]:
    # Each row is a string describing one problem
    for row in rows:
        print(row[0])
```

**`PRAGMA quick_check`** skips index consistency and foreign key checks. It verifies the structural integrity of b-trees and page allocation without the expensive cross-table work. It is fast — proportional to number of pages, not number of rows.

```python
rows = con.execute("PRAGMA quick_check").fetchall()
assert rows == [("ok",)]
```

**Recommended schedule:**

| When | Check |
|------|-------|
| Every startup (fast gate) | `quick_check` |
| After backup creation | `integrity_check` |
| Weekly on production | `integrity_check` |
| After any recovery operation | `integrity_check` |

Run `integrity_check` on the backup copy, not on the live database, to avoid holding a read lock for the duration of a full check on a large production file.

## What corruption looks like

Corruption manifests at the SQLite API level as error codes. When you see any of these, **stop writing immediately** — continued writes can extend damage into previously clean pages.

| Error code | Meaning |
|------------|---------|
| `SQLITE_CORRUPT` (code 11) | Database file structure is invalid. Page checksums mismatch, b-tree pointers are inconsistent, or the file header is garbled. |
| `SQLITE_NOTADB` (code 26) | The file does not begin with the SQLite magic bytes (`53 51 4c 69 74 65 ...`). Either the wrong file, a SQLCipher-encrypted file opened without a key, or complete header destruction. |
| `SQLITE_FULL` (code 13) | Disk is full. Not structural corruption, but if the write that hit this error was mid-transaction, the rollback journal or WAL may be in a partial state. Verify after freeing space. |
| `SQLITE_IOERR_*` (code 10 + extended) | I/O error from the OS. The database may or may not be corrupt. Run `integrity_check` after the underlying I/O issue is resolved. |

A `SQLITE_CORRUPT` error on a routine read means the damage is already present in the file on disk — it did not happen in this process. The question is whether the damage is localised (one page, one table) or widespread (free-list corruption, root-page corruption). `integrity_check` characterises it.

## Recovery from corruption

Apply these steps in order. Each step is cheaper than the next and recovers a larger fraction of undamaged data.

**Step 1: Characterise with integrity_check.**

```bash
sqlite3 database.db "PRAGMA integrity_check"
```

Read the output. If errors are confined to specific tables or indexes, a partial dump may recover everything else. If root-page or freelist errors appear, expect wider damage.

**Step 2: Dump and re-import.**

```bash
sqlite3 corrupt.db .dump > dump.sql
sqlite3 recovered.db < dump.sql
```

`.dump` reads the database sequentially and emits SQL for every row it can reach. It silently skips pages it cannot read. The output SQL is valid even if some data is missing. Re-import creates a structurally clean database from whatever `.dump` recovered. Verify with `integrity_check` after re-import.

**Step 3: sqlite3-recover extension (since SQLite 3.42, 2023).**

The SQLite development team ships `sqlite3-recover` as an extension (also exposed as the `.recover` command in the `sqlite3` shell since 3.41.0). Unlike `.dump`, it performs a lower-level page walk that can recover rows from pages that `.dump` skips because their b-tree pointers are invalid.

```bash
# sqlite3 shell .recover command (since 3.41.0)
sqlite3 corrupt.db ".recover" | sqlite3 recovered.db
```

This is best-effort: it cannot reconstruct data from pages that are physically overwritten or zeroed. Run `integrity_check` on the recovered database before using it.

**Step 4: Restore from backup.**

If the above steps do not produce a database with the required data integrity, restore from the most recent verified backup using the procedure in the Restore section above.

Document the data loss window: what was the timestamp of the backup? What was the timestamp of the corruption event? The gap is the data loss.

## WAL discipline around backups

WAL state affects backup correctness. Two PRAGMAs matter.

**`PRAGMA wal_checkpoint(TRUNCATE)`** copies all committed WAL frames into the main database file and truncates the WAL to zero bytes. After a successful TRUNCATE checkpoint, the main database file contains the complete committed state and the WAL is empty.

```python
busy, log_frames, checkpointed = con.execute(
    "PRAGMA wal_checkpoint(TRUNCATE)"
).fetchone()
# busy > 0 means an active reader held the WAL open — checkpoint is incomplete
```

Check that `busy == 0` and `log_frames == checkpointed`. If either condition fails, do not rely on the main file being complete.

**`PRAGMA wal_autocheckpoint(N)`** instructs SQLite to automatically run a passive checkpoint when the WAL reaches N pages (default: 1000). Passive checkpoints do not truncate; they copy what they can without blocking readers. A database under heavy write load may accumulate a large WAL between checkpoints. For backup purposes, a manual TRUNCATE checkpoint before backup is more predictable than relying on autocheckpoint.

See `pragma-discipline.md` for full coverage of `wal_autocheckpoint` and `synchronous` settings that affect checkpoint durability.

## Worked example

Complete backup function in Python. Performs checkpoint, backup via Online Backup API, integrity verification, row-count check, and atomic rename into final location.

```python
import sqlite3
import os
import time
from pathlib import Path


def backup_database(
    source_path: str,
    backup_dir: str,
    key_table_counts: dict[str, int] | None = None,
) -> str:
    """
    Back up a WAL-mode SQLite database.

    Args:
        source_path:       Path to the production database file.
        backup_dir:        Directory to write the backup into.
        key_table_counts:  Optional {table: minimum_expected_rows} check.

    Returns:
        Final path of the verified backup file.

    Raises:
        RuntimeError if checkpoint is incomplete, integrity check fails,
        or row counts are below expectations.
    """
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stem = Path(source_path).stem
    final_path = os.path.join(backup_dir, f"{stem}_{timestamp}.db")
    staging_path = final_path + ".tmp"

    # Step 1: Checkpoint — flush WAL into main file.
    src = sqlite3.connect(source_path)
    try:
        busy, log_frames, checkpointed = src.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchone()
        if busy != 0 or log_frames != checkpointed:
            raise RuntimeError(
                f"WAL checkpoint incomplete before backup: "
                f"busy={busy}, log_frames={log_frames}, "
                f"checkpointed={checkpointed}"
            )

        # Step 2: Copy via Online Backup API (handles WAL correctly).
        dst = sqlite3.connect(staging_path)
        try:
            src.backup(dst, pages=-1)
        finally:
            dst.close()
    finally:
        src.close()

    # Step 3: Verify the backup — open read-only to avoid modifying it.
    verify = sqlite3.connect(f"file:{staging_path}?mode=ro", uri=True)
    try:
        rows = verify.execute("PRAGMA integrity_check").fetchall()
        if rows != [("ok",)]:
            raise RuntimeError(
                f"Backup integrity_check failed: {rows}"
            )

        if key_table_counts:
            for table, minimum in key_table_counts.items():
                (count,) = verify.execute(
                    f"SELECT COUNT(*) FROM {table}"  # noqa: S608 — table name from caller
                ).fetchone()
                if count < minimum:
                    raise RuntimeError(
                        f"Row count check failed: {table} has {count} rows, "
                        f"expected >= {minimum}"
                    )
    finally:
        verify.close()

    # Step 4: Atomic rename into final location.
    # os.rename is atomic on POSIX when src and dst are on the same filesystem.
    # On Windows, use os.replace.
    os.replace(staging_path, final_path)
    return final_path


# Usage
if __name__ == "__main__":
    path = backup_database(
        source_path="/var/db/production.db",
        backup_dir="/backups/sqlite",
        key_table_counts={"orders": 1, "customers": 1},
    )
    print(f"Backup written and verified: {path}")
```

Note: `os.replace` is used in the final rename step. It is equivalent to `os.rename` on POSIX (both are atomic within the same filesystem) and additionally works on Windows, where `os.rename` raises `FileExistsError` if the destination exists.

## Anti-patterns

**cp under WAL.** Copying the main file while the WAL contains committed pages produces a backup that is missing those commits. On an active write workload, it may produce a structurally corrupt backup. Use `VACUUM INTO` or the Online Backup API.

**Backup never tested by restore.** A backup file that has never been opened and verified is a hypothesis, not a guarantee. Run `integrity_check` on every backup at creation time. Periodically do a full restore-to-staging test to verify the end-to-end procedure.

**"I'll restore from yesterday" without integrity_check.** Yesterday's backup may also be corrupt — especially if the corruption was caused by a bad write path that has been running for days. Always run `integrity_check` on the backup before swapping it into production.

**Assuming SQLCipher backups are different.** A SQLCipher-encrypted database is still a SQLite database. `VACUUM INTO`, the Online Backup API, and `PRAGMA integrity_check` all work the same way. The backup file is also encrypted. The same WAL discipline applies. See `encryption-with-sqlcipher.md`.

**Using `.dump` as the primary backup format.** `.dump` is a recovery tool. It produces a text SQL dump that is portable and human-readable, but it is not a binary backup — it cannot be opened as a database directly, it skips unreadable pages silently, and restoring it requires a full re-import which loses any SQLite-internal metadata (application_id, user_version if not set via SQL). Use `VACUUM INTO` or the Online Backup API for routine backups; reserve `.dump` for corruption recovery.

**Backup destination on the same physical disk as the source.** A disk failure that corrupts or destroys the source also destroys the backup. Write backups to a separate physical device or to remote storage. Even a local second drive is better than the same device.

## Cross-references

- `pragma-discipline.md` — `wal_autocheckpoint` tuning, `synchronous` levels, and the PRAGMA block that governs durability.
- `schema-migrations.md` — the "back up before migrating" rule; `VACUUM INTO` is the correct pre-migration backup.
- `concurrent-access-patterns.md` — WAL reader/writer coexistence during backup; why a long-running reader can stall a TRUNCATE checkpoint.
- `encryption-with-sqlcipher.md` — encrypted backup procedure; `VACUUM INTO` and the Online Backup API work identically on SQLCipher databases.
- `boundary-and-when-to-leave.md` — when SQLite's limitations (backup complexity, no incremental backup, single-writer) are a signal to migrate to a client/server database.
