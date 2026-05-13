---
name: schema-migrations
description: Use when changing a SQLite schema — adding constraints, changing column types, dropping columns on pre-3.35 builds, or building a versioned migration runner. Covers the 12-step rebuild-table pattern, user_version and application_id discipline, migration-as-code tradeoffs, three required tests per migration version, and anti-patterns that silently corrupt or stall a migration.
---

# Schema Migrations

**SQLite's ALTER TABLE is a feature list, not a guarantee. The disciplined pattern is rebuild-table-into-new-shape, swap, and version-bump — even when ALTER would technically work, because the discipline is what survives the next constraint change.**

## When this earns its cost

Use this sheet when:

- You are adding a `NOT NULL` constraint, changing a column type, changing a CHECK, changing a foreign key, or removing a column on SQLite < 3.35.0.
- You are inheriting a schema with no version tracking and need to add a migration runner without breaking existing databases.
- You want migration discipline that does not degrade when the next developer on the project does not know SQLite's ALTER TABLE limits.
- You are auditing a migration for correctness: missing `foreign_keys=OFF`, missing `foreign_key_check`, or a transaction boundary that allows partial migration to persist.

`sqlite-fundamentals.md` covers the connection model; read that first if you have not.

## What ALTER TABLE actually supports

**Pre-3.25.0**: only `RENAME TABLE`. One operation. Everything else is unsupported.

**3.25.0+**: adds `ALTER TABLE t RENAME COLUMN old TO new`. Column rename is supported. Foreign key references in schema text are silently updated, but old column names embedded in trigger bodies are not — SQLite's rewriter operates on schema strings, and trigger bodies are opaque to it. Verify with `PRAGMA integrity_check` after renaming if any trigger references the column.

**3.35.0+**: adds `ALTER TABLE t DROP COLUMN c`. The column must not be a primary key column, a generated column, or a column named in an index, foreign key, CHECK constraint, or trigger. If any of those are true, the drop fails. You must use the rebuild-table pattern instead.

**Everything else** — change a column type, change a default, add or remove `NOT NULL`, change a CHECK constraint, change a foreign key, change a primary key, rename a column that is referenced in an index or trigger — requires rebuild-table. There is no ALTER TYPE, no ADD CONSTRAINT, no DROP CONSTRAINT.

**Even when ALTER works**: SQLite's `RENAME COLUMN` rewrites schema strings by text substitution, not by semantic analysis. If a view or trigger body contains the old column name as a quoted string or as part of a larger identifier, the rename may miss it. After any ALTER, run `PRAGMA integrity_check` and visually inspect affected views and trigger bodies in `sqlite_schema`.

## The 12-step rebuild-table pattern

This is the canonical recipe from sqlite.org/lang_altertable.html. Every step is load-bearing; omitting any one produces either data loss, a corrupt schema, or a FK integrity violation that survives the migration.

**Why steps 1 and 12 bracket the transaction, not the reverse**: `PRAGMA foreign_keys` is silently ignored when issued inside a transaction. It must be set on the connection *before* the transaction opens. Turning it back on after commit is the paired close.

Worked example: changing `orders.status` from `TEXT` with no constraint to `TEXT NOT NULL CHECK(status IN ('pending','shipped','closed'))`, and adding a `shipped_at` column:

```sql
-- Step 1. Disable FK enforcement for this connection.
--         Must be outside the transaction — ignored inside.
PRAGMA foreign_keys = OFF;

-- Step 2. Open the migration transaction.
BEGIN TRANSACTION;

-- Step 3. Save dependent schema objects.
--         Query sqlite_schema now, before we drop anything.
--         (sqlite_schema is the modern name for sqlite_master; available since 3.33.0.
--          On older builds, use sqlite_master.)
-- In application code: SELECT type, name, sql FROM sqlite_schema
--                       WHERE tbl_name = 'orders' AND type IN ('index','trigger','view');
-- Store results; replay in step 8.

-- Step 4. Create the new table with the target shape.
CREATE TABLE orders_new (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    status      TEXT    NOT NULL CHECK(status IN ('pending','shipped','closed')),
    created_at  TEXT    NOT NULL,
    shipped_at  TEXT              -- new nullable column
);

-- Step 5. Copy data. Explicit column lists on both sides.
--         Transform the status value: existing NULLs become 'pending'.
INSERT INTO orders_new (id, customer_id, status, created_at, shipped_at)
SELECT
    id,
    customer_id,
    COALESCE(status, 'pending'),
    created_at,
    NULL          -- shipped_at did not exist before
FROM orders;

-- Step 6. Drop the old table.
DROP TABLE orders;

-- Step 7. Rename new table to the canonical name.
ALTER TABLE orders_new RENAME TO orders;

-- Step 8. Recreate indexes, triggers, and views saved in step 3.
--         Replay each saved CREATE statement here.
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
-- (triggers and views: replay their CREATE statements verbatim, then
--  check step 9 if any reference the old shape)

-- Step 9. If any view, trigger, or CHECK body references the old column
--         name or old constraints, rewrite and recreate it now.
--         No changes needed in this example.

-- Step 10. Verify FK integrity.
PRAGMA foreign_key_check;
-- Returns rows for each FK violation. Treat any row as a migration failure:
-- ROLLBACK and investigate before proceeding.

-- Step 11. Commit.
COMMIT;

-- Step 12. Re-enable FK enforcement for this connection.
PRAGMA foreign_keys = ON;
```

In application code, step 10 must assert the result set is empty:

```python
violations = conn.execute("PRAGMA foreign_key_check").fetchall()
if violations:
    conn.rollback()
    raise RuntimeError(f"FK violations after migration: {violations}")
```

The 12 steps above show the full envelope as a standalone recipe. When the migration runs under a versioned runner (next section), the runner owns steps 1, 2, 11, and 12 — and adds `PRAGMA user_version = N` between steps 10 and 11 so the version bump is inside the same transaction as the schema change. Individual migration functions then carry only steps 3–10.

## Versioning the schema

Two PRAGMAs in the database file header identify the file:

- `PRAGMA application_id` — a 32-bit integer that identifies your application. Set once when the database is first created; never changes. Prevents opening a file from a different application through the same code path. Conventionally chosen as a magic number (e.g. `0x7A6C7462` for "zsql" — pick something unique to your project and document it).
- `PRAGMA user_version` — a 32-bit integer that identifies the schema generation. Starts at 0. Your migration runner reads it, applies any migrations between current and target, and writes the new version. This is your migration version counter.

The migration runner pattern:

```python
LATEST_VERSION = 5  # bump this when you add a migration

MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    1: migrate_v1,
    2: migrate_v2,
    3: migrate_v3,
    4: migrate_v4,
    5: migrate_v5,
}


def migrate(conn: sqlite3.Connection) -> None:
    """Apply all pending migrations up to LATEST_VERSION."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current == LATEST_VERSION:
        return
    if current > LATEST_VERSION:
        raise RuntimeError(
            f"Database at version {current} is newer than code at {LATEST_VERSION}"
        )
    for v in range(current + 1, LATEST_VERSION + 1):
        apply_migration(conn, v)


def apply_migration(conn: sqlite3.Connection, v: int) -> None:
    """Apply a single migration version and bump user_version atomically.

    The runner owns the transaction. migrate_fn does its DDL and data work but
    must NOT issue BEGIN, COMMIT, or ROLLBACK; the version bump and the schema
    change must land in the same transaction.

    Rebuild-table migrations require foreign_keys=OFF, which is silently ignored
    inside a transaction — so the runner must set it before BEGIN. migrate_fn
    can assume FK enforcement is already off and need not touch it.
    """
    migrate_fn = MIGRATIONS[v]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            migrate_fn(conn)
            # user_version cannot be set via parameter binding — SQLite does not
            # support binding values in PRAGMA statements. This f-string is safe
            # because v is an int produced by range() from LATEST_VERSION, never
            # from user input. PRAGMA user_version writes the 4-byte file header
            # and participates in transactional rollback, so it belongs in the
            # same BEGIN…COMMIT as the schema change.
            conn.execute(f"PRAGMA user_version = {v}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
    actual = conn.execute("PRAGMA user_version").fetchone()[0]
    if actual != v:
        raise RuntimeError(f"user_version write failed: expected {v}, got {actual}")
```

**Atomicity requirement.** The runner owns the transaction; the `user_version` bump and the schema change land in the same `BEGIN … COMMIT`. `PRAGMA user_version = N` writes the 4-byte file header and participates in transactional rollback — it is one of the PRAGMAs that can be safely issued inside a transaction. If the version bump were outside the transaction, a crash between `COMMIT` and the bump would leave a fully-migrated schema at version N-1, and the runner would re-run migration N on next open, double-applying it. There is no exception to the atomicity rule for this PRAGMA.

`PRAGMA foreign_keys` is different — it is silently ignored inside a transaction — so the runner sets it before `BEGIN` and restores it after `COMMIT`, in a `try/finally` that survives errors. Individual migration functions (`migrate_v1`, `migrate_v2`, …) therefore must not issue `BEGIN`, `COMMIT`, `ROLLBACK`, or `PRAGMA foreign_keys` themselves; the runner has already arranged the envelope.

**The runner must be idempotent.** Running `migrate()` on a database at LATEST_VERSION must be a no-op. The check `if current == LATEST_VERSION: return` provides this; do not rely on catching errors from re-applying migrations.

**Test coverage required.** Every migration version N requires three tests (see `## Testing migrations` below for the full pattern with code): a forward test from N-1 to N with realistic data, a fresh-schema test that builds at N directly, and a backwards-equivalence test that asserts the two produce identical schemas. All three exercise different code paths; all three must pass before a migration ships.

## Migration as code, not as SQL files

Three approaches exist; none is universally correct:

**SQL-file migrations** (one `.sql` file per version, loaded and executed by the runner): simple to write, easy to review in a diff, and trivially portable to other SQLite clients. The limitation: data transformations that need application logic — filling a column from an external source, hashing a value, computing a derived field — cannot be expressed in SQL alone. Either you add a second pass in application code (two steps per migration: SQL then code) or you use code-migrations throughout.

**Code-migrations** (a Python or Rust function per version): can call any application helper, validate data after transformation, and integrate with observability. The cost: harder to review at a glance, harder to replay manually, and tied to the language runtime.

**Middle ground**: code-migrations that build their SQL via parameterized statements with comments explaining intent. The code function is the envelope; the SQL inside it is reviewable. This is the pattern shown in the worked example above and in the 12-step template.

Pick based on whether any of your migrations require application-layer data logic. If they never do, SQL-file migrations are simpler. If even one does, code-migrations are easier than maintaining a hybrid runner.

## Testing migrations

Every migration version N requires three tests:

**(a) Forward test from N-1 to N with realistic data.** Insert representative rows at schema version N-1, run `apply_migration(conn, N)`, assert the data is correct at version N. This catches data-transformation bugs and missing COALESCE guards.

**(b) Fresh-schema test.** Build the database at version N directly (skip all prior migrations: create tables at the N shape, set `user_version = N`). Verify the schema matches what a migrated database produces. This catches drift between the migration path and the authoritative schema definition.

**(c) Backwards-equivalence.** The schema produced by migrating from 0 to N must be identical to the schema created fresh at N. Compare `SELECT sql FROM sqlite_schema ORDER BY name` on both. This catches cases where a migration adds a column in a different position or with a different collation than the authoritative schema.

Pytest example:

```python
import sqlite3
import pytest
from myapp.db import apply_migration, create_schema_at_version


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def test_migrate_v5_forward_with_data(db):
    # Build schema at v4, insert realistic data.
    create_schema_at_version(db, 4)
    db.execute("INSERT INTO orders (id, customer_id, status, created_at) VALUES (1, 10, NULL, '2024-01-01')")
    db.commit()

    apply_migration(db, 5)

    row = db.execute("SELECT status FROM orders WHERE id = 1").fetchone()
    assert row[0] == "pending"  # NULL coerced to default
    assert db.execute("PRAGMA user_version").fetchone()[0] == 5


def test_migrate_v5_fresh_schema(db):
    create_schema_at_version(db, 5)
    schema = db.execute(
        "SELECT sql FROM sqlite_schema WHERE type = 'table' ORDER BY name"
    ).fetchall()
    assert any("shipped_at" in row[0] for row in schema)


def test_migrate_v5_backwards_equivalence(db_migrated, db_fresh):
    migrated_schema = db_migrated.execute(
        "SELECT name, sql FROM sqlite_schema WHERE type = 'table' ORDER BY name"
    ).fetchall()
    fresh_schema = db_fresh.execute(
        "SELECT name, sql FROM sqlite_schema WHERE type = 'table' ORDER BY name"
    ).fetchall()
    assert migrated_schema == fresh_schema
```

The backwards-equivalence test (c) is the one most often skipped. It catches the failure mode where `create_schema_at_version` and the migration path have diverged — which is silent unless you compare them.

## Worked example: adding a NOT NULL column

**Target change.** Add a `NOT NULL TEXT` column `category` to the `products` table, backfilling existing rows with `'uncategorized'`.

**Option A: ALTER TABLE ADD COLUMN with a default** (works since very early SQLite):

```sql
ALTER TABLE products ADD COLUMN category TEXT NOT NULL DEFAULT 'uncategorized';
```

This works. The default is written into the schema text permanently — `sqlite_schema` will contain `DEFAULT 'uncategorized'` even after backfill. Future rows inserted without specifying `category` will silently get `'uncategorized'` rather than raising a NOT NULL error. Whether that is correct depends on your application. If the default was a migration-time fill value not a valid application value, this encodes a mistake into the schema. The 12-step pattern avoids this.

**Option B: rebuild-table** (cleaner schema, no permanent default). The runner has already set `foreign_keys = OFF` and opened the transaction; this function only does the schema and data work.

```python
def migrate_v3(conn: sqlite3.Connection) -> None:
    # Precondition (provided by apply_migration): foreign_keys = OFF, inside
    # an open BEGIN IMMEDIATE. This function must not BEGIN/COMMIT/ROLLBACK.
    conn.execute("""
        CREATE TABLE products_new (
            id       INTEGER PRIMARY KEY,
            name     TEXT NOT NULL,
            price    REAL NOT NULL,
            category TEXT NOT NULL   -- no DEFAULT: application must supply it
        )
    """)
    conn.execute("""
        INSERT INTO products_new (id, name, price, category)
        SELECT id, name, price, COALESCE(category, 'uncategorized')
        FROM products
    """)
    conn.execute("DROP TABLE products")
    conn.execute("ALTER TABLE products_new RENAME TO products")
    # Recreate any indexes that existed on products.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)")
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        # Raising propagates to apply_migration, which issues ROLLBACK.
        raise RuntimeError(f"FK violations: {violations}")
```

**Three tests for migration v3**:

```python
def test_migrate_v3_forward(db):
    create_schema_at_version(db, 2)
    db.execute("INSERT INTO products (id, name, price) VALUES (1, 'Widget', 9.99)")
    db.commit()
    apply_migration(db, 3)
    row = db.execute("SELECT category FROM products WHERE id = 1").fetchone()
    assert row[0] == "uncategorized"
    assert db.execute("PRAGMA user_version").fetchone()[0] == 3


def test_migrate_v3_fresh(db):
    create_schema_at_version(db, 3)
    # Inserting without category must fail: no DEFAULT in the clean schema.
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO products (id, name, price) VALUES (2, 'Gadget', 4.99)")


def test_migrate_v3_equivalence(db):
    migrated = sqlite3.connect(":memory:")
    create_schema_at_version(migrated, 2)
    apply_migration(migrated, 3)

    fresh = sqlite3.connect(":memory:")
    create_schema_at_version(fresh, 3)

    m = migrated.execute("SELECT name, sql FROM sqlite_schema WHERE type='table' ORDER BY name").fetchall()
    f = fresh.execute("SELECT name, sql FROM sqlite_schema WHERE type='table' ORDER BY name").fetchall()
    assert m == f
```

The `test_migrate_v3_fresh` test is the one that catches Option A's footgun: if you used `ALTER TABLE ADD COLUMN ... DEFAULT 'uncategorized'`, inserting without `category` would succeed (the default fires), whereas the fresh-build schema has no default and correctly rejects it.

## Anti-patterns

**`ALTER TABLE ... ADD COLUMN ... NOT NULL DEFAULT ...` as a migration shortcut.** Works. Encodes the migration-time fill value as a permanent schema default. If `'uncategorized'` is not a valid future value, the schema will silently accept it forever. Use it only when the default is correct long-term application behaviour, not when it is a one-time backfill.

**Migrations not wrapped in transactions.** A multi-step migration (CREATE new table, INSERT, DROP old) that is not inside `BEGIN ... COMMIT` leaves the database in a half-migrated state on any failure. On next open, the runner sees `user_version` unchanged and re-runs from the same step — which may now fail differently because the old table is gone. Wrap every migration in a single transaction.

**`user_version` bump outside the migration transaction.** If `PRAGMA user_version = N` is executed after the migration's `COMMIT`, a crash between the two leaves a fully-migrated schema at version N-1. The runner re-runs migration N, which fails because the new table already exists (or succeeds silently and double-transforms the data). The version bump must be inside the same `BEGIN … COMMIT` as the schema change. SQLite supports `PRAGMA user_version = N` inside a transaction — it writes the 4-byte file header and participates in rollback. There is no exception to the atomicity rule for this PRAGMA. (The PRAGMAs with transaction restrictions are `foreign_keys`, which is silently ignored inside a transaction, and `journal_mode`, which cannot be changed inside one. `user_version` has neither restriction.)

**Running `PRAGMA foreign_keys = ON` during a rebuild-table.** `PRAGMA foreign_keys` is silently ignored when issued inside a transaction. The ordering in the 12-step pattern is not style — it is load-bearing. `PRAGMA foreign_keys = OFF` must precede `BEGIN`; `PRAGMA foreign_keys = ON` must follow `COMMIT`. If you issue `foreign_keys = OFF` inside the transaction, FK enforcement may still be on for the duration of the migration, causing the DROP TABLE or INSERT to fail on FK violations that the rebuild is in the process of fixing.

**Mutating data in a migration without a written-down rollback story.** A migration that backfills a column by computing values from application logic (calling an external API, hashing passwords, computing derived fields) is irreversible. If the migration is re-run after a partial commit, it may hash already-hashed values or overwrite valid data. Every data-mutating migration should document: what the precondition is, what transformation it applies, and what to do if it needs to be reversed. "ROLLBACK the migration transaction" is the answer only if the process crashes mid-migration; a migration that committed and then was found to be incorrect needs a forward-only compensating migration, not a rollback.

**Saving and replaying saved indexes/triggers without checking for staleness.** Step 3 (save) and step 8 (replay) assume the saved CREATE statements are valid for the new schema. If your migration renames a column, the saved CREATE INDEX statement still names the old column. Step 9 exists precisely for this: after replaying, re-check every saved object and rewrite any that reference the changed shape. Skipping step 9 produces an index pointing at a non-existent column — SQLite will create it silently and it will produce wrong results.

## Cross-references

- `pragma-discipline.md` — `application_id`, `user_version`, `foreign_keys`, and `foreign_key_check` in full PRAGMA context; the production connection setup block that sets FK enforcement on every open.
- `transactions-and-isolation.md` — migration transactions must use `BEGIN` (not `BEGIN DEFERRED`); `BEGIN IMMEDIATE` is appropriate when the migration runner is the only expected writer but you want to fail fast on lock contention rather than waiting.
- `backup-restore-and-corruption.md` — take a database backup before running migrations in production; the Online Backup API is safe with an active WAL, file-copy is not.
- `boundary-and-when-to-leave.md` — if your migration runway requires more than a few schema versions per quarter, or if schema evolution is the main driver of complexity, review whether SQLite is still the right fit.
