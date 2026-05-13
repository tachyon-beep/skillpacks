---
name: json1-and-structured-data
description: Use when storing or querying semi-structured data in SQLite — deciding between JSON columns and proper columns, indexing JSON paths via indexed expressions or generated columns, migrating JSON shape changes, and enforcing schema at the application boundary. Covers the full JSON1 function surface, the generated-columns pattern introduced in SQLite 3.31, and anti-patterns that make JSON columns silently slow or silently schemaless.
---

# JSON1 and Structured Data

**JSON1 turns SQLite into a partially-schemaless store, but the discipline is the same as always: declare what's queried, index what's queried, and never let a JSON path become a load-bearing schema by accident.**

## When this earns its cost

Read this sheet when:

- You are adding a column "just for now" to a JSON blob because you don't want to run a migration — and that column is starting to appear in WHERE clauses.
- A slow query turns out to be a full-table scan because the filter is `json_extract(metadata, '$.email') = ?` with no index behind it.
- A JSON column has become the place all new columns go. Six months later no one knows what fields exist, there is no migration history, and one path is load-bearing for a reporting query no one indexed.
- You need to store genuinely sparse or evolving data — a bag of user preferences, a plugin configuration, an event payload — and a JSON column is the right fit if you apply the right discipline around it.

`sqlite-fundamentals.md` covers the connection model; `schema-migrations.md` covers column-level schema changes. Read those first if you have not.

## When to choose JSON vs columns

The primary question is: will this field appear in a WHERE, ORDER BY, or GROUP BY?

**Separate column** when the field is:
- Filtered or sorted frequently — queries that hit it on every page load.
- Part of a foreign key, index, or unique constraint.
- Never NULL in practice — a sparse JSON field used as a required field is a design smell, not a feature.
- Stable in shape — it will not gain sub-keys or change type across rows.

**JSON column** when the field is:
- Sparse: only some rows have it; a column that is NULL 90% of the time is a reasonable candidate.
- Evolving: the set of sub-keys is not known at schema design time; a plugin system where each plugin adds its own keys.
- Accessed as a blob most of the time: the application reads the whole JSON object and deserializes it; individual paths are queried rarely or never.
- Read-once after a write, never queried: event log payloads, webhook bodies, audit records you will parse in application code but never filter by in SQL.

**The 80/20 hybrid** is usually correct for real data: hoist the fields that are queried into first-class columns, keep the rest in a JSON column. A `users` table might have `email TEXT NOT NULL` (indexed, queried on login), `created_at TEXT NOT NULL` (ordered by), and `preferences TEXT` (a JSON blob of display settings never used in SQL). The hybrid captures the benefits of both: typed, indexed hot fields and schemaless storage for everything else.

The anti-pattern is the reverse: putting everything in JSON and then indexing it piecemeal as queries appear. Each indexed expression is a band-aid that should have been a column. After three or four you should evaluate whether the schema should have been fully relational from the start.

## JSON1 essentials

JSON1 is compiled into SQLite by default. Prior to SQLite 3.38.0 (2022-02-22), JSON1 was a compile-time extension controlled by `SQLITE_ENABLE_JSON1`; it was enabled in most official distributions but not guaranteed in every build. From 3.38.0 onward, JSON1 is unconditionally part of the core. Verify availability with `SELECT json('{"ok":1}')` — on 3.38.0+ this always works.

JSON path syntax: `$.field` for a top-level key, `$.field.sub` for nested, `$.array[0]` for array element by index. Paths are case-sensitive.

### `json_extract(json, path [, path ...])`

Returns the value at the given path, or NULL if the path does not exist.

```sql
-- Single path
SELECT json_extract(metadata, '$.email') FROM users;

-- Multiple paths in one call — returns a JSON array of results
SELECT json_extract(metadata, '$.first_name', '$.last_name') FROM users;

-- Nested path
SELECT json_extract(config, '$.notifications.email') FROM user_prefs;

-- Array element
SELECT json_extract(payload, '$.tags[0]') FROM events;
```

`json_extract` returns the SQLite-typed value (TEXT, INTEGER, REAL) when the path resolves to a scalar, and a JSON text string when it resolves to an object or array. It returns NULL both when the path does not exist and when the value is JSON null — these are indistinguishable via `json_extract` alone. Use `json_type(json, path)` if you need to tell a missing path from an explicit null.

### `json_set(json, path, value [, path, value ...])`

Returns a new JSON string with the value at `path` set to `value`. Inserts the key if absent; replaces it if present. The original column is not mutated — you must UPDATE the row.

```sql
UPDATE users
SET metadata = json_set(metadata, '$.last_login', datetime('now'))
WHERE id = ?;

-- Set multiple paths in one call
UPDATE users
SET metadata = json_set(
    metadata,
    '$.last_login', datetime('now'),
    '$.login_count', COALESCE(json_extract(metadata, '$.login_count'), 0) + 1
)
WHERE id = ?;
```

### `json_insert`, `json_replace`, `json_remove`

`json_insert` inserts only — if the path exists, the existing value is unchanged. `json_replace` replaces only — if the path does not exist, no change is made. `json_remove` removes the named path. All three return a new JSON string; the column is not mutated until you UPDATE.

```sql
-- Insert only if $.onboarded_at is absent
UPDATE user_prefs
SET data = json_insert(data, '$.onboarded_at', datetime('now'))
WHERE id = ? AND json_extract(data, '$.onboarded_at') IS NULL;

-- Replace only (path must already exist)
UPDATE plugins
SET config = json_replace(config, '$.version', ?) WHERE plugin_id = ?;

-- Remove a deprecated field from all rows
UPDATE users SET metadata = json_remove(metadata, '$.legacy_token');
```

### `json_each(json [, path])`

A table-valued function that expands a JSON array or object into rows. Each row has columns `key`, `value`, `type`, `atom`, `id`, `parent`, `fullkey`, `path`. Used in a `FROM` clause or subquery.

```sql
-- Expand a JSON array column into one row per tag
SELECT u.id, t.value AS tag
FROM users u, json_each(u.metadata, '$.tags') t
WHERE t.value = 'admin';

-- Count elements in a JSON array
SELECT id, json_array_length(metadata, '$.tags') AS tag_count
FROM users;
```

`json_each` is shallow — it walks only the direct children of the path. For deep traversal, use `json_tree`.

### `json_tree(json [, path])`

Like `json_each` but recursive: walks the entire JSON tree from the root (or from `path` if provided) and returns a row for every node at every depth.

```sql
-- Find all paths in a document that contain the string 'error'
SELECT fullkey, value
FROM json_tree(?)
WHERE type = 'text' AND value LIKE '%error%';
```

`json_tree` is expensive on large documents. Do not use it in queries over large tables without a very narrow outer filter.

### `json_array_length(json [, path])`

Returns the length of a JSON array at the given path, or NULL if the path is absent or not an array.

```sql
SELECT id
FROM events
WHERE json_array_length(payload, '$.recipients') > 10;
```

## Indexed expressions on JSON paths

An index can be created on the result of a function applied to a column, including `json_extract`. This is the indexed expression pattern.

```sql
CREATE INDEX idx_users_email
    ON users(json_extract(metadata, '$.email'));
```

**The index is used if and only if the query contains the exact same expression** — same function, same column, same path string spelling. SQLite's query planner matches the index by structural identity of the expression, not by semantic equivalence. SQL function names are case-insensitive (`JSON_EXTRACT` and `json_extract` match), but the path string is a literal — case and exact spelling matter.

```sql
-- Uses the index: expression matches exactly.
SELECT id FROM users WHERE json_extract(metadata, '$.email') = ?;

-- Does NOT use the index: path string differs by case.
-- '$.email' (in the index) and '$.Email' (in the query) are different literals;
-- the planner does not normalise them.
SELECT id FROM users WHERE json_extract(metadata, '$.Email') = ?;
```

**Before: full-table scan.**

```
sqlite> EXPLAIN QUERY PLAN
   ...> SELECT id FROM users WHERE json_extract(metadata, '$.email') = 'x@example.com';
QUERY PLAN
`--SCAN users
```

**After: index seek.**

```
sqlite> EXPLAIN QUERY PLAN
   ...> SELECT id FROM users WHERE json_extract(metadata, '$.email') = 'x@example.com';
QUERY PLAN
`--SEARCH users USING INDEX idx_users_email (<expr>=?)
```

**Caveats:**

- The expression must be deterministic. `json_extract` is deterministic on a fixed column and path string; any expression involving `random()`, `datetime('now')`, or non-deterministic user functions cannot be indexed.
- SQLite does not automatically update an indexed expression definition when the underlying data changes shape. If you rename the JSON key from `$.email` to `$.email_address`, the index still exists on the old path and is silently no longer used. You must drop and recreate it.
- If the indexed expression evaluates to NULL (path absent), the NULL rows are stored in the index and found by `IS NULL` filters, but excluded from `= ?` filters (NULL is not equal to anything). This is correct SQLite behaviour but can be surprising if many rows have a missing path.

## Generated columns: the better pattern

Generated columns, introduced in SQLite 3.31.0 (2020-01-22), make the hoisted projection a first-class column with a name. You can then index it like any ordinary column and reference it by name in queries.

```sql
CREATE TABLE users (
    id       INTEGER PRIMARY KEY,
    metadata TEXT NOT NULL,          -- JSON blob
    -- STORED: value is computed at write time and saved on disk.
    -- Takes space; is immediately available; survives read-only opens.
    email    TEXT GENERATED ALWAYS AS (json_extract(metadata, '$.email')) STORED,
    -- VIRTUAL: value is computed at read time, not stored on disk.
    -- No extra disk space; recomputes on every read.
    language TEXT GENERATED ALWAYS AS (json_extract(metadata, '$.language')) VIRTUAL
);

-- Index the generated column normally — no json_extract in the CREATE INDEX.
CREATE INDEX idx_users_email    ON users(email);
CREATE INDEX idx_users_language ON users(language);
```

**STORED vs VIRTUAL tradeoff:**

| | STORED | VIRTUAL |
|---|---|---|
| Disk space | Column value stored on every row | No extra disk space |
| Write cost | json_extract runs on every INSERT/UPDATE | No write cost |
| Read cost | Ordinary column read | json_extract runs on every SELECT that touches the column |
| Index required | Index a normal column | Index a normal column |
| Survives read-only open | Yes | Yes |

STORED is correct for columns that are read frequently and written infrequently. VIRTUAL is correct for columns that are computed occasionally or where disk space is the constraint.

**Generated columns vs indexed expressions:** Generated columns are cleaner because the hoisted field has a name, is visible in `SELECT *`, and can be referenced in CHECK constraints and triggers. Indexed expressions are useful when you cannot alter the schema (adding a generated column requires a schema change) or when you need the index on a legacy table without a migration window.

**Queries use generated columns automatically.** A query `WHERE email = ?` will use `idx_users_email` without needing to write `json_extract(metadata, '$.email')` in the query. This is the main usability advantage over indexed expressions.

## Schema-at-the-boundary

A JSON column is schemaless in the database. It is not schemaless in the application. **Every read site must validate.** The database read returns a string; the application deserializes it; if the deserialized shape does not match the declared model, the error surfaces immediately and loudly rather than silently propagating bad data.

**Python — Pydantic:**

```python
from __future__ import annotations
import sqlite3
from pydantic import BaseModel, ValidationError


class NotificationPrefs(BaseModel):
    email: bool = True
    push: bool = False


class UserPrefs(BaseModel):
    theme: str = "light"
    language: str = "en"
    notifications: NotificationPrefs = NotificationPrefs()


def load_user_prefs(conn: sqlite3.Connection, user_id: int) -> UserPrefs:
    row = conn.execute(
        "SELECT prefs FROM user_prefs WHERE user_id = ?", (user_id,)
    ).fetchone()
    if row is None:
        return UserPrefs()   # default prefs for new users
    try:
        return UserPrefs.model_validate_json(row[0])
    except ValidationError as exc:
        # Fail loudly: corrupt prefs are a data integrity issue,
        # not a case that should silently fall back to defaults.
        raise RuntimeError(
            f"Corrupt prefs for user {user_id}: {exc}"
        ) from exc
```

**Rust:** same pattern with `serde::Deserialize` and `#[serde(default)]` on each field. Deserialization failure is surfaced as an error at the boundary, never silently defaulted. `serde_json::from_str` into the struct; handle `Err` explicitly.

The schema is in the Pydantic model or the Rust struct. The database enforces nothing beyond "this column contains a non-null string". The application boundary is the only enforcement point, which makes it mandatory.

## Migration of JSON-shape changes

JSON-shape changes are different from column migrations: there is no ALTER TABLE equivalent. The change is a data transformation: UPDATE rows to add, remove, or rename a key.

**Adding a field.** No migration required at write time. The new code writes the new key; old code ignores it; readers handle absence by defaulting. Add a `default` to the Pydantic model or `#[serde(default)]` on the Rust field before shipping the writer. No SQL migration needed until you decide to backfill — backfilling is only required if you need the field indexed.

**Removing a field.** Two-step:
1. Stop writing the field in application code. Readers must tolerate the key being present (they will, because they ignore unknown keys with `model_validate` / `serde`).
2. After all application instances have been deployed with step 1, run a migration to strip the field from existing rows:

```sql
UPDATE user_prefs
SET prefs = json_remove(prefs, '$.legacy_token')
WHERE json_extract(prefs, '$.legacy_token') IS NOT NULL;
```

Run this in batches on large tables to avoid locking.

**Renaming a field.** There is no `json_rename`. A rename is always: add new field with the value of the old field, then remove the old field. In a single UPDATE:

```sql
-- Rename $.lang to $.language
UPDATE user_prefs
SET prefs = json_remove(
    json_set(prefs, '$.language', json_extract(prefs, '$.lang')),
    '$.lang'
)
WHERE json_extract(prefs, '$.lang') IS NOT NULL;
```

The two-step (stop writing old / then migrate) still applies: if any writer still emits `$.lang`, the migration race-conditions it back in. Coordinate the write-side change and the migration.

**Changing a field type.** Treat as remove-and-add. Cast or transform the value in the `json_set` call.

## Worked example

A `user_prefs` table with theme, language, and email notification preferences. `language` is queried frequently enough to warrant an index; the rest is accessed as a blob.

```sql
CREATE TABLE user_prefs (
    user_id  INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    prefs    TEXT NOT NULL DEFAULT '{}',
    -- Generated column: hoisted projection, indexed normally.
    language TEXT GENERATED ALWAYS AS (json_extract(prefs, '$.language')) VIRTUAL
);

CREATE INDEX idx_user_prefs_language ON user_prefs(language);
```

The Pydantic model at the read boundary:

```python
from pydantic import BaseModel


class NotificationPrefs(BaseModel):
    email: bool = True


class UserPrefs(BaseModel):
    theme: str = "light"
    language: str = "en"
    notifications: NotificationPrefs = NotificationPrefs()
```

Querying by language — uses the generated column index:

```sql
SELECT u.id, u.email
FROM users u
JOIN user_prefs up ON up.user_id = u.id
WHERE up.language = ?;
```

EXPLAIN QUERY PLAN shows `SEARCH user_prefs USING INDEX idx_user_prefs_language (language=?)`. The query planner sees `language` as a normal column; the generated column definition handles the extraction transparently.

## Anti-patterns

**Querying a JSON path without an indexed expression or generated column.** `WHERE json_extract(metadata, '$.email') = ?` over a million-row table is a full scan every time. Run EXPLAIN QUERY PLAN before shipping any query that filters on a JSON path. If it shows SCAN instead of SEARCH, add the index.

**Using JSON columns to avoid migrations.** The reasoning is that adding a key to a JSON blob requires no schema change. This is true. The hidden cost: the key has no type, no NOT NULL constraint, no default enforced by the database, and no history in the migration log. When the key becomes load-bearing (appears in a WHERE clause), you discover the debt. JSON columns defer the migration cost; they do not eliminate it. The eventual migration is harder because the field was never declared.

**SELECT * the JSON column when you only need one field.** `SELECT metadata FROM users WHERE id = ?` reads and transfers the entire JSON blob to deserialize one field. Use `SELECT json_extract(metadata, '$.email') FROM users WHERE id = ?` to extract in the database; transfer one scalar.

**Reading JSON without a schema validator.** Accessing `data["notifications"]["email"]` without validating the shape first will raise a KeyError or AttributeError when the path is absent, and return a wrong type silently when the stored type does not match the expected type. Validate at every read site. There are no exceptions.

**Indexed expression on a non-deterministic expression.** SQLite will refuse to create an index on a non-deterministic expression and raise an error — but the error message is not always obvious. The common mistake is attempting to index an expression involving `json_each` (which is a table-valued function, not a scalar, and cannot be indexed at all) or an expression that calls a user-defined function without the `SQLITE_DETERMINISTIC` flag. If you need to index a value from inside a JSON array, extract it to a generated column first.

**Treating a JSON column as the application's schema history.** If `$.v1_field` and `$.v2_field` and `$.legacy_migration_flag` are all present in the same blob, the JSON column has become an archaeology site. Use the two-step removal process in `## Migration of JSON-shape changes` to clean up old fields. A JSON column should contain the current shape of the data, not the complete history of every field that ever existed.

## Cross-references

- `schema-migrations.md` — JSON-shape evolution (add/remove/rename fields) follows the same two-step pattern as column migrations; the SQL mechanism differs but the coordination requirement is identical.
- `parameterized-sql-only.md` — JSON path strings passed to `json_extract` are values and must be parameterized when they come from application variables; a path like `$.` + user_input is an injection vector.
- `pragma-discipline.md` — the production connection setup block; WAL mode and busy_timeout are prerequisites before any performance work on JSON columns matters.
- `fts5-full-text-search.md` — FTS5 cannot index JSON columns directly; to full-text-search a JSON field, extract the text to a generated STORED column or a separate FTS5 content table, then index that.
- `boundary-and-when-to-leave.md` — if a large fraction of your queries filter on JSON paths and generated columns are proliferating, review whether a document database or a Postgres JSONB column would be a better fit than SQLite JSON1.
