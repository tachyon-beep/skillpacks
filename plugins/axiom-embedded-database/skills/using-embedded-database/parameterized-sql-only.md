---
name: parameterized-sql-only
description: Use when writing any query that incorporates user input, API parameters, or data from any source outside the application binary itself. Covers Python sqlite3 qmark and named style, Rust rusqlite equivalents, what cannot be parameterized (identifiers), executescript hazards, bulk operations via executemany, and CI enforcement via AST checking.
---

# Parameterized SQL Only

**There is exactly one rule: every value that enters SQL is a parameter, every SQL string is a constant. The rule has no exception, and every codebase that breaks it eventually has the bug you expect.**

## When this earns its cost

Read this sheet when:

- You are writing any query that incorporates user input — form fields, URL parameters, command-line arguments, file contents read at runtime.
- You are building an admin tool that takes string inputs and queries the database. Admin tools have historically been the first place injection appears, because "only we use it" is not a threat model.
- You are accepting string inputs from an internal API and passing them to the database. The trust boundary is between your code and external data, not between your code and end users. A compromised upstream service is an attacker.
- You are reviewing code and want a mechanical checklist that catches the common violations before they reach production.

The argument for exceptions ("this value is already validated", "this column name comes from our own enum") is always locally coherent and usually wrong in six months. The rule is categorical because conditional rules produce conditional adherence.

## The rule

A parameterized query is a SQL string with placeholders where values go. The driver handles the binding. The SQL string is fixed at parse time — it never contains data.

**Correct:**

```python
cursor.execute(
    "SELECT id, email FROM users WHERE name = ?",
    (user_input,),
)
```

**Injection hole:**

```python
cursor.execute(f"SELECT id, email FROM users WHERE name = '{user_input}'")
```

The second form is a hole even when `user_input` looks safe. The driver receives a fully-formed SQL string. It cannot distinguish the structure from the data. A value of `' OR '1'='1` restructures the query. A value of `'; DROP TABLE users; --` appends a second statement (in contexts where multi-statement is enabled).

The fix is not to sanitise `user_input` before interpolation. Sanitisation is an arms race with the character set of SQL syntax. Parameterization removes the need for it entirely by keeping values outside the SQL parser's scope. Codebases that break this rule do not break it out of ignorance — they break it because a developer, under time pressure, made a local decision that looked safe. The rule exists to make that local decision impossible, not merely discouraged.

## Positional vs named parameters

Python's `sqlite3` module implements [PEP 249](https://peps.python.org/pep-0249/) and supports two parameter styles.

**Qmark style** — a `?` placeholder for each value, bound by position:

```python
conn.execute(
    "INSERT INTO events (user_id, action, ts) VALUES (?, ?, ?)",
    (user_id, action, timestamp),
)
```

**Named style** — a `:name` placeholder bound by key from a mapping:

```python
conn.execute(
    """
    INSERT INTO events (user_id, action, ts, session_id, ip_addr)
         VALUES (:user_id, :action, :ts, :session_id, :ip_addr)
    """,
    {
        "user_id": user_id,
        "action": action,
        "ts": timestamp,
        "session_id": session_id,
        "ip_addr": ip_addr,
    },
)
```

**Guidance:** use qmark for queries with three or fewer values; switch to named style when the parameter count makes positional order fragile. Both styles prevent injection equally. Named style is not more "safe" — it is more readable when the parameter list is long enough that position is no longer self-documenting.

Python's DB-API also specifies a numeric style (`:1`, `:2`) but sqlite3 does not implement it. Do not use it; it will raise `ProgrammingError`.

**Rust (rusqlite)** supports the same two styles with slightly different syntax:

```rust
// Positional — ? placeholders, bound by index
conn.execute(
    "INSERT INTO events (user_id, action) VALUES (?1, ?2)",
    params![user_id, action],
)?;

// Named — :name placeholders, bound by name mapping
conn.execute(
    "UPDATE users SET email = :email WHERE id = :id",
    named_params! { ":email": email, ":id": user_id },
)?;
```

In rusqlite the positional syntax is `?1`, `?2`, ... (1-indexed). Plain `?` (without an index) is also accepted and binds in order. Named parameters carry the colon prefix in the binding map key. The same readability heuristic applies: named style for four or more parameters.

## What you cannot parameterize

SQL drivers bind values, not identifiers. Table names, column names, schema names, and `ORDER BY` direction are part of the SQL structure — the parser resolves them before binding occurs. A driver will not accept a table name as a parameter and interpolate it into the query. Attempting to use a parameter where an identifier is expected raises an error, not a silent wrong query.

The situation where this bites: a function that accepts a sort column name or table name from the caller and builds the query dynamically. The caller-supplied identifier cannot be a `?` parameter. It must be interpolated — which is exactly what the rule prohibits for values.

The discipline for identifiers: **validate against an in-application allowlist, then interpolate.** The interpolation is safe because the allowlist removes attacker influence before interpolation occurs.

```python
_ALLOWED_SORT_COLUMNS = frozenset({"created_at", "updated_at", "name", "score"})
_ALLOWED_DIRECTIONS = frozenset({"ASC", "DESC"})


def safe_order_by(column: str, direction: str) -> str:
    """Return a safe ORDER BY clause after allowlist validation.

    Raises ValueError if either argument is not in the allowlist.
    Never call this with an unsanitised format string — the whole
    point is that validation happens here, not at the call site.
    """
    col = column.strip().lower()
    allowed_col = next((c for c in _ALLOWED_SORT_COLUMNS if c.lower() == col), None)
    if allowed_col is None:
        raise ValueError(f"Sort column not allowed: {column!r}")

    direction = direction.strip().upper()
    if direction not in _ALLOWED_DIRECTIONS:
        raise ValueError(f"Sort direction not allowed: {direction!r}")

    # Safe: both components passed the allowlist; neither is user data at this point.
    return f"ORDER BY {allowed_col} {direction}"


# Usage:
order_clause = safe_order_by(request.args["sort"], request.args["dir"])
sql = f"SELECT id, name, score FROM items WHERE active = ? {order_clause}"
cursor.execute(sql, (True,))
```

This is the only place in the codebase where an f-string is legitimate inside a SQL context: after the allowlist has stripped the attacker's ability to influence the content.

The common mistake is to implement the allowlist as a `if column not in allowed: column = 'default'` guard but then format with the original `column` variable due to copy-paste error. Use the canonical-form variable the allowlist returns, not the input variable.

## executescript() and the multi-statement hazard

`executescript(sql)` executes a SQL string as-is, with no parameter binding. The method signature does not accept a second argument for parameters. There is no way to use `executescript()` with parameterized values.

```python
# WRONG — executescript does not bind parameters.
# This is not even a well-formed call; sqlite3 will ignore any
# second argument if you pass one, or raise TypeError.
conn.executescript(
    "INSERT INTO log (msg) VALUES (?)",
    # (user_message,)  ← silently not used
)
```

`executescript()` also issues an implicit `COMMIT` before execution, which ends any open transaction. It is designed for schema migration scripts — sequences of DDL and DML statements that you control end-to-end. Its scope should be:

- Migration scripts applied at startup or via a migration tool.
- Seeding a test database from a fixture file you own.

**Never** pass any value derived from user input, API responses, or file contents into `executescript()`. If your migration script contains a comment that you are constructing from a variable, stop. The variable belongs in a subsequent parameterized `execute()`, not in the script string.

## Bulk parameterization

When inserting or updating N rows, use `executemany()`. It takes the same SQL string as `execute()` and a sequence of parameter tuples — one tuple per row. The SQL is parsed once; binding and execution happen N times.

```python
# Before — parses SQL on every iteration
for row in records:
    conn.execute(
        "INSERT INTO readings (device_id, value, ts) VALUES (?, ?, ?)",
        (row["device_id"], row["value"], row["ts"]),
    )

# After — parses once, binds N times
conn.executemany(
    "INSERT INTO readings (device_id, value, ts) VALUES (?, ?, ?)",
    [(row["device_id"], row["value"], row["ts"]) for row in records],
)
```

For 10,000 rows the parse overhead alone can be several times the bind-and-execute cost. `executemany()` still fully parameterizes — each tuple is bound independently. When `records` is a list of dicts, named style avoids a list comprehension entirely:

```python
conn.executemany(
    "INSERT INTO readings (device_id, value, ts) VALUES (:device_id, :value, :ts)",
    records,
)
```

## Lint and CI enforcement

Policy is not enforcement. The only reliable way to keep injection out of a codebase is to detect violations mechanically before they merge.

**ruff:** Rule `S608` (flake8-bandit, ruff `S` prefix) flags suspicious interpolation in SQL calls. Enable with `select = ["S608"]` in `pyproject.toml`. For full coverage — f-strings, string concat, `%`-formatting — supplement with a project AST checker.

**Minimal AST checker for CI:**

```python
"""check_sql_injection.py — flag f-strings and string concat in execute() calls.

Usage: python check_sql_injection.py path/to/src/
Exit 1 if any violations found.
"""
import ast
import sys
from pathlib import Path


_EXECUTE_METHODS = {"execute", "executemany", "executescript"}


def _is_unsafe_sql_arg(node: ast.expr) -> bool:
    """Return True if the node looks like a dynamically-built SQL string."""
    # f-string
    if isinstance(node, ast.JoinedStr):
        return True
    # string concatenation: "SELECT … " + variable
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return True
    # %-formatting: "SELECT … %s" % (value,)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            return True
    return False


def check_file(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        method_name = None
        if isinstance(func, ast.Attribute):
            method_name = func.attr
        elif isinstance(func, ast.Name):
            method_name = func.id
        if method_name not in _EXECUTE_METHODS:
            continue
        if node.args and _is_unsafe_sql_arg(node.args[0]):
            violations.append((node.lineno, method_name))
    return violations


def main() -> int:
    roots = [Path(p) for p in sys.argv[1:]] or [Path(".")]
    found = 0
    for root in roots:
        for py_file in root.rglob("*.py"):
            for lineno, method in check_file(py_file):
                print(f"{py_file}:{lineno}: unsafe SQL string in {method}()")
                found += 1
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
```

Add `python tools/check_sql_injection.py src/` to CI. This catches f-strings, concat, and `%`-format as the first SQL argument. It does not catch dynamic SQL assembled several lines before the `execute()` call — no static checker reliably handles that. For multi-line construction, the allowlist-then-interpolate discipline is the enforcement mechanism.

## Worked example: a "safe search" function

Input: user query string, sort column, sort direction. Output: matching rows. The query string is a value — parameterized. The column and direction are identifiers — allowlisted, then interpolated.

```python
import sqlite3
from typing import Any

_SEARCH_COLUMNS = frozenset({"created_at", "updated_at", "title", "score"})
_SORT_DIRECTIONS = frozenset({"ASC", "DESC"})


def search_items(
    conn: sqlite3.Connection,
    query: str,
    sort_col: str,
    sort_dir: str,
) -> list[dict[str, Any]]:
    col = next((c for c in _SEARCH_COLUMNS if c.lower() == sort_col.strip().lower()), None)
    if col is None:
        raise ValueError(f"Unknown sort column: {sort_col!r}")
    direction = sort_dir.strip().upper()
    if direction not in _SORT_DIRECTIONS:
        raise ValueError(f"Unknown sort direction: {sort_dir!r}")
    sql = f"SELECT id, title, score FROM items WHERE title LIKE ? ORDER BY {col} {direction}"
    return [dict(r) for r in conn.execute(sql, (f"%{query}%",)).fetchall()]
```

**Rejected version 1 — string concat:** `"… LIKE '%" + query + "%' ORDER BY " + sort_col` — driver receives a fully-formed string; injection via `query`.

**Rejected version 2 — f-string:** `f"… LIKE '%{query}%' ORDER BY {sort_col}"` — same hole, different syntax; the checker above catches it.

**Rejected version 3 — %-format:** `"… LIKE '%%%s%%' ORDER BY %s" % (query, sort_col)` — %-formatting is still interpolation.

All three allow a value like `%' UNION SELECT password, null, null FROM admins --` to restructure the query. The driver cannot undo interpolation that happened before the string was passed.


## Anti-patterns

**f-string interpolation inside execute().** The most common violation. Syntactically clean, semantically dangerous. The checker above catches it.

```python
# WRONG
cursor.execute(f"SELECT * FROM {table} WHERE id = {user_id}")
```

**executescript() with any user-derived value.** `executescript()` does not bind parameters. Any user value embedded in its argument is raw SQL.

```python
# WRONG
conn.executescript(f"INSERT INTO audit (msg) VALUES ('{message}')")
```

**"Parameterizing" an identifier via string concatenation under the assumption the driver handles it.** The driver will not reject this at bind time because the value is already in the SQL string before binding occurs.

```python
# WRONG — the attacker controls table_name; ? cannot fix it after the fact.
cursor.execute("SELECT * FROM " + table_name + " WHERE id = ?", (row_id,))
```

The correct form: validate `table_name` against an allowlist first, then interpolate.

**Trusting an upstream layer to have already sanitised.** "The API gateway validates this" or "the form library escapes that" are not guarantees you can rely on. Sanitisation for display and parameterization for SQL are different operations. Even if upstream sanitisation is correct, it may strip characters that are valid SQL-injection vectors in ways that appear safe but are not. Parameterize unconditionally.

**Logging a re-assembled SQL string for observability.** sqlite3 does not expose the resolved SQL (bind values go to the C library, not into the Python string). If you reconstruct `sql + str(params)` for logging, you leak parameter values into log storage and potentially into aggregation pipelines. Log the SQL template and parameter tuple separately: `log.debug("sql=%r params=%r", sql, params)`.

**`IN (?)` with a list argument.** `IN (?)` with a single `?` placeholder does not expand a list — sqlite3 raises `ProgrammingError` because the placeholder expects a scalar. The correct approach generates one placeholder per item:

```python
def get_items_by_ids(conn: sqlite3.Connection, ids: list[int]) -> list[sqlite3.Row]:
    if not ids:
        return []
    placeholders = ", ".join("?" * len(ids))
    sql = f"SELECT id, title FROM items WHERE id IN ({placeholders})"
    # Safe: len(ids) is a structural count, not user data. The values are parameterized.
    return conn.execute(sql, ids).fetchall()
```

If `ids` came from user input, validate each element as an integer before binding.

## Cross-references

- `sqlite-fundamentals.md` — connection model and the execution context in which parameters are bound.
- `transactions-and-isolation.md` — parameterization works the same way inside `BEGIN IMMEDIATE` transactions; the binding happens at execute time, not commit time.
- `optimistic-locking-and-leases.md` — version values used in compare-and-swap predicates must be parameterized; the version column is the concurrency safety mechanism and must not be interpolated.
- `json1-and-structured-data.md` — JSON path expressions passed to `json_extract()` are values and must be parameterized; a path like `$.name` is data, not SQL structure.
- `boundary-and-when-to-leave.md` — when parameterization discipline and other constraints together indicate that a server database with a prepared-statement cache would be preferable.
