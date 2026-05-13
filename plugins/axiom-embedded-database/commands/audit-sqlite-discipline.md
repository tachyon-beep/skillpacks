---
description: Sweep a project's SQLite usage for discipline violations against the `axiom-embedded-database` 13-sheet specification — PRAGMA settings absent or wrong, parameterized-SQL violations (f-strings inside execute), transaction discipline (DEFERRED-then-write, missing busy_timeout retry), locking gaps (no portable cross-process lock for app-level coordination), schema-versioning absent (`application_id` / `user_version` not set), backup procedure undocumented or untested. Produces a structured findings list with severity and the sheet that closes each gap. Optionally dispatches the `embedded-database-reviewer` agent for narrative synthesis.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[project_path]"
---

# Audit SQLite Discipline Command

You are sweeping a project's SQLite usage to find places where embedded-database discipline diverges from the `axiom-embedded-database` 13-sheet specification. The output is a structured findings list with severity, source location, the sheet that closes each gap, and a one-sentence remediation. This is a *finding-producing* command — it does not edit code, does not redesign schema, and does not run real workloads. For runtime profiling see `/profile-sqlite-workload`.

## Invocation Path

`/audit-sqlite-discipline` is a Claude Code slash command. It takes an optional project path argument; if omitted, it targets the current working directory. The sweep is pure static analysis — no database connection required.

Typical use cases: pre-merge discipline check on a new SQLite-backed service, brownfield gap analysis before upgrading a production database, or onboarding sweep when inheriting a codebase.

## Sweep dimensions

For each dimension: grep / glob pattern, heuristic, severity, and cross-reference sheet.

1. **PRAGMA block — presence and completeness**

   Pattern:
   ```bash
   grep -rn "PRAGMA" "${PROJECT_PATH}" --include="*.py" --include="*.rs"
   ```
   Heuristic: Confirm `journal_mode`, `synchronous`, `busy_timeout`, and `foreign_keys` are all set in the same connection-setup function. Flag any that are absent. Flag `synchronous = OFF` (data-loss risk) or `journal_mode = DELETE` (no WAL).

   Severity: **high** (absent `foreign_keys` or `journal_mode`), **med** (`synchronous = OFF`), **low** (missing `cache_size` or `temp_store`).

   Sheet: `pragma-discipline.md`

2. **String-format / f-string interpolation inside `execute()` calls**

   Pattern:
   ```bash
   grep -rn "execute\s*(f['\"]" "${PROJECT_PATH}" --include="*.py"
   grep -rn 'execute\s*(%\s*(' "${PROJECT_PATH}" --include="*.py"
   grep -rn "\.execute\b.*format(" "${PROJECT_PATH}" --include="*.py"
   grep -rn 'execute!.*format!' "${PROJECT_PATH}" --include="*.rs"
   ```
   Heuristic: Any non-constant first argument to `cursor.execute()`, `conn.execute()`, or `execute!()` is a potential SQL injection. String concatenation (`+`), f-strings, `.format()`, and `%`-interpolation are all violations.

   Severity: **high** (user-controlled input reaches the format expression), **med** (internal values formatted in, e.g. table or column names), **low** (integer literals formatted into non-parameterisable positions such as `PRAGMA user_version = {v}` with `v` from `range()`).

   Sheet: `parameterized-sql-only.md`

3. **`executescript()` with a non-constant first argument**

   Pattern:
   ```bash
   grep -rn "executescript(" "${PROJECT_PATH}" --include="*.py"
   ```
   Heuristic: `executescript()` disables parameterisation entirely. Flag every call where the script string is not a compile-time literal. Note that `executescript()` also issues an implicit `COMMIT` before running, breaking explicit transaction control.

   Severity: **high** (non-constant argument), **med** (constant argument — document the intentional use).

   Sheet: `parameterized-sql-only.md`

4. **Transaction flavour discipline — DEFERRED-then-write and missing `busy_timeout` retry**

   Pattern:
   ```bash
   grep -rn "BEGIN\b\|BEGIN DEFERRED\|BEGIN IMMEDIATE\|BEGIN EXCLUSIVE" "${PROJECT_PATH}" --include="*.py" --include="*.rs"
   grep -rn "INSERT\|UPDATE\|DELETE" "${PROJECT_PATH}" --include="*.py" --include="*.rs"
   ```
   Heuristic: A `BEGIN` without an explicit flavour defaults to `DEFERRED`. Trace each DEFERRED-flavoured transaction block for `INSERT`, `UPDATE`, or `DELETE` in the same scope. If writes appear without an upgrading `BEGIN IMMEDIATE`, the transaction promotes lazily under contention and produces `SQLITE_BUSY` errors that cannot be retried cleanly. Also flag any write loop that does not check for `sqlite3.OperationalError` / `rusqlite::Error::SqliteFailure(SQLITE_BUSY)` with an exponential-backoff retry.

   Severity: **high** (DEFERRED transaction containing writes with no retry handler), **med** (writes in a DEFERRED block where contention is unlikely but `busy_timeout` is 0), **low** (missing retry handler in a read-mostly path).

   Sheet: `transactions-and-isolation.md`

5. **Schema versioning absent — `application_id` and `user_version` never set**

   Pattern:
   ```bash
   grep -rn "application_id\|user_version" "${PROJECT_PATH}" --include="*.py" --include="*.rs"
   ```
   Heuristic: If neither `PRAGMA application_id` nor `PRAGMA user_version` appears anywhere in the project, the database has no identity and no migration guard. Flag the absence. Also flag `user_version` reads without a corresponding write path (migration runner never advances the version).

   Severity: **high** (neither set), **med** (`user_version` present but `application_id` absent), **low** (`application_id` set but no version assertion on open).

   Sheet: `schema-migrations.md`

6. **File-level locking — raw `open()` + `flock()` instead of `portalocker` / `fs2`**

   Pattern:
   ```bash
   grep -rn "import fcntl\|fcntl.flock\|fcntl.lockf" "${PROJECT_PATH}" --include="*.py"
   grep -rn "use std::fs::File\b" "${PROJECT_PATH}" --include="*.rs"
   ```
   Heuristic: `fcntl.flock()` is not portable across OS boundaries and does not compose safely with SQLite's own locking. Flag any use of `fcntl` or `msvcrt.locking` for cross-process coordination around a SQLite file. On Python, `portalocker` is the approved substitute; on Rust, `fs2`. Note: SQLite's WAL mode already provides single-writer, multi-reader at the page level — an app-level lock is only needed for coordinating multi-statement workflows across processes.

   Severity: **med** (fcntl used for cross-process coordination), **low** (fcntl used for intra-process serialisation already covered by WAL).

   Sheet: `concurrent-access-patterns.md`

7. **Claim-pattern — missing double-predicate guard on `UPDATE`**

   Pattern:
   ```bash
   grep -rn -A5 "UPDATE.*SET.*assignee\s*=" "${PROJECT_PATH}" --include="*.py" --include="*.rs"
   ```
   Heuristic: A claim `UPDATE` that sets `assignee = ?` without a `WHERE assignee IS NULL OR claim_expires_at < ?` guard creates a race: two workers can both read an unclaimed row and both write it. Flag any claim-style UPDATE that lacks both the identity guard (`assignee IS NULL`) and the expiry guard (`claim_expires_at < ?` or `claim_expires_at IS NULL`).

   Severity: **high** (missing both guards — double-claim race is certain under concurrent workers), **med** (missing expiry guard — zombie worker can block queue indefinitely).

   Sheet: `optimistic-locking-and-leases.md`

8. **JSON column queried with `json_extract` without a matching indexed expression**

   Pattern:
   ```bash
   grep -rn "json_extract\|->>" "${PROJECT_PATH}" --include="*.py" --include="*.rs" --include="*.sql"
   grep -rn "CREATE INDEX" "${PROJECT_PATH}" --include="*.py" --include="*.rs" --include="*.sql"
   ```
   Heuristic: A `WHERE json_extract(col, '$.key') = ?` predicate on a large table performs a full scan unless a matching expression index exists (`CREATE INDEX idx ON t (json_extract(col, '$.key'))`). Cross-reference every `json_extract` in a WHERE clause against the indexed expressions. Flag any that have no matching index.

   Severity: **med** (table expected to grow beyond 10K rows), **low** (small reference table where a full scan is acceptable).

   Sheet: `json1-and-structured-data.md`

9. **FTS5 virtual table without sync triggers**

   Pattern:
   ```bash
   grep -rn "CREATE VIRTUAL TABLE.*USING fts5\|fts5" "${PROJECT_PATH}" --include="*.py" --include="*.rs" --include="*.sql"
   grep -rn "CREATE TRIGGER" "${PROJECT_PATH}" --include="*.py" --include="*.rs" --include="*.sql"
   ```
   Heuristic: An FTS5 index on a shadow of a regular table stays consistent only if INSERT/UPDATE/DELETE triggers keep it in sync, or if all writes go through the FTS5 table directly. Flag any FTS5 virtual table that has no corresponding triggers and whose backing table is written through a non-FTS5 path.

   Severity: **high** (searches will silently return stale results), **med** (write path is documented as FTS5-direct but the regular table also receives writes).

   Sheet: `fts5-full-text-search.md`

10. **SQLCipher — hardcoded key string in source**

    Pattern:
    ```bash
    grep -rn "PRAGMA key\s*=\s*['\"]" "${PROJECT_PATH}" --include="*.py" --include="*.rs"
    grep -rn "pragma_key\|set_key\|cipher_key" "${PROJECT_PATH}" --include="*.py" --include="*.rs"
    ```
    Heuristic: A `PRAGMA key = 'literal-string'` in source code embeds the encryption key in the repository. Flag any literal key value. The key must be loaded at runtime from an environment variable, secrets manager, or OS keychain — never from a string literal in committed code.

    Severity: **high** (key literal in source), **med** (key loaded from a plaintext config file that may be committed).

    Sheet: `encryption-with-sqlcipher.md`

11. **Backup procedure — no `VACUUM INTO` or `Connection.backup()` invocation found**

    Pattern:
    ```bash
    grep -rn "VACUUM INTO\|\.backup(\|backup_db\|sqlite3.*backup" "${PROJECT_PATH}" --include="*.py" --include="*.rs"
    ```
    Heuristic: If neither `VACUUM INTO` nor the `sqlite3.Connection.backup()` API (Python) nor `rusqlite::backup::Backup` (Rust) appears anywhere in the project, no online backup path exists. Flag the absence. Also flag backup scripts that copy the raw `.db` file without first acquiring a read transaction (risks a corrupt copy mid-write under WAL mode).

    Severity: **med** (no backup path found), **low** (raw file copy without read transaction).

    Sheet: `backup-restore-and-corruption.md`

## Output format

Emit findings as structured JSON:

```json
{
  "summary": {"high": 3, "med": 4, "low": 1},
  "findings": [
    {
      "severity": "high",
      "sheet": "parameterized-sql-only.md",
      "anti_pattern": "f-string interpolation inside cursor.execute()",
      "location": "src/store.py:88",
      "evidence": "cursor.execute(f\"SELECT * FROM jobs WHERE name = '{name}'\")",
      "remediation": "Replace with a parameterized query: cursor.execute('SELECT * FROM jobs WHERE name = ?', (name,))"
    },
    {
      "severity": "high",
      "sheet": "pragma-discipline.md",
      "anti_pattern": "WAL mode not enabled",
      "location": "src/db.py:14",
      "evidence": "conn = sqlite3.connect(path)",
      "remediation": "Call conn.execute('PRAGMA journal_mode = WAL') immediately after connect(), before any queries."
    },
    {
      "severity": "med",
      "sheet": "transactions-and-isolation.md",
      "anti_pattern": "BEGIN DEFERRED with writes, no busy_timeout retry",
      "location": "src/worker.py:42",
      "evidence": "conn.execute('BEGIN'); conn.execute('UPDATE jobs SET status = ?', ('done',))",
      "remediation": "Upgrade to BEGIN IMMEDIATE for write transactions; add a retry loop on OperationalError with exponential backoff."
    }
  ]
}
```

Present HIGH findings first, then MED, then LOW. Within each severity band, order by file path then line number. After the JSON block, emit a plain-language triage summary (3–5 sentences maximum).

## Optional: dispatch reviewer agent

If the `--review` flag is present (or if the user asks for narrative synthesis), run the sweep, then pass the findings JSON to the `embedded-database-reviewer` agent for a prose summary with prioritized remediation guidance:

```
Task(subagent_type="embedded-database-reviewer",
     description="Narrative synthesis of SQLite discipline audit findings",
     prompt="The following JSON is the output of an /audit-sqlite-discipline sweep on ${PROJECT_PATH}. Produce a prioritized remediation plan: group related findings, explain the failure mode each group represents, and recommend a fix sequence. Cross-reference the relevant discipline sheets. Under 800 words.\n\n${FINDINGS_JSON}")
```

The agent's output supplements the structured JSON — it does not replace it. Present the JSON first so the engineer has machine-readable findings, then the narrative for context.

The `embedded-database-reviewer` agent is defined in Task 19 of the `axiom-embedded-database` plan; forward-reference it freely here.

## Verification

After generating findings:

1. Address HIGH findings before merging or deploying. A HIGH finding represents a correctness or security risk (SQL injection, data loss under concurrent write, corrupt backup) that is likely to manifest in production.
2. Schedule MED findings for the next planned touch of the affected code. They represent suboptimal patterns that will cause problems at scale or under contention.
3. Acknowledge LOW findings in the report. Close them as accepted patterns with a brief rationale, or fix them opportunistically.

Re-run the sweep after fixing each severity band. The command is cheap to re-run — no compilation, no database connection.

## Cross-references

- `using-embedded-database` — discipline router; load this first for design decisions
- `sqlite-fundamentals.md` — connection model, threading, row factory
- `pragma-discipline.md` — authoritative PRAGMA values and the production connection block (dimension 1)
- `parameterized-sql-only.md` — SQL injection prevention, `executescript()` constraints (dimensions 2, 3)
- `transactions-and-isolation.md` — BEGIN flavours, DEFERRED-then-write, busy_timeout retry (dimension 4)
- `schema-migrations.md` — `application_id`, `user_version`, migration runner contract (dimension 5)
- `concurrent-access-patterns.md` — WAL locking model, portalocker / fs2 (dimension 6)
- `optimistic-locking-and-leases.md` — claim-pattern double-predicate, heartbeat, dead-worker recovery (dimension 7)
- `json1-and-structured-data.md` — expression indexes, `json_extract` in WHERE clauses (dimension 8)
- `fts5-full-text-search.md` — sync trigger discipline, FTS5 consistency model (dimension 9)
- `encryption-with-sqlcipher.md` — key derivation, key storage discipline (dimension 10)
- `backup-restore-and-corruption.md` — `VACUUM INTO`, `Connection.backup()`, WAL-safe copy (dimension 11)
- `duckdb-for-analytics.md` — analytics companion; not audited by this command
- `boundary-and-when-to-leave.md` — signals that SQLite is the wrong tool
- `embedded-database-reviewer` agent — narrative synthesis and prioritized remediation (Task 19)
