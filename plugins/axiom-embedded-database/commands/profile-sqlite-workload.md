---
description: Profile a SQLite workload — run EXPLAIN QUERY PLAN over every query the application issues (extracted from source or from a runtime log), measure index hit rate via `sqlite_stat1`, sample WAL file size over time, identify slow queries via per-statement timing, and emit a structured report ranking queries by table-scan cost. Cross-references findings to the `axiom-embedded-database` sheet that closes each performance gap.
allowed-tools: ["Read", "Grep", "Glob", "Bash"]
argument-hint: "[db_path] [--queries-file=queries.sql | --log-file=app.log]"
---

# Profile SQLite Workload Command

You are profiling a SQLite workload to identify index gaps, WAL checkpoint pressure, and slow queries. The output is a structured measurement report — not a schema redesign, not a rewrite. Findings inform which indexes to add, which PRAGMAs to tune, and whether the query volume has outgrown SQLite's model. This command does **not** modify the schema, does **not** run a synthetic benchmark, and does **not** execute DML on the target database. All reads use read-only connections or shell stat tools; `ANALYZE` is the one exception and must be confirmed with the user before running.

## Invocation Path

`/profile-sqlite-workload` is a Claude Code slash command. It requires a path to the target database file. Queries to profile come from one of three sources — a queries file, a runtime log, or a heuristic source scan — resolved in that order. The profiling steps (EQP sweep, ANALYZE, WAL sampling, slow-query timing) each have clear preconditions; if a precondition is not met, skip the step gracefully and note the gap in the report.

For static discipline violations (wrong PRAGMAs, f-string SQL injection, missing `busy_timeout`) use `/audit-sqlite-discipline` instead. This command is strictly runtime-facing.

## Inputs

Three arguments control the profiling run:

**`db_path`** (required) — path to the `.db` file. The command will look for a `.db-wal` sidecar at `${db_path}-wal` automatically. Verify the file exists and is readable before proceeding:

```bash
stat "${DB_PATH}"
file "${DB_PATH}"  # should report "SQLite 3.x database"
```

**`--queries-file=queries.sql`** (optional) — a file containing one SQL statement per line. Parameter values may be supplied as SQL comments (e.g. `-- param: user_id=42`) or as placeholder-only statements (`SELECT * FROM orders WHERE user_id = ?`). The command runs EQP and timing against each statement. Blank lines and lines starting with `--` (comments without a statement) are skipped.

**`--log-file=app.log`** (optional) — an application log file. The command extracts query patterns by scanning for lines matching common log formats:

```bash
grep -Eo "(SELECT|INSERT|UPDATE|DELETE|WITH)[^;]+" "${LOG_FILE}" | sort -u
```

Deduplicate identical query shapes (parameter values differ; structure is the same) before profiling. The log parser is heuristic — verify extracted queries make sense before treating EQP results as authoritative.

**Neither argument supplied** — fall back to source scan. Grep the project source for query strings:

```bash
grep -rn --include="*.py" --include="*.rs" \
  -E '(execute|query|prepare)\s*\(' "${PROJECT_PATH}" |
  grep -v "^Binary"
```

Extract the SQL string argument from each call site. This heuristic misses dynamically constructed queries; note any gaps in the report.

## Profile dimensions

### 1. EXPLAIN QUERY PLAN sweep

For each query in the input set, run EQP against the target database using a read-only connection:

```python
import sqlite3

conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
cursor = conn.cursor()

for query in queries:
    try:
        rows = cursor.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
        # Each row: (id, parent, notused, detail)
        for row in rows:
            detail = row[3]
            if "SCAN" in detail and "SEARCH" not in detail:
                # Full table scan — flag it
                record_scan_finding(query, detail)
            if "USE TEMP B-TREE" in detail:
                # Implicit sort with no covering index — flag it
                record_btree_finding(query, detail)
    except sqlite3.OperationalError as e:
        # Parameterized queries with ? need dummy values substituted
        record_eqp_skip(query, str(e))
```

**Reading EQP output.** The `detail` column tells you what the planner chose:

- `SCAN TABLE orders` — full table scan; every row is visited. This is the primary signal for a missing index.
- `SEARCH TABLE orders USING INDEX idx_user (user_id=?)` — index used; cost is O(log n) plus row fetch.
- `SEARCH TABLE orders USING COVERING INDEX idx_user_status (user_id=? AND status=?)` — covering index; no row fetch required.
- `USE TEMP B-TREE FOR ORDER BY` — the sort could not be satisfied by an index scan order; a temporary sort structure was built. Add an index on the ORDER BY column(s) that matches the WHERE clause selectivity.

For queries with `?` placeholders, EQP cannot bind parameters and may choose a suboptimal plan. Substitute representative literal values if the query file includes them as comments; otherwise note the limitation.

**Cost estimate.** EQP does not emit numeric cost estimates. Use `sqlite_stat1` (dimension 2) to estimate table cardinality and derive scan cost:

```
scan_cost_estimate = "full scan of ~N rows" where N = sqlite_stat1.stat first token
```

### 2. ANALYZE and sqlite_stat1

`ANALYZE` updates the planner's table and index statistics. Without it, the planner uses hard-coded heuristics that frequently misorder joins and choose wrong indexes on real data distributions.

**Before running ANALYZE, confirm with the user.** `ANALYZE` is a write operation (it populates `sqlite_stat1`, and optionally `sqlite_stat4`). On a production database that is being actively written, prefer running `ANALYZE` on a backup copy. On a development or staging database, run it directly.

```python
conn = sqlite3.connect(db_path)  # read-write connection required
conn.execute("ANALYZE")
conn.commit()

# Read statistics
rows = conn.execute("""
    SELECT tbl, idx, stat
    FROM sqlite_stat1
    ORDER BY tbl, idx
""").fetchall()
```

`sqlite_stat1` columns:

- `tbl` — table name
- `idx` — index name (or the table name itself for the table's implicit rowid index)
- `stat` — space-separated integers: first integer is the approximate row count; subsequent integers are the average number of rows per distinct value for each column in the index (in left-to-right key order)

A `stat` value of `1000000 500000 1` on a two-column index means: ~1M rows, ~500K rows per distinct first-key value, ~1 row per distinct (first-key, second-key) pair — the index is highly selective on the second column only. A planner that did not run `ANALYZE` would not know this.

**Flag under-indexed columns.** Cross-reference EQP SCAN findings against `sqlite_stat1`: if a table appears in a SCAN and has no entry in `sqlite_stat1` for any index covering the WHERE predicate columns, the index is either missing or was never analyzed. Emit a finding for each such table.

### 3. WAL size sampling

The WAL file (`${db_path}-wal`) grows as writers append new page versions. A WAL checkpoint (`PRAGMA wal_checkpoint`) resets it. If checkpoints are infrequent or blocked by long-running readers, the WAL file grows unboundedly, which degrades read performance (readers must scan more of the WAL to reconstruct the current page version).

Sample WAL file size over a fixed window:

```bash
DB_WAL="${DB_PATH}-wal"
INTERVAL_SECS=5
SAMPLES=12  # 60-second window

for i in $(seq 1 $SAMPLES); do
    if [ -f "$DB_WAL" ]; then
        SIZE=$(stat -c %s "$DB_WAL" 2>/dev/null || stat -f %z "$DB_WAL" 2>/dev/null || echo 0)
        echo "$(date +%s) ${SIZE}"
    else
        echo "$(date +%s) 0  # WAL file not present — WAL mode may not be enabled"
    fi
    sleep $INTERVAL_SECS
done
```

Compute WAL growth rate in KiB/min from the sample series:

```python
samples = [(t1, s1), (t2, s2), ...]
if len(samples) >= 2:
    elapsed_min = (samples[-1][0] - samples[0][0]) / 60.0
    size_delta_kib = (samples[-1][1] - samples[0][1]) / 1024.0
    growth_kib_per_min = size_delta_kib / elapsed_min if elapsed_min > 0 else 0
```

**Thresholds.** These are rough guidelines, not hard limits:

| WAL growth | Signal |
|------------|--------|
| < 100 KiB/min | Normal; checkpoint keeping up |
| 100–1000 KiB/min | Elevated; verify `PRAGMA wal_autocheckpoint` is set |
| > 1000 KiB/min | Checkpoint pressure; investigate long-running read transactions blocking checkpoint |

**If WAL file is absent.** The database is not in WAL mode. Note this in the report — WAL mode is the production-correct setting per `pragma-discipline.md`. A missing WAL file is itself a configuration finding.

### 4. Page-cache hit rate

SQLite's page cache keeps recently accessed database pages in memory to avoid re-reading them from disk. A low cache hit rate means the working set exceeds the configured cache size (`PRAGMA cache_size`), and reads are hitting the OS page cache or disk on every access.

**Check current cache configuration via PRAGMA.** `PRAGMA cache_size` with a negative value sets a KiB ceiling (e.g., `-2000` = 2 MB); with a positive value it sets a page-count ceiling. The default is 2000 pages (typically 8 MB with a 4096-byte page size, per `PRAGMA page_size`). Read it to confirm the connection's configured ceiling before drawing any conclusions about whether the cache is undersized.

**Limitation: hit rate not measurable from Python's `sqlite3`.** The C-level functions `sqlite3_status(SQLITE_STATUS_PAGECACHE_HIT, ...)` and `sqlite3_db_status(SQLITE_DBSTATUS_CACHE_HIT, ...)` expose cache hit and miss counters, but Python's stdlib `sqlite3` module does not bind these functions. Cache hit rate cannot be measured directly without a C extension, `ctypes`, or an alternative driver such as `apsw` (which exposes `apsw.Connection.status()`).

**Proxy measurement.** Instead of measuring hit rate directly, compare query timing between runs:

- First run after a fresh connection: cold cache — reads come from OS page cache or disk.
- Subsequent runs: warm cache — reads come from SQLite's page cache.

A large ratio (e.g., first run 200 ms, second run 5 ms) suggests the working set fits in the page cache but the initial page load is expensive. A flat ratio (both runs slow) suggests the working set exceeds the page cache — consider increasing `PRAGMA cache_size` per `pragma-discipline.md`.

**Recommended action if cache appears undersized.** Increase `PRAGMA cache_size` to cover the hot working set. Use `PRAGMA page_count` and `PRAGMA page_size` to estimate database size:

```python
page_count = conn.execute("PRAGMA page_count").fetchone()[0]
page_size = conn.execute("PRAGMA page_size").fetchone()[0]
db_size_mb = (page_count * page_size) / (1024 * 1024)
# Set cache_size to cover ~25% of the database for typical OLTP hot sets
suggested_cache_kib = -(page_count * page_size * 0.25) // 1024
```

Note that `cache_size` is connection-scoped — it must be re-set on every connection open. It is not persisted in the database file.

### 5. Slow-query identification

Wrap each query in a timing harness using a read-only connection. Run each query five times (warm-up on first run, four measurement runs) and record wall-clock time per execution:

```python
import sqlite3
import time
import statistics

conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

timings = {}
for query in queries:
    try:
        sample_times = []
        for run in range(5):
            t0 = time.monotonic()
            conn.execute(query).fetchall()
            t1 = time.monotonic()
            if run > 0:  # discard first run (cold page cache)
                sample_times.append((t1 - t0) * 1000)  # ms
        if sample_times:
            timings[query] = {
                "mean_ms": statistics.mean(sample_times),
                "max_ms": max(sample_times),
            }
    except sqlite3.OperationalError as e:
        timings[query] = {"error": str(e)}
```

**Limitations.** The timing harness runs queries sequentially in a single-process context. It does not reproduce concurrent write pressure. The results are useful for identifying expensive queries but do not replicate production latency under load. For production-representative timing, instrument the application with a logging wrapper that records statement duration at the `execute()` call site.

## Output format

Emit findings as structured JSON:

```json
{
  "db_path": "/path/to/app.db",
  "queries_profiled": 12,
  "analyze_run": true,
  "wal_growth_kib_per_min": 42.3,
  "table_scan_findings": [
    {
      "query": "SELECT * FROM events WHERE user_id = ? AND created_at > ?",
      "table": "events",
      "eqp_detail": "SCAN TABLE events",
      "scan_cost_estimate": "full scan of ~2400000 rows",
      "remediation": "Add index: CREATE INDEX idx_events_user_created ON events (user_id, created_at)",
      "sheet": "sqlite-fundamentals.md"
    }
  ],
  "temp_btree_findings": [
    {
      "query": "SELECT * FROM orders ORDER BY created_at DESC LIMIT 20",
      "eqp_detail": "USE TEMP B-TREE FOR ORDER BY",
      "remediation": "Add index on (created_at DESC) or a covering index matching the WHERE + ORDER BY",
      "sheet": "sqlite-fundamentals.md"
    }
  ],
  "slow_queries": [
    {
      "query": "SELECT count(*) FROM events WHERE status = 'pending'",
      "mean_ms": 340.2,
      "max_ms": 412.7,
      "rank": 1
    }
  ],
  "wal_absent": false,
  "recommendations": [
    "Add index on events(user_id, created_at) — eliminates the highest-cost full scan.",
    "Run ANALYZE periodically (e.g., on deploy or after large bulk inserts) to keep planner statistics current.",
    "Verify wal_autocheckpoint is set to a value that keeps WAL below 10 MB; current growth rate projects 2.5 MB/hr."
  ]
}
```

After the JSON block, emit a plain-language triage summary (4–6 sentences maximum) covering: number of queries profiled, highest-cost scan findings, WAL health, and the single highest-priority action.

## Verification

After adding indexes or tuning PRAGMAs based on the report:

1. Re-run `/profile-sqlite-workload` with the same inputs. EQP output for previously flagged queries should change from `SCAN TABLE x` to `SEARCH TABLE x USING INDEX y`. If it does not, the index definition may not match the WHERE predicate (check column order and expression form).
2. Verify WAL growth rate dropped. If WAL growth persists after adding indexes, the cause is checkpoint pressure (long-running readers), not index selectivity.
3. Re-check slow-query timings. Expect a reduction proportional to the ratio of rows scanned before vs. after indexing. A query scanning 1M rows that now scans 100 rows via index should be ~100× faster. If improvement is smaller, examine whether the query fetches many columns not in the index (consider a covering index).
4. Run `ANALYZE` again after adding indexes so the planner has accurate statistics for the new index. Without re-running ANALYZE, the planner may not choose the new index immediately on fresh databases.

Do not skip the re-run. The EQP output is the canonical evidence that the fix worked — it is not sufficient to add the index and assume the planner uses it.

## Don't Use This Command When

- The database is on a network filesystem (NFS, SMB). WAL mode is unsafe on network filesystems; profiling WAL growth is misleading. Profile the application on a local copy instead.
- The query volume is dominated by analytics aggregations over millions of rows. For that workload, consider migrating the analytics path to DuckDB — see `duckdb-for-analytics.md`. This command profiles OLTP-shaped workloads.
- You want to audit static code patterns (parameterized SQL, PRAGMA block shape, transaction flavour). Use `/audit-sqlite-discipline` for static analysis.
- The database has never had `ANALYZE` run and the planner's plan choices look wrong even for simple queries. Run `ANALYZE` first (dimension 2), then re-run EQP — the plan may change and the original findings may be invalid.

## Cross-references

- `using-embedded-database` — discipline router; load for design and migration decisions
- `pragma-discipline.md` — `wal_autocheckpoint`, `cache_size`, `synchronous`, and `mmap_size` settings that directly affect the dimensions profiled here
- `json1-and-structured-data.md` — expression indexes for `json_extract` predicates; the EQP sweep will surface `SCAN` on JSON-filtered queries
- `fts5-full-text-search.md` — FTS5 queries appear in EQP as `SCAN TABLE x_fts5 VIRTUAL TABLE INDEX`; interpret FTS5 EQP output separately from B-tree index output
- `duckdb-for-analytics.md` — when the slow-query report is dominated by aggregation and GROUP BY queries rather than OLTP lookups, use this sheet to evaluate migration
- `/audit-sqlite-discipline` — static sweep for PRAGMA, parameterization, and transaction discipline violations; complement to this command
