---
name: duckdb-for-analytics
description: Use when a SQLite database is choking on aggregate queries over millions of rows, when you want to query Parquet files without a server, or when you need OLAP capability alongside an existing SQLite OLTP layer. Covers when DuckDB earns its cost, the OLTP vs OLAP engine choice, the hybrid SQLite+DuckDB pattern with ATTACH, Parquet-without-ingestion, schema design for columnar stores, memory and parallelism settings, and a worked telemetry pipeline example.
---

# DuckDB for Analytics

**DuckDB is not "SQLite for analytics" — it is a different engine optimised for columnar, scan-heavy reads. The discipline is to know when to reach for it instead of pushing SQLite past its query-shape budget.**

## When this earns its cost

Read this sheet when:

- A SQLite table has grown to millions of rows and aggregate queries — `GROUP BY`, `SUM`, `COUNT(DISTINCT …)`, window functions — are taking seconds rather than milliseconds, and adding indexes no longer helps because the query must touch most of the table.
- A Pandas-in-memory pipeline reads a file, filters it, and groups it — and that pipeline would be cleaner, faster, and less memory-hungry expressed as SQL over the file directly.
- Telemetry or event data accumulates continuously and the shape of the downstream analysis is OLAP: "how many events of type X occurred per day last week, broken by dimension Y?" SQLite can answer that, but it was not built for it.
- You want to query Parquet files on disk without ingesting them into any database. DuckDB treats Parquet files as tables, pushes predicates into the file format's column index, and reads only the columns your query touches.
- You are already writing Arrow-format data (pyarrow, polars) and want to query it in place without a format round-trip.

What this sheet does not cover: deploying DuckDB as a shared multi-host data warehouse, DuckDB's Python `duckdb` module API beyond what the examples need, or DuckDB's HTTP server mode. For multi-host analytics at scale, the answer is a real warehouse (BigQuery, ClickHouse, Snowflake); DuckDB's target is the single-process analytical workload.

## OLTP vs OLAP — the engine choice

The fundamental split is between **row-store** (data laid out row-by-row on disk) and **columnar store** (data laid out column-by-column). SQLite is row-store. DuckDB is columnar. The right choice is determined by query shape, not dataset size alone.

| Workload | Engine | Why |
|----------|--------|-----|
| Many small point-read/writes: `INSERT`, `SELECT` by PK, `UPDATE` one row | SQLite | Row-store is fast for fetching one complete row; columnar must reconstruct it from N column files |
| Aggregates over millions of rows: `SUM`, `AVG`, `COUNT(DISTINCT)`, window functions | DuckDB | Reads only the column(s) involved; SQLite reads every row's full width |
| Joins across two row-store tables on a shared key | SQLite | Row-store hash joins are cache-friendly for moderate cardinalities |
| `GROUP BY` over a wide telemetry table with 20+ columns | DuckDB | Columnar reads only the grouping columns and the aggregated column, skipping the rest |
| Querying Parquet or CSV files on disk | DuckDB | Native format support; predicate pushdown into the file format |
| Append-only audit log with chain hash, FTS5 lookup | SQLite | Row-store append is a simple page write; FTS5 is a SQLite extension; chain hash fits a per-row write model |
| Mixed: transactional writes + periodic analytical reports | Both | Hybrid pattern — see next section |

The rule of thumb: if a query must touch most rows of a table to produce its result, the engine that reads fewer bytes wins. DuckDB's columnar layout means a query touching 3 of 30 columns reads roughly 10% of the bytes SQLite would. That multiplier is why OLAP queries against large tables are categorically faster in DuckDB, not just incrementally so.

## DuckDB connection model

DuckDB embeds in-process, like SQLite, but the concurrency model differs:

**No per-connection write lock.** SQLite serialises writers through a file-level write lock. DuckDB uses optimistic MVCC: multiple writers can start transactions concurrently. At commit time, DuckDB checks for write–write conflicts. If two transactions modified the same rows, one commits and the other receives a transaction-conflict error and must retry. For workloads with low conflict rates, this is transparently faster than SQLite's lock-and-wait. For workloads with high conflict rates on the same rows, the application must handle the retry.

```python
import duckdb

# In-process, file-backed. Multiple connections within a single process are
# supported; but unlike SQLite, multiple processes cannot open the same
# DuckDB file in read-write mode concurrently.
conn = duckdb.connect("analytics.duckdb")

# :memory: — no persistence, useful for ad-hoc exploration or tests.
mem_conn = duckdb.connect(":memory:")
```

**Connections are cheap.** Open one per thread, or reuse a single connection in a single-threaded context. There is no connection pool tax comparable to a server database, because there is no IPC.

**File durability.** The `.duckdb` file is the durability boundary, analogous to SQLite's `.db` file. DuckDB writes a transaction log alongside the main file. On clean shutdown, the file is consistent. On crash, recovery runs at next open.

**File format stability.** DuckDB's on-disk format has not been stable across all versions. The v0.10 → v1.0 transition (2024) was a significant format break: databases created with v0.10 (and earlier) cannot be opened by v1.0 without export/re-import. As of DuckDB v1.0, the format is declared stable and the project has committed to forward compatibility. If you are pinning a DuckDB version in production, note the version in a comment next to the dependency and have an export path ready if the format changes.

## The hybrid pattern: SQLite OLTP + DuckDB OLAP

The most practical deployment for applications that need both transactional writes and analytical reads is to run OLTP in SQLite and OLAP in DuckDB, with a bridge between them.

**Why keep SQLite for OLTP.** SQLite handles concurrent small writes, row-level updates, FTS5, foreign key enforcement, and per-row trigger logic cleanly. DuckDB is slower for point writes and has historically lacked FK enforcement (added in v1.0 but not yet as reliable as SQLite's). Do not try to make DuckDB your application's write path.

**Option A: periodic ETL.** On a schedule (hourly, nightly), read new rows from SQLite, append them to DuckDB. Analytical queries run against DuckDB. The data is never perfectly fresh, but the latency is acceptable for dashboards and batch reports.

**Option B: `ATTACH` SQLite from DuckDB.** DuckDB's `sqlite_scanner` extension lets you attach a SQLite database file and query it as a DuckDB relation. The extension uses the SQLite library internally — it opens a real SQLite connection and delegates locking to SQLite's own lock machinery. This means:

- Locking is handled by the SQLite library: within the same process, SQLite uses mutexes; across processes, SQLite uses filesystem locks. A DuckDB read coexists with a SQLite writer under the same rules as any other SQLite reader. In WAL mode, DuckDB's scanner and a concurrent SQLite writer can proceed without blocking each other.
- The `ATTACH` read is done at query time, not cached — large OLAP queries over an `ATTACH`'d SQLite table still pay row-store costs for the SQLite side, then the DuckDB engine post-processes.
- Use it for moderate-size SQLite tables where freshness matters more than performance.
- Because the extension links the SQLite library into the DuckDB process, linking multiple copies of SQLite into the same application (e.g., your application already links libsqlite and so does DuckDB) can cause conflicts. Check the DuckDB documentation for your build's SQLite linkage before using `ATTACH` in a process that also loads its own SQLite.

```sql
-- In DuckDB, not SQLite.
-- The extension must be installed once per DuckDB installation.
INSTALL sqlite_scanner;
LOAD sqlite_scanner;

ATTACH 'app.sqlite' AS oltp (TYPE SQLITE);

-- Query a SQLite table from DuckDB.
SELECT user_id, COUNT(*) AS event_count
FROM oltp.events
GROUP BY user_id
ORDER BY event_count DESC
LIMIT 20;

-- Join a SQLite table with a DuckDB table.
SELECT e.user_id, u.display_name, COUNT(*) AS event_count
FROM oltp.events AS e
JOIN users AS u ON u.id = e.user_id   -- users is a DuckDB table
GROUP BY e.user_id, u.display_name;

-- Detach when done.
DETACH oltp;
```

The `ATTACH (TYPE SQLITE)` path is read-write in principle (DuckDB can write back to the SQLite file through the extension), but in practice treat it as read-only from the DuckDB side. Concurrent writes from both engines against the same file without coordination will corrupt the SQLite WAL state.

## Parquet without ingestion

DuckDB reads Parquet files as first-class tables, with no import step:

```sql
-- Single file.
SELECT event_type, COUNT(*) AS n
FROM read_parquet('events_2024_01.parquet')
WHERE event_type IN ('login', 'logout')
GROUP BY event_type;

-- Glob — all files matching the pattern are treated as one logical table.
SELECT DATE_TRUNC('day', ts) AS day, SUM(value) AS total
FROM read_parquet('metrics_*.parquet')
WHERE ts >= CURRENT_DATE - INTERVAL 7 DAY
GROUP BY day
ORDER BY day;

-- Hive-partitioned layout: DuckDB infers partitions from directory names.
SELECT region, SUM(revenue)
FROM read_parquet('sales/year=2024/month=*/data.parquet', hive_partitioning = true)
GROUP BY region;
```

DuckDB's Parquet reader pushes predicates into the file format's column statistics and row group index. A `WHERE ts >= X` on a column with row-group min/max statistics skips entire row groups without reading them. A `SELECT a, b` on a file with 20 columns reads only two column buffers. These are not DuckDB-level optimisations — they exploit the Parquet format's inherent structure.

**The architectural implication.** If you control the write side of an analytics pipeline, write Parquet instead of growing a database file. Parquet is an Arrow-compatible columnar format with broad external tooling support. Analysts can open it in pandas, polars, or any BI tool. The pipeline becomes: application writes Parquet → DuckDB queries it in place → no ETL, no separate analytical database to manage.

## Schema design for OLAP

OLAP schema differs from OLTP schema in ways that matter at query time.

**Wide, denormalised tables.** OLTP normalises to avoid update anomalies: separate tables for users, events, products. OLAP denormalises to avoid joins: a single wide `events` table with `user_name`, `user_region`, `product_category` baked in. In a columnar store, a wide table with 50 columns costs nothing if most queries only touch 4–6 of them. In a row-store, every query reads all 50.

**Fact/dimension pattern.** For larger analytical databases, the classic pattern is a `fact_events` table (high cardinality, one row per event) and `dim_user`, `dim_product`, `dim_date` dimension tables (low cardinality, one row per entity). Joins in DuckDB's query engine are hash joins that load dimension tables into memory — if dimension tables fit in memory (typical), fact-to-dimension joins are fast regardless of fact table size.

**Avoid per-row UPDATE.** The columnar layout makes full-column scans fast but makes point updates expensive — updating one row means rewriting a column segment. OLAP tables should be append-only or replace-partition patterns (drop and re-insert a day's data) rather than row-level UPDATE.

**Why a JOIN fine in SQLite can become a memory blowup in DuckDB.** A join between two 100-row lookup tables in SQLite is trivial. In DuckDB, the same join is also trivial, but if someone replaces one of those lookup tables with a 50M-row fact table and leaves the join in place, DuckDB's optimizer will try to build a hash table in memory from the larger side. The failure mode is memory exhaustion or spill-to-disk. Design OLAP queries assuming the engine will materialise intermediate results: keep join cardinalities bounded, filter before joining, and be explicit about which side of a join is the probe side.

## Memory and parallelism

DuckDB is designed to use all available system resources by default.

**Parallelism.** DuckDB uses all cores for query execution by default. A `GROUP BY COUNT(*)` on a 100M row table will spin up worker threads equal to the CPU count and partition the work across them. This is the right default for an analytics workload where one query per user runs at a time. If DuckDB is embedded in a multi-tenant service where many requests run concurrently, constrain the thread count:

```sql
-- Limit DuckDB to 4 worker threads.
SET threads = 4;
```

Or in Python:

```python
conn = duckdb.connect("analytics.duckdb", config={"threads": 4})
```

**Memory limit.** By default, DuckDB attempts to use as much RAM as needed. For a dedicated analytics host this is correct. For a process sharing memory with other services, set a limit:

```sql
SET memory_limit = '4GB';
```

With a memory limit set, DuckDB spills intermediate results to disk (a temp directory, configurable via `SET temp_directory`) when the limit is approached. Spill-to-disk works but degrades performance significantly — a query that spills is often 5–20x slower than one that fits in memory. Design query plans and memory limits together: either give DuckDB enough RAM to run the expected queries in memory, or restructure the queries to reduce working set size (filter earlier, avoid materialising huge intermediates).

**Configuring the temp directory:**

```sql
SET temp_directory = '/var/tmp/duckdb_spill';
```

## Worked example

A telemetry pipeline: events written continuously to SQLite, hourly ETL into DuckDB, dashboard queries run against DuckDB.

**Write path (SQLite — Python):**

```python
import sqlite3
import json
import time

def record_event(conn: sqlite3.Connection, event_type: str, user_id: int, payload: dict) -> None:
    """Append a single event. SQLite handles concurrent writers from multiple threads."""
    conn.execute(
        "INSERT INTO events (ts, event_type, user_id, payload) VALUES (?, ?, ?, ?)",
        (int(time.time()), event_type, user_id, json.dumps(payload)),
    )
```

**SQLite schema:**

```sql
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY,
    ts         INTEGER NOT NULL,          -- Unix timestamp; indexed for the ETL window query.
    event_type TEXT    NOT NULL,
    user_id    INTEGER NOT NULL,
    payload    TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_ts ON events (ts);
```

**ETL (Python, run hourly):**

```python
import duckdb
import sqlite3

def etl_hour(sqlite_path: str, duckdb_path: str, window_start: int, window_end: int) -> int:
    """Copy events in [window_start, window_end) from SQLite into DuckDB.

    Idempotent: uses INSERT OR IGNORE keyed on the SQLite id.
    Returns the number of rows inserted.
    """
    src = sqlite3.connect(sqlite_path)
    rows = src.execute(
        "SELECT id, ts, event_type, user_id, payload FROM events WHERE ts >= ? AND ts < ?",
        (window_start, window_end),
    ).fetchall()
    src.close()

    if not rows:
        return 0

    dst = duckdb.connect(duckdb_path)
    dst.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id         BIGINT PRIMARY KEY,
            ts         BIGINT NOT NULL,
            event_type VARCHAR NOT NULL,
            user_id    BIGINT NOT NULL,
            payload    VARCHAR NOT NULL
        )
    """)
    dst.executemany(
        "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    dst.close()
    return len(rows)
```

**Dashboard query (DuckDB):**

```sql
-- Events per day, by type, over the last 7 days.
SELECT
    DATE_TRUNC('day', TO_TIMESTAMP(ts)) AS day,
    event_type,
    COUNT(*)                             AS n,
    COUNT(DISTINCT user_id)             AS unique_users
FROM events
WHERE ts >= EPOCH(NOW() - INTERVAL 7 DAY)
GROUP BY day, event_type
ORDER BY day DESC, n DESC;
```

This query on a 10M-row DuckDB table runs in low single-digit seconds on commodity hardware because DuckDB reads only the `ts`, `event_type`, and `user_id` columns — three of five — and computes the aggregate in parallel across all cores.

The equivalent query in SQLite on the same 10M-row table reads every row's full width (including the `payload` JSON), serialises through the single query thread, and typically takes tens of seconds. The index on `ts` helps bound the scan to the 7-day window, but within that window it is full-row sequential.

## Anti-patterns

**Using DuckDB as the application's OLTP write path.** DuckDB's MVCC is optimistic, and point writes (single-row INSERT, UPDATE, DELETE) carry more overhead per operation than SQLite's simple B-tree append. FK enforcement existed only as a preview before v1.0. If your write shape is many small transactions hitting individual rows, SQLite is the right engine. DuckDB is for the read side.

**Doing full-row UPDATE at OLTP rates in DuckDB.** Columnar layout makes updates expensive: updating one column in one row requires rewriting a column segment. A workload that does thousands of UPDATE per second against a DuckDB table will degrade badly. If you find yourself needing frequent updates in DuckDB, that data belongs in SQLite, not DuckDB.

**Forgetting the WHERE and loading 100 GB of Parquet into memory.** `SELECT * FROM read_parquet('huge.parquet')` with no WHERE clause materialises the entire file into DuckDB's working memory. With a memory limit set, this spills and becomes very slow. Without a limit, it exhausts RAM. Always filter before materialising: add a WHERE, limit the columns with SELECT, or process in chunks. The `read_parquet` path is fast precisely because DuckDB can skip data it doesn't need — let it.

**Writing to the same SQLite file from both DuckDB and a native SQLite application simultaneously.** DuckDB's `sqlite_scanner` extension uses the SQLite library for locking, so a DuckDB read coexists safely with a SQLite writer in WAL mode — that is the normal SQLite reader/writer pattern. The hazard is concurrent *writes* from both sides: if DuckDB writes back to the SQLite file through the extension at the same time as your application is writing through its own SQLite connection, you have two writers competing through SQLite's single-writer serialisation. The result is write conflicts and potential corruption. In practice, treat `ATTACH (TYPE SQLITE)` as read-only from the DuckDB side. If you need to write from DuckDB, use the periodic ETL pattern to a separate DuckDB file instead.

**Treating the `.duckdb` file as portable across DuckDB versions without an export plan.** The v0.10 → v1.0 format break required `EXPORT DATABASE` / `IMPORT DATABASE` to migrate databases from any pre-v1.0 release. As of v1.0, the format is declared stable, but the lesson from that break is clear: pin the DuckDB version in your dependency manifest, test upgrades against real data before deploying, and have a working `EXPORT DATABASE` / `IMPORT DATABASE` path in your runbook. A `.duckdb` file is not a plain-text format; it is engine-specific.

**Treating DuckDB as a replacement for a real data warehouse at multi-host scale.** DuckDB is a single-process engine. It has no server mode, no cross-host replication, no distributed query execution, and no shared storage. If you need analytical queries from multiple hosts, or data that multiple writer processes contribute to simultaneously across machines, you need a warehouse (ClickHouse, BigQuery, Snowflake, Redshift). DuckDB is the right choice for the single-process analytical workload; it is not a drop-in warehouse substitute.

**Ignoring DuckDB's own encryption story and reaching for SQLCipher.** SQLCipher is a SQLite extension for transparent at-rest encryption. DuckDB has a different, separate encryption capability (via the `parquet` writer's encryption options and DuckDB extensions). Do not attempt to open a DuckDB file with SQLCipher or a SQLite+SQLCipher library — they are different formats and different encryption approaches. Encryption configuration for DuckDB files is outside this sheet; consult the DuckDB encryption extension documentation directly.

## Cross-references

- [`sqlite-fundamentals.md`](sqlite-fundamentals.md) — the SQLite connection model, ACID semantics, and thread/process safety rules that the hybrid pattern relies on.
- [`schema-migrations.md`](schema-migrations.md) — managing the SQLite OLTP schema that feeds the ETL path; `user_version` discipline for tracking migration state.
- [`pragma-discipline.md`](pragma-discipline.md) — configuring the SQLite side of the hybrid correctly so DuckDB's `ATTACH (TYPE SQLITE)` reads a well-configured database.
- [`encryption-with-sqlcipher.md`](encryption-with-sqlcipher.md) — SQLCipher is specific to SQLite; DuckDB has a different encryption mechanism that this sheet does not cover. If the OLTP layer needs at-rest encryption, apply it on the SQLite side; handle the DuckDB side separately.
- [`boundary-and-when-to-leave.md`](boundary-and-when-to-leave.md) — when neither SQLite nor DuckDB is the right answer: sustained high-concurrency multi-host writes, replication requirements, or analytical scale that exceeds what a single-process engine can handle.
