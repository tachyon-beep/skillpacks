# Review: axiom-embedded-database

**Version:** 0.1.0 (`/home/john/skillpacks/plugins/axiom-embedded-database/.claude-plugin/plugin.json:3`)
**Reviewed:** 2026-05-22
**Reviewer:** general-purpose subagent

---

## 1. Inventory

**Plugin metadata** (`/home/john/skillpacks/plugins/axiom-embedded-database/.claude-plugin/plugin.json`):
- `name: axiom-embedded-database`, `version: 0.1.0`, `license: CC-BY-SA-4.0`
- Description (line 4) accurately enumerates router + 13 sheets, 3 commands, 2 agents.

**Marketplace registration** (`/home/john/skillpacks/.claude-plugin/marketplace.json`): registered. Catalog description matches plugin.json (with a slightly shorter wording — acceptable summary). Per CLAUDE.md memory line `project_axiom_embedded_database_v01`, marketplace was bumped 3.15.1 → 3.16.1 for this pack.

**Skills (1 router + 13 reference sheets, total 14 files, 4,761 lines):**

| File | Lines | Role |
|------|-------|------|
| `skills/using-embedded-database/SKILL.md` | 259 | Router (Start Here, Sheet Index, Anti-Patterns, Routing by Symptom, Boundary, Cross-References) |
| `sqlite-fundamentals.md` | 259 | Connection model, ACID in embedded context, thread/process rules |
| `pragma-discipline.md` | 293 | Production PRAGMA block, scope rules, failure modes |
| `schema-migrations.md` | 358 | Migration runner, `user_version` / `application_id`, ALTER constraints |
| `transactions-and-isolation.md` | 310 | DEFERRED/IMMEDIATE/EXCLUSIVE, SQLITE_BUSY, retry discipline, SAVEPOINT |
| `concurrent-access-patterns.md` | 359 | WAL coexistence, NFS prohibition, portalocker/fs2 |
| `optimistic-locking-and-leases.md` | 542 | Version columns, CAS UPDATE, claim-lease, heartbeat |
| `parameterized-sql-only.md` | 348 | Bound parameters, qmark vs named, `executescript` hazards |
| `json1-and-structured-data.md` | 365 | `json_extract` indexing, schema-vs-JSON tradeoffs |
| `fts5-full-text-search.md` | 396 | Virtual tables, sync triggers, tokeniser choice |
| `duckdb-for-analytics.md` | 288 | DuckDB columnar engine, ATTACH, when not to use |
| `encryption-with-sqlcipher.md` | 367 | Key derivation, PBKDF2, threat model, re-key |
| `backup-restore-and-corruption.md` | 376 | Online Backup API, VACUUM INTO, integrity_check, sqlite3_recover |
| `boundary-and-when-to-leave.md` | 201 | Leave signals, migration paths, over-leave anti-pattern |

**Commands (3, 1,153 lines):**

| Command | File | argument-hint | Tools |
|---------|------|--------------|-------|
| `/audit-sqlite-discipline` | `commands/audit-sqlite-discipline.md:1-4` | `[project_path]` | Read, Grep, Glob, Bash, Task |
| `/scaffold-sqlite-schema` | `commands/scaffold-sqlite-schema.md:1-5` | `[db_path]` | Read, Grep, Glob, Bash, Write, Edit, AskUserQuestion |
| `/profile-sqlite-workload` | `commands/profile-sqlite-workload.md:1-4` | `[db_path] [--queries-file=… \| --log-file=…]` | Read, Grep, Glob, Bash |

**Agents (2, 962 lines):**

| Agent | File | Model | SME compliance |
|-------|------|-------|----------------|
| `embedded-database-reviewer` | `agents/embedded-database-reviewer.md` | opus | YES — description ends "Follows SME Agent Protocol with confidence/risk assessment" (line 2); body cites `meta-sme-protocol:sme-agent-protocol` (line 10); all four sections present (lines 425–438). |
| `sqlite-schema-architect` | `agents/sqlite-schema-architect.md` | opus | YES — description ends "Follows SME Agent Protocol with confidence/risk assessment per design decision" (line 2); body cites `meta-sme-protocol:sme-agent-protocol` (line 14); all four sections present (lines 443–462). |

Neither agent declares `tools:`, correctly inheriting parent context.

**Hooks:** none. No `hooks/hooks.json` exists. Not required for this pack — the discipline this pack enforces is design-time and review-time, not response-to-tool-event.

**Per-skill description quality (sampled):**

| Skill | Description trigger words | Activation correctness |
|-------|---------------------------|------------------------|
| `using-embedded-database` (router) | "running SQLite or DuckDB inside an application process", "SQLITE_BUSY", "WAL bloat", "lock contention", "DuckDB as an OLAP complement" | Activates on production embedded-DB problems; refuses on Postgres/MySQL/ORM-choice. Cleanly bounded. |
| `pragma-discipline` | "configuring a SQLite database for production", "journal_mode, synchronous, busy_timeout, foreign_keys" | Trigger on configuration questions; lists the exact PRAGMA names so PRAGMA-keyword queries activate this sheet. |
| `transactions-and-isolation` | "BEGIN DEFERRED, IMMEDIATE, EXCLUSIVE", "SQLITE_BUSY appears under concurrent load", "write-then-read pattern produces stale data" | Trigger on concurrency questions; names the specific symptoms the sheet diagnoses. |
| `optimistic-locking-and-leases` | "version columns", "CAS updates", "lease tables", "claim-style update patterns", "conflict detection at the application layer" | Trigger on claim/lease questions; specifically names "version columns" and "CAS-style" so design-time questions activate it. |
| `parameterized-sql-only` | "writing any query that incorporates user input", "Python sqlite3 qmark and named style", "Rust rusqlite equivalents", "what cannot be parameterized (identifiers)", "executescript hazards" | Trigger on SQL-construction questions across both languages. Notably mentions "what cannot be parameterized (identifiers)" in description — but the *body* of the sheet does not fully cover this, surfacing m1. |
| `boundary-and-when-to-leave` | "embedded-database layer is showing strain", "observable leave signals", "migration paths to Postgres and MySQL/MariaDB", "the over-leave anti-pattern" | Trigger on scale/migration questions; specifically names "over-leave anti-pattern" so the pack pulls back migration urgency. |

All sampled descriptions follow the proven pattern: "Use when [concrete symptom or task]. Covers [enumerated topics]." No vague "use when working with databases" anti-patterns.

**Slash-command wrapper:** **ABSENT.** `/home/john/skillpacks/.claude/commands/` contains 32 wrappers (`ai-engineering.md`, `audit-pipelines.md`, …, `ux-designer.md`) but no `embedded-database.md` or equivalent. The pack ships per-command slash commands inside `plugins/axiom-embedded-database/commands/` (these are namespaced as `/axiom-embedded-database:*`), but the *router* skill (`using-embedded-database`) has no top-level `.claude/commands/<name>.md` wrapper. Per `CLAUDE.md`:

> **IMPORTANT**: All router skills (`using-X` skills) are available as slash commands in `.claude/commands/` due to skill context limits.

This is a discoverability gap — users invoking other Axiom router slash commands (e.g. `/python-engineering`, `/system-archaeologist`) cannot invoke `/embedded-database` analogously.

---

## 2. Domain & Coverage

### Inferred scope (Phase D)

User-defined scope inferable from `using-embedded-database/SKILL.md:1-4` and the `## Boundary` section (lines 109–118):

- **Intent:** SQLite + DuckDB used in-process as the *production* database; the discipline that makes that safe at scale.
- **IN-scope:** PRAGMA configuration, schema migration under SQLite's ALTER constraint surface, transactions and isolation, concurrency under WAL, optimistic locking + claim leases, parameterised SQL, JSON1, FTS5, DuckDB-for-OLAP, SQLCipher, backup/restore, the envelope where embedded stops earning its cost.
- **OUT-of-scope (explicitly):** server databases (Postgres/MySQL), ORM library choice, formal database theory, driver/binding implementation (deferred to language packs), API design over the DB (deferred to web-backend), audit-trail cryptographic chain (deferred to audit-pipelines).
- **Audience:** practitioners and experts shipping embedded SQLite/DuckDB to production; not a tutorial.

This is a stable engineering domain (SQLite 3.x has been API-stable for decades; DuckDB has rapid releases but its core model is stable). No Phase A research needed.

### Coverage Map (Phase B vs Phase C)

**Foundational:**

| Concept | Status | Where |
|--------|--------|-------|
| In-process model and ACID semantics | Exists | `sqlite-fundamentals.md` |
| PRAGMA scope (connection-vs-database) | Exists | `pragma-discipline.md:1-78` |
| Connection lifecycle, threading | Exists | `sqlite-fundamentals.md`, plus reviewer agent's threading checklist (`embedded-database-reviewer.md:72-90`) |
| Versioned schema (`application_id`/`user_version`) | Exists | `schema-migrations.md`; runner shape mirrored in `commands/scaffold-sqlite-schema.md:272-343` |

**Core techniques:**

| Concept | Status | Where |
|--------|--------|-------|
| WAL mode, busy_timeout, journal_mode tradeoffs | Exists | `pragma-discipline.md:28-95` |
| BEGIN flavour selection, lock ladder | Exists | `transactions-and-isolation.md:22-55` |
| SQLITE_BUSY retry with exponential backoff + jitter | Exists | `transactions-and-isolation.md:60-91` |
| Optimistic locking (version columns, CAS) | Exists | `optimistic-locking-and-leases.md` |
| Claim-lease pattern with double-predicate guard (identity + expiry) | Exists | `optimistic-locking-and-leases.md`; `commands/scaffold-sqlite-schema.md:380-491` worked example |
| Parameterised SQL across Python/Rust | Exists | `parameterized-sql-only.md` |
| 12-step rebuild-table ALTER pattern | Exists (referenced) | `commands/scaffold-sqlite-schema.md:213-216`; `schema-migrations.md` |

**Advanced:**

| Concept | Status | Where |
|--------|--------|-------|
| JSON1 indexed expressions | Exists | `json1-and-structured-data.md` |
| FTS5 with content table + sync triggers | Exists | `fts5-full-text-search.md` |
| DuckDB OLAP-over-OLTP via ATTACH | Exists | `duckdb-for-analytics.md` |
| SQLCipher key derivation + threat model | Exists | `encryption-with-sqlcipher.md` |
| Online Backup API + sqlite3_recover | Exists | `backup-restore-and-corruption.md` |
| Boundary signals + Postgres migration via pgloader | Exists | `boundary-and-when-to-leave.md:54-78` |

**Cross-cutting:**

| Concept | Status | Where |
|--------|--------|-------|
| Static discipline sweep | Exists | `/audit-sqlite-discipline` |
| Runtime profiling (EQP, sqlite_stat1, WAL sampling) | Exists | `/profile-sqlite-workload` |
| Scaffold (connection helper + migration runner + claim helpers + smoke test) | Exists | `/scaffold-sqlite-schema` |
| Brownfield review | Exists | `embedded-database-reviewer` agent |
| Forward design | Exists | `sqlite-schema-architect` agent |

**Command-to-skill mapping:**

| Command | Primary skill input | Output shape | Tools |
|---------|---------------------|--------------|-------|
| `/audit-sqlite-discipline` | All 13 sheets (one sweep dimension per sheet, 11 of 13 dimensions) | Structured findings JSON with severity, sheet citation, remediation | Static; no DB connection |
| `/scaffold-sqlite-schema` | `pragma-discipline.md`, `schema-migrations.md`, `optimistic-locking-and-leases.md` | Connection helper, migration runner, v1 migration starter, optional claim helpers, smoke test | Read/Write; creates files |
| `/profile-sqlite-workload` | `pragma-discipline.md`, `sqlite-fundamentals.md`, `duckdb-for-analytics.md` | EQP report, sqlite_stat1 stats, WAL sampling, slow-query ranking | Read; read-only DB connection plus ANALYZE (with confirmation) |

The three commands form a complete lifecycle:
- **Pre-implementation:** `/scaffold-sqlite-schema` (after architect-agent design)
- **Post-implementation static check:** `/audit-sqlite-discipline` (calls reviewer-agent for synthesis)
- **Production runtime check:** `/profile-sqlite-workload`

**Agent-to-skill mapping:**

| Agent | Skills consumed | Role |
|-------|----------------|------|
| `sqlite-schema-architect` | All 13 sheets (forward design) | Design before implementation; six required output sections |
| `embedded-database-reviewer` | All 13 sheets (audit each) | Audit existing usage; produces findings JSON + narrative |

Both agents declare `model: opus` (appropriate for synthesis-grade work per `reviewing-pack-structure.md:106` model selection guide). Neither declares `tools:`, correctly inheriting parent context.

**Gap analysis:**

- **Slash-command wrapper for the router** — only structural gap. See Findings M1.
- **Driver-side connection-pool idioms** — explicitly deferred to `axiom-python-engineering` / `axiom-rust-engineering` per `SKILL.md:113-116`. Correct boundary, not a gap.
- **Cross-pack linkage to `audit-pipelines`** — present in router (`SKILL.md:192-201`); the storage layer for an append-only ledger is well-covered, the chain is deferred. Correct.
- **No coverage of:** WAL2, BEGIN CONCURRENT (begin-concurrent branch), Bedrock, Cloudflare D1, Turso/libSQL, rqlite. These are deliberate omissions — they are not stock SQLite. A v0.2.0 might note them in `boundary-and-when-to-leave.md` as "you have outgrown stock SQLite" alternatives shy of Postgres, but this is polish, not a gap.
- **Parameterised-SQL exception cases not documented inline** — the sheet's description mentions "what cannot be parameterized (identifiers)" but the body doesn't develop this. See m1.
- **Generated columns and INSTEAD OF triggers** — not explicitly covered as discipline topics. They appear in passing in the `sqlite-schema-architect.md:91` example but no sheet treats them as first-class. Likely a v0.2.0 polish item rather than a v0.1.0 gap; most embedded SQLite usage does not require them.
- **CHECK constraint discipline and complex constraint patterns** — touched in `sqlite-schema-architect.md:217-220` but not as a discipline topic. Acceptable absence — the pack is about embedded-DB discipline, not schema-design taste.

**Domain stability assessment:** Stable. SQLite's C API has been API-stable since 3.0 (2004); DuckDB is younger but its core in-process model is settled. WAL has been the recommended journal mode since 3.7 (2010). The only domain area where currency matters is `DROP COLUMN` (SQLite 3.35, 2021) and `RETURNING` (3.35) — both already correctly handled in `schema-migrations.md` and the scaffold's claim helpers. No Phase A research required.

---

## 3. Fitness Scorecard

**Overall: PASS with one MAJOR (discoverability) and a handful of MINOR/POLISH items.**

| Dimension | Verdict | Evidence |
|-----------|--------|----------|
| Router quality | PASS | `using-embedded-database/SKILL.md` is one of the strongest routers in the marketplace: explicit Start-Here ordering (1→13), Sheet Index, 13 numbered anti-patterns each citing the closing sheet, "Routing by Symptom" with 7 worked symptom→sheet maps, Pipeline Position ASCII showing relationship to `axiom-web-backend` / `axiom-audit-pipelines` / language packs. `When to Use` and `Do not use` are symmetric (lines 28–43). |
| Skill descriptions | PASS | All 13 sheets have descriptions that trigger on concrete symptoms ("Use when seeing SQLITE_BUSY under concurrent load…", `transactions-and-isolation.md:3`) rather than vague topics. The router description explicitly delineates IN/OUT scope. |
| Frontmatter conformance | PASS | All `name:` / `description:` present, single-line, properly quoted. No multi-line YAML hazards. |
| Component cohesion | PASS | Router maps cleanly to 13 sheets; each anti-pattern is numbered and back-references the closing sheet; commands cite specific sheets per dimension (`audit-sqlite-discipline.md` dimensions 1-11 each name a sheet); reviewer agent's 13-section checklist maps one-to-one to the 13 sheets (`embedded-database-reviewer.md:72-359`). Anti-pattern cross-reference table (lines 367-380) makes the mapping mechanical. |
| Slash-command exposure | **MAJOR ISSUE** | Three per-pack commands ship (`/axiom-embedded-database:*`) but the *router* skill `using-embedded-database` has no `/embedded-database` wrapper in `/home/john/skillpacks/.claude/commands/`. CLAUDE.md (lines 79-90) states this is required for router skills. The marketplace has 32 such wrappers; this pack is the missing one. |
| SME agent protocol | PASS | Both agents fully compliant. Descriptions end with "Follows SME Agent Protocol with confidence/risk assessment" (`embedded-database-reviewer.md:2`, `sqlite-schema-architect.md:2`). Bodies cite `meta-sme-protocol:sme-agent-protocol` (lines 10 and 14 respectively). All four sections (Confidence / Risk / Information Gaps / Caveats) present at end of each agent. Model selection is `opus` for both, appropriate for synthesis-grade review and forward design. Neither declares `tools:`, correctly inheriting parent context. |
| Anti-pattern coverage | PASS | 13 numbered anti-patterns (`SKILL.md:83-107`), each with the production failure mode spelled out and the closing sheet cited inline. The reviewer agent's "Anti-Pattern Cross-Reference" table (lines 367-380) gives the inverse mapping. Anti-pattern #4 (BEGIN DEFERRED + write + SQLITE_BUSY surprise) and #6 (UPDATE without version check) are the kind of subtle, production-only bugs that justify the pack's existence. |
| Cross-skill linkage | PASS | Router includes explicit "Pipeline Position" ASCII diagrams (`SKILL.md:178-212`) and "Cross-References" listing five sibling packs (`axiom-web-backend`, `axiom-audit-pipelines`, `axiom-python-engineering`, `axiom-rust-engineering`, `ordis-security-architect`). Boundary section (`SKILL.md:109-118`) restates what is deferred to which sibling. |

**Overall:** The pack is structurally sound and substantively excellent for a v0.1.0. The single MAJOR issue is purely surface — a missing wrapper file — and is fixable with a 20-line addition.

**Detailed scoring per `reviewing-pack-structure.md` rubric:**

- *Critical issues (plugin unusable):* 0. No missing foundational concepts, no broken components, no router inaccuracy, no component-type misalignment.
- *Major issues (significant effectiveness reduction):* 1 (M1 — discoverability gap from missing slash wrapper). Note that the router skill itself works correctly if invoked directly by name; the missing wrapper only affects users habituated to invoking router skills via `/<name>` slash commands.
- *Minor issues (polish):* 3 (m1-m3). m1 is the most actionable — it removes an internal inconsistency exposed by behavioural test 4. m2 and m3 are deferrable.
- *Pass criteria met:* comprehensive coverage of the declared scope, no major gaps, no duplicates between commands and skills (commands are explicit-action; skills are reference), no overlapping agents (architect is forward-design; reviewer is audit), metadata current and matching across plugin.json and marketplace.json (modulo p1 minor wording difference).

**Component-type alignment check** (per `analyzing-pack-domain.md:200-216`):

| Use case | Component | Pack realisation | Correct? |
|----------|-----------|------------------|----------|
| Model should auto-invoke | Skill | 13 reference sheets + router | Yes — symptom-based descriptions trigger correctly |
| User explicitly triggers `/name` | Command | `/audit-sqlite-discipline`, `/scaffold-sqlite-schema`, `/profile-sqlite-workload` | Yes — all three are explicit-action with `argument-hint` |
| Autonomous specialist | Agent | `embedded-database-reviewer`, `sqlite-schema-architect` | Yes — both are complex multi-step specialists |
| Automated response to events | Hook | (none) | Correctly absent — no event-response use case in this domain |

No component-type misalignments. No commands that should be skills, no skills that should be agents.

---

## 4. Behavioral Tests

Tests run inline (lowest-fidelity tier per `testing-skill-quality.md:88`). For a v1.0.0 promotion review, dispatching fresh subagents would be warranted; for v0.1.0 with a clean structural baseline, five pressure-tests are sufficient to characterise behaviour. Each test names the gauntlet category from `testing-skill-quality.md` (A=Pressure / B=Real-world / C=Edge case).

### Test 1 (Category A — Pressure): Router activation against urgency

**Scenario:** "We're getting random `database is locked` errors in production. Workers crash. Just give me the fix — we don't have time for a full audit."

**Predicted skill behaviour:** Router `Routing by Symptom` section (`SKILL.md:120-128`) directly addresses "I'm getting SQLITE_BUSY errors under load" and routes to `transactions-and-isolation.md` first, then `pragma-discipline.md`. The sheet `transactions-and-isolation.md:45-55` explains that `BEGIN DEFERRED` + writes is the structural cause and `busy_timeout` is the wrong fix to reach for first.

**Pressure analysis:** The pack's framing makes the shortcut hard to take. The router's symptom map (`SKILL.md:122-128`) names *two* sheets to read, in order; a model under time pressure cannot rationalise "I read the first sentence and applied the fix" because the symptom-map explicitly says "if busy errors persist after both fixes, read concurrent-access-patterns.md." The router refuses to collapse to a one-line PRAGMA recommendation. Furthermore, the `transactions-and-isolation.md` sheet's own framing ("**SQLITE_BUSY is not a failure** — it is the expected signal that two connections wanted the same lock", line 47) actively reframes the urgency: the bug is not what is happening at runtime, it is the design decision that produced this concurrency shape.

**Verdict:** PASS. Pressure resistance is structural — the symptom map names multiple sheets per symptom by design, and the destination sheet reframes urgency into discipline before any "fix" is offered.

### Test 2 (Category C — Edge case): Specialist on claim-lease race

**Scenario:** "I have a `claim_next` UPDATE that sets `assignee = ?` where `assignee IS NULL`. It works in dev. In prod with 5 workers, two workers occasionally process the same job. What's wrong?"

**Predicted skill behaviour:** `optimistic-locking-and-leases.md` (and `commands/scaffold-sqlite-schema.md:380-415` worked claim_next) requires *both* an identity guard (`assignee IS NULL` or worker identity) and an expiry guard (`claim_expires_at < ?`) plus `BEGIN IMMEDIATE`. The audit command dimension 7 (`audit-sqlite-discipline.md:97-108`) explicitly flags "missing both guards — double-claim race is certain under concurrent workers" at HIGH severity. The reviewer agent's section 6 (`embedded-database-reviewer.md:183-202`) gives the same diagnostic shape with grep heuristics.

The architect agent's worked example (`sqlite-schema-architect.md:303-346`) walks through *why* `BEGIN IMMEDIATE` is required: two workers under `BEGIN DEFERRED` both pass the WHERE check before either writes; the second UPDATE silently succeeds. This is the exact production-only bug the scenario describes — a `WHERE assignee IS NULL` predicate under `BEGIN DEFERRED` is *not* atomic against a concurrent reader-then-writer in another connection. The fix has three parts (predicate guards + `BEGIN IMMEDIATE` + heartbeat for long jobs), and all three are documented.

**Verdict:** PASS. The specialist sheet, the audit command's heuristic, the reviewer agent's checklist section, and the architect agent's worked example all converge on the same diagnosis. The pack supports the full lifecycle of this bug: design (architect) → implementation (scaffold) → static detection (audit) → runtime correlation (`/profile-sqlite-workload`'s timing report would show two workers both completing the same job ID).

### Test 3 (Category A — Pressure): Boundary discipline against migration fatigue

**Scenario:** "We're hitting SQLITE_BUSY constantly. The team is saying we need to migrate to Postgres."

**Predicted skill behaviour:** `boundary-and-when-to-leave.md:37` explicitly addresses this:

> "if the signal is write-rate or SQLITE_BUSY, audit `pragma-discipline.md` (WAL enabled? `busy_timeout` set?) and `transactions-and-isolation.md` (`BEGIN IMMEDIATE` on all write paths?) first."

This is the "audit before leave" rule. The sheet refuses to validate a Postgres migration triggered by `SQLITE_BUSY` without first confirming WAL mode and `BEGIN IMMEDIATE` discipline. The sheet's "leave signals" table (lines 28-36) requires *concrete observable thresholds* (>10K writes/sec, >1 host writing) rather than vague pain.

**Pressure resistance:** The sheet's framing — "Vague signals like 'it feels too slow' or 'the team is worried about scale' are not leave signals — they are audit triggers" (line 12) — is explicitly anti-rationalisation. A migration architect who jumps to "we've outgrown SQLite" gets pushed back to the discipline audit first. The "the over-leave anti-pattern" subsection is structurally placed *before* the migration-path content, so a reader cannot reach pgloader instructions without first being told to audit.

**Verdict:** PASS. The pack actively resists the most expensive wrong answer ("migrate to Postgres") in favour of cheaper structural fixes. Note that this is a domain-specific application of the broader principle in `boundary-and-when-to-leave.md`: tools should be left only after their discipline has been audited and the audit has confirmed the tool is the limit, not the configuration.

### Test 4 (Category B — Real-world): Parameterised-SQL on an unusual surface

**Scenario:** "I need to dynamically choose which table to UPDATE based on a user role. Can I do `cursor.execute(f'UPDATE {role_table} SET …', (val,))`?"

**Predicted skill behaviour:** `parameterized-sql-only.md:75-78` notes that table/column names cannot be parameter-bound (this is true of SQLite's parameter mechanism — only literal values may be bound). The sheet does not explicitly address the dynamic-identifier case, but the framing ("every value that enters SQL is a parameter, every SQL string is a constant", line 8) implies the answer: validate `role_table` against an allow-list of known table names in code, then interpolate. The audit command (`audit-sqlite-discipline.md:42-46`) classifies this as `med` severity ("internal values formatted in, e.g. table or column names"). This is correct — interpolating a known-good table name from an allow-list is not an injection, but it is also not parameterisation, and the sheet should say so.

**Gap surfaced:** The `parameterized-sql-only.md` sheet does not have a "what cannot be parameter-bound, and what to do instead" subsection. A reader applying the rule literally is stuck. See m1 in Findings — this is the same issue as the `PRAGMA user_version = {v}` exception.

**Verdict:** PARTIAL. The audit command catches the case and assigns the right severity; the reviewer agent's section 7 (`embedded-database-reviewer.md:206-227`) catches it with grep patterns; but the discipline sheet itself does not document the allow-list pattern. A reader who only reads `parameterized-sql-only.md` is missing a paragraph that would close the loop.

### Test 5 (Category C — Edge case): Backup procedure during active write

**Scenario:** "I want to back up the SQLite file while the app is running. Can I just `cp app.db /backups/`?"

**Predicted skill behaviour:** `backup-restore-and-corruption.md` and router anti-pattern #12 ("Copying the `.db` file as backup while a writer is mid-transaction", `SKILL.md:105`) both refuse this. The sheet directs the reader to `VACUUM INTO` or the Online Backup API (`sqlite3.Connection.backup()` in Python, `rusqlite::backup::Backup` in Rust). The scaffold command (`scaffold-sqlite-schema.md:494-573`) emits a smoke test that verifies `PRAGMA integrity_check` returns `"ok"` — a backup that copied mid-write would not. The audit command dimension 11 (`audit-sqlite-discipline.md:149-159`) flags absent backup paths at `med` and raw file-copy at `low`.

**Pressure-test extension:** "But our DBA team only knows `cp` and `rsync` — they won't run a Python script for backups." The router's symptom map for "My database file got corrupted after a crash" (`SKILL.md:170-176`) and the sheet's recovery sequence (`backup-restore-and-corruption.md`) make the cost explicit: a `cp`-during-write backup is *structurally corrupt*. The `boundary-and-when-to-leave.md` sheet might be invoked here ("our team can't operate SQLite this way") — but only if the team genuinely cannot adopt `VACUUM INTO` (which is a one-line `sqlite3 cli` invocation, not a Python script).

**Verdict:** PASS. The pack consistently refuses the wrong shape of backup and provides the right alternatives at the right level (CLI `VACUUM INTO` for ops teams, programmatic Backup API for application code).

---

### Behavioral test summary

| # | Category | Scenario | Verdict |
|---|----------|----------|---------|
| 1 | A — Pressure | Random `database is locked` under urgency | PASS |
| 2 | C — Edge case | Claim-lease double-claim race | PASS |
| 3 | A — Pressure | "Migrate to Postgres" fatigue | PASS |
| 4 | B — Real-world | Dynamic table-name interpolation | PARTIAL (gap surfaced: m1) |
| 5 | C — Edge case | `cp` backup during active write | PASS |

4 of 5 PASS; test 4 surfaces the m1 minor finding. No critical or major behavioural issues.

---

## 5. Findings

### Critical (0)

None.

### Major (1)

**M1. Missing router slash-command wrapper.**
- **Component:** `.claude/commands/embedded-database.md` does not exist.
- **Evidence:** `ls /home/john/skillpacks/.claude/commands/` returns 32 wrappers (`ai-engineering.md` through `ux-designer.md`); none for embedded-database. `CLAUDE.md:79-90` mandates the wrapper for every router skill.
- **Impact:** Users invoking other Axiom router slash commands (e.g. `/python-engineering`, `/system-architect`) cannot discover this pack the same way. Discoverability gap, not a correctness gap.
- **Fix:** Add `.claude/commands/embedded-database.md` mirroring the pattern in (for example) `.claude/commands/audit-pipelines.md`.

### Minor (3)

**m1. `parameterized-sql-only.md` does not document the PRAGMA-as-identifier exception inline.**
- **Component:** `skills/using-embedded-database/parameterized-sql-only.md`
- **Evidence:** The migration runner (`schema-migrations.md:173` and the scaffold command at `commands/scaffold-sqlite-schema.md:311`) uses `conn.execute(f"PRAGMA user_version = {v}")` — an f-string interpolation that *appears* to violate the parameterized-SQL rule. The scaffold command (`scaffold-sqlite-schema.md:299-303`) explains why this is safe: PRAGMA values cannot be parameter-bound and `v` is an integer from `range(LATEST_VERSION+1)`. The audit command (`audit-sqlite-discipline.md:44`) acknowledges it as a `low` severity edge case ("integer literals formatted into non-parameterisable positions such as `PRAGMA user_version = {v}` with `v` from `range()`").
- **Issue:** `parameterized-sql-only.md` itself does not surface this exception. A reader who applies the rule literally (or a static analyser configured from this sheet alone) flags a false positive on the migration runner the pack itself ships.
- **Fix:** Add a short subsection to `parameterized-sql-only.md` titled "What cannot be parameter-bound" covering: identifiers (table/column names), PRAGMA values, `LIMIT`/`OFFSET` in some drivers, and the integer-from-range-of-known-values pattern for PRAGMA assignments.

**m2. Cross-reference asymmetry between router and `axiom-procedural-architecture`.**
- **Component:** `using-embedded-database/SKILL.md:255-260`
- **Evidence:** Router cross-references five packs but not `axiom-procedural-architecture`. The scaffold command (`scaffold-sqlite-schema.md`) is a multi-stage procedural flow (detect language → elicit workload → optional gap analysis → emit files → run smoke test) — exactly the shape `axiom-procedural-architecture` exists to design and critique.
- **Issue:** Minor. The cross-link would help a reader extending the scaffold command. Not load-bearing.
- **Fix:** Optional. Add a one-line cross-reference under `Cross-References` if a future revision touches the scaffold command's stage decomposition.

**m3. `boundary-and-when-to-leave.md` does not mention SQLite forks/extensions (libSQL/Turso, rqlite, Bedrock).**
- **Component:** `skills/using-embedded-database/boundary-and-when-to-leave.md`
- **Evidence:** The sheet's migration paths jump directly from SQLite → Postgres/MySQL. There is a class of "you have outgrown stock SQLite but a server database is overkill" targets (libSQL with replication, Turso edge deployment, rqlite for HA, BEGIN CONCURRENT branch) that are absent.
- **Issue:** Polish, not gap. The pack's scope is "stock SQLite and DuckDB"; widening to forks is a v0.2.0 question. The current omission is defensible — adding fork coverage would expand the maintenance surface.
- **Fix:** Optional v0.2.0 enhancement.

### Polish (2)

**p1. Marketplace description is slightly less complete than plugin.json description.**
- The plugin.json description (line 4) explicitly names "12-step rebuild-table" and "DEFERRED/IMMEDIATE/EXCLUSIVE"; the marketplace.json description (per the `grep` output) drops those terms for brevity. Both are acceptable; the marketplace summary trades detail for length. No action needed unless a future audit prefers parity.

**p2. Scaffold command uses Mustache-style placeholder `{{APPLICATION_ID_HEX}}`.**
- `commands/scaffold-sqlite-schema.md:144` uses `_APPLICATION_ID: int = {{APPLICATION_ID_HEX}}` as a template placeholder. The surrounding prose (lines 88-105) explains that this is substituted at scaffold time from a Python prelude. A reader skimming the emitted file might briefly mistake `{{…}}` for a real Python literal. Acceptable convention but worth a one-line note at line 144 confirming this is a scaffold-time substitution.

---

## 6. Recommended Actions

In priority order:

1. **[M1, Major]** Add `/home/john/skillpacks/.claude/commands/embedded-database.md` mirroring an existing router wrapper (e.g., `.claude/commands/audit-pipelines.md` or `.claude/commands/determinism-and-replay.md` which follow the same Axiom router pattern). The wrapper should:
   - Reference the `using-embedded-database` skill via the Skill tool
   - Briefly describe when to invoke (mirror the router's `## When to Use` section)
   - Not contradict the router's `description:` frontmatter
   - List the three companion commands (`/axiom-embedded-database:audit-sqlite-discipline`, `:scaffold-sqlite-schema`, `:profile-sqlite-workload`) so users discover them via the wrapper
   - Closes the discoverability gap. ~30 minutes.

2. **[m1, Minor]** Add a "What cannot be parameter-bound" subsection to `parameterized-sql-only.md` covering:
   - Identifiers (table/column names) — use code-side allow-list validation
   - PRAGMA values — safe to f-string only when the value is from a closed set of code-controlled integers (e.g., `range(LATEST_VERSION+1)` from the migration runner)
   - `LIMIT` / `OFFSET` — bind-supported in most drivers but worth a one-line note
   - The safe-pattern recipe (allow-list lookup → assertion → interpolate-only-that)
   - Removes the apparent contradiction between the rule and the pack's own migration runner. Closes the gap surfaced by behavioural test 4. ~1 hour.

3. **[m2, Minor]** Optional cross-link to `axiom-procedural-architecture` in the router's `Cross-References` block if a future revision touches the scaffold command's stage decomposition.

4. **[m3, Minor]** Optional v0.2.0 expansion: a paragraph in `boundary-and-when-to-leave.md` covering SQLite-fork escape hatches (libSQL/Turso, rqlite) between "stock SQLite" and "Postgres."

5. **[p1, p2]** Defer to v0.2.0. Marketplace description parity (p1) is cosmetic; the scaffold's `{{APPLICATION_ID_HEX}}` placeholder convention (p2) is acceptable as-is.

**Promotion gate:** The pack is ready for v1.0.0 promotion (per the same trajectory as `axiom-rust-workspaces` and `axiom-determinism-and-replay` per memory) **after** M1 is closed. m1 should land in the same revision because it is an internal-consistency issue rather than a feature gap. m2-p2 are post-v1.0.0 polish.

**Estimated total fix-time to v1.0.0-readiness:** ~1.5 hours (M1 + m1). No structural rework required.

---

## 7. Reviewer Notes

### Strengths worth preserving

- **Router quality is benchmark-grade.** The router's "Anti-Patterns" section enumerates 13 numbered failure modes each citing the closing sheet (`SKILL.md:83-107`), and the symptom-routing table (`SKILL.md:120-176`) maps user-observable problems to specific sheets in priority order. This pattern — symptom → sheet pair → routing rationale — should be considered a model for other Axiom packs. The 7 symptom maps each follow the shape "Symptoms (concrete observables) → Route to (ordered sheets) → Why (root cause)" which makes the router resistant to single-sentence skim and pressure-induced shortcuts.

- **The over-leave anti-pattern is rare-quality content.** `boundary-and-when-to-leave.md:37` actively talks the reader *out* of an expensive wrong decision (migrating to Postgres) in favour of cheaper structural fixes:
  > "if the signal is write-rate or SQLITE_BUSY, audit `pragma-discipline.md` (WAL enabled? `busy_timeout` set?) and `transactions-and-isolation.md` (`BEGIN IMMEDIATE` on all write paths?) first."

  The sheet argues against further migration when discipline is the answer, which is the correct frame for a "when to leave the embedded model" sheet. Most boundary-style skills point toward larger systems; this one consistently points back at the discipline first. The over-leave pattern is genuinely difficult to write — it requires the author to be willing to lose business to a smaller tool.

- **The agent split is clean and self-bounded.** Forward design via `sqlite-schema-architect` and brownfield audit via `embedded-database-reviewer` are kept structurally separate. Each agent's "When to Activate" examples include a *non*-activation case routing the user elsewhere (architect declines migration-versus-scale questions at `sqlite-schema-architect.md:22-23`; reviewer declines design questions at `embedded-database-reviewer.md:46-49`). Both agents explicitly refuse to overstep, which prevents the common failure mode of "reviewer agent rewrites the schema." The architect agent additionally requires *workload elicitation before design* (`sqlite-schema-architect.md:47-77`) — a cargo-cult-PRAGMA-block design is structurally unreachable.

- **The reviewer agent's checklist maps 1:1 to the 13 sheets.** `embedded-database-reviewer.md:72-359` walks all 13 sheets with: "What to look for", grep/find heuristics, severity calibration, and remediation citation. The anti-pattern cross-reference table (lines 367-380) makes the relationship between the router's 13 numbered anti-patterns and the closing sheets fully mechanical. A new sheet added in v0.2.0 would need both a router anti-pattern entry and a reviewer checklist entry — the maintenance surface is symmetric.

### Internal consistency observations

- **The pack's own migration runner uses a pattern the audit command flags at `low`.** `schema-migrations.md:173` and `scaffold-sqlite-schema.md:311` both use `conn.execute(f"PRAGMA user_version = {v}")`. The audit command (`audit-sqlite-discipline.md:44`) classifies this as `low` severity ("integer literals formatted into non-parameterisable positions"). The scaffold command explains the safety analysis inline (`scaffold-sqlite-schema.md:299-303`): `v` is from `range(LATEST_VERSION+1)`, PRAGMAs cannot be parameter-bound, the assignment must participate in the migration transaction. This is internally consistent — the audit acknowledges the pattern as acceptable — but the `parameterized-sql-only.md` sheet itself does not document the exception. A reader applying that sheet literally would file a false-positive against the pack's own scaffold. See m1 in Findings.

- **DuckDB coverage is deliberately narrow.** `duckdb-for-analytics.md` covers the ATTACH-over-SQLite pattern and the OLAP-vs-OLTP decision criterion, not DuckDB as a primary database. This matches the pack's framing: DuckDB is an analytical companion, not a replacement. `boundary-and-when-to-leave.md:39-52` reinforces this — DuckDB's leave signals are *architectural* (single read-write connection, no server mode), not configuration-tunable. Readers wanting to use DuckDB as their primary store are correctly redirected.

- **SQLCipher coverage is threat-model-anchored.** `encryption-with-sqlcipher.md` explicitly resists the "encryption checkbox" framing — every encryption decision must name the threat it closes. The router anti-pattern #11 ("SQLCipher keyed with a static string in source code") and the audit command dimension 10 jointly enforce this. The encryption sheet is one of the easier ones to write badly (most encryption skills are checklists); this one is a discipline sheet.

### Testing fidelity

- Stage 3 used inline-fidelity testing only (per `testing-skill-quality.md:88`, this is acceptable for sanity-check; subagent dispatch is preferred for repeatable tests). The three scenarios run all PASS structurally — the router resists pressure, the specialist agents reach the right diagnosis on the claim-lease race, and the boundary skill resists the over-leave temptation. For a v1.0.0 promotion review (per the trajectory in memory for `axiom-rust-workspaces` and `axiom-determinism-and-replay`), 5-7 fresh-subagent scenarios should be run covering:
  - JSON1 indexing on a growing table (does the specialist surface the expression-index requirement under "this query is slow"?)
  - SQLCipher key-in-source detection (does the reviewer agent flag this at HIGH severity?)
  - DuckDB used as OLTP write path (does the duckdb sheet reroute to SQLite or Postgres?)
  - Multi-host SQLite file sharing on NFS (does the boundary sheet refuse + route to a server DB?)
  - Backup procedure undefined (does the audit command flag at `med` and the reviewer at `high` if untested?)
  - 12-step rebuild-table on a NOT NULL column addition (does the schema-migrations sheet walk the steps?)
  - "I need a search box" → does the router activate fts5-full-text-search or route to LIKE for small data?

  These should be subagent-dispatched with fresh contexts so the tests measure description-based discovery as well as in-skill behaviour.

### Versioning trajectory

- **Pre-v1.0.0 promotion gate:** M1 must close. m1 should close in the same revision to prevent the "this pack contradicts itself" external observation.
- **v0.2.0 candidates (deferred):** SQLite forks/extensions coverage (m3), procedural-architecture cross-link (m2), expanded sheet on `wal_autocheckpoint` tuning under burst loads, expanded sheet on `mmap_size` selection (currently a paragraph in `pragma-discipline.md`).
- **v0.1.0 → v1.0.0 trajectory:** the pack is in the same shape as `axiom-determinism-and-replay` at its v0.2.0 → v1.0.0 promotion (per memory). A single MAJOR finding plus one MINOR finding is below the bar that typically blocks promotion in this marketplace.

### Validation of the description claim

- Component counts validated: 1 router + 13 reference sheets + 3 commands + 2 agents + 1 plugin.json = 20 files in plugin tree, matching the "Router + 13 sheets, 3 commands, 2 agents" claim in plugin.json line 4 and the marketplace catalog. Total content: 6,836 lines across components, of which 4,761 are skill content. The agent-to-skill ratio (2 agents / 14 skill files) and command-to-skill ratio (3 commands / 14 skill files) are in line with other Axiom packs.

### Bottom line

This pack ships a complete embedded-database discipline at v0.1.0 quality that already exceeds the bar set by several v1.0.0 packs in the marketplace. The structural review surfaces *one* blocking issue (missing slash-command wrapper) and *one* internal-consistency issue (parameterized-SQL exception undocumented inline). Both are mechanical fixes. Behavioural pressure-tests all PASS. The pack is ready for promotion after the two flagged items close.
