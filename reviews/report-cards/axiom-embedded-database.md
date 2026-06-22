# Report Card — axiom-embedded-database

**Version:** 0.2.0 (`plugins/axiom-embedded-database/.claude-plugin/plugin.json:3`)
**Track:** H — Hard / Technical (SQLite + DuckDB; correctness = technically-accurate claims, sound code, current APIs)
**Date:** 2026-06-22
**Prior review:** `reviews/axiom-embedded-database.md` (2026-05-22) graded v0.1.0 with one MAJOR (missing slash wrapper). That defect is **resolved** in v0.2.0; this report supersedes it.

---

## Subject grades

| Subject | Grade | Load-bearing evidence |
|---------|-------|------------------------|
| **A — Substance** (40%) | **A** | Technically accurate at expert depth across the whole declared domain. The lock-ladder (SHARED→RESERVED→PENDING→EXCLUSIVE) and the WAL equivalence of `IMMEDIATE`/`EXCLUSIVE` are correct (`transactions-and-isolation.md:31-43`). The `synchronous` durability table (`pragma-discipline.md:51-54`) and the NORMAL-vs-FULL power-loss window (`:62`) are right. The 12-step rebuild-table pattern is faithful to sqlite.org including the subtle "`PRAGMA foreign_keys` is ignored inside a transaction, must be set outside" detail and `PRAGMA foreign_key_check` assertion (`schema-migrations.md:33-114`). Currency is good: `DROP COLUMN` 3.35+ guard, `sqlite3_recover` 3.41+, `VACUUM INTO`, `sqlite_schema` vs `sqlite_master` 3.33 note. Connection-scoped vs database-scoped PRAGMA table (`pragma-discipline.md:166-189`) is the kind of distinction that prevents real production bugs. Not S: depth gaps the prior review flagged remain (no WAL2 / BEGIN CONCURRENT / libSQL escape-hatch note in the boundary sheet; generated columns not first-class) — defensible scope choices, but they keep it short of authoritative-across-everything. |
| **B — Usefulness** (25%) | **A** | Router routes crisply: a "Routing by Symptom" section with seven concrete symptom→sheet mappings each carrying a *why* (`SKILL.md:120-176`), a Quick Reference symptom table (`:220-236`), and a Start-Here ordered path. Sheets are copy-paste actionable: production PRAGMA block in both Python and Rust (`pragma-discipline.md:196-271`), a complete retry-with-backoff context manager (`transactions-and-isolation.md:212-286`), allowlist-then-interpolate identifier helper and an AST CI checker (`parameterized-sql-only.md:105-260`). Reading it changes what you do. |
| **C — Discipline** (20%) | **A** | 13 numbered anti-patterns each with the production failure mode and closing sheet (`SKILL.md:81-107`). Per-sheet Anti-patterns sections name the rationalizations verbatim ("this value is already validated", "only we use it is not a threat model" — `parameterized-sql-only.md:16,20`). Both agents carry `model: opus` and the SME Agent Protocol with Confidence/Risk/Information-Gaps/Caveats sections (`embedded-database-reviewer.md:421-439`, `sqlite-schema-architect.md:1-4`). The reviewer agent has severity calibration per sheet, grep recipes, a machine-readable JSON contract, and a "Don't Do" guardrail list (`:441-447`). This is the discipline signature near-fully realized. |
| **D — Form** (15%) | **A** | Conformant frontmatter throughout; commands carry `allowed-tools` + `argument-hint`. Slash wrapper present, current, and *richer* than the SKILL in places (12-step rebuild, AST CI enforcement, claim-lease double-predicate — `.claude/commands/embedded-database.md:15-19`). Zero count drift: plugin.json, marketplace.json, and wrapper all say "Router + 13 sheets, 3 commands, 2 agents" and the filesystem matches (13 sheets, 3 commands, 2 agents). Clean sibling boundaries to `/web-backend`, `/audit-pipelines`, `/python-engineering`, `/security-architect`. |

---

## Gate analysis

1. **Discoverability gate:** PASS. Installs, registered (`marketplace.json:304`), slash wrapper present and current (was the v0.1.0 MAJOR; now closed). No ceiling cap.
2. **Substance-dominates gate:** Substance = A → overall ≤ A+. Not binding here.
3. **Honor-roll (S) gate:** Fails — Substance is A not S (boundary-sheet escape-hatch gap; generated columns not first-class). No subject reaches S, so the pack is not S-eligible.
4. **Honesty override:** N/A — not a scaffold; content matches marketing exactly.

**Overall: A.** Four A's with no Major+ defects. Reconciles with the existing system as **Pass** (the v0.1.0 single-MAJOR is resolved).

---

## Layered per-component grades

Uniformly strong; no weak tail drags the pack down. Surfaced for reference:

| Component | Grade | Note |
|-----------|-------|------|
| `embedded-database-reviewer.md` (agent) | **A/S** | Exemplar worth copying: per-sheet what-to-look-for + grep recipe + severity calibration + JSON contract + SME protocol + "Don't Do". The template other review agents should imitate. |
| `transactions-and-isolation.md` | **A** | Lock ladder, DEFERRED→write race, and the production retry context manager — the sheet that justifies the pack's existence. |
| `schema-migrations.md` | **A** | 12-step rebuild faithful to sqlite.org; FK-pragma-outside-transaction subtlety captured. |
| `boundary-and-when-to-leave.md` | **B+** | Sound, but the prior review's noted gap persists: no mention of SQLite-fork escape hatches (libSQL/Turso, rqlite) or WAL2/BEGIN CONCURRENT between stock SQLite and Postgres. The one place the pack is thinner than authoritative. |

---

## Verdict

A reference-quality embedded-database discipline pack: technically accurate, fully wired, exemplary review agent — held just short of S by a few deliberate scope omissions in the boundary sheet.

**Top finding:** Substance, Usefulness, Discipline, and Form are all A with zero Major+ defects; the v0.1.0 missing-wrapper MAJOR is resolved in v0.2.0 and all surfaces are drift-free.

**Top fix:** Add a short escape-hatch paragraph to `boundary-and-when-to-leave.md` covering libSQL/Turso, rqlite, and WAL2/BEGIN CONCURRENT as the rung between stock SQLite and a server database — the one gap keeping Substance from S.
