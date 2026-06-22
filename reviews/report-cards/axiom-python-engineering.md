# Report Card — axiom-python-engineering

**Version:** 1.6.0  ·  **Track:** H (Hard / Technical)  ·  **Graded:** 2026-06-22
**Unit:** pack (router + 10 reference sheets, 4 commands, 3 agents)

Prior review `reviews/axiom-python-engineering.md` is dated 2026-05-22 at v1.5.0 and is **partially stale**: its two Majors (M1 drifted slash wrapper, M2 router description not "Use when…") are both **resolved** in v1.6.0. Fresh reading weighted over the old verdict where they diverge.

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---------|-------|-----------------------|
| **A — Substance** | A− | Complete coverage of the declared domain (type system, mypy resolution, project/uv tooling, delinting, testing, async, sci-computing, ML workflows, profiling, Textual) across 10 deep sheets (~12.5k lines). Technically sound throughout: `modern-syntax-and-types.md` shows correct PEP 604/695/`@override`, list invariance→`Sequence` (lines 707-729), Protocol-vs-ABC. `scientific-computing-foundations.md` treats `iterrows()` as the headline anti-pattern and supplies `shift()`/lag patterns (line 830) for the previous-row case. Currency is strong for a fast-moving toolchain (uv-first, ty/pyrefly, PEP 735). **Deductions:** the type sheet pins ty/pyrefly "as of 2025 … pre-1.0" and frames Python 3.13 as "Oct 2024" experimental (lines 608-654) — by 2026-06 this reads slightly behind the frontier (3.13 GA, 3.14 emerging; the Rust checkers have matured). Minor not material — guidance still correct, just dated framing. |
| **B — Usefulness** | A | Router is a model symptom→specialist matcher: per-symptom routing tables, a "Common Routing Mistakes" table (SKILL.md:316-326), ordered cross-cutting sequences (profile→vectorize, setup→delint), and an "Ambiguous Queries — Ask First" block. Sheets are example-dense with ❌/✅ WRONG/CORRECT pairs and decision trees (e.g. `modern-syntax-and-types.md:846-872`). Reading it changes what you do. |
| **C — Discipline** | A | Strong rationalization-resistance: router "Common Rationalizations" table (SKILL.md:343-354 — "User is rushed, skip routing → Route anyway") and a 6-item Red-Flags self-check. Every sheet has an explicit anti-pattern register. Both SME agents (`python-code-reviewer`, `refactoring-architect`) cite `meta-sme-protocol:sme-agent-protocol` and require Confidence/Risk/Information-Gaps/Caveats; `refactoring-architect.md:19` holds the line ("a refactor that 'probably' preserves behavior is a rewrite in disguise"). `delinting-specialist` correctly non-SME on haiku. Three negative-activation examples on the refactoring agent. |
| **D — Form** | A | Frontmatter conformant; commands use quoted `allowed-tools`/`argument-hint`; agents declare `description`+`model`. Slash wrapper `.claude/commands/python-engineering.md` is **current** — thin pointer, all 10 sheets incl. textual, correct cross-refs, advertises ty/pyrefly/uv. Router `description` now follows "Use when…" idiom. Registered in marketplace. Counts (10/4/3) consistent across plugin.json, wrapper, marketplace, filesystem. **Nits:** SKILL.md:352 still says "8 focused skills" (should be 10); SKILL.md:436-443 "Future cross-references / Phase 1 — Standalone" framing is stale (superpowers TDD/debugging exist today); reference sheets begin with a stray blank line. Cosmetic only. |

---

## Gate analysis

1. **Discoverability gate:** Installs, registered, slash wrapper present and current → no cap. PASS.
2. **Substance-dominates:** Substance = A− → overall ≤ A. Not binding below A.
3. **Honor-roll (S):** Substance is A− (currency framing + "8 skills" nit), so S is unreachable. Correctly excluded.
4. **Honesty override:** N/A — fully built, no scaffold/vapor; marketing matches delivered content.

---

## Layered per-component grades

The pack is uniformly healthy; no weak tail drags it down. Notable points only:

| Component | Grade | Note |
|-----------|-------|------|
| `using-python-engineering/SKILL.md` (router) | A | Exemplary symptom routing + rationalization table; docked one notch by stale "8 focused skills" (line 352) and "Future cross-references / Phase 1" framing (436-443). |
| `modern-syntax-and-types.md` | A− | Correct and deep, but ty/pyrefly/3.13 currency framing reads ~12 months behind as of 2026-06 (lines 608-654). |
| `refactoring-architect.md` | **A (exemplar)** | Best component to copy: SME protocol, behavior-preservation-as-precondition, characterization-tests-first, and three negative-activation examples. Reference-grade agent discipline. |
| `scientific-computing-foundations.md` | A− | Anti-iterrows stance is unambiguous and `shift()`/lag patterns exist; the "I depend on the previous row" rationalization could be called out by name as the standard counter. Minor. |

No D/F components.

---

## Overall: **A−**

Blend: Substance A− (40), Usefulness A (25), Discipline A (20), Form A (15) → A; pulled to **A−** by the Substance currency drag (the gate caps at A anyway). Reconciles with the prior review's "Minor" verdict, upgraded because both of that review's Majors are now fixed.

**Verdict:** A production-ready, exemplary-router Python pack whose only real weakness is type-checker/version currency framing drifting ~a year behind.

**Top finding:** Substance is excellent but the type-tooling currency is dated — `modern-syntax-and-types.md` still pins ty/pyrefly as "pre-1.0 (2025)" and Python 3.13 as new/experimental, which is stale as of 2026-06.

**Top fix:** Refresh the "Emerging Type Checkers" and "Python 3.13 Notes" sections in `modern-syntax-and-types.md` to current ty/pyrefly maturity and the 3.13-GA / 3.14 landscape; while there, correct the "8 focused skills" count (SKILL.md:352) and un-future-ify the cross-references block (SKILL.md:436-443).
