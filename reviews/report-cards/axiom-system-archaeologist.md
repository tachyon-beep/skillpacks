# Report Card — axiom-system-archaeologist

**Version:** 1.6.2  **Track:** P (Process / Hybrid)  **Graded:** 2026-06-22
**Unit:** pack (router + 16 sheets, 5 commands, 4 agents)

Codebase-archaeology methodology pack: subagent-coordinated exploration that produces
architecture documentation (subsystem catalogs, C4 diagrams, quality/security/test/dependency
analysis, incremental delta, and an ultralarge-tier per-module track) under mandatory workspace
+ validation discipline.

---

## Subject grades

| Subject | Grade | Load-bearing evidence |
|---|---|---|
| **A — Substance** (P-track) | **A** | Methodology valid and maturity-appropriate across the whole declared domain. 16 sheets covering discovery → catalog → diagrams → quality → security → test → deps → incremental → ultralarge. Expert depth, not tutorial: `generating-architecture-diagrams.md` lines 29-77 define a mandatory Data Quality Gate (refuse if >30% subsystems Low-confidence, any critical-path subsystem Low, >50% deps Unknown) with confidence-based inclusion rules; `analyzing-unknown-codebases.md` pins an exact 8-field subsystem contract and forbids extra fields; `findings-schema.md` (469 LOC) is a load-bearing YAML schema with a "Why This Schema, Not A Different One" rationale. Stable domain (C4, dependency analysis) so no version-rot risk. Not S only because the bulk is well-executed application of mature technique rather than authoritative novelty across the full span. |
| **B — Usefulness** | **A** | Router (`SKILL.md`) routes crisply: Step-4 sequential/parallel/ultralarge decision tree keyed on concrete thresholds (LOC >100K, >12 subsystems, test≥source); Step 1.5 deliverable menu (A–G); explicit "Don't use" boundaries to `/solution-architect`, `/system-architect`, `/security-architect`, language packs (wrapper lines 20, 60-64). Reading it changes what you do — every phase names its output document and validation gate. |
| **C — Discipline** | **A** | Near-S. Verbatim rationalization-blocker tables throughout (SKILL.md lines 47-53, 230-239, 345-355) naming "I'll just quickly read this file", "We have 45 minutes, no time for validation", etc., each with a rebuttal. All 4 agents declare SME Agent Protocol in description + body. Defense-in-depth YAML self-validation (reviewer self-parse + scribe re-parse) grounded in cited empirical calibration ("reviewers can produce invalid YAML while reporting 'self-check pass'") — the v1.6.0→v1.6.1 hardening. Validator independence clause is non-negotiable with a 2-retry cap. |
| **D — Form** | **A−** | Conformant and fully wired. Slash wrapper `.claude/commands/system-archaeologist.md` is now a 64-line thin redirect, current and consistent with the router (resolves prior review's M1/M2 wrapper-drift Majors). Marketplace registered; "5 commands, 4 agents" count consistent across plugin.json, marketplace.json, wrapper. Commands use quoted-array `allowed-tools`; agents follow the omit-`tools` convention. One cosmetic nit only (P3, below). |

---

## Gate analysis

1. **Discoverability gate:** PASS. Installs, registered, router auto-invokes on "Use when…" description; slash wrapper present and current. No ceiling.
2. **Substance-dominates gate:** Substance = A → overall ≤ S. Not binding.
3. **Honor-roll (S) gate:** NOT met — requires Substance = S. Substance is A (mature-technique application, not domain-authoritative novelty). So overall caps below S.
4. **Honesty override:** N/A — complete pack, no scaffold; marketing matches reality.

No subject below A; zero Major+ defects. Overall lands at the top of A.

---

## Layered per-component grades

The pack is uniformly strong; no weak tail drags it down. Surfaced items:

| Component | Grade | Note |
|---|---|---|
| `SKILL.md` lines 74-78 | **A−** | **P3 (cosmetic):** path-disambiguation example cites `subsystem-discovery.md`, a file that does not exist (the analogous sheet is `analyzing-unknown-codebases.md`). Stale example from an earlier naming scheme. Only defect found. |
| `findings-schema.md` | **S (exemplar)** | Worth copying: load-bearing YAML schema with explicit "Why This Schema, Not A Different One" rationale (justifies YAML-over-JSON, per-focus partials, per-focus confidence, provenance block, no free-text notes) plus a self-validation checklist tied to an empirical ~6% failure rate. Reference-grade design-rationale prose. |
| `generating-architecture-diagrams.md` | **A** | Data Quality Gate with hard refuse-thresholds and "diagrams carry authority" framing — discipline rendered as concrete numeric gates. |
| 4 agents (explorer/validator/reviewer/scribe) | **A** | All SME-compliant; defense-in-depth validation; MIN-aggregation of confidence in the scribe. |

---

## Overall: **A**

### Verdict
Reference-quality process pack — disciplined, fully wired, current; the prior review's wrapper-drift Majors are resolved, leaving only one stale path example.

### Top finding
Wrapper-vs-router drift flagged by the 2026-05-22 review (M1/M2) is fixed: the slash wrapper is now a thin 64-line redirect consistent with the router. The sole remaining defect is the stale `subsystem-discovery.md` example at SKILL.md lines 74-78.

### Top fix
Replace `subsystem-discovery.md` at `SKILL.md` lines 74-78 with a real filename (e.g. `analyzing-unknown-codebases.md`), and consider promoting `findings-schema.md`'s "Why this, not that" rationale convention pack-wide.
