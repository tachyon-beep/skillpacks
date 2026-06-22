# Report Card — axiom-engineering-foundations

**Version:** 1.1.0 · **Track:** P (Process / Hybrid) · **Graded:** 2026-06-22

Methodology-only pack: router + 6 reference sheets, no commands, no agents (honestly declared as such). Slash wrapper present and current at `.claude/commands/software-engineering.md`. Registered in `.claude-plugin/marketplace.json` lines 109–111.

**Note on prior evidence:** `reviews/axiom-engineering-foundations.md` (2026-05-22, v1.0.1) is STALE. Its one Major finding — missing slash-command wrapper — has been resolved (wrapper now exists, lists all 6 sheets, includes the `superpowers:*` cross-refs the old review flagged as a Minor gap). The version bump to 1.1.0 tracks this fix. This card weights the fresh reading.

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---------|-------|-----------------------|
| **A — Substance** (P lens) | **A** | Valid, maturity-appropriate methodology at expert depth, complete across the declared 6-topic domain. `complex-debugging.md`: scientific method (Phase 0 static → reproduce → isolate via space/time binary search + git bisect + wolf-fence + Saff squeeze → falsifying-hypothesis loop → 5 Whys → test-driven fix → prevention), correlation-ID distributed tracing (l.473–504). `systematic-refactoring.md`: characterize-then-small-steps with revert-on-fail (l.231–243), strangler-fig / parallel-change / branch-by-abstraction (l.274–318). `technical-debt-triage.md`: Fowler reckless/prudent × deliberate/inadvertent quadrant (l.102–108), ROI = impact/cost scoring, prioritization matrix. `incident-response.md`: restore-first, severity matrix, blameless postmortem. `code-review-methodology.md`: context→high-level→correctness→quality ordering (l.45). No rot, nothing wrong. Ceiling: a focused methodology pack, not encyclopedic — but fully covers what it claims. |
| **B — Usefulness** | **A** | Router (`SKILL.md`) routes crisply: symptom tables per situation, "Common Routing Mistakes" table (l.211–218), "Ambiguous Queries — Ask First" interception (l.192–205), cross-cutting scenario playbooks (l.165–189), cross-plugin handoff tables (l.234–249). Every sheet ends in a Quick Reference checklist + decision tables + concrete code. Reading it changes what you do. |
| **C — Discipline** | **A** | Every sheet carries a "Red Flags" rationalization table with verbatim thoughts and counters — debugging: "Let me just try this…" / "I'll add a null check" (l.707–716); refactoring: "I'll write tests after refactoring" / "I know this won't break anything" (l.343–353); incident: "Let me understand the code first" → users suffering NOW (l.400–410); debt: "We'll fix it later" / "No time for debt" (l.363–372). Plus Heisenbug protocol (assertions over logging, l.516–568), Investigation Reset Protocol naming "I've tried everything" as a diagnostic (l.570–617), cognitive-bias table (l.443–449), blameless postmortem. Unusually robust pressure-resistance. No SME protocol needed — no agents, honestly methodology-only. |
| **D — Form** | **A−** | Router frontmatter conformant (`name` + `description`, "Use when…"). Reference sheets correctly frontmatter-free. Slash wrapper present, current, authority-pointer pattern correct. Marketplace registered. Intra-pack cross-refs all resolve; cross-plugin refs verified in prior review. Only nit: marketplace description (l.111) calls them "6 language-agnostic skills" when they are reference sheets, not skills — cosmetic, repo-wide convention drift. |

---

## Gate analysis

1. **Discoverability ceiling:** PASS — installs, router loads, slash wrapper present and current, registered, not a scaffold. No cap.
2. **Substance-dominates:** Substance = A → overall ≤ A+. Not binding (overall lands at A).
3. **Honor-roll (S):** Not met — S requires Substance = S. Substance is A (excellent, complete, but a focused 6-sheet methodology pack rather than reference-grade-authoritative across a sprawling domain). No subject below A; zero Major+ defects.
4. **Honesty override:** N/A — fully delivered, "Commands: None / Agents: None" stated plainly and truthfully.

---

## Layered per-component grades

The pack is uniformly strong; no weak tail drags it down.

| Component | Grade | Note |
|-----------|-------|------|
| `complex-debugging.md` | **A / S−** | Exemplar. Heisenbug protocol, Investigation Reset Protocol, cognitive-bias + red-flag tables, correlation-ID tracing. The sheet other process packs should copy for pressure-resistance signature. |
| `using-software-engineering/SKILL.md` (router) | **A** | Symptom routing + routing-mistakes table + ask-first interception + cross-plugin handoffs. Model router. |
| `incident-response.md` | **A** | Complete restore-first methodology, severity/comms/handoff templates, blameless postmortem. |
| `systematic-refactoring.md` | **A** | Characterize-first, revert-on-fail, three large-refactor patterns, strong red-flags table. |
| `technical-debt-triage.md` | **A−** | Fowler quadrant + ROI scoring + stakeholder-pitch framing; the numeric ROI scoring is slightly mechanical but defensible. |
| `code-review-methodology.md` | **A−** | Solid ordering + comment taxonomy + receiving-feedback discipline; the lightest sheet but still complete. |

**S-exemplar worth copying:** `complex-debugging.md` — its Heisenbug + Investigation Reset rationalization handling is best-in-marketplace for a process pack.

---

## Overall: **A**

**Verdict:** A disciplined, complete, well-wired methodology pack whose rationalization-resistance is best-in-class for the P track; the stale review's one Major (missing wrapper) is now fixed.

**Top finding:** Prior Major resolved — slash wrapper present and current at `.claude/commands/software-engineering.md`, including the previously-missing `superpowers:*` cross-refs; pack is fully discoverable. Content is uniformly A-grade with `complex-debugging.md` reaching exemplar quality.

**Top fix:** Cosmetic only — align the marketplace description (`.claude-plugin/marketplace.json` l.111) to call the deliverables "reference sheets" rather than "6 … skills" to match the router/plugin.json framing.
