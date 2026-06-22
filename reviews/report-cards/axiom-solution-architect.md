# Report Card — axiom-solution-architect

**Version:** 1.0.1
**Track:** P (Process / Hybrid) — methodology validity (forward solution-architecture workflow: triage → NFR → tech-selection → ADR → RTM → integration/migration → TOGAF/ArchiMate → SAD + consistency gate), borrowing the H lens where sheets touch concrete artifacts (C4, OpenAPI, ArchiMate 3.2).
**Structure:** 1 router + 8 reference sheets, 2 commands, 2 agents. ~3,500 lines total.
**Prior review:** `reviews/axiom-solution-architect.md` (2026-05-22) at the same version (1.0.1) reached "Pass with Minor." Not stale; my fresh reading concurs and independently reproduces its two flagged defects.

## Subject grades

| Subject | Grade | Load-bearing evidence |
|---------|-------|------------------------|
| **A — Substance** | **A** | Methodology is valid, complete against the declared domain, and current. NFR sheet (`quantifying-nfrs.md`) separates VER/VAL evidence modes per category, names measurement *environment* (prod / prod-shadow / staging rig vary 3–10×, line 116), and enforces a latency-budget sum check (lines 149–152). TOGAF/ArchiMate sheet is genuinely expert and current: ArchiMate 3.2 layer/aspect taxonomy (line 42), the 2.x→3.x `UsedBy`→`Serving` fold (line 90), ABB-vs-SBB distinction (line 301), ISO 42010 viewpoint/view distinction (line 100), `DataEntity`→`DataObject` metamodel bridge (line 63). ADR sheet cites Nygard (2011) + MADR provenance (line 13) and covers full lifecycle (supersession/deprecation/partial/amendment + bidirectional linking). No rot found. Held back from S only by minor depth gaps (e.g., interface-contracts and data-model live as router-owned quality-floor guidance rather than dedicated sheets). |
| **B — Usefulness** | **A** | Router (`SKILL.md`) routes crisply: Start-Here sequence, Scope-Tier table (XS→XL with per-tier required artifacts, lines 92–98), enterprise activation as four hard gates not keyword presence (lines 247–265), Stop Conditions table (lines 271–277), Update Workflows re-run/re-gate matrix (lines 285–294). Every sheet emits concrete templates with worked examples (the Kafka walk-back in `resisting-tech-and-scope-creep.md` lines 174–209 is a model of constraints-first reasoning). The assembling sheet's ceremonial-vs-substantive gate-report contrast (lines 153–181) directly changes what the executor does. |
| **C — Discipline** | **A** | Every sheet carries verbatim Pressure Responses ("It just needs to be fast", "We've already decided on Kubernetes", "The gate is too strict") and Anti-Patterns-to-Reject sections. Both agents declare `model: opus`, cite `meta-sme-protocol:sme-agent-protocol`, and mandate Confidence/Risk/Information-Gaps/Caveats — fully realized in `tech-selection-critic.md` (lines 128–142). The consistency gate has an explicit anti-rubber-stamp sampling protocol (`assembling-...md` lines 114–127). Marketing matches reality. |
| **D — Form** | **B** | Conformant frontmatter on commands (description/allowed-tools/argument-hint) and agents (description/model). Slash wrapper `.claude/commands/solution-architect.md` present, current (8 sheets named, correct cross-refs, content-authority note), and registered in marketplace (line 65). One Minor consistency drift: failure-mode count — router `SKILL.md:173` says "ten canonical failure modes" while both commands and the reviewer agent say/walk **eleven** (`solution-design-reviewer.md:81`). Plugin.json "9 skills" wording (1 router + 8 sheets) is loose-but-defensible. |

## Gate analysis

1. **Discoverability (ceiling):** Pack loads; slash wrapper present and current; registered and installable. No cap. PASS.
2. **Substance-dominates:** Substance = A → overall ≤ A+1 tier (S). Not binding.
3. **Honor-roll (S):** Requires Substance = S, no subject below A, zero Major+. Substance is A (not S) and Form is B → **S denied.**
4. **Honesty override:** N/A — fully realized pack, no scaffold claims.

## Layered per-component grades

The pack is uniformly strong; only the weak tail and one exemplar are surfaced.

| Component | Grade | Note |
|-----------|-------|------|
| `mapping-to-togaf-archimate.md` (Substance exemplar) | **A** | Currency benchmark for the marketplace: ArchiMate 3.2-accurate, names the 2.x→3.x relationship fold and the standard 23-viewpoint catalogue, distinguishes ABB/SBB. Worth copying as the template for "how to keep a framework sheet from rotting." |
| `writing-rigorous-adrs.md` (Discipline exemplar) | **A** | Constrained-decision escape with explicit anti-laundering rules, bidirectional supersession integrity, cost-as-first-class-driver, and a clean field-map handoff to the sibling SDLC pack. |
| `using-solution-architect/SKILL.md` (Form) | **B** | The single drift source: line 173 "ten canonical failure modes" contradicts the eleven everywhere else. Also: the reviewer agent's "failure mode check 1 (file presence)" reference (`solution-design-reviewer.md:67`) does not align cleanly with the 11-mode list that starts at "tech-before-problem." |

No sheet grades below B. No C/D/F components.

## Overall: **A**

## Verdict
A disciplined, current, reference-leaning forward-architecture pack; one cosmetic count-drift (ten vs eleven failure modes) is the only thing between it and an A+.

## Top finding
Failure-mode count is inconsistent across surfaces: router `SKILL.md:173` says "ten canonical failure modes" while both commands and the `solution-design-reviewer` agent define and walk **eleven** — a Minor Form drift that a reader hits immediately when reconciling the router against the review command.

## Top fix
Change `SKILL.md:173` from "ten canonical failure modes" to "eleven", and reconcile the reviewer agent's "failure mode check 1 (file presence)" wording (`solution-design-reviewer.md:67`) with its numbered 11-mode list so file-presence is unambiguously a mode or an explicit pre-check.
