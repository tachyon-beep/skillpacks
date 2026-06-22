# Changelog — axiom-procedural-architecture

## v0.2.1 (2026-06-22)

Polish fix from report-card review (`reviews/report-cards/axiom-procedural-architecture.md`):

- **Mi1 closed.** Corrected the self-description on both discovery surfaces from "Two roles (producer/critic)" to "Three roles (producer/critic/analyst)". The analyst role (analyst cluster sheets 9–12, `/analyze-procedure`) has shipped since v0.1.x but was invisible on the marketing surface. `plugin.json` updated in-pack; the matching `marketplace.json` catalog description change is requested separately (catalog file is owned outside this pack).

## v0.2.0 (2026-05-22)

Discoverability fix from external review (`reviews/axiom-procedural-architecture.md`):

- **Major (M1) closed.** Added the marketplace-standard slash-command wrapper at `.claude/commands/procedural-architecture.md`. The pack is now invokable as `/procedural-architecture` consistent with every other `using-X`-bearing pack in the marketplace and with the convention mandated in `CLAUDE.md`. Wrapper mirrors the sibling-pack style (sheets list, commands list, agents list, cross-references) and is a thin pointer to the router skill — content authority remains in `plugins/axiom-procedural-architecture/skills/using-procedural-architecture/SKILL.md`.
- Bumped plugin version 0.1.1 → 0.2.0. Promotion from v0.1.x reflects (a) the external-review pass closing the single Major issue and (b) the substantive cleanup already shipped in v0.1.1 (symmetric anti-failure protocols, +2 boundary smells, inbound section, expects-a-critic-finding) which the review confirmed reads past the v0.1 scaffold milestone. Marketplace catalog version is owned separately and not touched here.
- Minor/Polish findings (Mi1, Mi2, P1, P3) from the same review are deferred: marketplace description divergence (Mi1), marketplace keyword-list reconciliation (Mi2), worked multi-audience example (P1), and worked end-to-end `/analyze-procedure` example for the workflow-nets or DES route (P3). None block the major-fix-only scope of this bump.

## v0.1.1 (2026-05-12)

Cleanup pass — six minor fixes flagged during v0.1.0 build:

- Removed HTML-comment test-grep workarounds from the three commands (`decompose-procedure`, `review-decomposition`, `analyze-procedure`). The comments were inserted to satisfy a buggy `grep -qE "X\|Y"` task-test pattern; they were invisible in rendered markdown and vestigial.
- Boundary sheet: added Smell 6 (Site-IA Colonisation → `lyra-site-designer`) and Smell 7 (Emergent-Flow Colonisation → `bravos-simulation-tactics`). Closes the gap where two adjacent territories were named in the handoff list but absent from the smell catalog.
- Boundary sheet: added "Inbound Relationships" section documenting `axiom-system-archaeologist` as an upstream source. The relationship was named in SKILL.md but undocumented on the boundary sheet — asymmetry fixed.
- Producer agent (`decomposition-architect`): tightened Anti-Overconfidence into a 5-point pre-emit protocol matching the critic's Anti-Rubber-Stamp Protocol muscularity. The producer/critic disagreement promise is now encoded symmetrically.
- Producer agent: output contract now includes an explicit Closing Recommendation that recommends running `decomposition-critic` and expects at least one substantive finding. A finding-free critic audit is now treated as evidence of producer over-commitment rather than producer quality.
- Bumped plugin version 0.1.0 → 0.1.1. No marketplace version bump (catalog unchanged).

## v0.1.0 (2026-05-12)

Initial scaffold. Router + 13 reference sheets (4 producer / 4 critic / 4 analyst / 1 boundary), 3 commands (/decompose-procedure, /review-decomposition, /analyze-procedure), 2 agents (decomposition-architect, decomposition-critic). Structurally validated via integration smoke test; awaiting external feedback before promotion to v0.2.

### Reference sheets

Producer cluster:
- decomposition-fundamentals.md
- decision-flow-design.md
- granularity-calibration.md
- audience-modeling-for-procedures.md

Critic cluster:
- dependency-and-ordering-audit.md
- branching-and-mece-review.md
- decomposition-smells.md
- procedural-invariants-and-correctness.md

Analyst cluster:
- queueing-theory-for-procedures.md
- discrete-event-simulation-for-procedures.md
- process-algebra-and-workflow-nets.md
- flow-vs-state-vs-decision-modeling.md

Boundary:
- procedural-boundary-and-handoffs.md
