# Changelog — axiom-procedural-architecture

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
